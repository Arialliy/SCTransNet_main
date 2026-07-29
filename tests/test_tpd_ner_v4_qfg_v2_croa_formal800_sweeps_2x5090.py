from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO
    / "experiments/"
    "run_tpd_ner_v4_qfg_v2_croa_formal800_sweeps_2x5090.sh"
)
WORKER_NAME = "run_tpd_ner_v4_qfg_v2_croa_formal800_eval_lane.sh"
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"


def source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def fake_environment(
    tmp_path: Path,
) -> tuple[dict[str, str], Path, Path]:
    fake_repo = tmp_path / "repo"
    experiments = fake_repo / "experiments"
    result_root = fake_repo / "results"
    state_dir = fake_repo / "state"
    experiments.mkdir(parents=True)
    result_root.mkdir()
    state_dir.mkdir()
    source_lock = experiments / (
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
    )
    source_lock.write_text("{}\n", encoding="utf-8")
    worker = experiments / WORKER_NAME
    worker.write_text(
        """#!/usr/bin/env bash
set -eu
phase=run
if [[ "${1:-}" == "--preflight" ]]; then
    phase=preflight
    shift
fi
variant="$1"
checkpoint="$2"
physical_gpu="$3"
gpu_uuid="$4"
printf '%s %s %s %s %s\\n' \
    "$phase" "$variant" "$checkpoint" "$physical_gpu" "$gpu_uuid" \
    >>"$FAKE_CALL_LOG"
if [[ "$phase" == "preflight" \
    && "${FAKE_FAIL_PREFLIGHT:-}" == "$variant:$checkpoint" ]]; then
    exit 74
fi
if [[ "$phase" == "run" \
    && "${FAKE_FAIL_RUN:-}" == "$variant:$checkpoint" ]]; then
    exit 73
fi
if [[ "$phase" == "run" && "$checkpoint" == "best.pth.tar" ]]; then
    : >"$FAKE_STATE_DIR/$variant.best_done"
fi
if [[ "$phase" == "run" \
    && "$checkpoint" == "best_miou.pth.tar" \
    && ! -f "$FAKE_STATE_DIR/$variant.best_done" ]]; then
    exit 72
fi
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    call_log = fake_repo / "calls.log"
    environment = dict(os.environ)
    environment.update(
        {
            "TPD_NER_V4_QFG_V2_CROA_REPO": str(fake_repo),
            "TPD_NER_V4_QFG_V2_CROA_PYTHON": sys.executable,
            "TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK": str(source_lock),
            "TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT": str(result_root),
            "FAKE_CALL_LOG": str(call_log),
            "FAKE_STATE_DIR": str(state_dir),
        }
    )
    return environment, call_log, state_dir


def test_shell_source_parses_and_runner_is_executable() -> None:
    subprocess.run(
        ["/usr/bin/bash", "-n", str(RUNNER)],
        cwd=REPO,
        check=True,
    )
    assert RUNNER.stat().st_mode & 0o111


def test_defaults_are_optimized_v2_and_worker_is_single_checkpoint() -> None:
    text = source()
    assert WORKER_NAME in text
    assert "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized" in text
    assert (
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
        in text
    )
    assert "--all-four" not in text
    assert "--overwrite" not in text


def test_all_four_preflights_precede_any_output_lane() -> None:
    text = source()
    preflight_body = text[
        text.index("qfg_sweeps_preflight_all()")
        : text.index("qfg_sweeps_run_lane()")
    ]
    for call in (
        'qfg_only best.pth.tar 2 "$qfg_sweeps_gpu2_uuid"',
        'qfg_only best_miou.pth.tar 2 "$qfg_sweeps_gpu2_uuid"',
        'tss_qfg best.pth.tar 3 "$qfg_sweeps_gpu3_uuid"',
        'tss_qfg best_miou.pth.tar 3 "$qfg_sweeps_gpu3_uuid"',
    ):
        assert call in preflight_body
    preflight_call = text.index("qfg_sweeps_preflight_all\n")
    lock_creation = text.index('mkdir -p "$qfg_sweeps_lock_dir"')
    first_background = text.index(") &", lock_creation)
    assert preflight_call < lock_creation < first_background
    assert "outputs_started=false" in preflight_body


def test_each_fixed_gpu_lane_is_sequential_and_lanes_are_parallel() -> None:
    text = source()
    lane_body = text[
        text.index("qfg_sweeps_run_lane()")
        : text.index("qfg_sweeps_preflight_all\n")
    ]
    assert lane_body.index("best.pth.tar") < lane_body.index(
        "best_miou.pth.tar"
    )
    assert (
        'qfg_sweeps_run_lane \\\n        qfg_only 2 "$qfg_sweeps_gpu2_uuid"'
        in text
    )
    assert (
        'qfg_sweeps_run_lane \\\n        tss_qfg 3 "$qfg_sweeps_gpu3_uuid"'
        in text
    )
    assert text.count(") &") == 2
    assert 'wait "$qfg_sweeps_qfg_pid"' in text
    assert 'wait "$qfg_sweeps_tss_pid"' in text
    assert "TPDNERV4QFG_SWEEPS_FAILED" in text


def test_preflight_mode_calls_exactly_four_read_only_workers(
    tmp_path: Path,
) -> None:
    environment, call_log, state_dir = fake_environment(tmp_path)
    result = subprocess.run(
        [str(RUNNER), "--preflight"],
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TPDNERV4QFG_SWEEPS_PREFLIGHT_ONLY" in result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls == [
        f"preflight qfg_only best.pth.tar 2 {GPU2_UUID}",
        f"preflight qfg_only best_miou.pth.tar 2 {GPU2_UUID}",
        f"preflight tss_qfg best.pth.tar 3 {GPU3_UUID}",
        f"preflight tss_qfg best_miou.pth.tar 3 {GPU3_UUID}",
    ]
    assert list(state_dir.iterdir()) == []


def test_failed_preflight_starts_no_output_worker(tmp_path: Path) -> None:
    environment, call_log, state_dir = fake_environment(tmp_path)
    environment["FAKE_FAIL_PREFLIGHT"] = "tss_qfg:best.pth.tar"
    result = subprocess.run(
        [str(RUNNER)],
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "outputs_started=false" in result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls
    assert all(line.startswith("preflight ") for line in calls)
    assert list(state_dir.iterdir()) == []


def test_successful_run_and_idempotent_rerun_keep_lane_order(
    tmp_path: Path,
) -> None:
    environment, call_log, _ = fake_environment(tmp_path)
    for _attempt in range(2):
        result = subprocess.run(
            [str(RUNNER)],
            cwd=REPO,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "TPDNERV4QFG_SWEEPS_COMPLETE" in result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    run_calls = [line for line in calls if line.startswith("run ")]
    assert len(run_calls) == 8
    for variant, physical_gpu, gpu_uuid in (
        ("qfg_only", 2, GPU2_UUID),
        ("tss_qfg", 3, GPU3_UUID),
    ):
        variant_calls = [
            line for line in run_calls if line.split()[1] == variant
        ]
        assert variant_calls == [
            f"run {variant} best.pth.tar {physical_gpu} {gpu_uuid}",
            f"run {variant} best_miou.pth.tar {physical_gpu} {gpu_uuid}",
            f"run {variant} best.pth.tar {physical_gpu} {gpu_uuid}",
            f"run {variant} best_miou.pth.tar {physical_gpu} {gpu_uuid}",
        ]


def test_nonzero_lane_status_fails_after_both_lanes_are_reaped(
    tmp_path: Path,
) -> None:
    environment, call_log, _ = fake_environment(tmp_path)
    environment["FAKE_FAIL_RUN"] = "qfg_only:best.pth.tar"
    result = subprocess.run(
        [str(RUNNER)],
        cwd=REPO,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "TPDNERV4QFG_SWEEPS_FAILED" in result.stderr
    run_calls = [
        line
        for line in call_log.read_text(encoding="utf-8").splitlines()
        if line.startswith("run ")
    ]
    assert f"run tss_qfg best.pth.tar 3 {GPU3_UUID}" in run_calls
    assert f"run tss_qfg best_miou.pth.tar 3 {GPU3_UUID}" in run_calls
    assert not any(
        line.startswith("run qfg_only best_miou.pth.tar")
        for line in run_calls
    )


def test_no_gpu_zero_one_or_device_availability_polling() -> None:
    text = source()
    assert "physical_gpu=0" not in text
    assert "physical_gpu=1" not in text
    assert "qfg_only 0 " not in text
    assert "qfg_only 1 " not in text
    assert "tss_qfg 0 " not in text
    assert "tss_qfg 1 " not in text
    assert "nvidia-smi" not in text
    assert "query_compute_processes" not in text
    assert "assigned_gpu_busy" not in text
    assert "sleep " not in text
    assert "flock -n 7" in text
