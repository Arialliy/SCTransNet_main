from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = (
    REPO
    / "experiments/"
    "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_sweeps_2x5090.sh"
)
WORKER_NAME = (
    "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
)
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
    qfg_root = result_root / "qfg_dlr_lane"
    tss_root = result_root / "tss_qfg_dlr_lane"
    state_dir = fake_repo / "state"
    experiments.mkdir(parents=True)
    qfg_root.mkdir(parents=True)
    tss_root.mkdir(parents=True)
    state_dir.mkdir()
    source_lock = experiments / (
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
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
            "TPD_NER_DLR_RAMP100_REPO": str(fake_repo),
            "TPD_NER_DLR_RAMP100_PYTHON": str(
                Path(sys.executable).resolve()
            ),
            "TPD_NER_DLR_RAMP100_SOURCE_LOCK": str(source_lock),
            "TPD_NER_DLR_RAMP100_RESULT_ROOT": str(result_root),
            "TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT": str(qfg_root),
            "TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT": str(tss_root),
            "FAKE_CALL_LOG": str(call_log),
            "FAKE_STATE_DIR": str(state_dir),
        }
    )
    return environment, call_log, state_dir


def test_shell_source_parses_and_runner_is_executable() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], cwd=REPO, check=True)
    assert RUNNER.stat().st_mode & 0o111


def test_all_four_read_only_preflights_precede_output_lanes() -> None:
    text = source()
    preflight_body = text[
        text.index("ramp_sweeps_preflight_all()")
        : text.index("ramp_sweeps_run_lane()")
    ]
    for call in (
        'qfg_dlr best.pth.tar 2 "$ramp_sweeps_gpu2_uuid"',
        'qfg_dlr best_miou.pth.tar 2 "$ramp_sweeps_gpu2_uuid"',
        'tss_qfg_dlr best.pth.tar 3 "$ramp_sweeps_gpu3_uuid"',
        'tss_qfg_dlr best_miou.pth.tar 3 "$ramp_sweeps_gpu3_uuid"',
    ):
        assert call in preflight_body
    preflight_call = text.index("ramp_sweeps_preflight_all\n")
    lock_creation = text.index('mkdir -p "$ramp_sweeps_lock_dir"')
    first_background = text.index(") &", lock_creation)
    assert preflight_call < lock_creation < first_background
    assert "outputs_started=false" in preflight_body


def test_fixed_gpu_lanes_are_parallel_and_checkpoints_sequential() -> None:
    text = source()
    lane_body = text[
        text.index("ramp_sweeps_run_lane()")
        : text.index("ramp_sweeps_preflight_all\n")
    ]
    assert lane_body.index("best.pth.tar") < lane_body.index(
        "best_miou.pth.tar"
    )
    assert (
        'ramp_sweeps_run_lane \\\n'
        '        qfg_dlr 2 "$ramp_sweeps_gpu2_uuid"'
    ) in text
    assert (
        'ramp_sweeps_run_lane \\\n'
        '        tss_qfg_dlr 3 "$ramp_sweeps_gpu3_uuid"'
    ) in text
    assert text.count(") &") == 2
    assert 'wait "$ramp_sweeps_qfg_pid"' in text
    assert 'wait "$ramp_sweeps_tss_pid"' in text


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
    assert "TPDNER_DLR_RAMP100_SWEEPS_PREFLIGHT_ONLY" in result.stdout
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        f"preflight qfg_dlr best.pth.tar 2 {GPU2_UUID}",
        f"preflight qfg_dlr best_miou.pth.tar 2 {GPU2_UUID}",
        f"preflight tss_qfg_dlr best.pth.tar 3 {GPU3_UUID}",
        f"preflight tss_qfg_dlr best_miou.pth.tar 3 {GPU3_UUID}",
    ]
    assert list(state_dir.iterdir()) == []


def test_failed_preflight_starts_no_output_worker(tmp_path: Path) -> None:
    environment, call_log, state_dir = fake_environment(tmp_path)
    environment["FAKE_FAIL_PREFLIGHT"] = "tss_qfg_dlr:best.pth.tar"
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
    assert all(call.startswith("preflight ") for call in calls)
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
        assert "TPDNER_DLR_RAMP100_SWEEPS_COMPLETE" in result.stdout
    calls = call_log.read_text(encoding="utf-8").splitlines()
    run_calls = [call for call in calls if call.startswith("run ")]
    assert len(run_calls) == 8
    for variant, physical_gpu, gpu_uuid in (
        ("qfg_dlr", 2, GPU2_UUID),
        ("tss_qfg_dlr", 3, GPU3_UUID),
    ):
        variant_calls = [
            call for call in run_calls if call.split()[1] == variant
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
    environment["FAKE_FAIL_RUN"] = "qfg_dlr:best.pth.tar"
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
    assert "TPDNER_DLR_RAMP100_SWEEPS_FAILED" in result.stderr
    run_calls = [
        call
        for call in call_log.read_text(encoding="utf-8").splitlines()
        if call.startswith("run ")
    ]
    assert f"run tss_qfg_dlr best.pth.tar 3 {GPU3_UUID}" in run_calls
    assert f"run tss_qfg_dlr best_miou.pth.tar 3 {GPU3_UUID}" in run_calls
    assert not any(
        call.startswith("run qfg_dlr best_miou.pth.tar")
        for call in run_calls
    )


def test_no_gpu_zero_one_polling_or_wait_for_idle() -> None:
    text = source()
    assert "qfg_dlr 0 " not in text
    assert "qfg_dlr 1 " not in text
    assert "tss_qfg_dlr 0 " not in text
    assert "tss_qfg_dlr 1 " not in text
    assert "nvidia-smi" not in text
    assert "query_compute_processes" not in text
    assert "assigned_gpu_busy" not in text
    assert "sleep " not in text
    assert "flock -n 7" in text
