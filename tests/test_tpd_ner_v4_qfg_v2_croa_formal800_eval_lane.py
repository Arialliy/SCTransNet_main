from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = (
    REPO
    / "experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_eval_lane.sh"
)
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"


def source() -> str:
    return WORKER.read_text(encoding="utf-8")


def test_shell_source_parses_and_worker_is_executable() -> None:
    subprocess.run(
        ["/usr/bin/bash", "-n", str(WORKER)],
        cwd=REPO,
        check=True,
    )
    assert WORKER.stat().st_mode & 0o111


def test_cli_is_one_variant_times_one_checkpoint_times_one_gpu() -> None:
    text = source()
    assert (
        "{qfg_only|tss_qfg} "
        "{best.pth.tar|best_miou.pth.tar} {2|3} GPU_UUID"
    ) in text
    assert "qfg_eval_variant=\"$1\"" in text
    assert "qfg_eval_checkpoint=\"$2\"" in text
    assert "qfg_eval_physical_index=\"$3\"" in text
    assert "qfg_eval_gpu_uuid=\"$4\"" in text
    assert "qfg_only)" in text
    assert "tss_qfg)" in text
    assert "best.pth.tar)" in text
    assert "best_miou.pth.tar)" in text
    assert '"2:$qfg_eval_gpu2_uuid"|"3:$qfg_eval_gpu3_uuid"' in text
    assert "--all-four" not in text


def test_invalid_parameter_values_fail_before_artifact_or_cuda_work() -> None:
    invalid_variant = subprocess.run(
        [str(WORKER), "other", "best.pth.tar", "2", GPU2_UUID],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert invalid_variant.returncode == 64
    assert "reason=invalid_variant" in invalid_variant.stderr

    mismatched_uuid = subprocess.run(
        [str(WORKER), "qfg_only", "best.pth.tar", "2", GPU3_UUID],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert mismatched_uuid.returncode == 64
    assert "reason=invalid_gpu_uuid_mapping" in mismatched_uuid.stderr


def test_defaults_are_v2_optimized_and_never_v1() -> None:
    text = source()
    assert (
        "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
        in text
    )
    assert (
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
        in text
    )
    assert "tpd_ner_v4_qfg_v2_croa_exact_v1" not in text
    assert (
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock.json"
        not in text
    )


def test_freezer_and_complete_history_are_verified_before_cuda() -> None:
    text = source()
    assert '"$qfg_eval_python" "$qfg_eval_freezer" \\\n        --verify' in text
    assert "len(raw_lines) != exact.FORMAL_EPOCHS" in text
    assert "any(not line.strip() for line in raw_lines)" in text
    assert "list(range(1, exact.FORMAL_EPOCHS + 1))" in text
    assert "exact._load_complete_events(metrics_path, exact.FORMAL_EPOCHS)" in text
    assert "policy.recompute(events, require_flags=True)" in text
    assert "evaluator.validate_run_artifacts(run_dir, checkpoint_name)" in text
    assert 'summary.get("status") != "complete"' in text
    assert "QFG checkpoint/global-selection metrics" in text
    assert "QFG run is not bound to the V2 optimized source lock" in text

    locked_verify = text.rindex(
        "qfg_eval_verify_frozen_sources\nqfg_eval_validate_input"
    )
    gpu_probe = text.index("qfg_eval_gpu_probe\n", locked_verify)
    evaluator_run = text.index(
        '"$qfg_eval_python" "$qfg_eval_evaluator" \\\n',
        gpu_probe,
    )
    assert locked_verify < gpu_probe < evaluator_run


def test_existing_output_is_reverified_and_never_overwritten() -> None:
    text = source()
    assert "qfg_eval_validate_output" in text
    assert "evaluator.validate_output_identity(payload, artifact_audit=audit)" in text
    assert "QFG sweep evaluation source binding" in text
    assert "QFG sweep evaluator contract" in text
    assert "QFG sweep device assignment differs" in text
    assert "TPDNERV4QFG_EVAL_IDEMPOTENT_COMPLETE" in text
    assert "TPDNERV4QFG_EVAL_OUTPUT_VERIFIED" in text
    assert "--overwrite" not in text
    assert "flock -n 8" in text
    assert "flock -n 9" in text


def test_all_seven_thread_controls_and_uuid_environment_are_fixed() -> None:
    text = source()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        assert f"export {name}=1" in text
        assert f'"{name}",' in text
    assert 'export CUDA_VISIBLE_DEVICES="$qfg_eval_gpu_uuid"' in text
    assert (
        'export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX='
        '"$qfg_eval_physical_index"'
    ) in text
    assert (
        'export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$qfg_eval_gpu_uuid"'
        in text
    )
    assert GPU2_UUID in text
    assert GPU3_UUID in text
    assert "--device cuda:0" in text
    assert "--expected-epochs 800" in text


def test_worker_contains_no_legacy_paths_or_gpu_busy_wait() -> None:
    text = source()
    assert "/home/md0/" not in text
    assert "SCTransNet copy" not in text
    assert "query_compute_processes" not in text
    assert "assigned_gpu_busy" not in text
    assert "sleep " not in text
