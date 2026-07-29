from __future__ import annotations

import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = (
    REPO
    / "experiments/"
    "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
)
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"


def source() -> str:
    return WORKER.read_text(encoding="utf-8")


def test_shell_source_parses_and_worker_is_executable() -> None:
    subprocess.run(["bash", "-n", str(WORKER)], cwd=REPO, check=True)
    assert WORKER.stat().st_mode & 0o111


def test_cli_and_variant_gpu_mapping_are_fixed() -> None:
    text = source()
    assert (
        "{qfg_dlr|tss_qfg_dlr} "
        "{best.pth.tar|best_miou.pth.tar} {2|3} GPU_UUID"
    ) in text
    assert '"qfg_dlr:2:$ramp_eval_gpu2_uuid")' in text
    assert '"tss_qfg_dlr:3:$ramp_eval_gpu3_uuid")' in text
    invalid = subprocess.run(
        [str(WORKER), "qfg_dlr", "best.pth.tar", "3", GPU3_UUID],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert invalid.returncode == 64
    assert "reason=invalid_variant_gpu_mapping" in invalid.stderr


def test_lane_roots_match_the_training_launcher() -> None:
    text = source()
    assert "$ramp_eval_result_root/qfg_dlr_lane" in text
    assert "$ramp_eval_result_root/tss_qfg_dlr_lane" in text
    assert "seed_42_$ramp_eval_run_tag" in text
    assert 'ramp_eval_run_tag="formal800_qfg_dlr_control"' in text
    assert (
        'ramp_eval_run_tag="formal800_tss_qfg_dlr_ramp100"'
        in text
    )


def test_read_only_preflight_and_existing_output_verification_precede_gpu() -> None:
    text = source()
    assert "--preflight" in text
    assert "--device cpu" in text
    assert "evaluator.validate_run_artifacts(run_dir, checkpoint)" in text
    assert "evaluator.validate_existing_output(" in text
    assert "TPDNER_DLR_RAMP100_EVAL_IDEMPOTENT_COMPLETE" in text
    first_preflight = text.rindex("ramp_eval_preflight\n")
    gpu_probe = text.index("ramp_eval_gpu_probe\n", first_preflight)
    evaluator_run = text.index(
        '"$ramp_eval_python" "$ramp_eval_evaluator" \\\n',
        gpu_probe,
    )
    assert first_preflight < gpu_probe < evaluator_run
    assert "--overwrite" not in text


def test_uuid_environment_thread_controls_and_nonblocking_claims() -> None:
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
    assert 'export CUDA_VISIBLE_DEVICES="$ramp_eval_gpu_uuid"' in text
    assert (
        'export TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX='
        '"$ramp_eval_physical_index"'
    ) in text
    assert (
        'export TPD_NER_V4_QFG_PHYSICAL_GPU_UUID="$ramp_eval_gpu_uuid"'
        in text
    )
    assert "flock -n 8" in text
    assert "flock -n 9" in text
    assert "$ramp_eval_result_root/.evaluation_locks" in text


def test_worker_has_no_legacy_path_gpu_poll_or_idle_wait() -> None:
    text = source()
    assert "/home/md0/" not in text
    assert "SCTransNet copy" not in text
    assert "nvidia-smi" not in text
    assert "query_compute_processes" not in text
    assert "assigned_gpu_busy" not in text
    assert "sleep " not in text
    assert "physical_gpu=0" not in text
    assert "physical_gpu=1" not in text
