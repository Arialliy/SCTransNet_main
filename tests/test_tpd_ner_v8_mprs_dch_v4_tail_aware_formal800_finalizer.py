from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = (
    REPO
    / "experiments/"
    "run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_eval_lane.sh"
)
RUNNER = (
    REPO
    / "experiments/"
    "run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_finalizer.sh"
)
LAUNCHER = (
    REPO
    / "experiments/"
    "launch_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_finalizer.sh"
)
BASE_EVALUATOR = REPO / "experiments/evaluate_pd_fa_sweep.py"


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_shell_sources_parse() -> None:
    for path in (WORKER, RUNNER, LAUNCHER):
        subprocess.run(
            ["/usr/bin/bash", "-n", str(path)],
            cwd=REPO,
            check=True,
        )


def test_worker_hard_binds_checkpoint_roles_to_gpu2_and_gpu3() -> None:
    source = text(WORKER)
    assert "best.pth.tar:2:$v4_eval_gpu2_uuid" in source
    assert "best_miou.pth.tar:3:$v4_eval_gpu3_uuid" in source
    assert 'CUDA_VISIBLE_DEVICES="$v4_eval_gpu_uuid"' in source
    assert "--device cuda:0" in source
    assert "--expected-epochs 800" in source
    assert "--overwrite" not in source


def test_runner_requires_complete_contiguous_training_before_cuda() -> None:
    source = text(RUNNER)
    assert "validate_run_artifacts(run_dir, \"best.pth.tar\")" in source
    assert (
        "validate_run_artifacts(run_dir, \"best_miou.pth.tar\")"
        in source
    )
    assert 'best["metric_event_count"] != 800' in source
    readiness = source.index(
        'v4_finalizer_readiness="$(v4_finalizer_training_state)"'
    )
    launch = source.index("v4_finalizer_launch_eval()")
    assert readiness < launch


def test_runner_launches_two_parallel_role_specific_services() -> None:
    source = text(RUNNER)
    assert (
        "sctransnet-tpd-ner-v8-v4-tail-aware-eval-best-gpu2.service"
        in source
    )
    assert (
        "sctransnet-tpd-ner-v8-v4-tail-aware-eval-best-miou-gpu3.service"
        in source
    )
    assert 'v4_finalizer_launch_eval \\\n    "best.pth.tar"' in source
    assert 'v4_finalizer_launch_eval \\\n    "best_miou.pth.tar"' in source
    assert "parallel_sweeps_running" in source


def test_sweeps_and_publication_are_write_once_and_reverified() -> None:
    worker = text(WORKER)
    runner = text(RUNNER)
    assert "v4_eval_validate_output" in worker
    assert "TPDNERV8V4TAIL_EVAL_IDEMPOTENT_COMPLETE" in worker
    assert "v4_finalizer_validate_sweeps" in runner
    assert "postprocess.validate_v4_sweep" in runner
    assert "absent absent absent" in runner
    assert "regular regular regular" in runner
    assert "--aggregate" in runner
    assert "v4_finalizer_verify_report" in runner
    assert "partial_or_nonregular_postprocess_publish" in runner


def test_gpu23_guards_release_at_direct_posttraining_handoff() -> None:
    source = text(RUNNER)
    worker = text(WORKER)
    readiness = source.index("formal800_incomplete_after_lock")
    release = source.index("v4_finalizer_release_gpu23_reservations\n")
    launch = source.index("v4_finalizer_launch_eval()")
    assert readiness < release < launch
    assert "manage_gpu23_memory_reservation.sh" in source
    assert 'v4_finalizer_release_gpu_reservation 2 "$v4_finalizer_gpu2_guard_unit"' in source
    assert 'v4_finalizer_release_gpu_reservation 3 "$v4_finalizer_gpu3_guard_unit"' in source
    assert '--physical-gpu "$v4_finalizer_guard_index"' in source
    assert "holder_process_alive" in source
    assert "restart_after_eval=false" in source
    assert "query_compute_processes" not in source
    assert "query_compute_processes" not in worker
    assert "assigned_gpu_busy" not in worker
    assert (
        "DataLoader(validation_set, batch_size=1, shuffle=False, "
        "num_workers=0)"
    ) in text(BASE_EVALUATOR)


def test_launcher_supports_preflight_status_and_idempotent_launch() -> None:
    source = text(LAUNCHER)
    assert "--preflight" in source
    assert "--status" in source
    assert "TPDNERV8V4TAIL_FINALIZER_ALREADY_ACTIVE" in source
    assert "TPDNERV8V4TAIL_FINALIZER_IDEMPOTENT_COMPLETE" in source
    assert "--property=Restart=on-failure" in source
    assert "--property=RestartPreventExitStatus=64" in source


def test_no_legacy_paths_or_gpu0_gpu1_assignments() -> None:
    combined = "\n".join(text(path) for path in (WORKER, RUNNER, LAUNCHER))
    assert "/home/md0/" not in combined
    assert "SCTransNet copy" not in combined
    assert "physical_gpu=0" not in combined
    assert "physical_gpu=1" not in combined
    assert "GPU0" not in combined
    assert "GPU1" not in combined


def test_static_preflight_is_read_only_while_training_is_incomplete() -> None:
    env = dict(os.environ)
    env["TPD_NER_V8_V4_TAIL_AWARE_REPO"] = str(REPO)
    result = subprocess.run(
        [str(LAUNCHER), "--preflight"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "TPDNERV8V4TAIL_FINALIZER_PREFLIGHT_OK" in result.stdout
    assert "writes_performed=false" in result.stdout


def test_completed_preflight_keeps_model_progress_off_readiness_stdout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    experiments = repo / "experiments"
    experiments.mkdir(parents=True)
    (experiments / "__init__.py").write_text("", encoding="utf-8")
    evaluator = experiments / (
        "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py"
    )
    evaluator.write_text(
        "\n".join(
            (
                "def validate_run_artifacts(run_dir, checkpoint):",
                "    print('Deep-Supervision: True')",
                "    epoch = 422 if checkpoint == 'best.pth.tar' else 489",
                "    return {",
                "        'metric_event_count': 800,",
                "        'checkpoint_epoch': epoch,",
                "    }",
                "",
            )
        ),
        encoding="utf-8",
    )
    (experiments / (
        "postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800.py"
    )).write_text("", encoding="utf-8")
    for name in (
        "run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_eval_lane.sh",
        "manage_gpu23_memory_reservation.sh",
    ):
        path = experiments / name
        path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    run_dir = (
        repo
        / "results/NUDT-SIRST/"
        "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/"
        "seed_42_formal800_exact_v4_tail_aware_seed42"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text("{}\n", encoding="utf-8")

    env = dict(os.environ)
    env["TPD_NER_V8_V4_TAIL_AWARE_REPO"] = str(repo)
    env["TPD_NER_V8_V4_TAIL_AWARE_RESULT_ROOT"] = str(repo / "results")
    env["TPD_NER_V8_V4_TAIL_AWARE_PYTHON"] = sys.executable
    result = subprocess.run(
        [str(RUNNER), "--preflight"],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "training=ready events=800 last_epoch=800"
        " best_epoch=422 best_miou_epoch=489"
    ) in result.stdout
    assert "training=Deep-Supervision" not in result.stdout
    assert result.stderr.count("Deep-Supervision: True") == 2
