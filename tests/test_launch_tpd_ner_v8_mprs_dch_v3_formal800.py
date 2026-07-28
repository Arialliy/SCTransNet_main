from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LANE = (
    REPO_ROOT
    / "experiments/run_tpd_ner_v8_mprs_dch_v3_formal800_1x5090_lane.sh"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_ner_v8_mprs_dch_v3_formal800_1x5090.sh"
)

GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
GPU2_UNIT = "sctransnet-tpd-ner-v8-v3-relay-on-gpu2"
GPU3_UNIT = "sctransnet-tpd-ner-v8-v3-relay-on-gpu3"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _read_if_present(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _mock_fixture(
    root: Path,
    *,
    initialization: str = "fresh",
) -> dict[str, Any]:
    repo = root / "repo"
    experiments = repo / "experiments"
    experiments.mkdir(parents=True)
    (repo / "datasets").mkdir()

    copied_lane = experiments / LANE.name
    _write_executable(copied_lane, LANE.read_text(encoding="utf-8"))
    trainer = experiments / "train_tpd_ner_v8_mprs_dch_v3_exact.py"
    freezer = experiments / "freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py"
    trainer.write_text("# mock V3 trainer\n", encoding="utf-8")
    freezer.write_text("# mock V3 freezer\n", encoding="utf-8")

    locks = {
        "TPD_NER_V8_V3_TRAINING_LOCK": (
            root / "v3-training-source-lock.json"
        ),
        "TPD_NER_V8_V3_ACCEPTANCE_LOCK": (
            root / "v3-acceptance-source-lock.json"
        ),
        "TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK": (
            root / "upstream-v2-training-source-lock.json"
        ),
        "TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK": (
            root / "upstream-v2-acceptance-source-lock.json"
        ),
    }
    for label, path in locks.items():
        path.write_text(
            json.dumps({"fixture": label}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    python_log = root / "python.log"
    systemctl_log = root / "systemctl.log"
    systemd_run_log = root / "systemd-run.log"
    mock_bin = root / "mock-bin"
    mock_bin.mkdir()

    python_target = root / "mock-python-target"
    _write_executable(
        python_target,
        """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL'
    for v3_test_argument in "$@"; do
        printf '\\t%s' "$v3_test_argument"
    done
    printf '\\n'
    printf 'ENV\\t%s\\t%s\\t%s\\n' \
        "${CUDA_VISIBLE_DEVICES:-}" \
        "${TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX:-}" \
        "${TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID:-}"
} >> "${V3_TEST_PYTHON_LOG:?}"
case "${1:-}" in
    */freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py)
        exit "${V3_TEST_VERIFY_RC:-0}"
        ;;
    -)
        if [[ "${2:-}" == GPU-* ]]; then
            echo "TPDNERV8V3_GPU_OK logical_device=cuda:0"
        else
            printf '%s\\n' "${V3_TEST_INITIALIZATION:-fresh}"
        fi
        ;;
    */train_tpd_ner_v8_mprs_dch_v3_exact.py)
        echo "TPDNERV8V3_MOCK_TRAINER"
        ;;
    *)
        echo "unexpected mock-python command: $*" >&2
        exit 90
        ;;
esac
""",
    )
    python_link = root / "python-symlink"
    python_link.symlink_to(python_target)

    _write_executable(
        mock_bin / "systemctl",
        """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL'
    for v3_test_argument in "$@"; do
        printf '\\t%s' "$v3_test_argument"
    done
    printf '\\n'
} >> "${V3_TEST_SYSTEMCTL_LOG:?}"
if [[ "$*" == *"--property=ActiveState"* && "$*" == *"--value"* ]]; then
    if [[ -n "${V3_TEST_ACTIVE_UNIT:-}" && "$*" == *"${V3_TEST_ACTIVE_UNIT}"* ]]; then
        echo "active"
    else
        echo "inactive"
    fi
else
    echo "Id=${3:-unknown}"
    echo "ActiveState=inactive"
fi
""",
    )
    _write_executable(
        mock_bin / "systemd-run",
        """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL'
    for v3_test_argument in "$@"; do
        printf '\\t%s' "$v3_test_argument"
    done
    printf '\\n'
} >> "${V3_TEST_SYSTEMD_RUN_LOG:?}"
if [[ -n "${V3_TEST_SYSTEMD_RUN_ENTERED:-}" ]]; then
    touch "${V3_TEST_SYSTEMD_RUN_ENTERED}"
    while [[ ! -e "${V3_TEST_SYSTEMD_RUN_RELEASE:?}" ]]; do
        sleep 0.02
    done
fi
""",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{mock_bin}:{environment.get('PATH', '')}",
            "TPD_NER_V8_V3_REPO": str(repo),
            "TPD_NER_V8_V3_PYTHON": str(python_link),
            "TPD_NER_V8_V3_RESULT_ROOT": str(root / "v3-results"),
            "V3_TEST_INITIALIZATION": initialization,
            "V3_TEST_PYTHON_LOG": str(python_log),
            "V3_TEST_SYSTEMCTL_LOG": str(systemctl_log),
            "V3_TEST_SYSTEMD_RUN_LOG": str(systemd_run_log),
            # Legacy V2 settings must never steer the V3 lane or launcher.
            "TPD_NER_V8_V2_REPO": "/legacy/v2/repository",
            "TPD_NER_V8_V2_PYTHON": "/legacy/v2/python",
            "TPD_NER_V8_V2_SOURCE_LOCK": "/legacy/v2/training-lock.json",
            "TPD_NER_V8_V2_RESULT_ROOT": "/legacy/v2/results",
        }
    )
    environment.update({name: str(path) for name, path in locks.items()})
    return {
        "repo": repo,
        "lane": copied_lane,
        "trainer": trainer,
        "freezer": freezer,
        "locks": locks,
        "python": python_link,
        "python_log": python_log,
        "systemctl_log": systemctl_log,
        "systemd_run_log": systemd_run_log,
        "environment": environment,
    }


def _identity_lane_fixture(root: Path) -> dict[str, Any]:
    result_root = root / "results"
    command_log = root / "identity-python.log"
    locks = {
        "TPD_NER_V8_V3_TRAINING_LOCK": root / "v3-training.json",
        "TPD_NER_V8_V3_ACCEPTANCE_LOCK": root / "v3-acceptance.json",
        "TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK": (
            root / "v2-training.json"
        ),
        "TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK": (
            root / "v2-acceptance.json"
        ),
    }
    for label, path in locks.items():
        path.write_text(
            json.dumps({"fixture": label}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    python_target = root / "identity-aware-python"
    _write_executable(
        python_target,
        """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL'
    for v3_test_argument in "$@"; do
        printf '\\t%s' "$v3_test_argument"
    done
    printf '\\n'
} >> "${V3_TEST_PYTHON_LOG:?}"
case "${1:-}" in
    */freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py)
        exit 0
        ;;
    -)
        if [[ "${2:-}" == GPU-* ]]; then
            echo "TPDNERV8V3_GPU_OK logical_device=cuda:0"
            exit 0
        fi
        exec "${V3_TEST_REAL_PYTHON:?}" "$@"
        ;;
    */train_tpd_ner_v8_mprs_dch_v3_exact.py)
        echo "trainer must not run" >&2
        exit 99
        ;;
esac
""",
    )
    python_link = root / "identity-python-symlink"
    python_link.symlink_to(python_target)
    environment = os.environ.copy()
    environment.update(
        {
            "TPD_NER_V8_V3_REPO": str(REPO_ROOT),
            "TPD_NER_V8_V3_PYTHON": str(python_link),
            "TPD_NER_V8_V3_RESULT_ROOT": str(result_root),
            "V3_TEST_PYTHON_LOG": str(command_log),
            "V3_TEST_REAL_PYTHON": sys.executable,
        }
    )
    environment.update({label: str(path) for label, path in locks.items()})
    return {
        "result_root": result_root,
        "run_dir": (
            result_root
            / "NUDT-SIRST"
            / VARIANT
            / "seed_42_formal800_exact_v3_seed42"
        ),
        "training_lock": locks["TPD_NER_V8_V3_TRAINING_LOCK"],
        "locks": locks,
        "command_log": command_log,
        "environment": environment,
    }


def _valid_identity_and_metadata(
    training_lock: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact

    manifest = {
        "schema": exact.ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": VARIANT,
        "model": "tests.V3LaneCompletionFixture",
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_version": exact.V3_RELAY_VERSION,
        "relay_width": exact.RELAY_WIDTH,
        "relay_rms_eps": exact.RELAY_RMS_EPS,
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
        "eps": exact.FORMAL_EPS,
    }
    manifest_sha256 = exact.canonical_sha256(manifest)
    metadata = {
        "variant": VARIANT,
        "candidate_family": "fixture-v3-ner",
        "parent_variant": "tpd_clean_v8_mprs_dch_full",
        "relay_enabled": True,
        "relay_version": exact.V3_RELAY_VERSION,
        "relay_width": exact.RELAY_WIDTH,
        "relay_initialization_seed": exact.RELAY_INITIALIZATION_SEED,
        "gate_dc_offset": "learned_per_stage_post_centering",
        "gate_dc_offset_count": 3,
        "gate_dc_offset_initialization": "zero",
        "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
        "mask_mapping": "atan(pi*(centered+dc))/pi",
        "zero_gate_reference": "v2_and_relay_off_exact",
        "structural_predecessor": exact.V2_RELAY_ON_VARIANT,
        "required_control": exact.V8_PARENT_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "architecture_manifest": manifest,
        "architecture_id": manifest_sha256,
    }
    identity = {
        "dataset": "NUDT-SIRST",
        "variant": VARIANT,
        "seed": 42,
        "split_seed": 20260722,
        "run_id": (
            f"{exact.RUN_ID_PREFIX}NUDT-SIRST:{VARIANT}:"
            f"seed-42:split-20260722:{exact.FORMAL_RUN_TAG}"
        ),
        "architecture_id": manifest_sha256,
        "builder_manifest_sha256": manifest_sha256,
        "source_locks": {
            exact.SOURCE_LOCK_KEY: hashlib.sha256(
                training_lock.read_bytes()
            ).hexdigest()
        },
        "training_contract": {
            "determinism": {
                "entry_schema": exact.ENTRY_SCHEMA,
                "parent_variant": "tpd_clean_v8_mprs_dch_full",
                "relay_enabled": True,
                "relay_version": exact.V3_RELAY_VERSION,
                "relay_width": exact.RELAY_WIDTH,
                "relay_initialization_seed": (
                    exact.RELAY_INITIALIZATION_SEED
                ),
                "relay_rms_eps": exact.RELAY_RMS_EPS,
                "gate_bias": False,
                "gate_spatial_centering": "per_sample_mean_hw",
                "gate_dc_offset": "learned_per_stage_post_centering",
                "gate_dc_offset_count": 3,
                "gate_dc_offset_initialization": "zero",
                "gate_dc_offset_state_prefix": "tpd_ner.dc_offsets.",
                "mask_mapping": "atan(pi*(centered+dc))/pi",
                "zero_gate_reference": "v2_and_relay_off_exact",
                "scheduler_restore_mode": (
                    "identity_bound_manual_schedule_from_completed_epoch"
                ),
            }
        },
    }
    exact.require_v3_run_identity(
        identity,
        label="test lane completion identity",
        expected_variant=VARIANT,
    )
    return identity, metadata


def _validation_metrics() -> dict[str, float]:
    from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact

    return {
        name: float(index + 1)
        for index, name in enumerate(exact.STORED_VALIDATION_METRICS)
    }


def _write_completion_prefix(
    run_dir: Path,
    identity: dict[str, Any],
) -> None:
    from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact

    run_dir.mkdir(parents=True)
    (run_dir / "protocol.json").write_text(
        json.dumps(
            {
                "schema": exact.ENTRY_SCHEMA,
                "run_identity": identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": exact.COMPLETION_SUMMARY_SCHEMA,
                "status": "complete",
                "variant": VARIANT,
                "seed": 42,
                "split_seed": 20260722,
                "run_identity": identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = _validation_metrics()
    (run_dir / "metrics.jsonl").write_text(
        "".join(
            json.dumps(
                {"epoch": epoch, **metrics},
                sort_keys=True,
            )
            + "\n"
            for epoch in range(1, exact.FORMAL_EPOCHS + 1)
        ),
        encoding="utf-8",
    )


def _write_evaluator_checkpoint(
    path: Path,
    *,
    identity: dict[str, Any],
    metadata: dict[str, Any],
    epoch: int,
    role: str,
) -> None:
    import torch

    from experiments import tpd_exact_runner as exact_runner
    from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as exact

    metrics = _validation_metrics()
    payload = exact.EvaluatorCheckpointAdapter(
        model_metadata=metadata,
        split_hashes={"train": "a" * 64},
    )(
        exact_runner.CompatibilityPayloadContext(
            role=role,
            epoch=epoch,
            metrics=metrics,
            event={"epoch": epoch, **metrics},
            exact_payload={
                "model": {"state_dict": {"weight": torch.tensor([1.0])}},
                "optimizer": {"state_dict": {"state": {}}},
                "scaler": {"state_dict": {"updates": epoch}},
            },
            run_identity=identity,
            normalized_spec={},
        )
    )
    torch.save(payload, path)


def _run_identity_lane(
    fixture: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/bash",
            str(LANE),
            "--preflight",
            "2",
            GPU2_UUID,
        ],
        cwd=REPO_ROOT,
        env=fixture["environment"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TPDNERV8MPRSDCHV3FormalLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lane_text = LANE.read_text(encoding="utf-8")
        cls.launcher_text = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_and_v3_private_contract(self) -> None:
        for path in (LANE, LAUNCHER):
            with self.subTest(path=path.name):
                subprocess.run(
                    ["/usr/bin/bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

        self.assertIn(
            "train_tpd_ner_v8_mprs_dch_v3_exact.py",
            self.lane_text,
        )
        self.assertIn(
            'v3_variant="tpd_ner_v8_mprs_dch_v3_full_relay_on"',
            self.lane_text,
        )
        self.assertIn(
            "results/tpd_ner_v8_mprs_dch_v3_exact_v1",
            self.lane_text,
        )
        self.assertIn(
            'v3_run_tag="formal800_exact_v3_seed42"',
            self.lane_text,
        )
        self.assertIn("--device cuda:0", self.lane_text)
        self.assertNotIn("--amp", self.lane_text)
        self.assertNotIn(
            "train_tpd_ner_v8_mprs_dch_v2_exact.py",
            self.lane_text,
        )
        self.assertNotIn(
            "results/tpd_ner_v8_mprs_dch_v2_exact_v1",
            self.lane_text,
        )
        for text in (self.lane_text, self.launcher_text):
            with self.subTest(entrypoint="lane" if text is self.lane_text else "launcher"):
                self.assertIn("--mode verify", text)
                self.assertIn("--kind all", text)
                self.assertIn("--training-lock", text)
                self.assertIn("--acceptance-lock", text)
                self.assertIn("--upstream-v2-training-lock", text)
                self.assertIn("--upstream-v2-acceptance-lock", text)
                self.assertNotIn("TPD_NER_V8_V2_REPO", text)
                self.assertNotIn("TPD_NER_V8_V2_PYTHON", text)
                self.assertNotIn("TPD_NER_V8_V2_RESULT_ROOT", text)
        self.assertIn('[[ -x "$v3_python" ]]', self.lane_text)
        self.assertIn('[[ -x "$v3_python" ]]', self.launcher_text)
        self.assertNotIn('-f "$v3_python"', self.lane_text)
        self.assertNotIn('-f "$v3_python"', self.launcher_text)
        self.assertIn("exact.require_v3_run_identity(", self.lane_text)
        self.assertIn("exact.SOURCE_LOCK_KEY", self.lane_text)
        self.assertIn(
            "exact.require_evaluator_checkpoint_payload(",
            self.lane_text,
        )
        self.assertIn("metrics journal is truncated", self.lane_text)
        self.assertIn("metrics epochs are not contiguous 1..800", self.lane_text)
        self.assertIn(
            '("last.pth.tar", "last_evaluated_epoch")',
            self.lane_text,
        )
        self.assertIn(
            "active journal is not committed at epoch 800",
            self.lane_text,
        )
        self.assertIn("--fresh", self.lane_text)
        self.assertIn("--exact-resume", self.lane_text)
        self.assertIn(
            'v3_claim_path="$v3_claim_dir/formal800_v3_global.lock"',
            self.launcher_text,
        )
        self.assertIn("flock -n 9", self.launcher_text)
        self.assertIn("reason=v3_global_claim_busy", self.launcher_text)
        claim = self.launcher_text.index("flock -n 9")
        self.assertLess(
            claim,
            self.launcher_text.index(
                '"$v3_python" "$v3_manifest_tool"',
            ),
        )
        self.assertLess(
            claim,
            self.launcher_text.index(
                'for v3_unit in "$v3_gpu2_unit" "$v3_gpu3_unit"',
            ),
        )
        self.assertLess(claim, self.launcher_text.index("systemd-run --user"))

    def test_gpu_mapping_and_exact_training_command_are_frozen(self) -> None:
        invalid_mappings = (
            ("2", GPU3_UUID),
            ("3", GPU2_UUID),
            ("0", GPU2_UUID),
            ("1", GPU3_UUID),
        )
        for physical_index, gpu_uuid in invalid_mappings:
            with self.subTest(
                physical_index=physical_index,
                gpu_uuid=gpu_uuid,
            ):
                completed = subprocess.run(
                    ["/usr/bin/bash", str(LANE), physical_index, gpu_uuid],
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "reason=invalid_gpu_mapping",
                    completed.stderr,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(Path(directory))
            completed = subprocess.run(
                ["/usr/bin/bash", str(fixture["lane"]), "3", GPU3_UUID],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            command_log = _read_if_present(fixture["python_log"])
            self.assertIn(f"ENV\t{GPU3_UUID}\t3\t{GPU3_UUID}", command_log)
            trainer_call = next(
                line
                for line in command_log.splitlines()
                if "train_tpd_ner_v8_mprs_dch_v3_exact.py" in line
            )
            for fragment in (
                f"\t--variant\t{VARIANT}",
                "\t--dataset\tNUDT-SIRST",
                "\t--run-tag\tformal800_exact_v3_seed42",
                "\t--device\tcuda:0",
                "\t--epochs\t800",
                "\t--batch-size\t16",
                "\t--patch-size\t256",
                "\t--workers\t0",
                "\t--seed\t42",
                "\t--split-seed\t20260722",
                "\t--val-fraction\t0.20",
                "\t--eval-every\t1",
                "\t--base-lr\t0.001",
                "\t--min-lr\t0.00001",
                "\t--warmup-epochs\t10",
                "\t--threshold\t0.5",
                "\t--match-radius\t3.0",
                "\t--tiny-area\t9",
                "\t--eps\t0.000001",
                "\t--exact-source-lock\t",
                "\t--fresh",
            ):
                with self.subTest(fragment=fragment):
                    self.assertIn(fragment, trainer_call)

        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(
                Path(directory),
                initialization="exact-resume",
            )
            completed = subprocess.run(
                ["/usr/bin/bash", str(fixture["lane"]), "2", GPU2_UUID],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            command_log = _read_if_present(fixture["python_log"])
            trainer_call = next(
                line
                for line in command_log.splitlines()
                if "train_tpd_ner_v8_mprs_dch_v3_exact.py" in line
            )
            self.assertIn("\t--exact-resume", trainer_call)
            self.assertNotIn("\t--fresh", trainer_call)

    def test_missing_any_lock_fails_closed_before_commands(self) -> None:
        lock_environment_names = (
            "TPD_NER_V8_V3_TRAINING_LOCK",
            "TPD_NER_V8_V3_ACCEPTANCE_LOCK",
            "TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK",
            "TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK",
        )
        for environment_name in lock_environment_names:
            with self.subTest(environment_name=environment_name):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = _mock_fixture(Path(directory))
                    Path(
                        fixture["environment"][environment_name]
                    ).unlink()
                    completed = subprocess.run(
                        [
                            "/usr/bin/bash",
                            str(LAUNCHER),
                            "--preflight",
                            "--physical-gpu",
                            "2",
                        ],
                        cwd=REPO_ROOT,
                        env=fixture["environment"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(
                        "reason=missing_required_file",
                        completed.stderr,
                    )
                    self.assertEqual(
                        _read_if_present(fixture["python_log"]),
                        "",
                    )
                    self.assertEqual(
                        _read_if_present(fixture["systemd_run_log"]),
                        "",
                    )

    def test_launcher_verifies_all_locks_and_ignores_v2_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(Path(directory))
            before_locks = {
                label: path.read_bytes()
                for label, path in fixture["locks"].items()
            }
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--physical-gpu",
                    "2",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn("TPDNERV8V3_LAUNCHED", completed.stdout)

            python_log = _read_if_present(fixture["python_log"])
            verify_calls = [
                line
                for line in python_log.splitlines()
                if "freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py" in line
            ]
            self.assertEqual(len(verify_calls), 2)
            for call in verify_calls:
                with self.subTest(call=call):
                    self.assertIn("\t--mode\tverify", call)
                    self.assertIn("\t--kind\tall", call)
                    for option, environment_name in (
                        (
                            "--training-lock",
                            "TPD_NER_V8_V3_TRAINING_LOCK",
                        ),
                        (
                            "--acceptance-lock",
                            "TPD_NER_V8_V3_ACCEPTANCE_LOCK",
                        ),
                        (
                            "--upstream-v2-training-lock",
                            "TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK",
                        ),
                        (
                            "--upstream-v2-acceptance-lock",
                            "TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK",
                        ),
                    ):
                        self.assertIn(
                            f"\t{option}\t"
                            f"{fixture['environment'][environment_name]}",
                            call,
                        )
            self.assertNotIn("/legacy/v2/", python_log)

            systemctl_log = _read_if_present(fixture["systemctl_log"])
            self.assertIn(f"\tshow\t{GPU2_UNIT}.service", systemctl_log)
            self.assertIn(f"\tshow\t{GPU3_UNIT}.service", systemctl_log)
            self.assertIn(
                f"\treset-failed\t{GPU2_UNIT}.service",
                systemctl_log,
            )
            self.assertNotIn("sctransnet-tpd-ner-v8-v1", systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v2", systemctl_log)
            self.assertNotIn("\tstop\t", systemctl_log)
            self.assertNotIn("\trestart\t", systemctl_log)

            systemd_run_log = _read_if_present(
                fixture["systemd_run_log"]
            )
            self.assertIn(f"\t--unit\t{GPU2_UNIT}", systemd_run_log)
            self.assertIn(
                f"\t{fixture['lane']}\t2\t{GPU2_UUID}",
                systemd_run_log,
            )
            self.assertNotIn("/legacy/v2/", systemd_run_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v1", systemd_run_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v2", systemd_run_log)

            after_locks = {
                label: path.read_bytes()
                for label, path in fixture["locks"].items()
            }
            self.assertEqual(after_locks, before_locks)

    def test_active_peer_v3_unit_aborts_without_mutating_v1_or_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(Path(directory))
            fixture["environment"]["V3_TEST_ACTIVE_UNIT"] = GPU3_UNIT
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--physical-gpu",
                    "2",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("reason=v3_unit_already_active", completed.stderr)
            self.assertIn(GPU3_UNIT, completed.stderr)
            self.assertEqual(
                _read_if_present(fixture["systemd_run_log"]),
                "",
            )
            systemctl_log = _read_if_present(fixture["systemctl_log"])
            self.assertIn(GPU2_UNIT, systemctl_log)
            self.assertIn(GPU3_UNIT, systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v1", systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v2", systemctl_log)
            self.assertNotIn("reset-failed", systemctl_log)

    def test_global_claim_allows_only_one_concurrent_systemd_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _mock_fixture(root)
            entered = root / "systemd-run-entered"
            release = root / "systemd-run-release"
            fixture["environment"].update(
                {
                    "V3_TEST_SYSTEMD_RUN_ENTERED": str(entered),
                    "V3_TEST_SYSTEMD_RUN_RELEASE": str(release),
                }
            )
            first = subprocess.Popen(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--physical-gpu",
                    "2",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10.0
                while not entered.exists() and time.monotonic() < deadline:
                    if first.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(
                    entered.exists(),
                    msg="first launcher did not enter mocked systemd-run",
                )
                second = subprocess.run(
                    [
                        "/usr/bin/bash",
                        str(LAUNCHER),
                        "--physical-gpu",
                        "3",
                    ],
                    cwd=REPO_ROOT,
                    env=fixture["environment"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(second.returncode, 0)
                self.assertIn(
                    "reason=v3_global_claim_busy",
                    second.stderr,
                )
            finally:
                release.touch()
                first_stdout, first_stderr = first.communicate(timeout=10)
            self.assertEqual(
                first.returncode,
                0,
                msg=first_stdout + first_stderr,
            )
            systemd_run_calls = [
                line
                for line in _read_if_present(
                    fixture["systemd_run_log"]
                ).splitlines()
                if line.startswith("CALL")
            ]
            self.assertEqual(len(systemd_run_calls), 1)
            self.assertIn(f"\t--unit\t{GPU2_UNIT}", systemd_run_calls[0])
            systemctl_log = _read_if_present(fixture["systemctl_log"])
            self.assertNotIn("sctransnet-tpd-ner-v8-v1", systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v2", systemctl_log)

    def test_complete_run_and_preflight_never_create_a_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(
                Path(directory),
                initialization="complete",
            )
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--physical-gpu",
                    "3",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn(
                "TPDNERV8V3_IDEMPOTENT_COMPLETE",
                completed.stdout,
            )
            self.assertEqual(
                _read_if_present(fixture["systemd_run_log"]),
                "",
            )
            self.assertEqual(
                _read_if_present(fixture["systemctl_log"]),
                "",
            )

        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(Path(directory))
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--preflight",
                    "--physical-gpu",
                    "2",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn(
                "TPDNERV8V3_PREFLIGHT_COMPLETE",
                completed.stdout,
            )
            self.assertEqual(
                _read_if_present(fixture["systemd_run_log"]),
                "",
            )
            self.assertEqual(
                _read_if_present(fixture["systemctl_log"]),
                "",
            )

    def test_status_targets_only_the_selected_v3_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _mock_fixture(Path(directory))
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LAUNCHER),
                    "--status",
                    "--physical-gpu",
                    "3",
                ],
                cwd=REPO_ROOT,
                env=fixture["environment"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            systemctl_log = _read_if_present(fixture["systemctl_log"])
            self.assertIn(f"\tshow\t{GPU3_UNIT}.service", systemctl_log)
            self.assertNotIn(GPU2_UNIT, systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v1", systemctl_log)
            self.assertNotIn("sctransnet-tpd-ner-v8-v2", systemctl_log)
            self.assertEqual(
                _read_if_present(fixture["python_log"]),
                "",
            )
            self.assertEqual(
                _read_if_present(fixture["systemd_run_log"]),
                "",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "V3 completion checkpoint validation requires the project torch env",
    )
    def test_truncated_complete_summary_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _identity_lane_fixture(Path(directory))
            identity, _ = _valid_identity_and_metadata(
                fixture["training_lock"]
            )
            run_dir = fixture["run_dir"]
            run_dir.mkdir(parents=True)
            from experiments import (
                train_tpd_ner_v8_mprs_dch_v3_exact as exact,
            )

            (run_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": exact.ENTRY_SCHEMA,
                        "run_identity": identity,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            truncated = (
                '{"schema":"'
                + exact.COMPLETION_SUMMARY_SCHEMA
                + '","status":"complete"'
            )
            (run_dir / "summary.json").write_text(
                truncated,
                encoding="utf-8",
            )
            completed = _run_identity_lane(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "summary is invalid or truncated",
                completed.stderr,
            )
            self.assertNotIn("TPDNERV8V3_LANE_READY", completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "V3 completion checkpoint validation requires the project torch env",
    )
    def test_complete_summary_with_missing_checkpoint_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _identity_lane_fixture(Path(directory))
            identity, metadata = _valid_identity_and_metadata(
                fixture["training_lock"]
            )
            run_dir = fixture["run_dir"]
            _write_completion_prefix(run_dir, identity)
            _write_evaluator_checkpoint(
                run_dir / "best.pth.tar",
                identity=identity,
                metadata=metadata,
                epoch=700,
                role="best_validation_pd_primary",
            )
            _write_evaluator_checkpoint(
                run_dir / "last.pth.tar",
                identity=identity,
                metadata=metadata,
                epoch=800,
                role="last_evaluated_epoch",
            )
            completed = _run_identity_lane(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "lacks regular checkpoint: best_miou.pth.tar",
                completed.stderr,
            )
            self.assertNotIn("TPDNERV8V3_LANE_READY", completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "V3 completion checkpoint validation requires the project torch env",
    )
    def test_complete_summary_with_last_before_epoch_800_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _identity_lane_fixture(Path(directory))
            identity, metadata = _valid_identity_and_metadata(
                fixture["training_lock"]
            )
            run_dir = fixture["run_dir"]
            _write_completion_prefix(run_dir, identity)
            for filename, epoch, role in (
                (
                    "best.pth.tar",
                    700,
                    "best_validation_pd_primary",
                ),
                (
                    "best_miou.pth.tar",
                    750,
                    "best_validation_miou_secondary",
                ),
                ("last.pth.tar", 799, "last_evaluated_epoch"),
            ):
                _write_evaluator_checkpoint(
                    run_dir / filename,
                    identity=identity,
                    metadata=metadata,
                    epoch=epoch,
                    role=role,
                )
            completed = _run_identity_lane(fixture)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "last checkpoint is not evaluated epoch 800",
                completed.stderr,
            )
            self.assertNotIn("TPDNERV8V3_LANE_READY", completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "the exact V3 identity validator requires the project torch env",
    )
    def test_wrong_existing_identity_is_rejected_without_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            run_dir = (
                result_root
                / "NUDT-SIRST"
                / VARIANT
                / "seed_42_formal800_exact_v3_seed42"
            )
            run_dir.mkdir(parents=True)
            protocol_path = run_dir / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "sctransnet_tpd_ner_v8_mprs_dch_v3_"
                            "exact_entry_v1"
                        ),
                        "run_identity": {
                            "dataset": "NUDT-SIRST",
                            "variant": (
                                "tpd_ner_v8_mprs_dch_v2_full_relay_on"
                            ),
                            "seed": 42,
                            "split_seed": 20260722,
                            "run_id": "tpd-ner-v8-mprs-dch-v2-exact:foreign",
                            "source_locks": {
                                (
                                    "tpd_ner_v8_mprs_dch_v2_"
                                    "exact_source_lock"
                                ): "foreign"
                            },
                            "training_contract": {
                                "determinism": {
                                    "entry_schema": (
                                        "sctransnet_tpd_ner_v8_mprs_dch_"
                                        "v2_exact_entry_v1"
                                    )
                                }
                            },
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            original_protocol = protocol_path.read_bytes()

            locks = {
                "TPD_NER_V8_V3_TRAINING_LOCK": root / "v3-training.json",
                "TPD_NER_V8_V3_ACCEPTANCE_LOCK": root / "v3-acceptance.json",
                "TPD_NER_V8_V3_UPSTREAM_V2_TRAINING_LOCK": (
                    root / "v2-training.json"
                ),
                "TPD_NER_V8_V3_UPSTREAM_V2_ACCEPTANCE_LOCK": (
                    root / "v2-acceptance.json"
                ),
            }
            for path in locks.values():
                path.write_text("{}\n", encoding="utf-8")

            command_log = root / "python.log"
            python_target = root / "identity-aware-python"
            _write_executable(
                python_target,
                """#!/usr/bin/env bash
set -euo pipefail
{
    printf 'CALL'
    for v3_test_argument in "$@"; do
        printf '\\t%s' "$v3_test_argument"
    done
    printf '\\n'
} >> "${V3_TEST_PYTHON_LOG:?}"
case "${1:-}" in
    */freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py)
        exit 0
        ;;
    -)
        if [[ "${2:-}" == GPU-* ]]; then
            echo "TPDNERV8V3_GPU_OK logical_device=cuda:0"
            exit 0
        fi
        exec "${V3_TEST_REAL_PYTHON:?}" "$@"
        ;;
    */train_tpd_ner_v8_mprs_dch_v3_exact.py)
        echo "trainer must not run" >&2
        exit 99
        ;;
esac
""",
            )
            python_link = root / "python-symlink"
            python_link.symlink_to(python_target)
            environment = os.environ.copy()
            environment.update(
                {
                    "TPD_NER_V8_V3_REPO": str(REPO_ROOT),
                    "TPD_NER_V8_V3_PYTHON": str(python_link),
                    "TPD_NER_V8_V3_RESULT_ROOT": str(result_root),
                    "V3_TEST_PYTHON_LOG": str(command_log),
                    "V3_TEST_REAL_PYTHON": sys.executable,
                }
            )
            environment.update(
                {label: str(path) for label, path in locks.items()}
            )

            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    str(LANE),
                    "--preflight",
                    "2",
                    GPU2_UUID,
                ],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("V1/V2/V8 trajectory, not V3", completed.stderr)
            self.assertEqual(protocol_path.read_bytes(), original_protocol)
            command_log_text = _read_if_present(command_log)
            self.assertNotIn(
                "train_tpd_ner_v8_mprs_dch_v3_exact.py\t--variant",
                command_log_text,
            )


if __name__ == "__main__":
    unittest.main()
