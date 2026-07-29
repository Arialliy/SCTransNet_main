from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "launch_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_2x5090.sh"
)
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)
FROZEN_SHA256 = {
    "experiments/train_tpd_ner_v4_qfg_v2_croa_dlr_exact.py": (
        "67cee6d59740af7ccaa9d7d1abfe62fa3f0c31a728deb8d7c400a675fb7c7190"
    ),
    "experiments/tpd_ner_v4_qfg_v2_croa_dlr_exact_source_lock.json": (
        "048558ee8b751847bd3f27afa4376be4a08bd158e0b73cdf6872185bfd406f88"
    ),
    "experiments/train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact.py": (
        "b2c6842aaa7674ff8f23ff09034d22e5491f36825232e33ec2559c1e0b76e3f6"
    ),
    (
        "experiments/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
    ): (
        "88b4839b40484c881544614e60675c4d2805a4fd6de1cc2f0aad28bdcb1395e8"
    ),
    (
        "experiments/"
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
    ): (
        "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478"
    ),
}


FAKE_PYTHON = r"""#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def record(kind: str, argv: list[str]) -> None:
    value = {
        "kind": kind,
        "argv": argv,
        "cwd": os.getcwd(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                *THREAD_NAMES,
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX",
                "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            )
        },
    }
    with Path(os.environ["FAKE_CALL_LOG"]).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def fail(message: str, status: int = 9) -> None:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(status)


arguments = sys.argv[1:]
if arguments and arguments[0] == "-":
    sys.stdin.read()
    if len(arguments) < 2:
        fail("missing embedded marker")
    marker = arguments[1]
    record(marker, arguments)
    if marker == "paired-source-lock-verify":
        if os.environ.get("FAKE_SOURCE_LOCK_FAIL") == "1":
            fail("fake source-lock verification failure")
        lock = Path(arguments[2])
        if lock.is_symlink() or not lock.is_file():
            fail("fake source lock is not regular")
        statistics = Path(arguments[4])
        if statistics.is_symlink() or not statistics.is_file():
            fail("fake target statistics are not regular")
        requested_mode = arguments[5]
        if requested_mode not in {"verify", "write-once"}:
            fail("fake source-lock mode differs")
        print(
            "TPDNER_DLR_RAMP100_SOURCE_LOCK_OK"
            f" requested_mode={requested_mode}"
            " action=verify-existing source_count=51"
            " sha256=fake",
            flush=True,
        )
        raise SystemExit(0)
    if marker == "paired-gpu-preflight":
        physical_index, expected_uuid = arguments[2:4]
        if os.environ.get("FAKE_GPU_FAIL") in {"all", physical_index}:
            fail("fake GPU verification failure")
        if GPU_UUIDS.get(physical_index) != expected_uuid:
            fail("fake physical-to-UUID mapping differs")
        expected_environment = {
            "CUDA_VISIBLE_DEVICES": expected_uuid,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX": physical_index,
            "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID": expected_uuid,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
        }
        for name, expected in expected_environment.items():
            if os.environ.get(name) != expected:
                fail(
                    f"fake GPU environment differs: {name}="
                    f"{os.environ.get(name)!r}, expected={expected!r}"
                )
        for name in THREAD_NAMES:
            if os.environ.get(name) != "1":
                fail(f"fake CPU thread contract differs: {name}")
        print(
            "TPDNER_DLR_RAMP100_GPU_OK"
            f" physical_index={physical_index}"
            f" uuid={expected_uuid}"
            " model=NVIDIA_GeForce_RTX_5090"
            " logical_device=cuda:0",
            flush=True,
        )
        raise SystemExit(0)
    if marker == "paired-run-state":
        variant = arguments[3]
        completed = (
            Path(os.environ["FAKE_STATE_DIR"]) / f"{variant}.complete"
        )
        if completed.is_file():
            print("complete", flush=True)
            raise SystemExit(0)
        key = f"FAKE_INITIAL_{variant.upper()}"
        print(os.environ.get(key, "parent-warm-start"), flush=True)
        raise SystemExit(0)
    fail(f"unknown fake embedded marker: {marker}")

record("trainer", arguments)
try:
    variant = arguments[arguments.index("--variant") + 1]
except (ValueError, IndexError):
    fail("fake trainer has no variant")
if os.environ.get("FAKE_FAIL_VARIANT") == variant:
    fail(f"fake trainer failure for {variant}", status=17)
if os.environ.get("FAKE_NO_COMPLETE_VARIANT") != variant:
    state_dir = Path(os.environ["FAKE_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{variant}.complete").write_text(
        "complete\n", encoding="utf-8"
    )
print(f"FAKE_TRAINER_COMPLETE variant={variant}", flush=True)
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option_value(arguments: list[str], name: str) -> str:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise AssertionError(f"missing launcher option {name}") from error


class PairedRamp100LauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.experiments = self.repo / "experiments"
        self.datasets = self.repo / "datasets"
        self.experiments.mkdir(parents=True)
        self.datasets.mkdir()

        self.trainer = (
            self.experiments
            / "train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact.py"
        )
        self.source_lock = (
            self.experiments
            / "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
        )
        self.statistics = (
            self.experiments
            / "tpd_survival_target_statistics_nudt_sirst_v1.json"
        )
        self.parent = self.root / "parent.pth.tar"
        self.fake_python = self.root / "fake_python"
        self.result_root = self.root / "formal-results"
        self.qfg_output_root = self.result_root / "qfg-output"
        self.tss_output_root = self.result_root / "tss-output"
        self.call_log = self.root / "calls.jsonl"
        self.state_dir = self.root / "fake-state"
        self.state_dir.mkdir()

        self.trainer.write_text("# fake trainer source\n", encoding="utf-8")
        self.source_lock.write_text(
            '{"schema":"frozen-51-source-lock","count":51}\n',
            encoding="utf-8",
        )
        self.statistics.write_text(
            '{"schema":"fake-statistics"}\n', encoding="utf-8"
        )
        self.parent.write_bytes(b"fake parent checkpoint\n")
        self.fake_python.write_text(
            textwrap.dedent(FAKE_PYTHON), encoding="utf-8"
        )
        self.fake_python.chmod(
            self.fake_python.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        self.environment = {
            **os.environ,
            "TPD_NER_DLR_RAMP100_REPO": str(self.repo),
            "TPD_NER_DLR_RAMP100_PYTHON": str(self.fake_python),
            "TPD_NER_DLR_RAMP100_TRAINER": str(self.trainer),
            "TPD_NER_DLR_RAMP100_SOURCE_LOCK": str(self.source_lock),
            "TPD_NER_DLR_RAMP100_STATISTICS": str(self.statistics),
            "TPD_NER_DLR_RAMP100_PARENT": str(self.parent),
            "TPD_NER_DLR_RAMP100_RESULT_ROOT": str(self.result_root),
            "TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT": str(
                self.qfg_output_root
            ),
            "TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT": str(
                self.tss_output_root
            ),
            "FAKE_CALL_LOG": str(self.call_log),
            "FAKE_STATE_DIR": str(self.state_dir),
            # Prove that the launcher replaces inherited thread settings.
            **{name: "9" for name in THREAD_NAMES},
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_launcher(
        self,
        *arguments: str,
        changes: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(self.environment)
        if changes:
            environment.update(changes)
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )

    def calls(self) -> list[dict]:
        if not self.call_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.call_log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def calls_of_kind(self, kind: str) -> list[dict]:
        return [call for call in self.calls() if call["kind"] == kind]

    def trainer_calls(self) -> dict[str, dict]:
        result = {}
        for call in self.calls_of_kind("trainer"):
            variant = option_value(call["argv"], "--variant")
            result[variant] = call
        return result

    def assert_lane_environment(
        self,
        call: dict,
        *,
        physical_index: str,
        uuid: str,
    ) -> None:
        environment = call["environment"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], uuid)
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(
            environment["TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX"],
            physical_index,
        )
        self.assertEqual(
            environment["TPD_NER_V4_QFG_PHYSICAL_GPU_UUID"], uuid
        )
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(environment["PYTHONHASHSEED"], "42")
        for name in THREAD_NAMES:
            self.assertEqual(environment[name], "1", name)

    def assert_trainer_contract(
        self,
        call: dict,
        *,
        variant: str,
        run_tag: str,
        output_root: Path,
        weight_max: str,
        initialization: str,
        physical_index: str,
        uuid: str,
    ) -> None:
        arguments = call["argv"]
        self.assertEqual(arguments[0], str(self.trainer))
        expected_options = {
            "--variant": variant,
            "--dataset": "NUDT-SIRST",
            "--dataset-dir": str(self.datasets),
            "--output-root": str(output_root),
            "--run-tag": run_tag,
            "--device": "cuda:0",
            "--epochs": "800",
            "--batch-size": "16",
            "--patch-size": "256",
            "--workers": "0",
            "--seed": "42",
            "--split-seed": "20260722",
            "--val-fraction": "0.20",
            "--eval-every": "1",
            "--base-lr": "0.0001",
            "--min-lr": "0.000001",
            "--warmup-epochs": "10",
            "--threshold": "0.5",
            "--match-radius": "3.0",
            "--tiny-area": "9",
            "--eps": "0.000001",
            "--survival-weight-max": weight_max,
            "--survival-target-statistics": str(self.statistics),
            "--parent-checkpoint": str(self.parent),
            "--exact-source-lock": str(self.source_lock),
        }
        for name, expected in expected_options.items():
            self.assertEqual(option_value(arguments, name), expected, name)
        self.assertIn(initialization, arguments)
        other = (
            "--exact-resume"
            if initialization == "--parent-warm-start"
            else "--parent-warm-start"
        )
        self.assertNotIn(other, arguments)
        self.assert_lane_environment(
            call, physical_index=physical_index, uuid=uuid
        )

    def test_shell_parses_is_executable_and_frozen_sources_are_unchanged(
        self,
    ) -> None:
        parsed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertTrue(os.access(LAUNCHER, os.X_OK))
        for relative, expected in FROZEN_SHA256.items():
            self.assertEqual(sha256(REPO_ROOT / relative), expected, relative)

    def test_static_contract_has_fixed_mapping_lock_and_no_idle_wait(
        self,
    ) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for expected in (
            GPU2_UUID,
            GPU3_UUID,
            "qfg_dlr:2:",
            "tss_qfg_dlr:3:",
            "flock -n 9",
            'paired_exit_permanent=64',
            'paired_exit_retry=75',
            "action=verify-existing",
            "logical_device=cuda:0",
            "--epochs 800",
            "--seed 42",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("\nsleep ", source)
        self.assertNotIn("memory.free", source)
        self.assertNotIn("utilization.gpu", source)

    def test_preflight_is_read_only_and_checks_both_isolated_lanes(
        self,
    ) -> None:
        result = self.run_launcher("--preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIRED_PREFLIGHT_OK", result.stdout)
        self.assertIn("writes_performed=false", result.stdout)
        self.assertFalse(self.result_root.exists())
        self.assertEqual(len(self.calls_of_kind("trainer")), 0)
        self.assertEqual(len(self.calls_of_kind("paired-source-lock-verify")), 1)
        gpu_calls = self.calls_of_kind("paired-gpu-preflight")
        self.assertEqual(len(gpu_calls), 2)
        by_index = {call["argv"][2]: call for call in gpu_calls}
        self.assertEqual(set(by_index), {"2", "3"})
        self.assert_lane_environment(
            by_index["2"], physical_index="2", uuid=GPU2_UUID
        )
        self.assert_lane_environment(
            by_index["3"], physical_index="3", uuid=GPU3_UUID
        )
        states = self.calls_of_kind("paired-run-state")
        self.assertEqual(
            {call["argv"][3] for call in states},
            {"qfg_dlr", "tss_qfg_dlr"},
        )
        source_call = self.calls_of_kind("paired-source-lock-verify")[0]
        for name in THREAD_NAMES:
            self.assertEqual(source_call["environment"][name], "1", name)

    def test_write_once_spelling_only_verifies_existing_lock(self) -> None:
        before = self.source_lock.read_bytes()
        result = self.run_launcher(
            "--preflight", "--source-lock-mode", "write-once"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.source_lock.read_bytes(), before)
        self.assertIn("requested_mode=write-once", result.stdout)
        self.assertIn("source_lock_action=verify-existing", result.stdout)
        source_call = self.calls_of_kind("paired-source-lock-verify")[0]
        self.assertEqual(source_call["argv"][4], str(self.statistics))
        self.assertEqual(source_call["argv"][5], "write-once")

    def test_parent_warm_start_launches_exact_paired_commands(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIRED_LAUNCHED", result.stdout)
        self.assertIn("PAIRED_COMPLETE", result.stdout)
        trainers = self.trainer_calls()
        self.assertEqual(set(trainers), {"qfg_dlr", "tss_qfg_dlr"})
        self.assert_trainer_contract(
            trainers["qfg_dlr"],
            variant="qfg_dlr",
            run_tag="formal800_qfg_dlr_control",
            output_root=self.qfg_output_root,
            weight_max="0.0",
            initialization="--parent-warm-start",
            physical_index="2",
            uuid=GPU2_UUID,
        )
        self.assert_trainer_contract(
            trainers["tss_qfg_dlr"],
            variant="tss_qfg_dlr",
            run_tag="formal800_tss_qfg_dlr_ramp100",
            output_root=self.tss_output_root,
            weight_max="0.005",
            initialization="--parent-warm-start",
            physical_index="3",
            uuid=GPU3_UUID,
        )
        self.assertNotEqual(self.qfg_output_root, self.tss_output_root)
        claim = (
            self.result_root
            / ".launcher_locks"
            / "formal800_seed42_ramp100_paired.lock"
        )
        self.assertTrue(claim.is_file())
        self.assertTrue(
            (self.result_root / "logs" / "qfg_dlr_gpu2.log").is_file()
        )
        self.assertTrue(
            (self.result_root / "logs" / "tss_qfg_dlr_gpu3.log").is_file()
        )

    def test_exact_resume_is_selected_independently_for_both_lanes(
        self,
    ) -> None:
        result = self.run_launcher(
            changes={
                "FAKE_INITIAL_QFG_DLR": "exact-resume",
                "FAKE_INITIAL_TSS_QFG_DLR": "exact-resume",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        trainers = self.trainer_calls()
        self.assertEqual(set(trainers), {"qfg_dlr", "tss_qfg_dlr"})
        for call in trainers.values():
            self.assertIn("--exact-resume", call["argv"])
            self.assertNotIn("--parent-warm-start", call["argv"])

    def test_complete_lane_is_skipped_while_other_lane_runs(self) -> None:
        result = self.run_launcher(
            changes={"FAKE_INITIAL_QFG_DLR": "complete"}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        trainers = self.trainer_calls()
        self.assertEqual(set(trainers), {"tss_qfg_dlr"})
        self.assertIn("--parent-warm-start", trainers["tss_qfg_dlr"]["argv"])

    def test_two_complete_lanes_return_success_without_trainer(self) -> None:
        result = self.run_launcher(
            changes={
                "FAKE_INITIAL_QFG_DLR": "complete",
                "FAKE_INITIAL_TSS_QFG_DLR": "complete",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PAIRED_COMPLETE", result.stdout)
        self.assertEqual(self.calls_of_kind("trainer"), [])

    def test_busy_paired_claim_is_retryable_and_never_starts_trainer(
        self,
    ) -> None:
        claim_dir = self.result_root / ".launcher_locks"
        claim_dir.mkdir(parents=True)
        claim = claim_dir / "formal800_seed42_ramp100_paired.lock"
        with claim.open("w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_launcher()
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("reason=paired_launch_claim_busy", result.stderr)
        self.assertEqual(self.calls_of_kind("trainer"), [])

    def test_lane_failure_reaps_pair_and_returns_retryable_status(self) -> None:
        result = self.run_launcher(
            changes={"FAKE_FAIL_VARIANT": "qfg_dlr"}
        )
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("reason=lane_failed", result.stderr)
        self.assertEqual(
            set(self.trainer_calls()), {"qfg_dlr", "tss_qfg_dlr"}
        )

    def test_success_without_completion_is_retryable(self) -> None:
        result = self.run_launcher(
            changes={"FAKE_NO_COMPLETE_VARIANT": "tss_qfg_dlr"}
        )
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("reason=postrun_incomplete", result.stderr)
        self.assertEqual(
            set(self.trainer_calls()), {"qfg_dlr", "tss_qfg_dlr"}
        )

    def test_source_or_gpu_preflight_failure_is_permanent_and_read_only(
        self,
    ) -> None:
        for changes, reason in (
            (
                {"FAKE_SOURCE_LOCK_FAIL": "1"},
                "source_lock_live_verify_failed",
            ),
            ({"FAKE_GPU_FAIL": "3"}, "tss_qfg_dlr_preflight_failed"),
        ):
            with self.subTest(changes=changes):
                if self.call_log.exists():
                    self.call_log.unlink()
                result = self.run_launcher("--preflight", changes=changes)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn(f"reason={reason}", result.stderr)
                self.assertEqual(self.calls_of_kind("trainer"), [])
                self.assertFalse(self.result_root.exists())

    def test_usage_and_invalid_mode_have_distinct_permanent_contracts(
        self,
    ) -> None:
        usage = self.run_launcher("--unknown")
        self.assertEqual(usage.returncode, 2)
        invalid = self.run_launcher(
            "--preflight", "--source-lock-mode", "replace"
        )
        self.assertEqual(invalid.returncode, 64)
        self.assertIn("reason=invalid_source_lock_mode", invalid.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
