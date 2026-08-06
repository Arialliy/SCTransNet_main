from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from experiments import train_three_dataset_gcsf_tss_off_seed42_v1 as trainer
from experiments import train_three_dataset_seed42_global_tss_v2 as positive
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf import (
    GCSF_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFSurvivalSCTransNet,
)


torch.set_num_threads(1)


def _formal_args(dataset: str = "NUAA-SIRST") -> argparse.Namespace:
    return trainer.parse_args(
        [
            "--dataset",
            dataset,
            "--method",
            "final",
            "--physical-gpu-index",
            "0",
            "--expected-gpu-uuid",
            trainer.GPU_UUIDS["0"],
        ]
    )


def _minimal_authorized_payload() -> dict[str, object]:
    keys = [
        f"{dataset}::{role}"
        for dataset in trainer.DATASETS
        for role in trainer.comparator.CHECKPOINT_ROLES
    ]
    return {
        "schema": trainer.comparator.SCHEMA,
        "status": "complete",
        "decision": trainer.comparator.DECISION_AUTHORIZE,
        "trigger_a": {
            "implemented": True,
            "sole_training_authorization_trigger": True,
            "passed": True,
            "qualifying_modes": ["gneg025_l1_only"],
        },
        "trigger_b": {"authorizes_training": False},
        "trigger_c": {"authorizes_training": False},
        "gcsf_v1_implementation_and_pilot_authorized": True,
        "input_bindings": {key: {} for key in keys},
        "per_unit": {key: {} for key in keys},
        "source_sha256": trainer._expected_comparator_sources(),
    }


class GCSFProtocolAndBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = trainer._build_method_model(
            "final",
            42,
            dataset_name="NUAA-SIRST",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        del cls.metadata

    def test_formal_constants_recipe_and_selection_contract(self) -> None:
        args = _formal_args()
        self.assertEqual(trainer.DATASETS, ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"))
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.epochs, 1000)
        self.assertEqual(args.begin_test, 10)
        self.assertEqual(args.eval_every, 10)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.patch_size, 256)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.base_lr, 1e-3)
        self.assertEqual(args.min_lr, 1e-5)
        self.assertEqual(args.warmup_epochs, 10)
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(trainer.CHECKPOINT_ROLES, ("best_miou", "best_pd"))
        recipe = trainer.recipe_identity(args)
        self.assertEqual(recipe["recipe_id"], "final_tss_off_gcsf_v1")
        self.assertEqual(recipe["requested_tss_weight"], 0.0)
        self.assertFalse(recipe["tss_enabled"])
        self.assertTrue(recipe["fresh_seed42_scratch"])
        self.assertIsNone(recipe["parent_checkpoint"])
        self.assertFalse(recipe["warm_start_used"])

    def test_builder_is_fresh_exact_gcsf_graph_and_optimizer_owns_gate(self) -> None:
        self.assertIs(
            type(self.model),
            TPDNERV8MPRSDCHV4QFGV2CROAGCSFSurvivalSCTransNet,
        )
        self.assertEqual(len(self.model.state_dict()), trainer.TRAINING_STATE_KEY_COUNT)
        self.assertEqual(
            self.metadata["initialization_mode"],
            "fresh_seed42_paired_scratch_extension",
        )
        self.assertEqual(self.metadata["construction"], "scratch_seed42_no_parent_checkpoint")
        self.assertIsNone(self.metadata["parent_checkpoint"])
        self.assertFalse(self.metadata["learned_state_loaded"])
        self.assertTrue(self.metadata["all_pre_gcsf_state_bitwise_equal_to_reference"])
        self.assertTrue(self.metadata["gcsf_new_state_zero_initialized"])
        for key in GCSF_STATE_KEYS:
            self.assertEqual(int(torch.count_nonzero(self.model.state_dict()[key])), 0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        owned = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            {id(parameter) for parameter in self.model.global_skip_fusion.parameters()} <= owned
        )

    def test_runtime_closure_is_explicit_and_contains_gcsf_and_protocol_sources(self) -> None:
        paths = trainer.runtime_source_paths()
        observed = {key.split("::", 1)[1] for key in paths}
        required = {
            "experiments/train_three_dataset_gcsf_tss_off_seed42_v1.py",
            "experiments/train_three_dataset_tss_off_seed42_v1.py",
            "experiments/train_three_dataset_seed42_global_tss_v2.py",
            "model/tpd_global_constant_sum_skip_fusion.py",
            "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf.py",
            "analysis/compare_three_dataset_gcsf_branch_audit_v1.py",
        }
        self.assertTrue(required <= observed)
        self.assertTrue(all(path.is_file() for path in paths.values()))
        self.assertLess(
            len([item for item in observed if item.startswith("model/")]),
            100,
        )


class GCSFTriggerAGuardTests(unittest.TestCase):
    def test_contract_rejects_failed_or_nonsole_trigger_a(self) -> None:
        failed = _minimal_authorized_payload()
        failed["decision"] = trainer.comparator.DECISION_NO_AUTHORIZATION
        failed["gcsf_v1_implementation_and_pilot_authorized"] = False
        failed["trigger_a"]["passed"] = False  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "decision"):
            trainer._validate_trigger_a_authorization(failed)

        not_sole = _minimal_authorized_payload()
        not_sole["trigger_a"]["sole_training_authorization_trigger"] = False  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "sole_training_authorization_trigger"):
            trainer._validate_trigger_a_authorization(not_sole)

    def test_load_guard_checks_each_bound_input_hash_and_replays_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _minimal_authorized_payload()
            bindings = payload["input_bindings"]
            self.assertIsInstance(bindings, dict)
            for index, key in enumerate(tuple(bindings)):
                path = root / f"input_{index}.json"
                path.write_text(json.dumps({"key": key}), encoding="utf-8")
                bindings[key] = {
                    "path": str(path),
                    "sha256": trainer.engine.file_sha256(path),
                }
            decision_path = root / "decision.json"
            decision_path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(
                trainer.comparator,
                "compare_payloads",
                return_value=copy.deepcopy(payload),
            ) as replay:
                binding = trainer.load_trigger_a_decision(decision_path)
            self.assertTrue(binding["trigger_a_passed"])
            self.assertTrue(binding["replayed_from_six_bound_inputs"])
            self.assertEqual(binding["sha256"], trainer.engine.file_sha256(decision_path))
            replay.assert_called_once()

            first = Path(next(iter(bindings.values()))["path"])
            first.write_text('{"tampered": true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input SHA"):
                trainer.load_trigger_a_decision(decision_path)

    def test_formal_requires_the_fixed_comparator_path(self) -> None:
        args = _formal_args()
        args.gcsf_decision_json = Path("/tmp/not-the-frozen-decision.json")
        with self.assertRaisesRegex(ValueError, "formal comparator path"):
            trainer.validate_args(args)


class GCSFProtocolArtifactTests(unittest.TestCase):
    def test_protocol_binds_trigger_scratch_fp32_adam_and_pause_resume(self) -> None:
        args = _formal_args()
        decision = {
            "path": str(trainer.DEFAULT_DECISION_JSON),
            "sha256": "a" * 64,
            "schema": trainer.comparator.SCHEMA,
            "decision": trainer.comparator.DECISION_AUTHORIZE,
            "trigger_a_passed": True,
            "qualifying_modes": ["gneg025_l1_only"],
            "replayed_from_six_bound_inputs": True,
            "source_sha256": {},
        }
        seed_payload = {
            "training": {},
            "runtime_sources": {"forbidden": {}},
            "rolling_resume_state": {},
        }
        args.gcsf_trigger_a_decision_binding = decision
        with mock.patch.object(
            trainer, "_BASE_PROTOCOL_PAYLOAD", return_value=seed_payload
        ):
            payload = trainer._protocol_payload(
                args,
                model_metadata=self._metadata_stub(),
                tss_metadata={},
                data_manifests={},
                train_count=1,
                test_count=1,
                device=torch.device("cpu"),
            )
        self.assertEqual(payload["schema"], trainer.SCHEMA)
        self.assertEqual(payload["gcsf_trigger_a_authorization"], decision)
        self.assertEqual(payload["training"]["optimizer"], "Adam")
        self.assertEqual(payload["training"]["precision"], "FP32")
        self.assertFalse(payload["training"]["amp"])
        self.assertEqual(payload["training"]["initialization"], "fresh_seed42_scratch")
        self.assertIsNone(payload["training"]["parent_checkpoint"])
        pause = payload["pause_resume_contract"]
        self.assertEqual(pause["pause_epoch"], 200)
        self.assertTrue(pause["pilot_is_prefix_of_same_run"])
        self.assertFalse(pause["pilot_creates_additional_run"])
        self.assertEqual(set(payload["runtime_sources"]), set(trainer.runtime_source_records()))

    @staticmethod
    def _metadata_stub() -> dict[str, object]:
        return {
            "architecture_id": "architecture",
            "architecture_manifest": {},
            "initialization_mode": "fresh_seed42_scratch",
            "parent_checkpoint": None,
        }

    def test_resume_rejects_foreign_schema_before_state_load(self) -> None:
        class Dummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))

            def architecture_manifest(self) -> dict[str, object]:
                return {"graph": "dummy"}

        args = _formal_args()
        args.resume = "required"
        args.gcsf_trigger_a_decision_binding = {"sha256": "d" * 64}
        model = Dummy()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pth.tar"
            torch.save({"schema": "historical-current"}, path)
            with self.assertRaisesRegex(ValueError, "resume schema"):
                trainer._load_resume_gcsf(
                    args=args,
                    path=path,
                    model=model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="protocol",
                )

    def test_orphaned_partial_run_is_not_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _formal_args()
            args.results_root = Path(directory)
            run_dir = trainer._run_directory(args)
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints/best_miou.pth.tar").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "refusing implicit overwrite"):
                trainer._validate_existing_run_artifacts(args)


class GCSFCpuPauseResumeTests(unittest.TestCase):
    class TinyTrain(Dataset):
        def __init__(self, dataset: str, **kwargs: object) -> None:
            self.normalization = positive.data_protocol.get_legacy_normalization(dataset)
            self.epoch = 0

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            image = torch.full((1, 256, 256), float(index) / 10.0)
            return image, torch.zeros_like(image)

    class TinyTest(Dataset):
        def __init__(self, train_dataset: str, test_dataset: str, **kwargs: object) -> None:
            self.normalization = positive.data_protocol.get_legacy_normalization(train_dataset)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            image = torch.zeros(1, 32, 32)
            return image, torch.zeros_like(image), (32, 32), "mock_0"

    class TinyGCSF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit = nn.Parameter(torch.tensor(-1.0))

        def architecture_manifest(self) -> dict[str, object]:
            return {"graph": "tiny_gcsf", "seed": 42}

        def forward(self, image: torch.Tensor):
            probability = torch.sigmoid(self.logit) * torch.ones_like(image)
            return tuple(probability for _ in range(6))

    @staticmethod
    def _builder(method: str, seed: int, *, dataset_name: str):
        return GCSFCpuPauseResumeTests.TinyGCSF(), {
            "method": method,
            "seed": seed,
            "dataset_name": dataset_name,
            "formal_training_objective": {"tss_enabled": False},
        }

    @staticmethod
    def _args(root: Path, *, pause: int | None, resume: str) -> argparse.Namespace:
        argv = [
            "--dataset",
            "NUAA-SIRST",
            "--method",
            "final",
            "--results-root",
            str(root),
            "--gcsf-decision-json",
            str(root / "decision.json"),
            "--smoke",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--begin-test",
            "1",
            "--eval-every",
            "1",
            "--batch-size",
            "1",
            "--max-train-images",
            "2",
            "--max-test-images",
            "1",
            "--resume",
            resume,
        ]
        if pause is not None:
            argv.extend(["--pause-after-epoch", str(pause)])
        return trainer.parse_args(argv)

    @staticmethod
    def _decision_binding(path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "sha256": "d" * 64,
            "schema": trainer.comparator.SCHEMA,
            "decision": trainer.comparator.DECISION_AUTHORIZE,
            "trigger_a_passed": True,
            "qualifying_modes": ["gneg025_l1_only"],
            "replayed_from_six_bound_inputs": True,
            "source_sha256": {},
        }

    def tearDown(self) -> None:
        trainer.base._AUDIT.reset()

    def test_pause_at_durable_epoch_then_exact_same_run_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            continuous_root = root / "continuous"
            split_root = root / "split"
            components = (
                self._builder,
                self.TinyTrain,
                self.TinyTest,
            )

            def execute(args: argparse.Namespace) -> Path:
                binding = self._decision_binding(args.gcsf_decision_json)
                with (
                    mock.patch.object(trainer, "load_trigger_a_decision", return_value=binding),
                    mock.patch.object(trainer, "_import_runtime_components", return_value=components),
                ):
                    return trainer.run(args)

            continuous_summary_path = execute(
                self._args(continuous_root, pause=None, resume="never")
            )
            paused_path = execute(self._args(split_root, pause=1, resume="never"))
            paused = json.loads(paused_path.read_text(encoding="utf-8"))
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["completed_epoch"], 1)
            self.assertTrue(paused["resume_required"])
            latest = Path(paused["rolling_resume_state"]["path"])
            self.assertTrue(latest.is_file())
            rolling = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(rolling["epoch"], 1)
            self.assertIn("optimizer", rolling)
            self.assertIn("rng_state", rolling)

            split_summary_path = execute(
                self._args(split_root, pause=None, resume="required")
            )
            continuous = json.loads(continuous_summary_path.read_text(encoding="utf-8"))
            split = json.loads(split_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(continuous["status"], split["status"])
            self.assertEqual(set(continuous["checkpoints"]), {"best_miou", "best_pd"})
            self.assertEqual(set(split["checkpoints"]), {"best_miou", "best_pd"})
            self.assertFalse(latest.exists())

            for role in ("best_miou", "best_pd"):
                left = torch.load(
                    continuous_summary_path.parent / "checkpoints" / f"{role}.pth.tar",
                    map_location="cpu",
                    weights_only=False,
                )
                right = torch.load(
                    split_summary_path.parent / "checkpoints" / f"{role}.pth.tar",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(left["epoch"], right["epoch"])
                self.assertEqual(left["test_metrics"], right["test_metrics"])
                self.assertEqual(set(left["state_dict"]), set(right["state_dict"]))
                for name in left["state_dict"]:
                    self.assertTrue(torch.equal(left["state_dict"][name], right["state_dict"][name]))


if __name__ == "__main__":
    unittest.main()
