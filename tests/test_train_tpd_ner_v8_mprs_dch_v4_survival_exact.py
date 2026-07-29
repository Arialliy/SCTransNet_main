from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from experiments import tpd_exact_runner as exact_runner
from experiments import tpd_training_loss
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as entry,
)
from model.tpd_forward_contract import TPDForwardOutput
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    TPDNERV8MPRSDCHV4SurvivalSCTransNet,
)


class StateScaler:
    def state_dict(self) -> dict[str, int]:
        return {"updates": 0}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if state != {"updates": 0}:
            raise ValueError("unexpected scaler fixture state")


class TinySurvivalModel(nn.Module):
    """A tiny forward graph with the formal 544+4 state-key topology."""

    def __init__(self) -> None:
        super().__init__()
        self.parent_state = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(544)]
        )
        self.target_survival = nn.Module()
        self.target_survival.heads = nn.ModuleDict(
            {
                "emb1": nn.ModuleDict(
                    {"classifier": nn.Conv2d(1, 1, 1)}
                ),
                "emb2": nn.ModuleDict(
                    {"classifier": nn.Conv2d(2, 1, 1)}
                ),
            }
        )
        with torch.no_grad():
            for parameter in self.target_survival.parameters():
                parameter.zero_()

    def forward(self, images: torch.Tensor):
        scalar = self.parent_state[0]
        probability = torch.sigmoid(images * 0.0 + scalar)
        segmentation = tuple(probability for _ in range(6))
        if not self.training:
            return segmentation
        endpoint1 = nn.functional.avg_pool2d(images, 16)
        endpoint2 = torch.cat((endpoint1, endpoint1), dim=1)
        logits1 = self.target_survival.heads["emb1"][
            "classifier"
        ](endpoint1)
        logits2 = self.target_survival.heads["emb2"][
            "classifier"
        ](endpoint2)
        return TPDForwardOutput(
            segmentation=segmentation,
            emb1_endpoint=endpoint1,
            emb2_endpoint=endpoint2,
            emb1_survival_logits=logits1,
            emb2_survival_logits=logits2,
        )


class TinyTrainingSet(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        image = torch.full((1, 32, 32), float(index) / 16.0)
        mask = torch.zeros(1, 32, 32)
        mask[:, 4:7, 5:8] = 1.0
        return image, mask


class TinyValidationSet(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        image = torch.zeros(1, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 4:7, 5:8] = 1.0
        return image, mask, torch.tensor([32, 32]), "validation"


def parse(variant: str, trailing: list[str] | None = None):
    return entry.parse_args(
        [
            "--variant",
            variant,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--fresh",
            *(trailing or []),
        ]
    )


def extension_provenance() -> dict:
    return {
        "schema": entry.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(
            entry.PARENT_CHECKPOINT_PATH.resolve()
        ),
        "parent_checkpoint_sha256": entry.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": ["state_dict"],
        "parent_state_key_count": 544,
        "preserved_new_state_key_count": 4,
        "new_module_prefixes": ["target_survival"],
        "zero_init_prefixes": [
            "target_survival.heads.emb1.classifier",
            "target_survival.heads.emb2.classifier",
        ],
    }


def validation_metrics() -> dict[str, int | float]:
    count_fields = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    return {
        name: 1 if name in count_fields else 0.5
        for name in entry.STORED_VALIDATION_METRICS
    }


class V4SurvivalExactTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.args = parse(entry.TSS_CONTROL_VARIANT)
        cls.model, cls.metadata = entry.build_selected_model(
            entry.TSS_CONTROL_VARIANT,
            42,
        )
        cls.initial_sha = exact_runner.initial_model_state_sha256(cls.model)
        cls.initialization_contract = (
            exact_runner.extension_parent_initialization_contract(
                extension_provenance(),
                loaded_child_model_state_sha256=cls.initial_sha,
            )
        )
        cls.optimizer = torch.optim.Adam(
            cls.model.parameters(),
            lr=entry.FORMAL_BASE_LR,
        )
        statistics = entry.load_survival_target_statistics()
        cls.spec = entry.make_exact_run_spec(
            cls.args,
            model=cls.model,
            model_metadata=cls.metadata,
            optimizer=cls.optimizer,
            scaler=StateScaler(),
            initialization_contract=cls.initialization_contract,
            initial_model_state_sha256=cls.initial_sha,
            initial_rng=exact_runner.initial_rng_contract(),
            selection_policy=exact_runner.pd_miou_selection_policy(
                stored_metrics=entry.STORED_VALIDATION_METRICS
            ).normalized(),
            source_locks={
                entry.SOURCE_LOCK_KEY: "1" * 64,
                "training_data": "2" * 64,
                "survival_target_statistics": statistics["sha256"],
                "parent_checkpoint": entry.PARENT_CHECKPOINT_SHA256,
            },
            split_records={
                "train": exact_runner.OrderedFingerprint.from_values(
                    "train",
                    ("a", "b"),
                ),
                "validation": exact_runner.OrderedFingerprint.from_values(
                    "validation",
                    ("c",),
                ),
            },
            data_records={
                "train_samples": (
                    exact_runner.OrderedFingerprint.from_values(
                        "train_samples",
                        ("a:image", "b:image"),
                    )
                ),
            },
            environment={"name": "cpu-survival-fixture"},
        )
        cls.identity = exact_runner.build_run_identity(
            cls.model,
            cls.spec,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del (
            cls.identity,
            cls.spec,
            cls.optimizer,
            cls.initialization_contract,
            cls.metadata,
            cls.model,
            cls.args,
        )
        torch.set_num_threads(cls.previous_threads)

    def test_parser_freezes_paired_axes_and_only_weight_differs(self) -> None:
        control = self.args
        enabled = parse(entry.TSS_ON_VARIANT)
        self.assertEqual(control.survival_weight, 0.0)
        self.assertEqual(enabled.survival_weight, 0.005)
        self.assertEqual(
            control.survival_pos_weight,
            enabled.survival_pos_weight,
        )
        for name, expected in (
            ("seed", 42),
            ("split_seed", 20260722),
            ("epochs", 800),
            ("batch_size", 16),
            ("patch_size", 256),
            ("workers", 0),
            ("base_lr", 1e-4),
            ("min_lr", 1e-6),
            ("warmup_epochs", 10),
            ("amp", False),
        ):
            self.assertEqual(getattr(control, name), expected, msg=name)
            self.assertEqual(getattr(enabled, name), expected, msg=name)
        self.assertEqual(control.run_tag, entry.FORMAL_CONTROL_RUN_TAG)
        self.assertEqual(enabled.run_tag, entry.FORMAL_TSS_RUN_TAG)
        self.assertTrue(control.parent_warm_start)
        self.assertTrue(control.fresh)

        invalid = (
            ["--seed", "7"],
            ["--base-lr", "0.001"],
            ["--survival-weight", "0.005"],
            ["--survival-pos-weight", "1.0"],
            ["--run-tag", "changed"],
        )
        for trailing in invalid:
            with self.subTest(trailing=trailing):
                with self.assertRaises((ValueError, SystemExit)):
                    parse(entry.TSS_CONTROL_VARIANT, trailing)

    def test_protocol_arguments_do_not_depend_on_transient_startup_mode(
        self,
    ) -> None:
        warm_start = self.args
        exact_resume = entry.parse_args(
            [
                "--variant",
                entry.TSS_CONTROL_VARIANT,
                "--device",
                "cpu",
                "--allow-cpu-smoke",
                "--exact-resume",
            ]
        )
        warm_arguments = entry.training_arguments(warm_start)
        resume_arguments = entry.training_arguments(exact_resume)
        self.assertEqual(warm_arguments, resume_arguments)
        self.assertNotIn("parent_warm_start", warm_arguments)
        self.assertNotIn("exact_resume", warm_arguments)

    def test_statistics_and_locked_parent_identity_are_authoritative(
        self,
    ) -> None:
        statistics = entry.load_survival_target_statistics()
        self.assertEqual(statistics["positive_cells"], 1313)
        self.assertEqual(statistics["negative_cells"], 134367)
        self.assertEqual(statistics["total_cells"], 135680)
        self.assertEqual(
            statistics["survival_pos_weight"],
            134367 / 1313,
        )
        parent = entry.validate_parent_checkpoint()
        self.assertEqual(parent["checkpoint_epoch"], 489)
        self.assertEqual(parent["checkpoint_role"], "best_miou")
        self.assertEqual(
            parent["checkpoint_sha256"],
            entry.PARENT_CHECKPOINT_SHA256,
        )
        self.assertEqual(
            parent["state_dict_sha256"],
            entry.PARENT_STATE_DICT_SHA256,
        )
        self.assertEqual(parent["state_key_count"], 544)

    def test_builder_manifest_and_exact_identity_are_tss_owned(self) -> None:
        self.assertIs(
            type(self.model),
            TPDNERV8MPRSDCHV4SurvivalSCTransNet,
        )
        self.assertEqual(len(self.model.state_dict()), 548)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            10_854_544,
        )
        self.assertEqual(
            {
                name
                for name in self.model.state_dict()
                if name.startswith(entry.SURVIVAL_STATE_PREFIX)
            },
            set(entry.SURVIVAL_STATE_KEYS),
        )
        manifest = self.metadata["architecture_manifest"]
        self.assertEqual(
            manifest["schema"],
            entry.ARCHITECTURE_MANIFEST_SCHEMA,
        )
        self.assertEqual(manifest["state_key_count"], 548)
        self.assertEqual(manifest["total_parameters"], 10_854_544)

        normalized = self.spec.normalized()
        loss = normalized["loss"]
        self.assertEqual(
            loss["segmentation"]["aggregate"],
            "python_ordered_sum",
        )
        self.assertEqual(loss["survival"]["survival_weight"], 0.0)
        self.assertFalse(
            loss["survival"]["disabled_path_builds_target"]
        )
        identity = entry.require_tss_run_identity(
            self.identity,
            label="fixture",
            expected_variant=entry.TSS_CONTROL_VARIANT,
        )
        determinism = identity["training_contract"]["determinism"]
        self.assertEqual(
            determinism["tss_run_identity_schema"],
            entry.RUN_IDENTITY_SCHEMA,
        )
        self.assertEqual(
            determinism["initialization_mode"],
            exact_runner.EXTENSION_PARENT_MODE,
        )
        self.assertTrue(determinism["continued_training_control"])

        wrong = copy.deepcopy(self.identity)
        wrong["training_contract"]["loss"]["survival"][
            "survival_weight"
        ] = 0.005
        with self.assertRaisesRegex(ValueError, "loss contract"):
            entry.require_tss_run_identity(wrong, label="wrong weight")

    def test_control_loss_and_log_fields_do_not_build_survival_target(
        self,
    ) -> None:
        target = torch.zeros(2, 1, 32, 32)
        probability = torch.full(
            (2, 1, 32, 32),
            0.25,
            requires_grad=True,
        )
        legacy_output = tuple(probability for _ in range(6))
        with mock.patch.object(
            tpd_training_loss,
            "build_survival_target",
            side_effect=AssertionError("Y16 must not be built"),
        ):
            losses = entry.compute_stage_loss(
                legacy_output,
                target,
                nn.BCELoss(),
                survival_weight=0.0,
                survival_pos_weight=self.args.survival_pos_weight,
            )
        self.assertEqual(losses.survival_terms, ())
        accumulator = entry.EpochLossAccumulator(
            survival_enabled=False
        )
        accumulator.update(losses, 2)
        fields = accumulator.fields()
        self.assertEqual(
            fields["train_total_loss"],
            fields["train_segmentation_loss"],
        )
        self.assertEqual(fields["train_survival_loss"], 0.0)
        self.assertIsNone(fields["train_survival_emb1_loss"])
        self.assertIsNone(fields["train_survival_emb2_loss"])

    def test_checkpoint_adapter_emits_full_extension_state_contract(
        self,
    ) -> None:
        state = {
            f"parent.fixture.{index}": torch.tensor(float(index))
            for index in range(544)
        }
        state.update(
            {
                name: torch.zeros(1)
                for name in entry.SURVIVAL_STATE_KEYS
            }
        )
        adapter = entry.EvaluatorCheckpointAdapter(
            model_metadata=self.metadata,
            split_hashes={"train": "a" * 64},
        )
        checkpoint = adapter(
            exact_runner.CompatibilityPayloadContext(
                role="best_validation_miou_secondary",
                epoch=1,
                metrics=validation_metrics(),
                event={"epoch": 1, **validation_metrics()},
                exact_payload={
                    "model": {"state_dict": state},
                    "optimizer": {"state_dict": {"state": {}}},
                    "scaler": {"state_dict": {"updates": 0}},
                },
                run_identity=self.identity,
                normalized_spec=self.spec.normalized(),
            )
        )
        self.assertEqual(checkpoint["schema"], entry.CHECKPOINT_SCHEMA)
        self.assertEqual(len(checkpoint["state_dict"]), 548)
        self.assertTrue(checkpoint["continued_training_control"])
        self.assertEqual(checkpoint["survival_weight"], 0.0)
        self.assertEqual(
            checkpoint["initialization_mode"],
            exact_runner.EXTENSION_PARENT_MODE,
        )
        entry.require_evaluator_checkpoint_payload(
            checkpoint,
            expected_variant=entry.TSS_CONTROL_VARIANT,
        )

    def test_cli_control_runs_one_tiny_cpu_epoch_and_commits(self) -> None:
        args = parse(entry.TSS_CONTROL_VARIANT)
        args.epochs = 1
        toy_model = TinySurvivalModel()
        self.assertEqual(len(toy_model.state_dict()), 548)
        initial_sha = exact_runner.initial_model_state_sha256(toy_model)
        request = exact_runner.InitializationRequest.extension_parent(
            extension_provenance(),
            loaded_child_model_state_sha256=initial_sha,
        )
        initialization_contract = request.initialization_contract()
        self.assertIsNotNone(initialization_contract)
        plan = entry.InitializationPlan(
            request=request,
            contract=initialization_contract,
            initial_model_state_sha256=initial_sha,
        )
        prepared = SimpleNamespace(
            dataset_dir=Path("/unused"),
            dataset_root=Path("/unused/NUDT-SIRST"),
            train_ids=tuple(f"train-{index}" for index in range(16)),
            val_ids=("validation",),
            normalization={"mean": 0.0, "std": 1.0},
            training_data_sha256="2" * 64,
            split_hashes={
                "used_train_sha256": "3" * 64,
                "used_val_sha256": "4" * 64,
            },
            split_manifest={"schema": "tiny-split"},
        )
        split_records = {
            "train": exact_runner.OrderedFingerprint.from_values(
                "train",
                prepared.train_ids,
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation",
                prepared.val_ids,
            ),
        }
        data_records = {
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples",
                tuple(f"{name}:mask" for name in prepared.train_ids),
            )
        }
        statistics = entry.load_survival_target_statistics()
        sources = {
            entry.SOURCE_LOCK_KEY: "1" * 64,
            "training_data": prepared.training_data_sha256,
            "survival_target_statistics": statistics["sha256"],
            "parent_checkpoint": entry.PARENT_CHECKPOINT_SHA256,
        }

        with tempfile.TemporaryDirectory() as directory_text:
            args.output_root = Path(directory_text)
            args.exact_source_lock = entry.DEFAULT_TARGET_STATISTICS_PATH
            # ``run_training`` still executes its real loop, exact runner,
            # validation, checkpoint adapter, journal commit, and summaries.
            # Only heavyweight data/model construction and immutable preflight
            # I/O are replaced by contract-equivalent tiny fixtures.
            with (
                mock.patch.object(entry, "FORMAL_EPOCHS", 1),
                mock.patch.object(entry, "FORMAL_WARMUP_EPOCHS", 0),
                mock.patch.object(
                    entry,
                    "_validate_formal_args",
                    return_value=None,
                ),
                mock.patch.object(
                    entry,
                    "resolve_device",
                    return_value=torch.device("cpu"),
                ),
                mock.patch.object(
                    entry,
                    "prepare_data",
                    return_value=prepared,
                ),
                mock.patch.object(
                    entry,
                    "_require_prepared_statistics_match",
                    return_value=None,
                ),
                mock.patch.object(
                    entry,
                    "source_lock_contract",
                    return_value=sources,
                ),
                mock.patch.object(
                    entry,
                    "build_selected_model",
                    return_value=(toy_model, self.metadata),
                ),
                mock.patch.object(
                    entry,
                    "initialization_plan",
                    return_value=plan,
                ),
                mock.patch.object(
                    entry,
                    "split_fingerprints",
                    return_value=split_records,
                ),
                mock.patch.object(
                    entry,
                    "data_fingerprints",
                    return_value=data_records,
                ),
                mock.patch.object(
                    entry,
                    "environment_contract",
                    return_value={"name": "tiny-cpu"},
                ),
                mock.patch.object(
                    entry.base,
                    "TrainingSubset",
                    return_value=TinyTrainingSet(),
                ),
                mock.patch.object(
                    entry.base,
                    "ValidationSubset",
                    return_value=TinyValidationSet(),
                ),
            ):
                directory = entry.run_training(args)

            events = [
                json.loads(line)
                for line in (
                    directory / exact_runner.METRICS_FILENAME
                ).read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["epoch"], 1)
            self.assertIn("train_total_loss", event)
            self.assertIn("train_segmentation_loss", event)
            self.assertEqual(event["train_survival_loss"], 0.0)
            self.assertIsNone(event["train_survival_emb1_loss"])
            self.assertIsNone(event["train_survival_emb2_loss"])
            checkpoint = torch.load(
                directory / exact_runner.LAST_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["schema"], entry.CHECKPOINT_SCHEMA)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(len(checkpoint["state_dict"]), 548)
            self.assertTrue((directory / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
