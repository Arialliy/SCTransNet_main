from __future__ import annotations

import copy
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn

from experiments import pbdr_v4_original_models as registry


torch.set_num_threads(1)


class FakeOriginalModel(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._installed_state = dict(state)
        self._parameter_count_anchor = nn.Parameter(
            torch.empty(
                registry.four_dataset_models.ORIGINAL_PARAMETER_COUNT,
                device="meta",
            ),
            requires_grad=False,
        )
        self.mode = "train"

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        del args, kwargs
        return dict(self._installed_state)

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> SimpleNamespace:
        if not strict:
            raise AssertionError("the registry must request strict loading")
        self._installed_state = dict(state_dict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def synthetic_original_state(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        f"tensor_{index:03d}": torch.tensor(
            [offset + float(index)], dtype=torch.float32
        )
        for index in range(registry.four_dataset_models.ORIGINAL_STATE_KEY_COUNT)
    }


def fake_raw_builder_metadata() -> dict[str, object]:
    return {
        "schema": registry.four_dataset_models.BUILDER_SCHEMA,
        "method": "original_scratch",
        "training_graph_requested": False,
        "dataset_name": "NUAA-SIRST",
        "training_seed": 42,
        "selected_model_parameter_count": (
            registry.four_dataset_models.ORIGINAL_PARAMETER_COUNT
        ),
        "selected_model_state_key_count": (
            registry.four_dataset_models.ORIGINAL_STATE_KEY_COUNT
        ),
        "warm_start_used": False,
        "parent_checkpoint": None,
    }


class UnifiedOriginalRegistryTests(unittest.TestCase):
    def test_scope_and_public_api_have_no_path_override(self) -> None:
        self.assertEqual(
            registry.DATASETS,
            ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"),
        )
        self.assertEqual(registry.CHECKPOINT_ROLES, ("best_miou", "best_pd"))
        self.assertEqual(registry.TRAINING_SEED, 42)
        self.assertEqual(
            tuple(inspect.signature(registry.load_original_checkpoint).parameters),
            ("dataset_name", "checkpoint_role"),
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    registry.build_original_inference_model
                ).parameters
            ),
            ("dataset_name", "checkpoint_role"),
        )

    def test_two_dataset_loads_delegate_exact_dataset_and_role(self) -> None:
        calls: list[tuple[str, str]] = []

        def delegated_loader(
            dataset: str, role: str
        ) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, object]]:
            calls.append((dataset, role))
            return (
                {"dataset": dataset, "checkpoint_role": role},
                {"weight": torch.tensor([1.0])},
                {"dataset": dataset, "checkpoint_role": role},
            )

        with mock.patch.object(
            registry.two_dataset_models,
            "load_original_checkpoint",
            side_effect=delegated_loader,
        ):
            for dataset in registry.DELEGATED_DATASETS:
                for role in registry.CHECKPOINT_ROLES:
                    with self.subTest(dataset=dataset, role=role):
                        payload, state, record = registry.load_original_checkpoint(
                            dataset, role
                        )
                        self.assertEqual(payload["dataset"], dataset)
                        self.assertEqual(payload["checkpoint_role"], role)
                        self.assertEqual(set(state), {"weight"})
                        self.assertEqual(record["checkpoint_role"], role)

        self.assertEqual(
            calls,
            [
                (dataset, role)
                for dataset in registry.DELEGATED_DATASETS
                for role in registry.CHECKPOINT_ROLES
            ],
        )

    def test_all_three_datasets_and_six_roles_build_with_mocked_authorities(self) -> None:
        delegated_calls: list[tuple[str, str, int]] = []

        def delegated_builder(
            dataset: str, role: str, *, seed: int
        ) -> tuple[nn.Module, dict[str, object]]:
            delegated_calls.append((dataset, role, seed))
            model = FakeOriginalModel({"weight": torch.tensor([1.0])})
            model.eval()
            model.mode = "test"
            return model, {
                "dataset": dataset,
                "checkpoint_role": role,
                "strict_load": True,
            }

        built: set[tuple[str, str]] = set()
        with mock.patch.object(
            registry.two_dataset_models,
            "build_original_inference_model",
            side_effect=delegated_builder,
        ):
            for dataset in registry.DELEGATED_DATASETS:
                for role in registry.CHECKPOINT_ROLES:
                    model, metadata = registry.build_original_inference_model(
                        dataset, role
                    )
                    built.add((dataset, role))
                    self.assertFalse(model.training)
                    self.assertEqual(model.mode, "test")
                    self.assertEqual(metadata["checkpoint_role"], role)

        states = {
            role: synthetic_original_state(float(index) * 1000.0)
            for index, role in enumerate(registry.CHECKPOINT_ROLES)
        }

        def nuaa_loader(
            dataset: str, role: str
        ) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, object]]:
            state = states[role]
            return (
                {"dataset": dataset, "checkpoint_role": role},
                state,
                {
                    "dataset": dataset,
                    "checkpoint_role": role,
                    "state_sha256": (
                        registry.four_dataset_models.state_dict_sha256(state)
                    ),
                },
            )

        four_calls: list[tuple[str, str, int, bool]] = []

        def four_builder(
            method: str,
            dataset: str,
            *,
            seed: int,
            training: bool,
        ) -> tuple[nn.Module, dict[str, object]]:
            four_calls.append((method, dataset, seed, training))
            return FakeOriginalModel(synthetic_original_state(-1000.0)), (
                fake_raw_builder_metadata()
            )

        with mock.patch.object(
            registry, "load_original_checkpoint", side_effect=nuaa_loader
        ), mock.patch.object(
            registry.four_dataset_models,
            "build_paper_model",
            side_effect=four_builder,
        ):
            for role in registry.CHECKPOINT_ROLES:
                model, metadata = registry.build_original_inference_model(
                    "NUAA-SIRST", role
                )
                built.add(("NUAA-SIRST", role))
                self.assertFalse(model.training)
                self.assertEqual(model.mode, "test")
                self.assertTrue(metadata["strict_load"])
                self.assertEqual(metadata["checkpoint_role"], role)
                self.assertEqual(
                    metadata["state_sha256"],
                    registry.four_dataset_models.state_dict_sha256(states[role]),
                )

        self.assertEqual(
            built,
            {
                (dataset, role)
                for dataset in registry.DATASETS
                for role in registry.CHECKPOINT_ROLES
            },
        )
        self.assertEqual(
            delegated_calls,
            [
                (dataset, role, 42)
                for dataset in registry.DELEGATED_DATASETS
                for role in registry.CHECKPOINT_ROLES
            ],
        )
        self.assertEqual(
            four_calls,
            [
                ("original", "NUAA-SIRST", 42, False),
                ("original", "NUAA-SIRST", 42, False),
            ],
        )

    def test_actual_nuaa_roles_pass_file_metadata_and_state_audit(self) -> None:
        expected = {
            "best_miou": (
                830,
                "b5edfe46fc54d5e74c1896a43f0f44c8970c143d90eaebb1098cac760f119ead",
                "48a9ada9fae4b7e0fe9068916bf3a7011ca7379d4dfa8f3cd84cd593ebd986ac",
            ),
            "best_pd": (
                440,
                "9638f92d6aac6114a5cfb7b8124f90e94c7569053735558c4b9cde73fb8ebd7d",
                "ebb31ad2e621ea12a479572600fcfadecab016dac2234e591efd17c999a88ee1",
            ),
        }
        for role, identity in expected.items():
            with self.subTest(role=role):
                payload, state, record = registry.load_original_checkpoint(
                    "NUAA-SIRST", role
                )
                self.assertEqual(
                    (payload["epoch"], record["sha256"], record["state_sha256"]),
                    identity,
                )
                self.assertEqual(len(state), 510)
                self.assertEqual(payload["checkpoint_role"], role)
                self.assertEqual(record["checkpoint_role"], role)
                self.assertFalse(record["official_test_data_accessed"])
                self.assertFalse(record["dataset_loader_imported"])
                del payload, state, record
                gc.collect()

    def test_wrong_dataset_role_and_delegated_role_swap_are_rejected(self) -> None:
        for dataset, role in (
            ("SIRST3", "best_miou"),
            ("NUAA-SIRST", "latest"),
            ("NUDT-SIRST", "best_f1"),
        ):
            with self.subTest(dataset=dataset, role=role), self.assertRaises(
                registry.PBDRV4OriginalModelRegistryError
            ):
                registry.load_original_checkpoint(dataset, role)

        swapped = (
            {"dataset": "NUDT-SIRST", "checkpoint_role": "best_pd"},
            {"weight": torch.tensor([1.0])},
            {"dataset": "NUDT-SIRST", "checkpoint_role": "best_pd"},
        )
        with mock.patch.object(
            registry.two_dataset_models,
            "load_original_checkpoint",
            return_value=swapped,
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError, "payload role differs"
        ):
            registry.load_original_checkpoint("NUDT-SIRST", "best_miou")

    def test_duplicate_cross_role_and_path_tampered_authority_records_fail(self) -> None:
        authority = json.loads(
            registry.NUAA_AUTHORITY_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        original = next(
            item
            for item in authority["records"]
            if item["dataset"] == "NUAA-SIRST" and item["method"] == "original"
        )

        duplicate = copy.deepcopy(authority)
        duplicate["records"].append(copy.deepcopy(original))
        with mock.patch.object(
            registry, "_load_nuaa_manifest", return_value=duplicate
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError, "exactly one"
        ):
            registry.load_original_checkpoint("NUAA-SIRST", "best_miou")

        wrong_role = copy.deepcopy(authority)
        wrong_record = next(
            item
            for item in wrong_role["records"]
            if item["dataset"] == "NUAA-SIRST" and item["method"] == "original"
        )
        wrong_record["checkpoints"]["best_pd"]["checkpoint_role"] = "best_miou"
        with mock.patch.object(
            registry, "_load_nuaa_manifest", return_value=wrong_role
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError,
            "best_pd authority checkpoint_role differs",
        ):
            registry.load_original_checkpoint("NUAA-SIRST", "best_pd")

        wrong_path = copy.deepcopy(authority)
        path_record = next(
            item
            for item in wrong_path["records"]
            if item["dataset"] == "NUAA-SIRST" and item["method"] == "original"
        )
        path_record["checkpoints"]["best_pd"]["frozen_path"] = (
            path_record["checkpoints"]["best_miou"]["frozen_path"]
        )
        with mock.patch.object(
            registry, "_load_nuaa_manifest", return_value=wrong_path
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError,
            "best_pd authority frozen_path differs",
        ):
            registry.load_original_checkpoint("NUAA-SIRST", "best_pd")

    def test_manifest_file_and_loaded_state_tampering_are_rejected(self) -> None:
        with mock.patch.object(
            registry, "NUAA_AUTHORITY_MANIFEST_SHA256", "0" * 64
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError,
            "manifest SHA-256 differs",
        ):
            registry.load_original_checkpoint("NUAA-SIRST", "best_miou")

        payload, state, _ = registry.load_original_checkpoint(
            "NUAA-SIRST", "best_miou"
        )
        tampered_payload = dict(payload)
        tampered_state = dict(state)
        first_name = sorted(tampered_state)[0]
        changed = tampered_state[first_name].clone()
        changed.reshape(-1)[0] += 1
        tampered_state[first_name] = changed
        tampered_payload["state_dict"] = tampered_state
        pin = registry.NUAA_CHECKPOINT_PINS["best_miou"]

        def trusted_file_identity(path: Path) -> str:
            if path.name == "checkpoint_manifest.json":
                return registry.NUAA_AUTHORITY_MANIFEST_SHA256
            return pin.file_sha256

        with mock.patch.object(
            registry, "file_sha256", side_effect=trusted_file_identity
        ), mock.patch.object(
            registry.torch, "load", return_value=tampered_payload
        ), self.assertRaisesRegex(
            registry.PBDRV4OriginalModelRegistryError,
            "state SHA-256 differs",
        ):
            registry.load_original_checkpoint("NUAA-SIRST", "best_miou")
        del payload, state, tampered_payload, tampered_state
        gc.collect()

    def test_source_has_no_dataset_index_evaluation_or_training_access(self) -> None:
        source = inspect.getsource(registry)
        self.assertNotIn("load_" + "index(", source)
        self.assertNotIn("resolve_" + "sample", source)
        self.assertNotIn("from " + "dataset import", source)
        self.assertNotIn("torch." + "save(", source)
        self.assertNotIn("optimizer." + "step(", source)


if __name__ == "__main__":
    unittest.main()
