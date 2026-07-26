from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

from experiments import tpd_extension_warm_start as warm_start
from experiments.train_tpd_clean_v4 import build_clean_v4_model


class ScaleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context_scale = nn.Parameter(torch.zeros(channels))
        self.saliency_scale = nn.Parameter(torch.zeros(channels))


class ParentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.kcs_blocks = nn.ModuleList(
            ScaleBlock(2 + index) for index in range(7)
        )


class NewModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.zeros(1))


class ExtensionModel(ParentModel):
    def __init__(self) -> None:
        super().__init__()
        self.new_module = NewModule()


class IncompleteExtension(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.kcs_blocks = nn.ModuleList(
            ScaleBlock(2 + index) for index in range(6)
        )
        self.new_module = NewModule()


def build_parent() -> ParentModel:
    torch.manual_seed(42)
    model = ParentModel()
    with torch.no_grad():
        scale_index = 0
        for name, parameter in model.named_parameters():
            if name.endswith((".context_scale", ".saliency_scale")):
                parameter.fill_(0.10 + 0.01 * scale_index)
                scale_index += 1
    assert scale_index == 14
    return model


def build_extension() -> ExtensionModel:
    torch.manual_seed(3407)
    return ExtensionModel()


def write_checkpoint(
    directory: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    name: str,
) -> Path:
    path = directory / name
    torch.save(
        {
            "epoch": 800,
            "state_dict": state_dict,
            "checkpoint_role": "parent_fixture",
        },
        path,
    )
    return path


class TPDExtensionWarmStartTests(unittest.TestCase):
    def test_real_v4_parent_keeps_all_fourteen_nonzero_kcs_scales(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory_text,
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            directory = Path(directory_text)
            parent, _ = build_clean_v4_model("tpd_clean_v4_full", seed=42)
            extension, _ = build_clean_v4_model(
                "tpd_clean_v4_full",
                seed=3407,
            )
            extension.future_extension = NewModule()

            scale_names = [
                name
                for name, _ in parent.named_parameters()
                if name.endswith((".context_scale", ".saliency_scale"))
            ]
            self.assertEqual(len(scale_names), 14)
            parent_parameters = dict(parent.named_parameters())
            with torch.no_grad():
                for index, name in enumerate(scale_names):
                    parent_parameters[name].fill_(0.20 + 0.01 * index)

            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="real_v4_parent.pth.tar",
            )
            new_state_before = {
                key: value.detach().clone()
                for key, value in extension.state_dict().items()
                if key.startswith("future_extension.")
            }
            result = warm_start.load_parent_into_extension(
                checkpoint,
                parent_model=parent,
                extension_model=extension,
                new_module_prefixes=("future_extension",),
                zero_init_prefixes=("future_extension.gate",),
            )

            self.assertEqual(result.parent_state_key_count, 532)
            for name in scale_names:
                restored = extension.state_dict()[name]
                self.assertTrue(
                    torch.equal(restored, parent.state_dict()[name]),
                    msg=name,
                )
                self.assertEqual(
                    torch.count_nonzero(restored).item(),
                    restored.numel(),
                    msg=name,
                )
            for name, expected in new_state_before.items():
                self.assertTrue(
                    torch.equal(extension.state_dict()[name], expected),
                    msg=name,
                )

    def test_strict_transfer_preserves_new_state_and_all_kcs_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            extension = build_extension()
            parent_before = copy.deepcopy(parent.state_dict())
            extension_before = copy.deepcopy(extension.state_dict())
            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="parent.pth.tar",
            )
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            result = warm_start.load_parent_into_extension(
                checkpoint,
                parent_model=parent,
                extension_model=extension,
                new_module_prefixes=("new_module",),
                zero_init_prefixes=("new_module.gate",),
                expected_parent_checkpoint_sha256=digest,
            )

            parent_keys = set(parent.state_dict())
            new_keys = set(extension.state_dict()) - parent_keys
            self.assertEqual(result.parent_checkpoint_sha256, digest)
            self.assertEqual(result.parent_state_key_count, len(parent_keys))
            self.assertEqual(
                result.preserved_new_state_key_count,
                len(new_keys),
            )
            self.assertEqual(result.new_module_prefixes, ("new_module",))
            self.assertEqual(
                result.zero_init_prefixes,
                ("new_module.gate",),
            )
            self.assertEqual(
                result.provenance()["parent_checkpoint_sha256"],
                digest,
            )

            for key, expected in parent_before.items():
                self.assertTrue(
                    torch.equal(extension.state_dict()[key], expected),
                    msg=key,
                )
                self.assertTrue(
                    torch.equal(parent.state_dict()[key], expected),
                    msg=f"parent mutated: {key}",
                )
            for key in new_keys:
                self.assertTrue(
                    torch.equal(
                        extension.state_dict()[key],
                        extension_before[key],
                    ),
                    msg=f"new state overwritten: {key}",
                )

            scale_names = [
                key
                for key in parent_keys
                if key.endswith((".context_scale", ".saliency_scale"))
            ]
            self.assertEqual(len(scale_names), 14)
            for key in scale_names:
                restored = extension.state_dict()[key]
                self.assertEqual(
                    torch.count_nonzero(restored).item(),
                    restored.numel(),
                    msg=key,
                )

    def test_parent_state_keys_shape_and_dtype_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            original = copy.deepcopy(parent.state_dict())
            cases: list[tuple[str, str, dict[str, torch.Tensor]]] = []

            omitted = copy.deepcopy(original)
            omitted.pop("stem.bias")
            cases.append(("omitted", "omits required parent key", omitted))

            unexpected = copy.deepcopy(original)
            unexpected["fabricated.weight"] = torch.zeros(1)
            cases.append(("unexpected", "unexpected key", unexpected))

            wrong_shape = copy.deepcopy(original)
            wrong_shape["stem.weight"] = wrong_shape["stem.weight"][:-1]
            cases.append(("shape", "shape mismatch", wrong_shape))

            wrong_dtype = copy.deepcopy(original)
            wrong_dtype["stem.bias"] = wrong_dtype["stem.bias"].double()
            cases.append(("dtype", "dtype mismatch", wrong_dtype))

            for name, message, state in cases:
                with self.subTest(name=name):
                    checkpoint = write_checkpoint(
                        directory,
                        state,
                        name=f"{name}.pth.tar",
                    )
                    with self.assertRaisesRegex(
                        warm_start.ExtensionWarmStartError,
                        message,
                    ):
                        warm_start.load_parent_into_extension(
                            checkpoint,
                            parent_model=parent,
                            extension_model=build_extension(),
                            new_module_prefixes=("new_module",),
                            zero_init_prefixes=("new_module.gate",),
                        )

    def test_missing_parent_key_in_extension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="parent.pth.tar",
            )
            with self.assertRaisesRegex(
                warm_start.ExtensionWarmStartError,
                "extension model omits parent key",
            ):
                warm_start.load_parent_into_extension(
                    checkpoint,
                    parent_model=parent,
                    extension_model=IncompleteExtension(),
                    new_module_prefixes=("new_module",),
                    zero_init_prefixes=("new_module.gate",),
                )

    def test_only_real_explicit_new_prefixes_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="parent.pth.tar",
            )
            cases = (
                (
                    "undeclared",
                    ("new_module.proj",),
                    "not explicitly declared new",
                ),
                (
                    "fabricated",
                    ("new_module", "fabricated"),
                    "does not identify extension-only state",
                ),
                (
                    "old_module",
                    ("new_module", "stem"),
                    "overlaps parent state",
                ),
            )
            for name, prefixes, message in cases:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        warm_start.ExtensionWarmStartError,
                        message,
                    ):
                        warm_start.load_parent_into_extension(
                            checkpoint,
                            parent_model=parent,
                            extension_model=build_extension(),
                            new_module_prefixes=prefixes,
                        )

    def test_nonzero_declared_zero_initialization_is_rejected_before_load(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="parent.pth.tar",
            )
            extension = build_extension()
            with torch.no_grad():
                extension.new_module.gate.fill_(0.25)
            before = copy.deepcopy(extension.state_dict())

            with self.assertRaisesRegex(
                warm_start.ExtensionWarmStartError,
                "zero-initialization contract violated",
            ):
                warm_start.load_parent_into_extension(
                    checkpoint,
                    parent_model=parent,
                    extension_model=extension,
                    new_module_prefixes=("new_module",),
                    zero_init_prefixes=("new_module.gate",),
                )

            for key, expected in before.items():
                self.assertTrue(
                    torch.equal(extension.state_dict()[key], expected),
                    msg=f"state changed before rejection: {key}",
                )

    def test_parent_checkpoint_digest_can_be_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            parent = build_parent()
            checkpoint = write_checkpoint(
                directory,
                copy.deepcopy(parent.state_dict()),
                name="parent.pth.tar",
            )
            with self.assertRaisesRegex(
                warm_start.ExtensionWarmStartError,
                "SHA-256 mismatch",
            ):
                warm_start.load_parent_into_extension(
                    checkpoint,
                    parent_model=parent,
                    extension_model=build_extension(),
                    new_module_prefixes=("new_module",),
                    zero_init_prefixes=("new_module.gate",),
                    expected_parent_checkpoint_sha256="f" * 64,
                )


if __name__ == "__main__":
    unittest.main()
