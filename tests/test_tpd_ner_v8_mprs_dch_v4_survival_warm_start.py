from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.tpd_extension_warm_start import (
    ExtensionWarmStartError,
    load_parent_into_extension,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_V4_PARENT_STATE_KEY_COUNT,
    SURVIVAL_STATE_KEYS,
    build_formal_v4_reference,
    build_formal_v4_survival_model,
    validate_formal_survival_model,
)


torch.set_num_threads(1)


def _write_parent_checkpoint(
    directory: Path,
    state_dict: dict[str, torch.Tensor],
) -> tuple[Path, str]:
    path = directory / "v4_best_miou_fixture.pth.tar"
    torch.save(
        {
            "epoch": 489,
            "checkpoint_role": "best_validation_miou_secondary",
            "state_dict": state_dict,
        },
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


class V4SurvivalWarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, _ = build_formal_v4_reference()
        cls.extension, _ = build_formal_v4_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.reference
        del cls.extension

    def test_strict_warm_start_preserves_only_four_zero_head_tensors(
        self,
    ) -> None:
        reference = copy.deepcopy(self.reference)
        extension = copy.deepcopy(self.extension)
        first_name, first_parameter = next(iter(reference.named_parameters()))
        with torch.no_grad():
            first_parameter.fill_(0.03125)

        parent_state = copy.deepcopy(reference.state_dict())
        head_before = {
            key: extension.state_dict()[key].detach().clone()
            for key in SURVIVAL_STATE_KEYS
        }
        with tempfile.TemporaryDirectory() as directory_text:
            checkpoint, digest = _write_parent_checkpoint(
                Path(directory_text),
                parent_state,
            )
            result = load_parent_into_extension(
                parent_checkpoint=checkpoint,
                parent_model=reference,
                extension_model=extension,
                new_module_prefixes=("target_survival",),
                zero_init_prefixes=(
                    "target_survival.heads.emb1.classifier",
                    "target_survival.heads.emb2.classifier",
                ),
                parent_state_dict_path=("state_dict",),
                expected_parent_checkpoint_sha256=digest,
            )

        self.assertEqual(
            result.parent_state_key_count,
            FORMAL_V4_PARENT_STATE_KEY_COUNT,
        )
        self.assertEqual(result.preserved_new_state_key_count, 4)
        self.assertEqual(result.parent_checkpoint_sha256, digest)
        self.assertTrue(
            torch.equal(extension.state_dict()[first_name], parent_state[first_name])
        )
        for key, expected in parent_state.items():
            self.assertTrue(
                torch.equal(extension.state_dict()[key], expected),
                msg=key,
            )
        for key, expected in head_before.items():
            self.assertTrue(
                torch.equal(extension.state_dict()[key], expected),
                msg=key,
            )
            self.assertEqual(
                int(torch.count_nonzero(extension.state_dict()[key])),
                0,
                msg=key,
            )
        validate_formal_survival_model(
            extension,
            require_zero_initialized_heads=True,
        )

    def test_nonzero_declared_head_is_rejected_without_mutation(self) -> None:
        reference = copy.deepcopy(self.reference)
        extension = copy.deepcopy(self.extension)
        with torch.no_grad():
            extension.target_survival.heads["emb1"].classifier.weight.fill_(
                0.25
            )
        before = copy.deepcopy(extension.state_dict())

        with tempfile.TemporaryDirectory() as directory_text:
            checkpoint, _ = _write_parent_checkpoint(
                Path(directory_text),
                copy.deepcopy(reference.state_dict()),
            )
            with self.assertRaisesRegex(
                ExtensionWarmStartError,
                "zero-initialization contract violated",
            ):
                load_parent_into_extension(
                    parent_checkpoint=checkpoint,
                    parent_model=reference,
                    extension_model=extension,
                    new_module_prefixes=("target_survival",),
                    zero_init_prefixes=(
                        "target_survival.heads.emb1.classifier",
                        "target_survival.heads.emb2.classifier",
                    ),
                )

        for key, expected in before.items():
            self.assertTrue(
                torch.equal(extension.state_dict()[key], expected),
                msg=key,
            )


if __name__ == "__main__":
    unittest.main()
