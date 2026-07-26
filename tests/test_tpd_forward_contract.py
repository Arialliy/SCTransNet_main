from __future__ import annotations

import unittest

import torch

from model.tpd_forward_contract import (
    TPDForwardOutput,
    evaluator_prediction,
    legacy_output,
)


class TPDForwardContractTests(unittest.TestCase):
    def test_single_map_preserves_baseline_evaluation_output(self) -> None:
        prediction = torch.rand(2, 1, 45, 73)
        result = TPDForwardOutput(segmentation=prediction)

        self.assertIs(result.segmentation, prediction)
        self.assertIs(result.final_prediction, prediction)
        self.assertIs(result.legacy_output(), prediction)
        self.assertIs(result.evaluator_prediction(), prediction)
        self.assertIs(legacy_output(result), prediction)
        self.assertIs(evaluator_prediction(result), prediction)
        self.assertIs(legacy_output(prediction), prediction)
        self.assertIs(evaluator_prediction(prediction), prediction)

    def test_six_full_resolution_maps_preserve_training_output(self) -> None:
        segmentation = tuple(torch.rand(2, 1, 40, 68) for _ in range(6))
        result = TPDForwardOutput(segmentation=segmentation)

        self.assertIs(result.legacy_output(), segmentation)
        self.assertIs(result.final_prediction, segmentation[-1])
        self.assertIs(evaluator_prediction(segmentation), segmentation[-1])

    def test_low_resolution_auxiliary_cannot_enter_six_outputs(self) -> None:
        invalid = (
            torch.rand(1, 1, 48, 80),
            torch.rand(1, 1, 48, 80),
            torch.rand(1, 1, 24, 40),
            torch.rand(1, 1, 48, 80),
            torch.rand(1, 1, 48, 80),
            torch.rand(1, 1, 48, 80),
        )
        with self.assertRaisesRegex(
            ValueError,
            "low-resolution auxiliary outputs",
        ):
            TPDForwardOutput(segmentation=invalid)

        for wrong_length in ((), (torch.rand(1, 1, 7, 11),) * 5):
            with self.subTest(length=len(wrong_length)):
                with self.assertRaisesRegex(ValueError, "exactly six"):
                    legacy_output(wrong_length)  # type: ignore[arg-type]

    def test_optional_endpoints_and_survival_logits_share_token_space(
        self,
    ) -> None:
        prediction = torch.rand(3, 1, 52, 84)
        emb1 = torch.rand(3, 8, 5, 7)
        emb2 = torch.rand(3, 16, 5, 7)
        logit1 = torch.randn(3, 1, 5, 7)
        logit2 = torch.randn(3, 1, 5, 7)
        result = TPDForwardOutput(
            segmentation=prediction,
            emb1_endpoint=emb1,
            emb2_endpoint=emb2,
            emb1_survival_logits=logit1,
            emb2_survival_logits=logit2,
        )

        self.assertEqual(result.token_endpoints, (emb1, emb2))
        self.assertEqual(result.survival_logits, (logit1, logit2))
        self.assertIs(result.legacy_output(), prediction)
        self.assertIs(result.evaluator_prediction(), prediction)

    def test_rejects_unpaired_or_invalid_token_fields(self) -> None:
        prediction = torch.rand(2, 1, 37, 61)
        valid_emb1 = torch.rand(2, 4, 3, 5)
        valid_emb2 = torch.rand(2, 8, 3, 5)

        cases = (
            (
                "paired endpoints",
                {
                    "emb1_endpoint": valid_emb1,
                },
                "both be present",
            ),
            (
                "endpoint batch",
                {
                    "emb1_endpoint": torch.rand(1, 4, 3, 5),
                    "emb2_endpoint": torch.rand(1, 8, 3, 5),
                },
                "batch size",
            ),
            (
                "endpoint space",
                {
                    "emb1_endpoint": valid_emb1,
                    "emb2_endpoint": torch.rand(2, 8, 2, 5),
                },
                "share one spatial",
            ),
            (
                "endpoint channels",
                {
                    "emb1_endpoint": valid_emb1,
                    "emb2_endpoint": torch.rand(2, 7, 3, 5),
                },
                "twice",
            ),
            (
                "paired logits",
                {
                    "emb1_survival_logits": torch.rand(2, 1, 3, 5),
                },
                "both be present",
            ),
            (
                "logit channels",
                {
                    "emb1_survival_logits": torch.rand(2, 2, 3, 5),
                    "emb2_survival_logits": torch.rand(2, 1, 3, 5),
                },
                "1 channel",
            ),
            (
                "logit endpoint space",
                {
                    "emb1_endpoint": valid_emb1,
                    "emb2_endpoint": valid_emb2,
                    "emb1_survival_logits": torch.rand(2, 1, 2, 5),
                    "emb2_survival_logits": torch.rand(2, 1, 2, 5),
                },
                "endpoint spatial space",
            ),
        )
        for name, fields, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    TPDForwardOutput(
                        segmentation=prediction,
                        **fields,  # type: ignore[arg-type]
                    )

    def test_rejects_invalid_segmentation_batch_shape_and_channels(self) -> None:
        cases = (
            (torch.rand(1, 2, 19, 31), ValueError, "1 channel"),
            (torch.rand(1, 19, 31), ValueError, "BxCxHxW"),
            (torch.ones(1, 1, 19, 31, dtype=torch.int64), TypeError, "floating"),
            ([torch.rand(1, 1, 19, 31)], TypeError, "tuple"),
        )
        for output, error, message in cases:
            with self.subTest(shape=getattr(output, "shape", None)):
                with self.assertRaisesRegex(error, message):
                    legacy_output(output)  # type: ignore[arg-type]

        mismatched_batch = (
            torch.rand(2, 1, 19, 31),
            torch.rand(1, 1, 19, 31),
            *(torch.rand(2, 1, 19, 31) for _ in range(4)),
        )
        with self.assertRaisesRegex(ValueError, "same Bx1xHxW shape"):
            legacy_output(mismatched_batch)


if __name__ == "__main__":
    unittest.main()
