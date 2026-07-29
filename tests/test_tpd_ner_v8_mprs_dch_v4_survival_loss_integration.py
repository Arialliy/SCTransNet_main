from __future__ import annotations

import copy
import unittest
from typing import Iterable

import torch
import torch.nn as nn

from experiments.tpd_training_loss import compute_tpd_training_loss
from model.tpd_forward_contract import TPDForwardOutput
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_PREFIX,
    build_formal_v4_survival_model,
    validate_formal_survival_model,
)


torch.set_num_threads(1)


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260729)
    images = torch.randn(2, 1, 32, 32, generator=generator)
    target = torch.zeros(2, 1, 32, 32)
    target[0, 0, 2:5, 3:6] = 1.0
    target[0, 0, 19:22, 20:23] = 1.0
    target[1, 0, 8:11, 24:27] = 1.0
    return images, target


def _parameters_with_prefixes(
    model: nn.Module,
    prefixes: Iterable[str],
) -> list[tuple[str, nn.Parameter]]:
    normalized = tuple(prefixes)
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(normalized)
    ]


def _has_nonzero_gradient(
    parameters: Iterable[tuple[str, nn.Parameter]],
) -> bool:
    return any(
        parameter.grad is not None
        and int(torch.count_nonzero(parameter.grad)) > 0
        for _, parameter in parameters
    )


def _all_gradients_none_or_zero(
    parameters: Iterable[tuple[str, nn.Parameter]],
) -> bool:
    return all(
        parameter.grad is None
        or int(torch.count_nonzero(parameter.grad)) == 0
        for _, parameter in parameters
    )


class V4SurvivalLossIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template, _ = build_formal_v4_survival_model()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.template

    def test_zero_weight_matches_segmentation_gradients_and_first_adam_step(
        self,
    ) -> None:
        manual_model = copy.deepcopy(self.template).train()
        control_model = copy.deepcopy(self.template).train()
        images, target = _fixed_batch()
        criterion = nn.BCELoss(reduction="mean")
        manual_optimizer = torch.optim.Adam(
            manual_model.parameters(),
            lr=1e-4,
        )
        control_optimizer = torch.optim.Adam(
            control_model.parameters(),
            lr=1e-4,
        )

        manual_output = manual_model(images)
        control_output = control_model(images)
        self.assertIsInstance(manual_output, TPDForwardOutput)
        self.assertIsInstance(control_output, TPDForwardOutput)
        manual_segmentation = manual_output.segmentation
        self.assertIsInstance(manual_segmentation, tuple)
        manual_loss = sum(
            criterion(probability, target)
            for probability in manual_segmentation
        )
        control_losses = compute_tpd_training_loss(
            control_output,
            target,
            criterion,
            survival_weight=0.0,
            survival_pos_weight=1.0,
        )
        self.assertTrue(torch.equal(control_losses.total, manual_loss))
        self.assertTrue(
            torch.equal(control_losses.segmentation, manual_loss)
        )
        self.assertEqual(control_losses.survival_terms, ())
        self.assertEqual(float(control_losses.survival), 0.0)

        manual_loss.backward()
        control_losses.total.backward()
        manual_parameters = dict(manual_model.named_parameters())
        control_parameters = dict(control_model.named_parameters())
        self.assertEqual(
            tuple(manual_parameters),
            tuple(control_parameters),
        )
        for name in manual_parameters:
            left = manual_parameters[name].grad
            right = control_parameters[name].grad
            if name.startswith(SURVIVAL_STATE_PREFIX):
                self.assertIsNone(left, msg=name)
                self.assertIsNone(right, msg=name)
            elif left is None or right is None:
                self.assertIsNone(left, msg=name)
                self.assertIsNone(right, msg=name)
            else:
                self.assertTrue(torch.equal(left, right), msg=name)

        manual_optimizer.step()
        control_optimizer.step()
        manual_state = manual_model.state_dict()
        control_state = control_model.state_dict()
        for name in manual_state:
            self.assertTrue(
                torch.equal(manual_state[name], control_state[name]),
                msg=name,
            )

    def test_zero_head_then_second_forward_routes_survival_gradient(self) -> None:
        model = copy.deepcopy(self.template).train()
        validate_formal_survival_model(
            model,
            require_zero_initialized_heads=True,
        )
        images, target = _fixed_batch()
        criterion = nn.BCELoss(reduction="mean")
        head_optimizer = torch.optim.Adam(
            model.target_survival.parameters(),
            lr=1e-3,
        )

        first_output = model(images)
        self.assertIsInstance(first_output, TPDForwardOutput)
        first_losses = compute_tpd_training_loss(
            first_output,
            target,
            criterion,
            survival_weight=0.01,
            survival_pos_weight=1.0,
        )
        first_losses.survival.backward()

        heads = _parameters_with_prefixes(model, ("target_survival.",))
        embeddings1 = _parameters_with_prefixes(
            model,
            ("mtc.embeddings_1.",),
        )
        embeddings2 = _parameters_with_prefixes(
            model,
            ("mtc.embeddings_2.",),
        )
        shallow = _parameters_with_prefixes(
            model,
            ("inc.", "down_encoder1."),
        )
        self.assertTrue(_has_nonzero_gradient(heads))
        self.assertTrue(_all_gradients_none_or_zero(embeddings1))
        self.assertTrue(_all_gradients_none_or_zero(embeddings2))
        self.assertTrue(_all_gradients_none_or_zero(shallow))

        head_optimizer.step()
        for endpoint_name in ("emb1", "emb2"):
            weight = model.target_survival.heads[
                endpoint_name
            ].classifier.weight
            self.assertGreater(int(torch.count_nonzero(weight)), 0)

        model.zero_grad(set_to_none=True)
        second_output = model(images)
        self.assertIsInstance(second_output, TPDForwardOutput)
        second_losses = compute_tpd_training_loss(
            second_output,
            target,
            criterion,
            survival_weight=0.01,
            survival_pos_weight=1.0,
        )
        second_losses.survival.backward()

        self.assertTrue(_has_nonzero_gradient(heads))
        self.assertTrue(_has_nonzero_gradient(embeddings1))
        self.assertTrue(_has_nonzero_gradient(embeddings2))
        self.assertTrue(_has_nonzero_gradient(shallow))

        downstream_groups = {
            "tpd_ner": ("tpd_ner.",),
            "sctb_encoder": ("mtc.encoder.",),
            "decoder": (
                "up_decoder4.",
                "up_decoder3.",
                "up_decoder2.",
                "up_decoder1.",
            ),
            "segmentation_heads": (
                "gt_conv5.",
                "gt_conv4.",
                "gt_conv3.",
                "gt_conv2.",
                "outconv.",
                "outc.",
            ),
        }
        for label, prefixes in downstream_groups.items():
            self.assertTrue(
                _all_gradients_none_or_zero(
                    _parameters_with_prefixes(model, prefixes)
                ),
                msg=label,
            )


if __name__ == "__main__":
    unittest.main()
