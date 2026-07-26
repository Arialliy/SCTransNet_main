from __future__ import annotations

import copy
import inspect
import random
import unittest
from typing import Any
from unittest import mock

import numpy as np
import torch

from experiments import tpd_exact_resume as exact
from experiments.train_tpd_clean_v4 import build_clean_v4_model


def _assert_nested_equal(
    case: unittest.TestCase,
    actual: Any,
    expected: Any,
    *,
    path: str,
) -> None:
    if isinstance(expected, torch.Tensor):
        case.assertIsInstance(actual, torch.Tensor, msg=path)
        case.assertTrue(torch.equal(actual, expected), msg=path)
        return
    if isinstance(expected, dict):
        case.assertIsInstance(actual, dict, msg=path)
        case.assertEqual(set(actual), set(expected), msg=path)
        for key in expected:
            _assert_nested_equal(
                case,
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, (tuple, list)):
        case.assertIsInstance(actual, type(expected), msg=path)
        case.assertEqual(len(actual), len(expected), msg=path)
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_nested_equal(
                case,
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    case.assertEqual(actual, expected, msg=path)


class TPDExactResumeV4IntegrationTests(unittest.TestCase):
    def test_real_v4_full_exact_resume_contract_on_cpu(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.cuda, "device_count", return_value=0),
            mock.patch.object(torch.cuda, "get_rng_state_all") as get_cuda_rng,
            mock.patch.object(torch.cuda, "set_rng_state_all") as set_cuda_rng,
        ):
            model, metadata = build_clean_v4_model(
                "tpd_clean_v4_full",
                seed=42,
            )
            self.assertEqual(metadata["variant"], "tpd_clean_v4_full")
            self.assertEqual(metadata["mainline_contract"], "Keep-Context-Saliency")
            self.assertFalse(metadata["fourth_parallel_branch_added"])
            self.assertTrue(
                all(parameter.device.type == "cpu" for parameter in model.parameters())
            )

            named_parameters = dict(model.named_parameters())
            scale_names = [
                name
                for name in named_parameters
                if name.endswith((".context_scale", ".saliency_scale"))
            ]
            self.assertEqual(len(scale_names), 14)
            with torch.no_grad():
                for index, name in enumerate(scale_names):
                    named_parameters[name].fill_(0.10 + 0.01 * index)

            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            for name in scale_names:
                parameter = named_parameters[name]
                parameter.grad = torch.full_like(parameter, 0.01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            scaler = torch.amp.GradScaler(
                "cpu",
                init_scale=128.0,
                growth_interval=17,
            )
            loader_generator = torch.Generator(device="cpu")
            loader_generator.manual_seed(404)
            identity = {
                "run_id": "tpd-clean-v4-full-seed42-cpu-integration",
                "variant": "tpd_clean_v4_full",
                "architecture_id": (
                    "spd_anchored_tpd_clean_v4_single_logit_kcs"
                ),
                "dataset": "synthetic-no-dataset",
                "seed": 42,
                "split_seed": 20260722,
                "split_sha256": "1" * 64,
                "source_lock_sha256": "2" * 64,
            }
            metrics = {
                "pd": 188 / 189,
                "fa": 1.0e-6,
                "miou": 0.94,
                "val_loss": 0.06,
                "tiny_pd": 1.0,
                "tiny_target_count": 39,
            }
            best_selection = {
                "primary": {
                    "role": "best_validation_pd_primary",
                    "epoch": 2,
                    "key": [188 / 189, -1.0e-6, 1.0, 0.94, -0.06],
                    "metrics": metrics,
                },
                "secondary": {
                    "role": "best_validation_miou_secondary",
                    "epoch": 1,
                    "key": [0.94, 188 / 189, -1.0e-6, 1.0, -0.06],
                    "metrics": metrics,
                },
            }
            metrics_boundary = {
                "completed_epoch": 2,
                "event_count": 2,
                "last_event_epoch": 2,
                "metrics_sha256": "3" * 64,
                "last_event_sha256": "4" * 64,
            }

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            checkpoint = exact.build_exact_resume_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=2,
                run_identity=identity,
                best_selection=best_selection,
                metrics_boundary=metrics_boundary,
                loader_generator=loader_generator,
                extra_state={"variant_metadata": metadata},
            )

            self.assertEqual(
                checkpoint["model"]["layout"],
                exact.model_layout(model),
            )
            bound_parameter_names = [
                name
                for group in checkpoint["optimizer"]["parameter_names"]
                for name in group
            ]
            self.assertEqual(
                bound_parameter_names,
                [name for name, _ in model.named_parameters()],
            )
            self.assertTrue(set(scale_names).issubset(bound_parameter_names))

            expected_python = random.getrandbits(63)
            expected_numpy = np.random.randint(0, 2**63 - 1, dtype=np.int64)
            expected_torch = torch.rand(5)
            expected_loader = torch.rand(5, generator=loader_generator)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.25)
            optimizer.param_groups[0]["lr"] = 0.75
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.add_(7)
            mutated_scaler = copy.deepcopy(scaler.state_dict())
            mutated_scaler["scale"] = 2.0
            scaler.load_state_dict(mutated_scaler)
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            loader_generator.manual_seed(14)

            result = exact.restore_exact_resume(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader_generator=loader_generator,
                expected_run_identity=identity,
                expected_epoch=2,
                expected_metrics_boundary=metrics_boundary,
                expected_best_selection=best_selection,
            )

            self.assertEqual(result.epoch, 2)
            self.assertEqual(result.metrics_boundary, metrics_boundary)
            self.assertEqual(result.best_selection, best_selection)
            for name, expected in checkpoint["model"]["state_dict"].items():
                actual = model.state_dict()[name]
                self.assertTrue(torch.equal(actual, expected), msg=name)
            _assert_nested_equal(
                self,
                optimizer.state_dict(),
                checkpoint["optimizer"]["state_dict"],
                path="optimizer",
            )
            _assert_nested_equal(
                self,
                scaler.state_dict(),
                checkpoint["scaler"]["state_dict"],
                path="scaler",
            )
            for name in scale_names:
                restored = model.state_dict()[name]
                self.assertTrue(torch.equal(
                    restored,
                    checkpoint["model"]["state_dict"][name],
                ))
                self.assertEqual(
                    torch.count_nonzero(restored).item(),
                    restored.numel(),
                    msg=name,
                )

            self.assertEqual(random.getrandbits(63), expected_python)
            self.assertEqual(
                int(np.random.randint(0, 2**63 - 1, dtype=np.int64)),
                int(expected_numpy),
            )
            self.assertTrue(torch.equal(torch.rand(5), expected_torch))
            self.assertTrue(
                torch.equal(
                    torch.rand(5, generator=loader_generator),
                    expected_loader,
                )
            )

            restore_parameters = inspect.signature(
                exact.restore_exact_resume
            ).parameters
            self.assertIs(
                restore_parameters["expected_metrics_boundary"].default,
                inspect.Parameter.empty,
            )
            self.assertIs(
                restore_parameters["expected_best_selection"].default,
                inspect.Parameter.empty,
            )
            with self.assertRaisesRegex(TypeError, "expected_best_selection"):
                exact.restore_exact_resume(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loader_generator=loader_generator,
                    expected_run_identity=identity,
                    expected_metrics_boundary=metrics_boundary,
                )
            with self.assertRaisesRegex(TypeError, "expected_metrics_boundary"):
                exact.restore_exact_resume(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    loader_generator=loader_generator,
                    expected_run_identity=identity,
                    expected_best_selection=best_selection,
                )

            reversed_optimizer = torch.optim.Adam(
                list(model.parameters())[::-1],
                lr=1.0e-3,
            )
            with self.assertRaisesRegex(
                exact.ExactResumeValidationError,
                "parameter name/order binding mismatch",
            ):
                exact.restore_exact_resume(
                    checkpoint,
                    model=model,
                    optimizer=reversed_optimizer,
                    scaler=scaler,
                    loader_generator=loader_generator,
                    expected_run_identity=identity,
                    expected_epoch=2,
                    expected_metrics_boundary=metrics_boundary,
                    expected_best_selection=best_selection,
                )

            get_cuda_rng.assert_not_called()
            set_cuda_rng.assert_not_called()


if __name__ == "__main__":
    unittest.main()
