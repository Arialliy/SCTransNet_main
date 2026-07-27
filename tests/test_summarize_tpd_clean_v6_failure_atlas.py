from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from analysis import summarize_tpd_clean_v6_failure_atlas as subject


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(
    *,
    matched: int = 188,
    fa: float = 2e-6,
    miou: float = 0.93,
    fragments: int = 3,
    threshold: float = 0.5,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189.0,
        "matched_tiny_target_count": 39,
        "tiny_target_count": 39,
        "tiny_pd": 1.0,
        "fa": fa,
        "miou": miou,
        "niou": miou - 0.001,
        "pixel_precision": 0.95,
        "pixel_recall": 0.96,
        "pixel_f1": 0.955,
        "predicted_object_count": matched + fragments,
        "unmatched_predicted_object_count": fragments,
        "gt_topology": {
            "fragmented_gt_count": fragments,
            "split_target_count": fragments,
            "fragment_excess_total": fragments,
            "largest_fragment_fraction_mean": 0.99,
            "largest_fragment_fraction_p10": 0.98,
        },
        "component_taxonomy": {
            "unmatched_component_count_by_class": {
                "in_gt_fragment": fragments,
                "near_gt_duplicate": 0,
                "attached_or_near_gt": 0,
                "background_false_object": 1,
            },
            "unmatched_component_pixels_by_class": {
                "in_gt_fragment": fragments,
                "near_gt_duplicate": 0,
                "attached_or_near_gt": 0,
                "background_false_object": 2,
            },
            "fragment_fa_fraction": 0.6,
            "background_fa_fraction": 0.4,
        },
    }


def _copy_point(point: dict[str, Any], **changes: Any) -> dict[str, Any]:
    copied = json.loads(json.dumps(point))
    fragments = changes.pop("fragments", None)
    copied.update(changes)
    if "matched_target_count" in changes:
        copied["pd"] = copied["matched_target_count"] / 189.0
    if fragments is not None:
        copied["gt_topology"]["fragmented_gt_count"] = fragments
        copied["gt_topology"]["split_target_count"] = fragments
        copied["gt_topology"]["fragment_excess_total"] = fragments
        copied["component_taxonomy"]["unmatched_component_count_by_class"][
            "in_gt_fragment"
        ] = fragments
        copied["component_taxonomy"]["unmatched_component_pixels_by_class"][
            "in_gt_fragment"
        ] = fragments
    return copied


class SyntheticAtlas:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.payloads: dict[str, dict[str, Any]] = {}
        self.paths: dict[str, Path] = {}
        self._build()

    def _build(self) -> None:
        for lane, lane_spec in subject.EXPECTED_LANES.items():
            output_paths = []
            for seed in subject.SEEDS:
                for role in subject.ROLES:
                    path = (
                        self.root
                        / lane
                        / lane_spec["variant"]
                        / f"seed_{seed}"
                        / f"{role}.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = self._payload(
                        path,
                        lane_spec["variant"],
                        seed,
                        role,
                    )
                    path.write_text(
                        json.dumps(payload, sort_keys=True),
                        encoding="utf-8",
                    )
                    key = subject.checkpoint_key(
                        lane_spec["variant"],
                        seed,
                        role,
                    )
                    self.payloads[key] = payload
                    self.paths[key] = path
                    output_paths.append(str(path.resolve()))
            matrix = {
                "schema": subject.MATRIX_SCHEMA,
                "mode": "run",
                "training_performed": False,
                "official_test_accessed": False,
                "formal_gate_replacement": False,
                "complete_validation_split": True,
                "output_count": 4,
                "outputs": output_paths,
                "requested_variants": [lane_spec["variant"]],
                "requested_seeds": list(subject.SEEDS),
                "requested_checkpoint_roles": list(subject.ROLES),
                "requested_modes": list(subject.MODES),
                "fixed_thresholds": [0.5, 0.58, 0.999],
                "fa_budgets": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                "device": {"physical_gpu": lane_spec["physical_gpu"]},
                "decision_inputs": {"status": "INCONCLUSIVE"},
            }
            (self.root / lane / "matrix_summary.json").write_text(
                json.dumps(matrix, sort_keys=True),
                encoding="utf-8",
            )

    def _payload(
        self,
        output: Path,
        variant: str,
        seed: int,
        role: str,
    ) -> dict[str, Any]:
        run_dir = self.root / "formal_inputs" / variant / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = run_dir / f"{role}.pth.tar"
        formal_sweep = run_dir / f"{role}_sweep.json"
        artifacts = {
            "protocol": run_dir / "protocol.json",
            "split": run_dir / "split.json",
            "summary": run_dir / "summary.json",
            "metrics": run_dir / "metrics.jsonl",
            "checkpoint": checkpoint,
            "formal_sweep": formal_sweep,
        }
        for name, path in artifacts.items():
            if not path.exists():
                path.write_text(
                    f"{variant}-{seed}-{role}-{name}",
                    encoding="utf-8",
                )
        hashes = {name: _sha(path) for name, path in artifacts.items()}
        state = f"state-{variant}-{seed}-{role}"
        modes = {}
        for mode in subject.MODES:
            base = _point()
            fixed = {
                "0.5": base,
                "0.58": _copy_point(base, threshold=0.58, fa=1.5e-6),
                "0.999": _copy_point(
                    base,
                    threshold=0.999,
                    matched_target_count=20,
                    fa=5e-7,
                    miou=0.2,
                    fragments=5,
                ),
            }
            budgets = {
                budget: _copy_point(
                    fixed["0.58"],
                    threshold=0.58,
                )
                for budget in subject.FA_BUDGETS
            }
            modes[mode] = {
                "counterfactual_provenance": {
                    "mode": mode,
                    "implementation": (
                        subject.EXPECTED_MODE_IMPLEMENTATIONS[mode]
                    ),
                    "block_count": 7,
                    "state_sha256_before": state,
                    "state_sha256_after_restore": state,
                    "state_restored_exactly": True,
                    "zero_training": True,
                },
                "fixed_threshold_points": fixed,
                "best_points_under_fa_budget": budgets,
            }
        static_blocks = []
        for index in range(7):
            static_blocks.append(
                {
                    "block": f"blocks.{index}",
                    "channels": 32 if index < 4 else 64,
                    "use_context_headroom": variant == subject.FULL_VARIANT,
                    "saliency_scale_raw": {
                        "median": 0.1,
                        "p90": 0.2,
                        "max": 0.3,
                    },
                    "saliency_scale_effective_abs_tanh": {
                        "median": 0.09,
                        "p90": 0.19,
                        "max": 0.29,
                    },
                    "phase_sum_cancellation": {
                        "rho_l1": 0.53,
                        "rho_l2": 0.54,
                    },
                }
            )
        return {
            "schema": subject.DIAGNOSTIC_SCHEMA,
            "variant": variant,
            "seed": seed,
            "checkpoint_role": role,
            "checkpoint_epoch": 800,
            "output": str(output.resolve()),
            "run_directory": str(run_dir.resolve()),
            "checkpoint": str(checkpoint.resolve()),
            "formal_sweep": str(formal_sweep.resolve()),
            "training_performed": False,
            "official_test_accessed": False,
            "formal_gate_replacement": False,
            "checkpoint_reselection_permitted": False,
            "complete_validation_split": True,
            "validation": {
                "validation_count": 133,
                "formal_validation_count": 133,
                "validation_split_sha256": "split-sha",
            },
            "loaded_model_state_sha256": state,
            "formal_inputs_unchanged": True,
            "input_sha256_before": hashes,
            "input_sha256_after": dict(hashes),
            "as_trained_formal_sweep_consistency": {
                "max_abs_numeric_delta": 0.0,
                "all_count_fields_match": True,
                "fixed_threshold_0_5_numeric_deltas_diagnostic_minus_formal": {
                    "pd": 0.0,
                    "fa": 0.0,
                    "miou": 0.0,
                },
                "fixed_threshold_0_5_exact_count_matches": {
                    "matched_target_count": True,
                },
                "formal_sweep_checkpoint_sha256": hashes["checkpoint"],
            },
            "modes": modes,
            "checkpoint_static_diagnostics": {
                "block_count": 7,
                "blocks": static_blocks,
                "aggregate": {
                    "saliency_scale_effective_abs_tanh": {
                        "median": 0.09,
                        "p90": 0.19,
                        "max": 0.29,
                    },
                    "phase_sum_cancellation": {
                        "rho_l1": 0.53,
                        "rho_l2": 0.54,
                    },
                },
                "definitions": {
                    "saliency_scale": "absolute tanh",
                    "rho_l1": "synthetic",
                    "rho_l2": "synthetic",
                },
                "scope_limit": "static only",
            },
        }

    def rewrite(self, key: str) -> None:
        self.paths[key].write_text(
            json.dumps(self.payloads[key], sort_keys=True),
            encoding="utf-8",
        )


class FailureAtlasSummaryTests(unittest.TestCase):
    def test_complete_neutral_atlas_authorizes_trajectory_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            report = subject.build_summary(atlas.root)
        self.assertEqual(
            report["decision"]["status"],
            "GO_DCH_TRAJECTORY_TEST",
        )
        self.assertTrue(report["validation"]["all_checks_passed"])
        self.assertEqual(
            report["validation"]["as_trained_formal_sweep_exact_count"],
            8,
        )
        self.assertFalse(
            report["decision"]["conditions"]["context_direct_support"]
        )
        self.assertFalse(
            report["decision"]["implementation_state"][
                "dch_causal_mechanism_established"
            ]
        )
        markdown = subject.render_markdown(report)
        self.assertIn("GO_DCH_TRAJECTORY_TEST", markdown)
        self.assertIn("不建立 DCH 因果机制", markdown)

    def test_context_fragment_improvement_is_direct_support_not_causality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            key = subject.checkpoint_key(
                subject.FULL_VARIANT,
                3407,
                "pd_primary",
            )
            payload = atlas.payloads[key]
            for collection in (
                payload["modes"]["same_weights_context_off"][
                    "fixed_threshold_points"
                ],
                payload["modes"]["same_weights_context_off"][
                    "best_points_under_fa_budget"
                ],
            ):
                for point in collection.values():
                    if point["matched_target_count"] == 188:
                        point["gt_topology"]["fragment_excess_total"] = 2
            atlas.rewrite(key)
            report = subject.build_summary(atlas.root)
        self.assertEqual(report["decision"]["status"], "CONTEXT_DIRECT_SUPPORT")
        self.assertTrue(
            report["decision"]["conditions"][
                "context_off_improves_seed3407_primary_fragmentation"
            ]
        )
        self.assertFalse(
            report["decision"]["implementation_state"][
                "dch_causal_mechanism_established"
            ]
        )

    def test_residual_off_only_registered_explanation_returns_no_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            for role in subject.ROLES:
                key = subject.checkpoint_key(
                    subject.FULL_VARIANT,
                    3407,
                    role,
                )
                payload = atlas.payloads[key]
                for collection in (
                    payload["modes"]["same_weights_residual_off"][
                        "fixed_threshold_points"
                    ],
                    payload["modes"]["same_weights_residual_off"][
                        "best_points_under_fa_budget"
                    ],
                ):
                    for point in collection.values():
                        if point["matched_target_count"] == 188:
                            point["gt_topology"]["fragment_excess_total"] = 1
                            point["fa"] = 1e-6
                atlas.rewrite(key)
            report = subject.build_summary(atlas.root)
        self.assertEqual(report["decision"]["status"], "NO_GO_DCH")
        self.assertTrue(
            report["decision"]["conditions"][
                "residual_off_only_explains_registered_failure"
            ]
        )
        self.assertFalse(
            report["decision"]["implementation_state"][
                "v7_dch_implementation_authorized"
            ]
        )

    def test_nonzero_formal_delta_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            key = subject.checkpoint_key(
                subject.FULL_VARIANT,
                42,
                "pd_primary",
            )
            atlas.payloads[key][
                "as_trained_formal_sweep_consistency"
            ]["max_abs_numeric_delta"] = 1e-8
            atlas.rewrite(key)
            with self.assertRaisesRegex(ValueError, "Non-zero"):
                subject.build_summary(atlas.root)

    def test_changed_input_sha_and_unrestored_state_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            key = subject.checkpoint_key(
                subject.FULL_VARIANT,
                42,
                "pd_primary",
            )
            atlas.payloads[key]["input_sha256_after"]["metrics"] = "changed"
            atlas.rewrite(key)
            with self.assertRaisesRegex(ValueError, "Input SHA changed"):
                subject.build_summary(atlas.root)

        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            key = subject.checkpoint_key(
                subject.FULL_VARIANT,
                42,
                "pd_primary",
            )
            atlas.payloads[key]["modes"]["same_weights_residual_off"][
                "counterfactual_provenance"
            ]["state_sha256_after_restore"] = "changed"
            atlas.rewrite(key)
            with self.assertRaisesRegex(ValueError, "not restored"):
                subject.build_summary(atlas.root)

    def test_write_outputs_are_deterministic_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            atlas = SyntheticAtlas(Path(directory))
            report = subject.build_summary(atlas.root)
            json_path, markdown_path = subject.write_outputs(
                atlas.root,
                report,
                overwrite=False,
            )
            first_json = json_path.read_bytes()
            first_markdown = markdown_path.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_outputs(
                    atlas.root,
                    report,
                    overwrite=False,
                )
            subject.write_outputs(atlas.root, report, overwrite=True)
            self.assertEqual(first_json, json_path.read_bytes())
            self.assertEqual(first_markdown, markdown_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
