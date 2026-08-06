from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis import compare_three_dataset_ner_l4_tpr_posttraining_v1 as subject


FOREGROUND_PIXELS = 1000
VALID_PIXELS = 10000


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _metrics(
    *,
    matched: int,
    component_fp: int,
    true_positive_pixels: int,
    background_fp: int,
    miou_override: float | None = None,
    include_pixel_metrics: bool = True,
    explicit_background_fp: bool = False,
) -> dict[str, object]:
    target_count = 100
    tiny_count = 20
    matched_tiny = min(18, matched)
    precision = true_positive_pixels / (true_positive_pixels + background_fp)
    recall = true_positive_pixels / FOREGROUND_PIXELS
    f1 = 2.0 * precision * recall / (precision + recall)
    miou = true_positive_pixels / (FOREGROUND_PIXELS + background_fp)
    payload: dict[str, object] = {
        "fa": component_fp / VALID_PIXELS,
        "false_objects_per_image": 0.5,
        "matched_target_count": matched,
        "matched_tiny_target_count": str(matched_tiny),
        "miou": miou if miou_override is None else miou_override,
        "niou": min(1.0, miou + 0.01),
        "pd": matched / target_count,
        "predicted_object_count": matched + 5,
        "target_count": target_count,
        "test_loss": 0.01,
        "tiny_pd": matched_tiny / tiny_count,
        "tiny_target_count": str(tiny_count),
        "unmatched_predicted_object_count": 5,
        "valid_pixel_count": VALID_PIXELS,
    }
    if include_pixel_metrics:
        payload.update(
            {
                "pixel_precision": precision,
                "pixel_recall": recall,
                "pixel_f1": f1,
            }
        )
    if explicit_background_fp:
        payload["false_positive_pixels"] = background_fp
    return payload


def _method_settings(method: str) -> tuple[str, str, str]:
    if method == subject.CANDIDATE_METHOD:
        return subject.CANDIDATE_SCHEMA, "final", subject.CANDIDATE_RECIPE
    if method == subject.CURRENT_METHOD:
        return subject.CURRENT_SCHEMA, "final", subject.CURRENT_RECIPE
    return subject.ORIGINAL_SCHEMA, "original", "original_no_tss"


def _write_summary(
    *,
    root: Path,
    method: str,
    dataset: str,
    role_metrics: dict[str, dict[str, object]],
) -> Path:
    run_dir = subject._run_directory(root, method, dataset)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: dict[str, object] = {}
    roles: dict[str, object] = {}
    for role_index, role in enumerate(subject.CHECKPOINT_ROLES, start=1):
        raw = f"{method}/{dataset}/{role}".encode("utf-8")
        checkpoint = checkpoint_dir / f"{role}.pth.tar"
        checkpoint.write_bytes(raw)
        record = {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(raw),
            "bytes": len(raw),
        }
        checkpoints[role] = record
        roles[role] = {
            "epoch": 500 + role_index * 10,
            "path": str(checkpoint.resolve()),
            "metrics": role_metrics[role],
        }
    schema, summary_method, recipe_id = _method_settings(method)
    payload: dict[str, object] = {
        "schema": schema,
        "status": "complete",
        "dataset": dataset,
        "method": summary_method,
        "epochs": subject.FORMAL_EPOCHS,
        "seed": subject.SEED,
        "checkpoint_roles": list(subject.CHECKPOINT_ROLES),
        "requested_tss_weight": 0.0,
        "test_selected": True,
        "selection_is_optimistic": True,
        "recipe": {
            "recipe_id": recipe_id,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
        },
        "checkpoints": checkpoints,
        **roles,
    }
    if method == subject.CANDIDATE_METHOD:
        payload["planned_total_epochs"] = subject.FORMAL_EPOCHS
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    return summary


def _reference_role_metrics(method: str, role: str) -> dict[str, object]:
    # Current best_miou has reduced recall denominator 1000.  Together with
    # the other reference rows this fixes the dataset GT foreground count.
    if method == subject.CURRENT_METHOD:
        true_positive = 801 if role == "best_miou" else 825
        return _metrics(
            matched=90 if role == "best_miou" else 94,
            component_fp=20 if role == "best_miou" else 25,
            true_positive_pixels=true_positive,
            background_fp=100 if role == "best_miou" else 120,
        )
    true_positive = 700 if role == "best_miou" else 750
    return _metrics(
        matched=85 if role == "best_miou" else 92,
        component_fp=35 if role == "best_miou" else 40,
        true_positive_pixels=true_positive,
        background_fp=180 if role == "best_miou" else 200,
    )


def _candidate_role_metrics(*, mixed: bool = False) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for role in subject.CHECKPOINT_ROLES:
        output[role] = _metrics(
            matched=95 if role == "best_miou" else 97,
            component_fp=10 if not mixed else 50,
            true_positive_pixels=880 if role == "best_miou" else 900,
            background_fp=60 if not mixed else 240,
        )
    return output


def _populate_roots(base: Path, *, candidate: bool = True, mixed: bool = False) -> dict[str, Path]:
    roots = {
        subject.CANDIDATE_METHOD: base / "candidate",
        subject.CURRENT_METHOD: base / "current",
        subject.ORIGINAL_METHOD: base / "original",
    }
    for dataset in subject.DATASETS:
        for method in (subject.CURRENT_METHOD, subject.ORIGINAL_METHOD):
            _write_summary(
                root=roots[method],
                method=method,
                dataset=dataset,
                role_metrics={
                    role: _reference_role_metrics(method, role)
                    for role in subject.CHECKPOINT_ROLES
                },
            )
        if candidate:
            _write_summary(
                root=roots[subject.CANDIDATE_METHOD],
                method=subject.CANDIDATE_METHOD,
                dataset=dataset,
                role_metrics=_candidate_role_metrics(mixed=mixed),
            )
    return roots


def _build_from_roots(
    roots: dict[str, Path], *, allow_partial: bool = False
) -> dict[str, object]:
    loaded, completion = subject.load_inputs(
        candidate_root=roots[subject.CANDIDATE_METHOD],
        current_root=roots[subject.CURRENT_METHOD],
        original_root=roots[subject.ORIGINAL_METHOD],
        allow_partial=allow_partial,
        verify_checkpoint_files=True,
    )
    return subject.build_comparison(
        loaded,
        completion,
        roots=roots,
        allow_partial=allow_partial,
    )


class NERL4TPRPosttrainingComparisonTests(unittest.TestCase):
    def test_complete_candidate_uses_own_roles_and_dominates_both_references(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw))
            payload = _build_from_roots(roots)
        subject.validate_output_payload(payload)
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(payload["final_comparison_classification_made"])
        self.assertEqual(
            payload["classification"],
            "NER_L4_TPR_NON_INFERIOR_TO_BOTH_REFERENCES_REPORTED_VECTOR",
        )
        aggregate = payload["aggregate_comparisons"]
        self.assertEqual(
            aggregate["candidate_vs_current_final_tss_off"]["relation"],
            "candidate_dominates",
        )
        self.assertEqual(
            aggregate["candidate_vs_original"]["relation"],
            "candidate_dominates",
        )
        role = payload["per_dataset"][subject.DATASETS[0]]["checkpoint_roles"][
            "best_pd"
        ]["methods"][subject.CANDIDATE_METHOD]
        self.assertEqual(role["epoch"], 520)
        self.assertTrue(role["checkpoint"]["path"].endswith("best_pd.pth.tar"))
        point = role["metrics"]
        self.assertEqual(point["matched_target_count"], 97)
        self.assertEqual(point["target_count"], 100)
        self.assertEqual(point["component_false_positive_pixels"], 10)
        self.assertEqual(point["background_false_positive_pixels"], 60)
        self.assertEqual(point["foreground_positive_pixels"], FOREGROUND_PIXELS)

    def test_mixed_candidate_is_incomparable_not_scalarized(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw), mixed=True)
            payload = _build_from_roots(roots)
        current = payload["aggregate_comparisons"][
            "candidate_vs_current_final_tss_off"
        ]
        self.assertEqual(current["relation"], "incomparable")
        self.assertGreater(current["candidate_better_cell_count"], 0)
        self.assertGreater(current["candidate_worse_cell_count"], 0)
        self.assertEqual(
            payload["classification"],
            "NER_L4_TPR_MIXED_TRADEOFF_REPORTED_VECTOR",
        )
        policy = payload["relation_policy"]
        self.assertFalse(policy["weighted_sum_used"])
        self.assertFalse(policy["raw_metric_sum_used"])

    def test_incomplete_runs_fail_closed_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw), candidate=False)
            with self.assertRaisesRegex(
                subject.FormalRunsIncompleteError, "no final comparison"
            ):
                subject.load_inputs(
                    candidate_root=roots[subject.CANDIDATE_METHOD],
                    current_root=roots[subject.CURRENT_METHOD],
                    original_root=roots[subject.ORIGINAL_METHOD],
                    allow_partial=False,
                )

    def test_allow_partial_is_preview_without_final_classification(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw), candidate=False)
            payload = _build_from_roots(roots, allow_partial=True)
        subject.validate_output_payload(payload)
        self.assertEqual(payload["status"], "partial_preview")
        self.assertEqual(
            payload["classification"], "NOT_EVALUATED_INCOMPLETE_FORMAL_RUNS"
        )
        self.assertFalse(payload["final_comparison_classification_made"])
        self.assertEqual(payload["completion"]["complete_candidate_run_count"], 0)
        self.assertEqual(
            payload["aggregate_comparisons"][
                "candidate_vs_current_final_tss_off"
            ]["relation"],
            "not_available",
        )

    def test_pd_count_mismatch_and_cross_role_checkpoint_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw))
            dataset = subject.DATASETS[0]
            summary_path = subject._run_directory(
                roots[subject.CANDIDATE_METHOD], subject.CANDIDATE_METHOD, dataset
            ) / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["best_miou"]["metrics"]["pd"] = 0.01
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.NERL4TPRPosttrainingComparisonError,
                "pd differs from matched/total counts",
            ):
                _build_from_roots(roots)

            roots = _populate_roots(Path(raw) / "second")
            summary_path = subject._run_directory(
                roots[subject.CANDIDATE_METHOD], subject.CANDIDATE_METHOD, dataset
            ) / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["best_miou"]["path"] = payload["best_pd"]["path"]
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.NERL4TPRPosttrainingComparisonError,
                "checkpoint path differs from its own run directory",
            ):
                _build_from_roots(roots)

    def test_background_fp_requires_integer_confusion_cross_check(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            roots = _populate_roots(Path(raw))
            dataset = subject.DATASETS[0]
            summary_path = subject._run_directory(
                roots[subject.CANDIDATE_METHOD], subject.CANDIDATE_METHOD, dataset
            ) / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["best_miou"]["metrics"]["miou"] += 0.001
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                subject.NERL4TPRPosttrainingComparisonError,
                "miou fails integer confusion-count cross-check",
            ):
                _build_from_roots(roots)

    def test_explicit_background_fp_supports_absent_pixel_metrics(self) -> None:
        metrics = _metrics(
            matched=90,
            component_fp=12,
            true_positive_pixels=800,
            background_fp=70,
            include_pixel_metrics=False,
            explicit_background_fp=True,
        )
        normalized = subject._normalize_metrics(metrics, "synthetic")
        bound = subject._bind_background_fp(
            normalized,
            foreground_pixels=FOREGROUND_PIXELS,
            label="synthetic",
        )
        self.assertFalse(bound["pixel_metrics_available"])
        self.assertEqual(bound["background_false_positive_pixels"], 70)
        relation = subject.compare_points(bound, bound)
        self.assertEqual(relation["relation"], "equal")
        self.assertFalse(relation["cells"]["pixel_precision"]["available"])

    def test_markdown_json_roundtrip_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            roots = _populate_roots(directory / "inputs")
            payload = _build_from_roots(roots)
            before = copy.deepcopy(payload)
            json_path = directory / "output" / "comparison.json"
            markdown_path = directory / "output" / "comparison.md"
            subject.write_outputs(json_path, markdown_path, payload)
            roundtrip = json.loads(json_path.read_text(encoding="utf-8"))
            subject.validate_output_payload(roundtrip)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Pd（计数）", markdown)
            self.assertIn("component FP(px)", markdown)
            self.assertIn("background FP(px)", markdown)
            self.assertIn("Precision", markdown)
            self.assertIn("best_miou 只与 best_miou 比", markdown)
            self.assertEqual(payload, before)
            with self.assertRaises(FileExistsError):
                subject.write_outputs(json_path, markdown_path, payload)


if __name__ == "__main__":
    unittest.main()
