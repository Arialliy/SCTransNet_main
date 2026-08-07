"""Post-hoc, zero-margin PBDR-V3 adjudication against Original SCTransNet.

This program is deliberately metadata-only.  It reads existing JSON and
checkpoint file identities, never imports a dataset module, never constructs a
model, and never accesses the official test index.  Earlier evaluation and
deployment artifacts remain immutable; the output is a hash-bound advisory
metric overlay requested after those results were observed.  Because the
historical Original run did not attest both TF32 switches, the overlay never
turns its metric winner into a binding deployment decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "NUAA-SIRST"
ROLES = ("best_miou", "best_pd")
FIXED_THRESHOLD = 0.5
TARGET_COUNT = 263
VALID_PIXEL_COUNT = 14_577_078
TEST_COUNT = 214
ORDERED_TEST_IDS_SHA256 = (
    "b8a1b96c74306247d56da4abbaf7871619bc2210c710be51816f40002dc5ad1d"
)
INFERENCE_ORDER_NEWLINE_SHA256 = (
    "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5"
)
ORIGINAL_MANIFEST = (
    REPO_ROOT
    / "results/four_dataset_seed42_v1/selected_checkpoints/checkpoint_manifest.json"
)
EXPECTED_ORIGINAL_MANIFEST_SHA256 = (
    "f286c2f07113be079a2f447b3a2a4e868c81df58ac06cc4acda2de2210249799"
)
V3_RESULTS_ROOT = REPO_ROOT / "results/nuaa_pbdr_v3_stage1_v1"
DEFAULT_OUTPUT = (
    V3_RESULTS_ROOT / "original_zero_margin_role_adjudication_v1.json"
)
POLICY_DOCUMENT = (
    REPO_ROOT / "experiments/PBDR_V3_VS_ORIGINAL_ZERO_MARGIN_POLICY.md"
)
ORIGINAL_EVALUATION_ROOT = (
    REPO_ROOT
    / "results/four_dataset_seed42_v1/evaluations/fixed_0_5/NUAA-SIRST/original"
)
EXPECTED_ORIGINAL_PROTOCOL_SHA256 = (
    "350509403ea8c08d7b354fa40e83d17773d4014b44d838948f93560ce2e16a0e"
)
EXPECTED_ORIGINAL_EVALUATION_SHA256 = {
    "best_miou": "a6e518d45c7f7bc973b905b2d85cca84fba8b45cf46ebc6ace447f81ca450344",
    "best_pd": "0d15be0746251c9b6bb8c66d24a7e37cf56051699f4cf5a1ea2568fa55ad6e8f",
}
EXPECTED_V3_ARTIFACT_SHA256 = {
    "best_miou": {
        "claim": "61f83d1d39588e53f71c1984e4ebce440ceb781afa025d431da073f0cb184790",
        "evaluation": "e640c1eb2a3fb5363e54e9087018ad35b324085a974c40b96ba91c7fa3fef46a",
        "deployment": "1a29abf990778fe20378242919ebeee8c6a2d278879076ff5857dc73ee4cb16f",
        "summary": "ddff2b11cbd2fe36dc9f9c61d97c68532794bccc26a06507b84b130ea37e50e9",
        "protocol": "e6034570c8862912f92529d7d17c1949d26034b3a0d982289e93f14b065d92f8",
        "split": "318a7bfb1c692f08f3dfbfc26de5e78f3ac7282e99e9e59e91fd71e7e308fde1",
        "candidate": "0ac2222c76771a9a7bccec2b67837308034af054b1e2ccc846b80cc7569fade3",
    },
    "best_pd": {
        "claim": "551354f57fb75d7063cb6cb58c07991aa210d294a9e03f7da1d1dd94fa3fac37",
        "evaluation": "689925beb13e0c5a7e0f720751d8067f09fa803910d66aa72bec90c09b075395",
        "deployment": "045120c98f49f4747471c63f117f0ba302ca3e510650f70ac67d692b3cbea4cf",
        "summary": "fe44bd20d84538422895689b83592300005052b83a4c4f5cc860ca32dc5acfbc",
        "protocol": "88ec0c83625e216fed3f5f5276d64d6af8ef2aad560d5f4aeec33e9c6d27483b",
        "split": "318a7bfb1c692f08f3dfbfc26de5e78f3ac7282e99e9e59e91fd71e7e308fde1",
        "candidate": "ab77c41f40bcdd07b9f71b377b9e19c4af4e00de945217b07dec696b758535ad",
    },
}
EXPECTED_EVALUATOR_SHA256 = (
    "3af14c20f683288f9cc8ac5b5a1b0cbd201503fb5278ab2a90c90c05ca40d43f"
)
EXPECTED_TRAINER_SHA256 = (
    "6dff4315c5c43121d0d401c3eb827b4dafd02f698b4e8281b0bf37be858dd064"
)
SCHEMA = "sctransnet_nuaa_pbdr_v3_vs_original_zero_margin_role_v1/v1"

ROLE_ORDERS: dict[str, tuple[str, ...]] = {
    "best_miou": (
        "higher_miou",
        "higher_pd",
        "lower_fa",
        "higher_niou",
        "higher_tiny_pd",
        "lower_test_loss",
        "earlier_epoch",
    ),
    "best_pd": (
        "higher_pd",
        "lower_fa",
        "higher_tiny_pd",
        "higher_miou",
        "higher_niou",
        "lower_test_loss",
        "earlier_epoch",
    ),
}


class OriginalZeroMarginPolicyError(ValueError):
    """An input artifact or comparison boundary is not the frozen authority."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OriginalZeroMarginPolicyError(message)


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OriginalZeroMarginPolicyError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise OriginalZeroMarginPolicyError(f"{label} must be numeric") from error
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _integer(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be int")
    return int(value)


def role_key(role: str, metrics: Mapping[str, Any], epoch: int) -> tuple[float, ...]:
    """Replay the Original manifest's exact same-role selection order."""

    _require(role in ROLES, "unknown checkpoint role")
    _require(isinstance(epoch, int) and not isinstance(epoch, bool), "epoch must be int")
    values = {
        "pd": _finite(metrics.get("pd"), "pd"),
        "fa": _finite(metrics.get("fa"), "fa"),
        "miou": _finite(metrics.get("miou"), "miou"),
        "niou": _finite(metrics.get("niou"), "niou"),
        "tiny_pd": _finite(metrics.get("tiny_pd"), "tiny_pd"),
        "test_loss": _finite(metrics.get("test_loss"), "test_loss"),
    }
    if role == "best_miou":
        return (
            values["miou"],
            values["pd"],
            -values["fa"],
            values["niou"],
            values["tiny_pd"],
            -values["test_loss"],
            -float(epoch),
        )
    return (
        values["pd"],
        -values["fa"],
        values["tiny_pd"],
        values["miou"],
        values["niou"],
        -values["test_loss"],
        -float(epoch),
    )


def compare_role(
    role: str,
    original_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    original_epoch: int,
    candidate_epoch: int,
) -> dict[str, Any]:
    """Return a strict zero-margin, same-role lexicographic decision."""

    original_key = role_key(role, original_metrics, original_epoch)
    candidate_key = role_key(role, candidate_metrics, candidate_epoch)
    decisive_index: int | None = None
    for index, (candidate_value, original_value) in enumerate(
        zip(candidate_key, original_key, strict=True)
    ):
        if candidate_value != original_value:
            decisive_index = index
            break
    advisory_metric_winner = (
        "candidate" if candidate_key > original_key else "original"
    )

    directions = {
        "matched_target_count": "higher",
        "pd": "higher",
        "fa": "lower",
        "miou": "higher",
        "niou": "higher",
        "matched_tiny_target_count": "higher",
        "tiny_pd": "higher",
        "unmatched_predicted_object_count": "lower",
        "unmatched_predicted_pixel_count": "lower",
        "test_loss": "lower",
    }
    deltas: dict[str, dict[str, Any]] = {}
    for name, direction in directions.items():
        original_value = _finite(original_metrics.get(name), f"original.{name}")
        candidate_value = _finite(candidate_metrics.get(name), f"candidate.{name}")
        delta = candidate_value - original_value
        improved = delta > 0.0 if direction == "higher" else delta < 0.0
        regressed = delta < 0.0 if direction == "higher" else delta > 0.0
        deltas[name] = {
            "direction": direction,
            "original": original_value,
            "candidate": candidate_value,
            "candidate_minus_original": delta,
            "improved": improved,
            "tied": delta == 0.0,
            "regressed": regressed,
        }

    return {
        "advisory_metric_winner": advisory_metric_winner,
        "candidate_strictly_better_on_role_order": candidate_key > original_key,
        "exact_tie": candidate_key == original_key,
        "minimum_gain": 0.0,
        "comparison": "strict_lexicographic_no_positive_margin",
        "selection_order": list(ROLE_ORDERS[role]),
        "decisive_index": decisive_index,
        "decisive_term": (
            ROLE_ORDERS[role][decisive_index]
            if decisive_index is not None
            else None
        ),
        "tail_terms_not_reached": (
            list(ROLE_ORDERS[role][decisive_index + 1 :])
            if decisive_index is not None
            else []
        ),
        "original_key": list(original_key),
        "candidate_key": list(candidate_key),
        "metric_deltas": deltas,
        "improved_metrics": [name for name, value in deltas.items() if value["improved"]],
        "regressed_metrics": [name for name, value in deltas.items() if value["regressed"]],
        "tied_metrics": [name for name, value in deltas.items() if value["tied"]],
    }


def _artifact(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise OriginalZeroMarginPolicyError(f"artifact cannot be a symlink: {candidate}")
    ready = candidate.resolve(strict=True)
    return {
        "path": str(ready),
        "sha256": file_sha256(ready),
        "bytes": ready.stat().st_size,
    }


def _original_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    records = manifest.get("records")
    _require(isinstance(records, list), "Original manifest records are malformed")
    matches = [
        value
        for value in records
        if isinstance(value, Mapping)
        and value.get("dataset") == DATASET
        and value.get("method") == "original"
    ]
    _require(len(matches) == 1, "Original NUAA record is not unique")
    record = matches[0]
    _require(record.get("audit_passed") is True, "Original audit did not pass")
    disclosure = record.get("selection_disclosure")
    _require(isinstance(disclosure, Mapping), "Original selection disclosure is missing")
    for role in ROLES:
        _require(
            disclosure.get(f"{role}_order") == list(ROLE_ORDERS[role]),
            f"Original {role} selection order differs",
        )
    _require(
        disclosure.get("selection_threshold") == FIXED_THRESHOLD,
        "Original selection threshold differs",
    )
    protocol = record.get("protocol_audit")
    _require(isinstance(protocol, Mapping), "Original protocol audit is missing")
    _require(
        protocol.get("sha256") == EXPECTED_ORIGINAL_PROTOCOL_SHA256,
        "Original protocol SHA differs",
    )
    _require(file_sha256(Path(str(protocol.get("path")))) == EXPECTED_ORIGINAL_PROTOCOL_SHA256, "Original protocol file differs")
    return record


def _fa_numerator(metrics: Mapping[str, Any], *, label: str) -> int:
    fa = _finite(metrics.get("fa"), f"{label}.fa")
    pixels = _integer(metrics.get("valid_pixel_count"), f"{label}.valid_pixel_count")
    numerator = fa * pixels
    rounded = int(round(numerator))
    _require(abs(numerator - rounded) <= 1.0e-9, f"{label} Fa numerator is not integral")
    return rounded


def _enriched_metrics(metrics: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    ready = dict(metrics)
    ready["unmatched_predicted_pixel_count"] = _fa_numerator(metrics, label=label)
    return ready


def _validate_original_role(
    role: str,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = record.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "Original checkpoints are missing")
    checkpoint = checkpoints.get(role)
    _require(isinstance(checkpoint, Mapping), f"Original {role} is missing")
    _require(checkpoint.get("checkpoint_role") == role, "Original role differs")
    _require(checkpoint.get("test_selected") is True, "Original test-selected disclosure differs")
    _require(
        checkpoint.get("selection_is_optimistic") is True,
        "Original optimistic-selection disclosure differs",
    )
    metrics = checkpoint.get("fixed_threshold_0_5_metrics")
    _require(isinstance(metrics, Mapping), "Original fixed metrics are missing")
    _require(_finite(metrics.get("threshold"), "Original threshold") == FIXED_THRESHOLD, "Original threshold differs")
    _require(_integer(metrics.get("target_count"), "Original target_count") == TARGET_COUNT, "Original target count differs")
    _require(_integer(metrics.get("valid_pixel_count"), "Original valid pixels") == VALID_PIXEL_COUNT, "Original valid pixels differ")
    frozen = Path(str(checkpoint.get("frozen_path"))).resolve(strict=True)
    _require(file_sha256(frozen) == checkpoint.get("sha256"), "Original checkpoint SHA differs")

    evaluation_path = ORIGINAL_EVALUATION_ROOT / f"{role}.json"
    _require(
        file_sha256(evaluation_path) == EXPECTED_ORIGINAL_EVALUATION_SHA256[role],
        f"Original {role} formal evaluation SHA differs",
    )
    evaluation = _json_object(evaluation_path, label=f"Original {role} formal evaluation")
    _require(evaluation.get("status") == "complete", "Original formal evaluation is incomplete")
    _require(evaluation.get("evaluation_dataset") == DATASET, "Original formal evaluation dataset differs")
    _require(evaluation.get("method") == "original", "Original formal evaluation method differs")
    _require(evaluation.get("checkpoint_role") == role, "Original formal evaluation role differs")
    _require(evaluation.get("test_selected") is True, "Original formal test-selection disclosure differs")
    _require(evaluation.get("selection_is_optimistic") is True, "Original formal optimism disclosure differs")
    evaluation_checkpoint = evaluation.get("checkpoint")
    _require(isinstance(evaluation_checkpoint, Mapping), "Original formal checkpoint binding is missing")
    _require(evaluation_checkpoint.get("sha256") == checkpoint.get("sha256"), "Original formal checkpoint SHA differs")
    _require(file_sha256(Path(str(evaluation_checkpoint.get("path")))) == checkpoint.get("sha256"), "Original formal checkpoint file differs")
    evaluation_data = evaluation.get("data")
    _require(isinstance(evaluation_data, Mapping), "Original formal data binding is missing")
    _require(evaluation_data.get("test_count") == TEST_COUNT, "Original formal test count differs")
    _require(
        evaluation_data.get("ordered_sample_id_sha256") == INFERENCE_ORDER_NEWLINE_SHA256,
        "Original formal ordered IDs differ",
    )
    formal_metrics = evaluation.get("fixed_threshold_0_5")
    _require(isinstance(formal_metrics, Mapping), "Original formal fixed metrics are missing")
    for field in (
        "matched_target_count",
        "pd",
        "fa",
        "miou",
        "niou",
        "matched_tiny_target_count",
        "tiny_pd",
        "unmatched_predicted_object_count",
        "target_count",
        "valid_pixel_count",
    ):
        _require(
            _finite(formal_metrics.get(field), f"formal.{field}")
            == _finite(metrics.get(field), f"manifest.{field}"),
            f"Original formal {field} differs from manifest",
        )
    _require(
        abs(_finite(formal_metrics.get("test_loss"), "formal.test_loss") - _finite(metrics.get("test_loss"), "manifest.test_loss")) <= 1.0e-7,
        "Original formal test loss exceeds its frozen recomputation tolerance",
    )
    return dict(checkpoint), {
        "metrics": _enriched_metrics(formal_metrics, label=f"Original.{role}"),
        "formal_evaluation": _artifact(evaluation_path),
        "manifest_selection_metrics": dict(metrics),
    }


def _validate_v3_role(role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = V3_RESULTS_ROOT / "formal" / role / "core"
    evaluation_path = run_dir / "evaluation.json"
    deployment_path = run_dir / "deployment.json"
    summary_path = run_dir / "summary.json"
    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split_manifest.json"
    claim_path = run_dir / "official_test_access_claim.json"
    expected = EXPECTED_V3_ARTIFACT_SHA256[role]
    for name, path in (
        ("evaluation", evaluation_path),
        ("deployment", deployment_path),
        ("summary", summary_path),
        ("protocol", protocol_path),
        ("split", split_path),
        ("claim", claim_path),
    ):
        _require(file_sha256(path) == expected[name], f"V3 {role} {name} SHA differs")
    evaluation = _json_object(evaluation_path, label=f"V3 {role} evaluation")
    deployment = _json_object(deployment_path, label=f"V3 {role} deployment")
    summary = _json_object(summary_path, label=f"V3 {role} summary")
    protocol = _json_object(protocol_path, label=f"V3 {role} protocol")
    claim = _json_object(claim_path, label=f"V3 {role} official-test claim")
    _require(evaluation.get("status") == "complete", "V3 evaluation is incomplete")
    _require(evaluation.get("dataset") == DATASET, "V3 dataset differs")
    _require(evaluation.get("parent_role") == role, "V3 role differs")
    _require(evaluation.get("recipe") == "core", "V3 recipe differs")
    _require(evaluation.get("official_test_accessed") is True, "V3 official evaluation is missing")
    data = evaluation.get("data")
    _require(isinstance(data, Mapping), "V3 data binding is missing")
    _require(data.get("test_count") == TEST_COUNT, "V3 test count differs")
    _require(data.get("img_idx_test_ordered_ids_sha256") == ORDERED_TEST_IDS_SHA256, "V3 ordered-ID SHA differs")
    _require(data.get("inference_order_newline_sha256") == INFERENCE_ORDER_NEWLINE_SHA256, "V3 inference order differs")
    candidate_metrics = evaluation.get("metrics", {}).get("candidate", {}).get("fixed_0_5")
    _require(isinstance(candidate_metrics, Mapping), "V3 fixed candidate metrics are missing")
    _require(_finite(candidate_metrics.get("threshold"), "V3 threshold") == FIXED_THRESHOLD, "V3 threshold differs")
    _require(_integer(candidate_metrics.get("target_count"), "V3 target_count") == TARGET_COUNT, "V3 target count differs")
    _require(_integer(candidate_metrics.get("valid_pixel_count"), "V3 valid pixels") == VALID_PIXEL_COUNT, "V3 valid pixels differ")
    checkpoint = evaluation.get("candidate_checkpoint")
    _require(isinstance(checkpoint, Mapping), "V3 candidate checkpoint is missing")
    checkpoint_path = Path(str(checkpoint.get("path"))).resolve(strict=True)
    _require(file_sha256(checkpoint_path) == checkpoint.get("sha256"), "V3 candidate SHA differs")
    _require(checkpoint.get("sha256") == expected["candidate"], "V3 candidate frozen SHA differs")
    _require(deployment.get("evaluation", {}).get("sha256") == file_sha256(evaluation_path), "V3 deployment/evaluation binding differs")
    _require(evaluation.get("official_test_access_claim", {}).get("sha256") == expected["claim"], "V3 evaluation/claim binding differs")
    _require(deployment.get("official_test_access_claim", {}).get("sha256") == expected["claim"], "V3 deployment/claim binding differs")
    _require(deployment.get("trainer_protocol", {}).get("sha256") == expected["protocol"], "V3 deployment/protocol binding differs")
    _require(deployment.get("trainer_summary", {}).get("sha256") == expected["summary"], "V3 deployment/summary binding differs")
    _require(deployment.get("internal_split_manifest", {}).get("sha256") == expected["split"], "V3 deployment/split binding differs")
    _require(summary.get("selected_epoch") == checkpoint.get("epoch"), "V3 selected epoch differs")
    _require(summary.get("selected_checkpoint", {}).get("sha256") == expected["candidate"], "V3 summary/candidate binding differs")
    protocol_sha = protocol.get("protocol_sha256")
    _require(isinstance(protocol_sha, str), "V3 protocol canonical SHA is missing")
    _require(summary.get("protocol_sha256") == protocol_sha, "V3 summary/protocol canonical binding differs")
    _require(evaluation.get("protocol_sha256") == protocol_sha, "V3 evaluation/protocol canonical binding differs")
    _require(claim.get("protocol_sha256") == protocol_sha, "V3 claim/protocol canonical binding differs")
    _require(claim.get("status") == "claimed_before_dataset_construction", "V3 claim status differs")
    _require(claim.get("maximum_official_test_evaluations") == 1, "V3 one-use claim differs")
    _require(claim.get("dataset") == DATASET and claim.get("parent_role") == role and claim.get("recipe") == "core", "V3 claim identity differs")
    _require(claim.get("candidate_sha256") == expected["candidate"], "V3 claim/candidate binding differs")
    _require(protocol.get("precision") == "fp32", "V3 protocol precision differs")
    protocol_sources = protocol.get("source_locks", {}).get("runtime_sources", {})
    evaluation_sources = evaluation.get("runtime_sources", {})
    evaluator_rel = "experiments/evaluate_nuaa_pbdr_v3_stage1_v1.py"
    trainer_rel = "experiments/train_nuaa_pbdr_v3_stage1_v1.py"
    _require(protocol_sources.get(evaluator_rel, {}).get("sha256") == EXPECTED_EVALUATOR_SHA256, "V3 protocol evaluator lock differs")
    _require(protocol_sources.get(trainer_rel, {}).get("sha256") == EXPECTED_TRAINER_SHA256, "V3 protocol trainer lock differs")
    _require(evaluation_sources.get(evaluator_rel, {}).get("sha256") == EXPECTED_EVALUATOR_SHA256, "V3 evaluation evaluator lock differs")
    _require(evaluation_sources.get(trainer_rel, {}).get("sha256") == EXPECTED_TRAINER_SHA256, "V3 evaluation trainer lock differs")
    return {
        "run_dir": str(run_dir.resolve()),
        "evaluation": _artifact(evaluation_path),
        "earlier_deployment": _artifact(deployment_path),
        "summary": _artifact(summary_path),
        "protocol": _artifact(protocol_path),
        "split_manifest": _artifact(split_path),
        "official_test_access_claim": _artifact(claim_path),
        "candidate_checkpoint": _artifact(checkpoint_path),
        "candidate_epoch": int(checkpoint["epoch"]),
    }, _enriched_metrics(candidate_metrics, label=f"V3.{role}")


def build_adjudication() -> dict[str, Any]:
    """Build the complete overlay without reading any dataset or model."""

    _require(file_sha256(ORIGINAL_MANIFEST) == EXPECTED_ORIGINAL_MANIFEST_SHA256, "Original manifest SHA differs")
    manifest = _json_object(ORIGINAL_MANIFEST, label="Original checkpoint manifest")
    _require(manifest.get("status") == "complete", "Original manifest is incomplete")
    _require(manifest.get("no_fabricated_results") is True, "Original manifest integrity differs")
    record = _original_record(manifest)

    roles: dict[str, Any] = {}
    for role in ROLES:
        original_checkpoint, original = _validate_original_role(role, record)
        v3_binding, candidate_metrics = _validate_v3_role(role)
        original_metrics = original["metrics"]
        comparison = compare_role(
            role,
            original_metrics,
            candidate_metrics,
            original_epoch=int(original_checkpoint["epoch"]),
            candidate_epoch=int(v3_binding["candidate_epoch"]),
        )
        advisory_metric_winner = comparison["advisory_metric_winner"]
        advisory_winner_artifact = (
            {
                "kind": "pbdr_v3_candidate",
                **v3_binding["candidate_checkpoint"],
            }
            if advisory_metric_winner == "candidate"
            else {
                "kind": "original_sctransnet",
                "path": str(Path(str(original_checkpoint["frozen_path"])).resolve(strict=True)),
                "sha256": str(original_checkpoint["sha256"]),
                "bytes": Path(str(original_checkpoint["frozen_path"])).stat().st_size,
            }
        )
        roles[role] = {
            "advisory_metric_winner": advisory_metric_winner,
            "comparison_threshold": FIXED_THRESHOLD,
            "advisory_reason": "strict_same_role_original_key_comparison_no_positive_margin",
            "advisory_winner_artifact": advisory_winner_artifact,
            "binding_decision": {
                "status": "blocked_precision_provenance",
                "binding_eligible": False,
                "binding_selected": None,
                "effective_deployment": "earlier_current_based_deployment_unchanged",
            },
            "comparison": comparison,
            "original": {
                "checkpoint": {
                    "path": str(Path(str(original_checkpoint["frozen_path"])).resolve(strict=True)),
                    "sha256": str(original_checkpoint["sha256"]),
                    "bytes": Path(str(original_checkpoint["frozen_path"])).stat().st_size,
                    "epoch": int(original_checkpoint["epoch"]),
                    "test_selected": True,
                    "selection_is_optimistic": True,
                },
                "metrics": dict(original_metrics),
                "formal_evaluation": original["formal_evaluation"],
                "manifest_selection_metrics": original["manifest_selection_metrics"],
            },
            "candidate": {
                "checkpoint": v3_binding["candidate_checkpoint"],
                "epoch": int(v3_binding["candidate_epoch"]),
                "metrics": candidate_metrics,
                "source_artifacts": {
                    key: value
                    for key, value in v3_binding.items()
                    if key not in ("candidate_checkpoint", "candidate_epoch")
                },
            },
        }

    return {
        "schema": SCHEMA,
        "status": "advisory_complete_binding_blocked",
        "dataset": DATASET,
        "policy": {
            "baseline": "same_role_original_sctransnet",
            "threshold": FIXED_THRESHOLD,
            "minimum_gain": 0.0,
            "comparison": "strict_role_specific_lexicographic_order",
            "timing": "post_official_result_user_revision",
            "confirmatory_status": "post_hoc_not_preregistered",
            "decision_scope": "advisory_metric_comparison_only",
            "binding_eligible": False,
            "binding_blocker": "original_dual_tf32_off_not_attested",
            "earlier_deployment_artifacts_overwritten": False,
            "official_test_reaccessed": False,
            "dataset_or_model_loaded": False,
            "policy_document": _artifact(POLICY_DOCUMENT),
        },
        "original_authority": {
            "manifest": _artifact(ORIGINAL_MANIFEST),
            "audit_passed": True,
            "test_selected": True,
            "selection_is_optimistic": True,
        },
        "precision_provenance": {
            "v3": "explicit_cuda_matmul_and_cudnn_tf32_off",
            "original": "historical_artifact_does_not_explicitly_attest_dual_tf32_off",
            "claim": "authoritative_historical_fixed_0_5_comparison_not_fully_matched_precision_attestation",
        },
        "roles": roles,
        "adjudicator": _artifact(Path(__file__)),
        "no_fabricated_results": True,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is atomic and refuses an existing target.
        # Unlike os.replace(), it cannot overwrite another adjudication created
        # between the initial check and this commit point.
        os.link(temporary, destination)
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate existing NUAA V3 artifacts against Original without test re-access."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_adjudication()
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output),
                "official_test_reaccessed": False,
                "advisory_metric_winners": {
                    role: payload["roles"][role]["advisory_metric_winner"]
                    for role in ROLES
                },
                "binding_status": "blocked_precision_provenance",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "OriginalZeroMarginPolicyError",
    "ROLE_ORDERS",
    "ROLES",
    "build_adjudication",
    "compare_role",
    "file_sha256",
    "main",
    "role_key",
]
