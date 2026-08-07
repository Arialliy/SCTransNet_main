"""Strict zero-margin selection against the Original+Current envelope.

Every record is bound to one immutable evaluation context.  Integer sufficient
statistics are used for mIoU, Pd, tiny-Pd, and Fa comparisons; only nIoU and
loss use deterministic floating-point values.  There is deliberately no
epsilon, minimum gain, percentage margin, or pass/fail gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Literal, Mapping, Sequence


Role = Literal["best_miou", "best_pd"]
CandidateFamily = Literal[
    "Original",
    "Current",
    "V3-calibrated",
    "V4-Stage1",
    "V4-Stage2",
]
SUPPORTED_ROLES: tuple[Role, ...] = ("best_miou", "best_pd")
FROZEN_TIE_ORDER: tuple[CandidateFamily, ...] = (
    "Original",
    "Current",
    "V3-calibrated",
    "V4-Stage1",
    "V4-Stage2",
)
_TIE_RANK = {name: index for index, name in enumerate(FROZEN_TIE_ORDER)}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ZeroMarginSelectionError(ValueError):
    """A metric record or envelope comparison is invalid."""


def _require_plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ZeroMarginSelectionError(f"{name} must be non-negative")
    return value


def _require_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(ready):
        raise ZeroMarginSelectionError(f"{name} must be finite")
    return ready


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ZeroMarginSelectionError(f"{name} must be lowercase SHA-256")
    return value


def _require_role(role: str) -> Role:
    if role not in SUPPORTED_ROLES:
        raise ZeroMarginSelectionError(f"unsupported role: {role!r}")
    return role  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EvaluationBinding:
    """Immutable identity of the split and metric semantics used by a record."""

    dataset: str
    role: Role
    evaluation_context_sha256: str
    sample_id_order_sha256: str
    target_sha256: str
    metric_core_sha256: str
    threshold_numerator: int = 1
    threshold_denominator: int = 2
    connectivity: int = 2
    match_radius: float = 3.0
    tiny_area: int = 9

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ZeroMarginSelectionError("dataset must be a non-empty string")
        _require_role(self.role)
        for field in (
            "evaluation_context_sha256",
            "sample_id_order_sha256",
            "target_sha256",
            "metric_core_sha256",
        ):
            _require_sha256(getattr(self, field), name=field)
        _require_plain_int(self.threshold_numerator, name="threshold_numerator")
        _require_plain_int(self.threshold_denominator, name="threshold_denominator")
        if self.threshold_denominator == 0:
            raise ZeroMarginSelectionError("threshold_denominator must be positive")
        threshold = Fraction(self.threshold_numerator, self.threshold_denominator)
        if threshold != Fraction(1, 2):
            raise ZeroMarginSelectionError("V4 selection threshold must be exactly 0.5")
        if self.connectivity != 2:
            raise ZeroMarginSelectionError("V4 matcher connectivity must be 2 (8-connected)")
        radius = _require_finite_float(self.match_radius, name="match_radius")
        if radius != 3.0:
            raise ZeroMarginSelectionError("V4 match radius must be exactly 3")
        _require_plain_int(self.tiny_area, name="tiny_area")
        if self.tiny_area != 9:
            raise ZeroMarginSelectionError("V4 tiny-area definition must be exactly 9")

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "role": self.role,
            "evaluation_context_sha256": self.evaluation_context_sha256,
            "sample_id_order_sha256": self.sample_id_order_sha256,
            "target_sha256": self.target_sha256,
            "metric_core_sha256": self.metric_core_sha256,
            "threshold": {
                "numerator": self.threshold_numerator,
                "denominator": self.threshold_denominator,
            },
            "connectivity": self.connectivity,
            "match_radius": self.match_radius,
            "tiny_area": self.tiny_area,
        }


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One fixed-threshold result with exact sufficient statistics."""

    name: str
    family: CandidateFamily
    binding: EvaluationBinding
    intersection_pixels: int
    union_pixels: int
    matched_target_count: int
    target_count: int
    unmatched_component_pixels: int
    valid_pixels: int
    niou: float
    matched_tiny_target_count: int
    tiny_target_count: int
    loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ZeroMarginSelectionError("name must be a non-empty string")
        if self.family not in _TIE_RANK:
            raise ZeroMarginSelectionError(f"unsupported candidate family: {self.family!r}")
        if not isinstance(self.binding, EvaluationBinding):
            raise TypeError("binding must be an EvaluationBinding")
        integer_fields = (
            "intersection_pixels",
            "union_pixels",
            "matched_target_count",
            "target_count",
            "unmatched_component_pixels",
            "valid_pixels",
            "matched_tiny_target_count",
            "tiny_target_count",
        )
        for field in integer_fields:
            _require_plain_int(getattr(self, field), name=field)
        if self.union_pixels <= 0:
            raise ZeroMarginSelectionError("union_pixels must be positive")
        if self.target_count <= 0:
            raise ZeroMarginSelectionError("target_count must be positive")
        if self.valid_pixels <= 0:
            raise ZeroMarginSelectionError("valid_pixels must be positive")
        if self.intersection_pixels > self.union_pixels:
            raise ZeroMarginSelectionError("intersection_pixels exceeds union_pixels")
        if self.union_pixels > self.valid_pixels:
            raise ZeroMarginSelectionError("union_pixels exceeds valid_pixels")
        if self.matched_target_count > self.target_count:
            raise ZeroMarginSelectionError("matched_target_count exceeds target_count")
        if self.tiny_target_count > self.target_count:
            raise ZeroMarginSelectionError("tiny_target_count exceeds target_count")
        if self.matched_tiny_target_count > self.tiny_target_count:
            raise ZeroMarginSelectionError(
                "matched_tiny_target_count exceeds tiny_target_count"
            )
        if self.matched_tiny_target_count > self.matched_target_count:
            raise ZeroMarginSelectionError(
                "matched_tiny_target_count exceeds matched_target_count"
            )
        if self.unmatched_component_pixels > self.valid_pixels:
            raise ZeroMarginSelectionError(
                "unmatched_component_pixels exceeds valid_pixels"
            )
        niou = _require_finite_float(self.niou, name="niou")
        if not 0.0 <= niou <= 1.0:
            raise ZeroMarginSelectionError("niou must lie in [0, 1]")
        loss = _require_finite_float(self.loss, name="loss")
        if loss < 0.0:
            raise ZeroMarginSelectionError("loss must be non-negative")

    @classmethod
    def from_mapping(
        cls,
        *,
        name: str,
        family: CandidateFamily,
        binding: EvaluationBinding,
        value: Mapping[str, object],
    ) -> "MetricRecord":
        """Build from evaluator fields without accepting rounded Pd/Fa/mIoU."""

        if not isinstance(value, Mapping):
            raise TypeError("metric payload must be a mapping")
        component_pixels = value.get("unmatched_component_pixels")
        if component_pixels is None:
            component_pixels = value.get("unmatched_predicted_pixels")
        if component_pixels is None:
            component_pixels = value.get("unmatched_predicted_pixel_count")
        if component_pixels is None:
            component_pixels = value.get("component_false_positive_pixels")
        if component_pixels is None:
            raise ZeroMarginSelectionError(
                "metric payload lacks exact unmatched component pixels"
            )
        loss = value.get("test_loss", value.get("val_loss", value.get("loss")))
        if loss is None:
            raise ZeroMarginSelectionError("metric payload lacks loss")
        return cls(
            name=name,
            family=family,
            binding=binding,
            intersection_pixels=value["intersection_pixels"],  # type: ignore[arg-type]
            union_pixels=value["union_pixels"],  # type: ignore[arg-type]
            matched_target_count=value["matched_target_count"],  # type: ignore[arg-type]
            target_count=value["target_count"],  # type: ignore[arg-type]
            unmatched_component_pixels=component_pixels,  # type: ignore[arg-type]
            valid_pixels=value["valid_pixel_count"],  # type: ignore[arg-type]
            niou=value["niou"],  # type: ignore[arg-type]
            matched_tiny_target_count=value["matched_tiny_target_count"],  # type: ignore[arg-type]
            tiny_target_count=value["tiny_target_count"],  # type: ignore[arg-type]
            loss=loss,  # type: ignore[arg-type]
        )

    @property
    def miou(self) -> Fraction:
        return Fraction(self.intersection_pixels, self.union_pixels)

    @property
    def pd(self) -> Fraction:
        return Fraction(self.matched_target_count, self.target_count)

    @property
    def tiny_pd(self) -> Fraction:
        if self.tiny_target_count == 0:
            return Fraction(0, 1)
        return Fraction(self.matched_tiny_target_count, self.tiny_target_count)

    @property
    def fa(self) -> Fraction:
        return Fraction(self.unmatched_component_pixels, self.valid_pixels)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "binding": self.binding.as_dict(),
            "intersection_pixels": self.intersection_pixels,
            "union_pixels": self.union_pixels,
            "miou": float(self.miou),
            "matched_target_count": self.matched_target_count,
            "target_count": self.target_count,
            "pd": float(self.pd),
            "unmatched_component_pixels": self.unmatched_component_pixels,
            "valid_pixels": self.valid_pixels,
            "fa": float(self.fa),
            "niou": float(self.niou),
            "matched_tiny_target_count": self.matched_tiny_target_count,
            "tiny_target_count": self.tiny_target_count,
            "tiny_pd": float(self.tiny_pd),
            "loss": float(self.loss),
        }


def role_key(role: Role, record: MetricRecord) -> tuple[object, ...]:
    """Return the exact frozen role key; no epoch or positive margin appears."""

    ready_role = _require_role(role)
    if record.binding.role != ready_role:
        raise ZeroMarginSelectionError("record role differs from selector role")
    if ready_role == "best_miou":
        return (
            record.miou,
            record.pd,
            -record.fa,
            float(record.niou),
            record.tiny_pd,
            -float(record.loss),
        )
    return (
        record.pd,
        -record.fa,
        record.tiny_pd,
        record.miou,
        float(record.niou),
        -float(record.loss),
    )


def _ordered_and_validated(
    role: Role,
    candidates: Sequence[MetricRecord],
) -> tuple[MetricRecord, ...]:
    if not candidates:
        raise ZeroMarginSelectionError("candidate pool is empty")
    names = [record.name for record in candidates]
    if len(set(names)) != len(names):
        raise ZeroMarginSelectionError("candidate names must be unique")
    families = [record.family for record in candidates]
    if len(set(families)) != len(families):
        raise ZeroMarginSelectionError("candidate families must be unique")
    bindings = {record.binding for record in candidates}
    if len(bindings) != 1:
        raise ZeroMarginSelectionError(
            "all records must bind the identical evaluation context"
        )
    binding = next(iter(bindings))
    if binding.role != role:
        raise ZeroMarginSelectionError("evaluation binding role differs from selector role")
    exact_counts = {
        (record.target_count, record.tiny_target_count, record.valid_pixels)
        for record in candidates
    }
    if len(exact_counts) != 1:
        raise ZeroMarginSelectionError(
            "all records must bind identical target/tiny/valid-pixel counts"
        )
    return tuple(sorted(candidates, key=lambda item: _TIE_RANK[item.family]))


def select_best(role: Role, candidates: Sequence[MetricRecord]) -> MetricRecord:
    """Return the strict role-key maximum; a tie follows frozen family order."""

    ready_role = _require_role(role)
    ordered = _ordered_and_validated(ready_role, candidates)
    winner = ordered[0]
    winner_key = role_key(ready_role, winner)
    for record in ordered[1:]:
        key = role_key(ready_role, record)
        if key > winner_key:
            winner = record
            winner_key = key
    return winner


def _envelope_pool(
    *,
    original: MetricRecord,
    current: MetricRecord,
    candidates: Sequence[MetricRecord],
) -> tuple[MetricRecord, ...]:
    if original.family != "Original" or current.family != "Current":
        raise ZeroMarginSelectionError(
            "baseline envelope requires Original and Current family bindings"
        )
    return (original, current, *tuple(candidates))


def select_against_baseline_envelope(
    role: Role,
    *,
    original: MetricRecord,
    current: MetricRecord,
    candidates: Sequence[MetricRecord],
) -> MetricRecord:
    """Select strictly from the baseline envelope and all frozen candidates."""

    return select_best(
        role,
        _envelope_pool(original=original, current=current, candidates=candidates),
    )


def selection_report(
    role: Role,
    *,
    original: MetricRecord,
    current: MetricRecord,
    candidates: Sequence[MetricRecord],
    operational_test_selected: bool = False,
) -> dict[str, object]:
    """Return an audit trace without inventing a performance gate."""

    ready_role = _require_role(role)
    pool = _ordered_and_validated(
        ready_role,
        _envelope_pool(original=original, current=current, candidates=candidates),
    )
    winner = select_best(ready_role, pool)
    return {
        "schema": "sctransnet_pbdr_v4_zero_margin_selector/v2",
        "role": ready_role,
        "comparison": "strict_lexicographic_no_positive_margin",
        "performance_acceptance_margin": None,
        "exact_tie_order": list(FROZEN_TIE_ORDER),
        "pool_order": [record.family for record in pool],
        "winner": winner.name,
        "winner_family": winner.family,
        "operational_test_selected": bool(operational_test_selected),
        "selection_is_optimistic": bool(operational_test_selected),
        "binding": pool[0].binding.as_dict(),
        "records": [record.as_dict() for record in pool],
    }


__all__ = [
    "CandidateFamily",
    "EvaluationBinding",
    "FROZEN_TIE_ORDER",
    "MetricRecord",
    "Role",
    "SUPPORTED_ROLES",
    "ZeroMarginSelectionError",
    "role_key",
    "select_against_baseline_envelope",
    "select_best",
    "selection_report",
]
