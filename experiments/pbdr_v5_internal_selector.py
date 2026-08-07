"""Pure zero-margin selector for the frozen PBDR-V5 internal candidate pool.

The selector consumes already-computed metric mappings only.  It never imports
or opens a dataset, index, loader, checkpoint, or official-test artifact.
Integer sufficient statistics are used for mIoU, Pd, Fa, and tiny-Pd so that a
strict improvement of any size is retained without an epsilon or margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Literal, Mapping


Role = Literal["best_miou", "best_pd"]
CandidateFamily = Literal[
    "Original",
    "Current",
    "V3-calibrated",
    "V4-Stage1",
    "V4-Stage2",
    "V5",
]

SUPPORTED_ROLES: tuple[Role, ...] = ("best_miou", "best_pd")
FROZEN_FAMILY_ORDER: tuple[CandidateFamily, ...] = (
    "Original",
    "Current",
    "V3-calibrated",
    "V4-Stage1",
    "V4-Stage2",
    "V5",
)
EXISTING_ENVELOPE_FAMILIES: tuple[CandidateFamily, ...] = FROZEN_FAMILY_ORDER[:-1]

_EXACT_STAT_FIELDS = (
    "intersection_pixels",
    "union_pixels",
    "matched_target_count",
    "target_count",
    "unmatched_component_pixels",
    "valid_pixel_count",
    "matched_tiny_target_count",
    "tiny_target_count",
)


class InternalSelectionError(ValueError):
    """The frozen family pool or one of its metric mappings is invalid."""


def _require_role(role: str) -> Role:
    if role not in SUPPORTED_ROLES:
        raise InternalSelectionError(f"unsupported role: {role!r}")
    return role  # type: ignore[return-value]


def _require_plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise InternalSelectionError(f"{name} must be non-negative")
    return value


def _require_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(ready):
        raise InternalSelectionError(f"{name} must be finite")
    return ready


def _first_present(
    metrics: Mapping[str, object],
    names: tuple[str, ...],
    *,
    description: str,
) -> object:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    raise InternalSelectionError(f"metric payload lacks {description}")


@dataclass(frozen=True, slots=True)
class _MetricRecord:
    family: CandidateFamily
    intersection_pixels: int
    union_pixels: int
    matched_target_count: int
    target_count: int
    unmatched_component_pixels: int
    valid_pixel_count: int
    niou: float
    matched_tiny_target_count: int
    tiny_target_count: int
    loss: float

    @classmethod
    def from_mapping(
        cls,
        family: CandidateFamily,
        metrics: Mapping[str, object],
    ) -> "_MetricRecord":
        if not isinstance(metrics, Mapping):
            raise TypeError(f"metrics for {family} must be a mapping")

        component_pixels = _first_present(
            metrics,
            (
                "unmatched_component_pixels",
                "unmatched_predicted_pixels",
                "unmatched_predicted_pixel_count",
                "component_false_positive_pixels",
            ),
            description="exact unmatched component pixels",
        )
        valid_pixels = _first_present(
            metrics,
            ("valid_pixel_count", "valid_pixels"),
            description="exact valid-pixel count",
        )
        loss = _first_present(
            metrics,
            ("test_loss", "val_loss", "loss"),
            description="loss",
        )

        record = cls(
            family=family,
            intersection_pixels=_require_plain_int(
                metrics.get("intersection_pixels"), name="intersection_pixels"
            ),
            union_pixels=_require_plain_int(
                metrics.get("union_pixels"), name="union_pixels"
            ),
            matched_target_count=_require_plain_int(
                metrics.get("matched_target_count"), name="matched_target_count"
            ),
            target_count=_require_plain_int(
                metrics.get("target_count"), name="target_count"
            ),
            unmatched_component_pixels=_require_plain_int(
                component_pixels, name="unmatched_component_pixels"
            ),
            valid_pixel_count=_require_plain_int(
                valid_pixels, name="valid_pixel_count"
            ),
            niou=_require_finite_float(metrics.get("niou"), name="niou"),
            matched_tiny_target_count=_require_plain_int(
                metrics.get("matched_tiny_target_count"),
                name="matched_tiny_target_count",
            ),
            tiny_target_count=_require_plain_int(
                metrics.get("tiny_target_count"), name="tiny_target_count"
            ),
            loss=_require_finite_float(loss, name="loss"),
        )
        record._validate_relations()
        return record

    def _validate_relations(self) -> None:
        if self.union_pixels <= 0:
            raise InternalSelectionError("union_pixels must be positive")
        if self.target_count <= 0:
            raise InternalSelectionError("target_count must be positive")
        if self.valid_pixel_count <= 0:
            raise InternalSelectionError("valid_pixel_count must be positive")
        if self.intersection_pixels > self.union_pixels:
            raise InternalSelectionError("intersection_pixels exceeds union_pixels")
        if self.union_pixels > self.valid_pixel_count:
            raise InternalSelectionError("union_pixels exceeds valid_pixel_count")
        if self.matched_target_count > self.target_count:
            raise InternalSelectionError("matched_target_count exceeds target_count")
        if self.tiny_target_count > self.target_count:
            raise InternalSelectionError("tiny_target_count exceeds target_count")
        if self.matched_tiny_target_count > self.tiny_target_count:
            raise InternalSelectionError(
                "matched_tiny_target_count exceeds tiny_target_count"
            )
        if self.matched_tiny_target_count > self.matched_target_count:
            raise InternalSelectionError(
                "matched_tiny_target_count exceeds matched_target_count"
            )
        if self.unmatched_component_pixels > self.valid_pixel_count:
            raise InternalSelectionError(
                "unmatched_component_pixels exceeds valid_pixel_count"
            )
        if not 0.0 <= self.niou <= 1.0:
            raise InternalSelectionError("niou must lie in [0, 1]")
        if self.loss < 0.0:
            raise InternalSelectionError("loss must be non-negative")

    @property
    def miou(self) -> Fraction:
        return Fraction(self.intersection_pixels, self.union_pixels)

    @property
    def pd(self) -> Fraction:
        return Fraction(self.matched_target_count, self.target_count)

    @property
    def fa(self) -> Fraction:
        return Fraction(self.unmatched_component_pixels, self.valid_pixel_count)

    @property
    def tiny_pd(self) -> Fraction:
        if self.tiny_target_count == 0:
            return Fraction(0, 1)
        return Fraction(self.matched_tiny_target_count, self.tiny_target_count)

    def exact_statistics(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in _EXACT_STAT_FIELDS}

    def metrics_dict(self) -> dict[str, object]:
        return {
            **self.exact_statistics(),
            "miou": float(self.miou),
            "pd": float(self.pd),
            "fa": float(self.fa),
            "niou": self.niou,
            "tiny_pd": float(self.tiny_pd),
            "loss": self.loss,
        }


def _record_role_key(role: Role, record: _MetricRecord) -> tuple[object, ...]:
    """Return the complete V4-equivalent role key with exact count ratios."""

    if role == "best_miou":
        return (
            record.miou,
            record.pd,
            -record.fa,
            record.niou,
            record.tiny_pd,
            -record.loss,
        )
    return (
        record.pd,
        -record.fa,
        record.tiny_pd,
        record.miou,
        record.niou,
        -record.loss,
    )


def role_key(role: Role, metrics: Mapping[str, object]) -> tuple[object, ...]:
    """Build the exact V4-equivalent role key for one metric mapping."""

    ready_role = _require_role(role)
    return _record_role_key(
        ready_role,
        _MetricRecord.from_mapping("V5", metrics),
    )


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _serialized_role_key(role: Role, record: _MetricRecord) -> list[dict[str, object]]:
    if role == "best_miou":
        fields = ("miou", "pd", "negative_fa", "niou", "tiny_pd", "negative_loss")
    else:
        fields = ("pd", "negative_fa", "tiny_pd", "miou", "niou", "negative_loss")
    serialized: list[dict[str, object]] = []
    for field, value in zip(fields, _record_role_key(role, record), strict=True):
        if isinstance(value, Fraction):
            serialized.append(
                {"field": field, "representation": "exact_fraction", **_fraction_dict(value)}
            )
        else:
            serialized.append(
                {
                    "field": field,
                    "representation": "binary64_hex",
                    "hex": float(value).hex(),
                }
            )
    return serialized


def _select_first_strict_maximum(
    role: Role,
    records: tuple[_MetricRecord, ...],
) -> _MetricRecord:
    winner = records[0]
    winner_key = _record_role_key(role, winner)
    for record in records[1:]:
        candidate_key = _record_role_key(role, record)
        if candidate_key > winner_key:
            winner = record
            winner_key = candidate_key
    return winner


def _validated_records(
    metrics_by_family: Mapping[str, Mapping[str, object]],
) -> tuple[_MetricRecord, ...]:
    if not isinstance(metrics_by_family, Mapping):
        raise TypeError("metrics_by_family must be a mapping")
    expected = set(FROZEN_FAMILY_ORDER)
    observed = set(metrics_by_family)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise InternalSelectionError(
            "candidate families must exactly match the frozen pool; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )

    records = tuple(
        _MetricRecord.from_mapping(family, metrics_by_family[family])
        for family in FROZEN_FAMILY_ORDER
    )
    denominator_bindings = {
        (record.target_count, record.tiny_target_count, record.valid_pixel_count)
        for record in records
    }
    if len(denominator_bindings) != 1:
        raise InternalSelectionError(
            "all candidates must share target, tiny-target, and valid-pixel counts"
        )
    return records


def select_internal_candidate(
    role: Role,
    metrics_by_family: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Select the frozen six-family internal winner and report V5's exact delta.

    Equal keys never replace the earlier family.  The V5 improvement flag is
    true only when V5's complete role key is strictly greater than the winner
    of the five pre-existing families.
    """

    ready_role = _require_role(role)
    records = _validated_records(metrics_by_family)
    existing_records = records[:-1]
    v5 = records[-1]
    existing_winner = _select_first_strict_maximum(ready_role, existing_records)
    winner = _select_first_strict_maximum(ready_role, records)
    strictly_improves = _record_role_key(ready_role, v5) > _record_role_key(
        ready_role, existing_winner
    )

    exact_stat_delta = {
        field: int(getattr(v5, field)) - int(getattr(existing_winner, field))
        for field in _EXACT_STAT_FIELDS
    }
    exact_metric_delta = {
        "miou": _fraction_dict(v5.miou - existing_winner.miou),
        "pd": _fraction_dict(v5.pd - existing_winner.pd),
        "fa": _fraction_dict(v5.fa - existing_winner.fa),
        "tiny_pd": _fraction_dict(v5.tiny_pd - existing_winner.tiny_pd),
    }

    return {
        "schema": "sctransnet_pbdr_v5_internal_selector/v1",
        "role": ready_role,
        "comparison": "strict_lexicographic_full_role_key_no_positive_margin",
        "performance_acceptance_margin": None,
        "exact_tie_order": list(FROZEN_FAMILY_ORDER),
        "candidate_families": list(FROZEN_FAMILY_ORDER),
        "candidates": [
            {
                "family": record.family,
                "metrics": record.metrics_dict(),
                "role_key": _serialized_role_key(ready_role, record),
            }
            for record in records
        ],
        "existing_envelope_families": list(EXISTING_ENVELOPE_FAMILIES),
        "existing_envelope_winner": existing_winner.family,
        "winner": winner.family,
        "v5_strictly_improves_existing_envelope": strictly_improves,
        "v5_vs_existing_envelope_winner": {
            "direction": "V5_minus_existing_envelope_winner",
            "reference_family": existing_winner.family,
            "exact_sufficient_statistics_delta": exact_stat_delta,
            "exact_role_metric_delta": exact_metric_delta,
            "floating_metric_delta": {
                "niou": v5.niou - existing_winner.niou,
                "loss": v5.loss - existing_winner.loss,
            },
        },
    }


selection_report = select_internal_candidate


__all__ = [
    "CandidateFamily",
    "EXISTING_ENVELOPE_FAMILIES",
    "FROZEN_FAMILY_ORDER",
    "InternalSelectionError",
    "Role",
    "SUPPORTED_ROLES",
    "role_key",
    "select_internal_candidate",
    "selection_report",
]
