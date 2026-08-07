"""Strict role-objective comparison with no positive performance margin.

This gate is used only on the frozen internal-validation split of the
cross-dataset PBDR-V3 extension.  It compares complete role-ordered metric
keys.  The fixed probability threshold (0.5) is part of metric measurement;
there is no minimum effect-size threshold or epsilon in this decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA = "sctransnet_pbdr_v3_zero_margin_role_gate_v1/v1"
ROLES = ("best_miou", "best_pd")
CHECK_NAMES = (
    "target_count_equal",
    "tiny_target_count_equal",
    "strict_role_performance_gain",
)
ROLE_ORDERS = {
    "best_miou": (
        "higher_miou",
        "higher_pd",
        "lower_fa",
        "higher_niou",
        "higher_tiny_pd",
        "lower_test_loss",
    ),
    "best_pd": (
        "higher_pd",
        "lower_fa",
        "higher_tiny_pd",
        "higher_miou",
        "higher_niou",
        "lower_test_loss",
    ),
}


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    raise TypeError(f"{name} must be an integer")


def _metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        ready = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not math.isfinite(ready):
        raise ValueError(f"{name} must be finite")
    return ready


@dataclass(frozen=True, slots=True)
class CertificationMetrics:
    matched_target_count: int
    target_count: int
    fa: float
    miou: float
    niou: float
    matched_tiny_target_count: int
    tiny_target_count: int
    tiny_pd: float
    test_loss: float

    def __post_init__(self) -> None:
        integer_fields = (
            "matched_target_count",
            "target_count",
            "matched_tiny_target_count",
            "tiny_target_count",
        )
        for name in integer_fields:
            object.__setattr__(self, name, _count(getattr(self, name), name))
        for name in ("fa", "miou", "niou", "tiny_pd", "test_loss"):
            object.__setattr__(self, name, _metric(getattr(self, name), name))
        if not 0 <= self.matched_target_count <= self.target_count:
            raise ValueError("matched_target_count is outside target_count")
        if not 0 <= self.matched_tiny_target_count <= self.tiny_target_count:
            raise ValueError("matched_tiny_target_count is outside tiny_target_count")
        if self.tiny_target_count > self.target_count:
            raise ValueError("tiny_target_count exceeds target_count")
        if self.matched_tiny_target_count > self.matched_target_count:
            raise ValueError(
                "matched_tiny_target_count exceeds matched_target_count"
            )
        for name in ("fa", "miou", "niou", "tiny_pd"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.test_loss < 0.0:
            raise ValueError("test_loss must be non-negative")
        expected_tiny_pd = (
            self.matched_tiny_target_count / self.tiny_target_count
            if self.tiny_target_count
            else 0.0
        )
        if self.tiny_pd != expected_tiny_pd:
            raise ValueError(
                "tiny_pd differs from matched_tiny_target_count / tiny_target_count"
            )

    @property
    def pd(self) -> float:
        if self.target_count == 0:
            return 0.0
        return self.matched_target_count / self.target_count

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CertificationMetrics":
        if not isinstance(value, Mapping):
            raise TypeError("certification metrics must be a mapping")
        names = (
            "matched_target_count",
            "target_count",
            "fa",
            "miou",
            "niou",
            "matched_tiny_target_count",
            "tiny_target_count",
            "tiny_pd",
            "test_loss",
        )
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError("certification metrics missing: " + ", ".join(missing))
        ready = cls(**{name: value[name] for name in names})
        if "pd" in value and _metric(value["pd"], "pd") != ready.pd:
            raise ValueError("pd differs from matched_target_count / target_count")
        return ready


def role_key(role: str, metrics: CertificationMetrics) -> tuple[float, ...]:
    if role == "best_miou":
        return (
            metrics.miou,
            metrics.pd,
            -metrics.fa,
            metrics.niou,
            metrics.tiny_pd,
            -metrics.test_loss,
        )
    if role == "best_pd":
        return (
            metrics.pd,
            -metrics.fa,
            metrics.tiny_pd,
            metrics.miou,
            metrics.niou,
            -metrics.test_loss,
        )
    raise ValueError(f"unsupported role: {role!r}")


@dataclass(frozen=True, slots=True)
class CertificationDecision:
    role: str
    passed: bool
    selected: str
    checks: Mapping[str, bool]
    current: CertificationMetrics
    candidate: CertificationMetrics
    decisive_index: int | None = None
    decisive_term: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError("unsupported role")
        if set(self.checks) != set(CHECK_NAMES):
            raise ValueError("decision checks differ")
        if any(not isinstance(value, bool) for value in self.checks.values()):
            raise TypeError("decision checks must be boolean")
        expected = all(self.checks.values())
        if self.passed is not expected:
            raise ValueError("passed differs from checks")
        if self.selected != ("candidate" if expected else "current"):
            raise ValueError("selected differs from checks")


def certify(
    role: str,
    current: CertificationMetrics,
    candidate: CertificationMetrics,
) -> CertificationDecision:
    if not isinstance(current, CertificationMetrics) or not isinstance(
        candidate, CertificationMetrics
    ):
        raise TypeError("current and candidate must be CertificationMetrics")
    if role not in ROLES:
        raise ValueError("unsupported role")
    current_key = role_key(role, current)
    candidate_key = role_key(role, candidate)
    decisive_index = next(
        (
            index
            for index, (candidate_value, current_value) in enumerate(
                zip(candidate_key, current_key, strict=True)
            )
            if candidate_value != current_value
        ),
        None,
    )
    checks = {
        "target_count_equal": candidate.target_count == current.target_count,
        "tiny_target_count_equal": (
            candidate.tiny_target_count == current.tiny_target_count
        ),
        "strict_role_performance_gain": candidate_key > current_key,
    }
    passed = all(checks.values())
    return CertificationDecision(
        role=role,
        passed=passed,
        selected="candidate" if passed else "current",
        checks=checks,
        current=current,
        candidate=candidate,
        decisive_index=decisive_index,
        decisive_term=(
            ROLE_ORDERS[role][decisive_index]
            if decisive_index is not None
            else None
        ),
    )


class RoleGateAdapter:
    """Expose the legacy trainer's no-role gate interface for one run role."""

    CertificationMetrics = CertificationMetrics

    def __init__(self, role: str) -> None:
        if role not in ROLES:
            raise ValueError("unsupported role")
        self.role = role

    def CertificationDecision(self, **kwargs: Any) -> CertificationDecision:  # noqa: N802
        current = kwargs.get("current")
        candidate = kwargs.get("candidate")
        expected = certify(self.role, current, candidate)
        for name in ("passed", "selected", "checks"):
            if kwargs.get(name) != getattr(expected, name):
                raise ValueError(f"reconstructed decision {name} differs")
        return expected

    def certify(
        self,
        current: CertificationMetrics,
        candidate: CertificationMetrics,
    ) -> CertificationDecision:
        return certify(self.role, current, candidate)

    def write_decision(self, path: Path, decision: CertificationDecision) -> None:
        write_decision(path, decision)


def _json_payload(decision: CertificationDecision) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scope": "frozen_certification_split_only",
        "policy": "strict_role_performance_key_no_positive_margin",
        "minimum_gain": 0.0,
        "role": decision.role,
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "decisive_index": decision.decisive_index,
        "decisive_term": decision.decisive_term,
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "unseen_test_guarantee": False,
    }


def write_decision(path: Path, decision: CertificationDecision) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_payload(decision), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CertificationDecision",
    "CertificationMetrics",
    "ROLE_ORDERS",
    "ROLES",
    "RoleGateAdapter",
    "certify",
    "role_key",
    "write_decision",
]
