"""Select PBDR-V3 only when it dominates Current on a frozen split.

This aggregate gate is deliberately conservative and scoped to the supplied
certification split.  It does not claim non-regression on unseen data.
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


SCHEMA = "sctransnet_non_regression_gate/v1"
CHECK_NAMES = (
    "pd_non_regression",
    "fa_non_regression",
    "miou_strict_gain",
    "niou_non_regression",
)


def _coerce_count(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
    raise TypeError(f"{name} must be an integer")


def _coerce_metric(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CertificationMetrics:
    matched_target_count: int
    target_count: int
    fa: float
    miou: float
    niou: float

    def __post_init__(self) -> None:
        matched = _coerce_count(
            self.matched_target_count,
            "matched_target_count",
        )
        total = _coerce_count(self.target_count, "target_count")
        fa = _coerce_metric(self.fa, "fa")
        miou = _coerce_metric(self.miou, "miou")
        niou = _coerce_metric(self.niou, "niou")

        if total < 0:
            raise ValueError("target_count must be non-negative")
        if not 0 <= matched <= total:
            raise ValueError(
                "matched_target_count must be in [0, target_count]"
            )
        if not 0.0 <= fa <= 1.0:
            raise ValueError("fa must be in [0, 1]")
        if not 0.0 <= miou <= 1.0:
            raise ValueError("miou must be in [0, 1]")
        if not 0.0 <= niou <= 1.0:
            raise ValueError("niou must be in [0, 1]")

        object.__setattr__(self, "matched_target_count", matched)
        object.__setattr__(self, "target_count", total)
        object.__setattr__(self, "fa", fa)
        object.__setattr__(self, "miou", miou)
        object.__setattr__(self, "niou", niou)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CertificationMetrics":
        if not isinstance(value, Mapping):
            raise TypeError("certification metrics must be a mapping")
        required = (
            "matched_target_count",
            "target_count",
            "fa",
            "miou",
            "niou",
        )
        missing = [name for name in required if name not in value]
        if missing:
            raise ValueError(
                "certification metrics missing keys: " + ", ".join(missing)
            )
        return cls(
            matched_target_count=value["matched_target_count"],
            target_count=value["target_count"],
            fa=value["fa"],
            miou=value["miou"],
            niou=value["niou"],
        )


@dataclass(frozen=True, slots=True)
class CertificationDecision:
    passed: bool
    selected: str
    checks: Mapping[str, bool]
    current: CertificationMetrics
    candidate: CertificationMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be boolean")
        if self.selected not in ("current", "candidate"):
            raise ValueError("selected must be 'current' or 'candidate'")
        if not isinstance(self.checks, Mapping):
            raise TypeError("checks must be a mapping")
        missing = [name for name in CHECK_NAMES if name not in self.checks]
        extra = [name for name in self.checks if name not in CHECK_NAMES]
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if extra:
                detail.append("extra=" + ",".join(extra))
            raise ValueError("invalid certification checks: " + "; ".join(detail))
        checks = {}
        for name in CHECK_NAMES:
            result = self.checks[name]
            if not isinstance(result, bool):
                raise TypeError(f"checks[{name!r}] must be boolean")
            checks[name] = result
        expected_passed = all(checks.values())
        if self.passed != expected_passed:
            raise ValueError("passed must equal the conjunction of checks")
        expected_selected = "candidate" if expected_passed else "current"
        if self.selected != expected_selected:
            raise ValueError("selected conflicts with the gate result")
        if not isinstance(self.current, CertificationMetrics):
            raise TypeError("current must be CertificationMetrics")
        if not isinstance(self.candidate, CertificationMetrics):
            raise TypeError("candidate must be CertificationMetrics")
        # Copy the mapping so later mutations of the caller's dictionary do
        # not change an already-issued decision.  Keep a plain dictionary so
        # standard dataclass serialization continues to work.
        object.__setattr__(self, "checks", checks)


def certify(
    current: CertificationMetrics,
    candidate: CertificationMetrics,
    *,
    minimum_miou_gain: float = 0.002,
    maximum_fa_ratio: float = 1.0,
    require_niou_non_decrease: bool = True,
) -> CertificationDecision:
    """Apply the frozen-split aggregate non-regression gate."""

    if not isinstance(current, CertificationMetrics):
        raise TypeError("current must be CertificationMetrics")
    if not isinstance(candidate, CertificationMetrics):
        raise TypeError("candidate must be CertificationMetrics")
    miou_gain = _coerce_metric(minimum_miou_gain, "minimum_miou_gain")
    fa_ratio = _coerce_metric(maximum_fa_ratio, "maximum_fa_ratio")
    if miou_gain < 0.0:
        raise ValueError("minimum_miou_gain must be non-negative")
    if not 0.0 <= fa_ratio <= 1.0:
        raise ValueError("maximum_fa_ratio must be in [0, 1]")
    if not isinstance(require_niou_non_decrease, bool):
        raise TypeError("require_niou_non_decrease must be boolean")
    if current.target_count != candidate.target_count:
        raise ValueError("Current and candidate target counts differ")

    checks = {
        "pd_non_regression": (
            candidate.matched_target_count >= current.matched_target_count
        ),
        "fa_non_regression": candidate.fa <= current.fa * fa_ratio,
        "miou_strict_gain": candidate.miou >= current.miou + miou_gain,
        "niou_non_regression": (
            not require_niou_non_decrease or candidate.niou >= current.niou
        ),
    }
    passed = all(checks.values())
    return CertificationDecision(
        passed=passed,
        selected="candidate" if passed else "current",
        checks=checks,
        current=current,
        candidate=candidate,
    )


def _decision_payload(decision: CertificationDecision) -> dict[str, Any]:
    if not isinstance(decision, CertificationDecision):
        raise TypeError("decision must be CertificationDecision")
    return {
        "schema": SCHEMA,
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "scope": "frozen_certification_split_only",
        "unseen_test_guarantee": False,
    }


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_decision(path: Path, decision: CertificationDecision) -> None:
    """Durably publish a decision by same-directory atomic replacement."""

    payload = _decision_payload(decision)
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(Path(path), content)


__all__ = [
    "CHECK_NAMES",
    "SCHEMA",
    "CertificationDecision",
    "CertificationMetrics",
    "certify",
    "write_decision",
]
