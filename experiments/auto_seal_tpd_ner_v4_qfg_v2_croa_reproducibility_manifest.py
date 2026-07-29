#!/usr/bin/env python3
"""Route a terminal fallback receipt to the reproducibility manifest sealer.

This wrapper is CPU-only.  ``--dry-run`` performs the generator preflight
without publishing anything.  Only the explicit ``--worker`` mode may ask
the generator to seal its atomic JSON/Markdown bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (
    generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest as generator,
)


SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "reproducibility_auto_sealer_action_v1"
)
EXIT_PERMANENT = 64
EXIT_PENDING = 75
DEFAULT_RECEIPT = generator.default_layout().controller_receipt

RECEIPT_KEYS = {
    "schema",
    "status",
    "phase",
    "official_test_accessed",
    "authoritative_action",
    "selected_method_id",
    "selected_variant",
    "selected_candidate_status",
    "decision",
    "query_fg_stage_success",
    "final_model_engineering_selected",
    "final_model_established",
    "meaningful_overall_improvement_by_frozen_policy",
    "meaningful_improvement_basis",
    "paired_required",
    "current_terminal",
    "launcher",
    "paired",
    "terminal_for_fallback_controller",
    "terminal_for_reproducibility_manifest",
    "receipt_write_policy",
}


class ReceiptPending(RuntimeError):
    """The receipt has not reached a usable terminal phase."""


class ReceiptConflict(RuntimeError):
    """The receipt conflicts with its fixed v1 state-machine contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptConflict(message)


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = Path(path).expanduser()
    if receipt_path.is_symlink():
        raise ReceiptConflict(f"receipt must not be a symlink: {receipt_path}")
    if not receipt_path.exists():
        raise ReceiptPending(f"fallback receipt is missing: {receipt_path}")
    if not receipt_path.is_file():
        raise ReceiptConflict(
            f"receipt is not a regular file: {receipt_path}"
        )
    raw = receipt_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptConflict(f"receipt is invalid JSON: {error}") from error
    _require(isinstance(payload, dict), "receipt must contain one object")
    try:
        expected = _canonical_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ReceiptConflict(
            f"receipt contains a noncanonical value: {error}"
        ) from error
    _require(
        raw == expected,
        "receipt is not canonical pretty JSON",
    )
    _require(set(payload) == RECEIPT_KEYS, "receipt field set differs")
    _require(
        payload.get("schema") == generator.CONTROLLER_SCHEMA,
        "receipt schema differs",
    )
    _require(
        payload.get("official_test_accessed") is False,
        "receipt official-test field must be false",
    )
    _require(
        payload.get("receipt_write_policy") == "atomic_state_transition",
        "receipt write policy differs",
    )
    _require(
        isinstance(payload.get("current_terminal"), Mapping),
        "receipt current-terminal binding is missing",
    )
    _require(
        isinstance(payload.get("launcher"), Mapping),
        "receipt launcher binding is missing",
    )
    _require(
        isinstance(payload.get("paired"), Mapping),
        "receipt paired binding is missing",
    )
    return payload, {
        "path": str(receipt_path.resolve()),
        "sha256": _sha256(receipt_path),
        "schema": generator.CONTROLLER_SCHEMA,
    }


def route_receipt(path: Path) -> tuple[str, dict[str, Any]]:
    receipt, binding = _load_receipt(path)
    status = receipt.get("status")
    phase = receipt.get("phase")
    if status == "in_progress" and phase == "paired_launching":
        _require(
            receipt.get("terminal_for_fallback_controller") is False
            and receipt.get("terminal_for_reproducibility_manifest") is False,
            "in-progress receipt terminal flags differ",
        )
        raise ReceiptPending("paired fallback training is still launching")
    if (
        status == "retryable"
        and phase == "paired_launch_retryable_failure"
    ):
        _require(
            receipt.get("terminal_for_fallback_controller") is False
            and receipt.get("terminal_for_reproducibility_manifest") is False,
            "retryable receipt terminal flags differ",
        )
        raise ReceiptPending("paired fallback is waiting for a retry")
    if status != "complete":
        raise ReceiptConflict(
            f"receipt has an unusable state: status={status!r}, "
            f"phase={phase!r}"
        )
    _require(
        receipt.get("terminal_for_fallback_controller") is True,
        "complete receipt is not controller-terminal",
    )
    paired = receipt["paired"]
    if phase == "no_fallback":
        _require(
            receipt.get("authoritative_action") == "no_fallback",
            "no-fallback action differs",
        )
        _require(
            receipt.get("paired_required") is False
            and paired.get("required") is False
            and paired.get("training_complete") is False,
            "no-fallback paired state differs",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest") is True,
            "no-fallback receipt is not reproducibility-terminal",
        )
        family = "current"
    elif phase == "paired_training_complete":
        _require(
            receipt.get("authoritative_action") == "launch_paired",
            "paired terminal action differs",
        )
        _require(
            receipt.get("paired_required") is True
            and paired.get("required") is True
            and paired.get("training_complete") is True,
            "paired training completion state differs",
        )
        _require(
            paired.get("posttraining_selection_complete") is False
            and paired.get("posttraining_deployment_complete") is False,
            "controller receipt must defer paired post-training closure",
        )
        _require(
            receipt.get("terminal_for_reproducibility_manifest") is False,
            "paired-training receipt prematurely seals reproducibility",
        )
        family = "paired"
    else:
        raise ReceiptConflict(f"complete receipt phase is invalid: {phase!r}")
    return family, {
        **binding,
        "status": status,
        "phase": phase,
        "terminal_family": family,
    }


LayoutFactory = Callable[..., generator.EvidenceLayout]
Executor = Callable[..., dict[str, Any]]


def auto_seal(
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
    output_dir: Path | None = None,
    worker: bool,
    layout_factory: LayoutFactory = generator.default_layout,
    executor: Executor = generator.execute,
) -> dict[str, Any]:
    family, receipt_binding = route_receipt(receipt_path)
    layout = layout_factory(output_dir=output_dir)
    layout = layout._replace(
        controller_receipt=Path(receipt_path).expanduser().resolve()
    )
    generator_result = executor(
        layout,
        terminal_family=family,
        preflight=not worker,
        verify=False,
    )
    return {
        "schema": SCHEMA,
        "status": generator_result.get("status"),
        "mode": "worker" if worker else "dry_run",
        "terminal_family": family,
        "receipt": receipt_binding,
        "generator": generator_result,
        "seal_requested": worker,
        "writes_performed": bool(
            worker and generator_result.get("writes_performed")
        ),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        result = auto_seal(
            receipt_path=args.receipt,
            output_dir=args.output_dir,
            worker=args.worker,
        )
    except (ReceiptPending, generator.IncompleteEvidence) as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "pending",
                    "writes_performed": False,
                    "reason": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_PENDING
    except (ReceiptConflict, generator.EvidenceConflict) as error:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "status": "conflict",
                    "writes_performed": False,
                    "reason": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return EXIT_PERMANENT
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "DEFAULT_RECEIPT",
    "EXIT_PENDING",
    "EXIT_PERMANENT",
    "ReceiptConflict",
    "ReceiptPending",
    "SCHEMA",
    "auto_seal",
    "main",
    "route_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
