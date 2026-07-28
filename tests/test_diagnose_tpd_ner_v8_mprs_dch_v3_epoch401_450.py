from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    diagnose_tpd_ner_v8_mprs_dch_v3_epoch401_450 as subject,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_fingerprint(name: str, values: list[str]) -> dict:
    return {
        "schema": subject.ORDERED_FINGERPRINT_SCHEMA,
        "name": name,
        "count": len(values),
        "sha256": _canonical_sha256(values),
    }


def _split_payload() -> dict:
    train = [f"train-{index:03d}" for index in range(530)]
    validation = [f"validation-{index:03d}" for index in range(133)]
    return {
        "source": "img_idx/train_NUDT-SIRST.txt",
        "dataset": subject.DATASET,
        "split_seed": subject.SPLIT_SEED,
        "val_fraction": 0.2,
        "full_official_train_count": 663,
        "full_internal_train_count": 530,
        "full_internal_val_count": 133,
        "used_train_count": 530,
        "used_val_count": 133,
        "official_test_accessed": False,
        "full_internal_train_ids": train,
        "full_internal_val_ids": validation,
        "used_train_ids": list(train),
        "used_val_ids": list(validation),
    }


def _split_fingerprints(split: dict) -> dict:
    return {
        "full_train": _ordered_fingerprint(
            "full_train",
            split["full_internal_train_ids"],
        ),
        "full_validation": _ordered_fingerprint(
            "full_validation",
            split["full_internal_val_ids"],
        ),
        "train": _ordered_fingerprint("train", split["used_train_ids"]),
        "validation": _ordered_fingerprint(
            "validation",
            split["used_val_ids"],
        ),
    }


def _data_fingerprints() -> dict:
    return {
        name: {
            "schema": subject.ORDERED_FINGERPRINT_SCHEMA,
            "name": name,
            "count": count,
            "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        }
        for name, count in (
            ("normalization", 1),
            ("official_training_data", 1),
            ("train_samples", 530),
            ("validation_samples", 133),
        )
    }


def _protocol_payload(
    run_dir: Path,
    *,
    variant: str,
    run_tag: str,
    version: str,
    schema: str,
    split: dict,
) -> dict:
    split_fingerprints = _split_fingerprints(split)
    data_fingerprints = _data_fingerprints()
    return {
        "schema": schema,
        "arguments": {
            "dataset": subject.DATASET,
            "variant": variant,
            "seed": subject.TRAINING_SEED,
            "split_seed": subject.SPLIT_SEED,
            "run_tag": run_tag,
            "epochs": 800,
            "eval_every": 1,
        },
        "formal_contract": {
            "dataset": subject.DATASET,
            "training_seed": subject.TRAINING_SEED,
            "split_seed": subject.SPLIT_SEED,
            "epochs": 800,
            "eval_every": 1,
            "candidate_variants": [variant],
            "multi_seed_scheduled": False,
        },
        "official_test_accessed": False,
        "run_directory": str(run_dir.resolve()),
        "run_identity": {
            "schema": subject.RUN_IDENTITY_SCHEMA,
            "dataset": subject.DATASET,
            "variant": variant,
            "seed": subject.TRAINING_SEED,
            "split_seed": subject.SPLIT_SEED,
            "run_id": (
                f"tpd-ner-v8-mprs-dch-{version}-exact:"
                f"{subject.DATASET}:{variant}:seed-{subject.TRAINING_SEED}:"
                f"split-{subject.SPLIT_SEED}:{run_tag}"
            ),
            "ordered_split_fingerprints": split_fingerprints,
            "split_sha256": _canonical_sha256(split_fingerprints),
            "ordered_data_fingerprints": data_fingerprints,
            "data_sha256": _canonical_sha256(data_fingerprints),
        },
    }


def _metric_rows(variant: str, *, v3: bool) -> list[dict]:
    rows = []
    valid_pixels = 10_000_000
    for epoch in range(1, subject.WINDOW_END + 1):
        in_first_half = subject.WINDOW_START <= epoch <= 425
        if v3 and epoch >= subject.WINDOW_START:
            false_pixels = 30 if in_first_half else 50
            matched = 182 if in_first_half else 184
            unmatched = 7 if in_first_half else 8
            miou = 0.897
        else:
            false_pixels = 40
            matched = 184
            unmatched = 7
            miou = 0.9
        rows.append(
            {
                "epoch": epoch,
                "variant": variant,
                "fa": false_pixels / valid_pixels,
                "pd": matched / subject.TARGET_COUNT,
                "miou": miou,
                "target_count": subject.TARGET_COUNT,
                "matched_target_count": matched,
                "unmatched_predicted_object_count": unmatched,
                "valid_pixel_count": valid_pixels,
            }
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class _CanonicalFixture:
    def __init__(self, root: Path) -> None:
        self.v3_dir = root / "v3"
        self.v2_dir = root / "v2"
        self.v3_dir.mkdir()
        self.v2_dir.mkdir()
        self.split = _split_payload()
        split_bytes = (
            json.dumps(self.split, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        (self.v3_dir / "split.json").write_bytes(split_bytes)
        (self.v2_dir / "split.json").write_bytes(split_bytes)
        self.v3_protocol = _protocol_payload(
            self.v3_dir,
            variant=subject.V3_VARIANT,
            run_tag=subject.V3_RUN_TAG,
            version="v3",
            schema=subject.V3_ENTRY_SCHEMA,
            split=self.split,
        )
        self.v2_protocol = _protocol_payload(
            self.v2_dir,
            variant=subject.V2_VARIANT,
            run_tag=subject.V2_RUN_TAG,
            version="v2",
            schema=subject.V2_ENTRY_SCHEMA,
            split=self.split,
        )
        _write_json(self.v3_dir / "protocol.json", self.v3_protocol)
        _write_json(self.v2_dir / "protocol.json", self.v2_protocol)
        self.v3_rows = _metric_rows(subject.V3_VARIANT, v3=True)
        self.v2_rows = _metric_rows(subject.V2_VARIANT, v3=False)
        self.write_metrics()

    def patch_paths(self):
        return mock.patch.multiple(
            subject,
            V3_RUN_DIR=self.v3_dir,
            V2_RUN_DIR=self.v2_dir,
        )

    def write_metrics(self) -> None:
        _write_metrics(self.v3_dir / "metrics.jsonl", self.v3_rows)
        _write_metrics(self.v2_dir / "metrics.jsonl", self.v2_rows)

    def rewrite_v2_split_and_binding(self, split: dict) -> None:
        _write_json(self.v2_dir / "split.json", split)
        self.v2_protocol = _protocol_payload(
            self.v2_dir,
            variant=subject.V2_VARIANT,
            run_tag=subject.V2_RUN_TAG,
            version="v2",
            schema=subject.V2_ENTRY_SCHEMA,
            split=split,
        )
        _write_json(self.v2_dir / "protocol.json", self.v2_protocol)


class V3Epoch401To450DiagnosticTests(unittest.TestCase):
    def test_boundary_pass_publishes_exact_five_gates_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _CanonicalFixture(Path(temporary))
            with fixture.patch_paths():
                first = subject.diagnose_and_write()
                json_path = fixture.v3_dir / subject.JSON_OUTPUT_NAME
                markdown_path = fixture.v3_dir / subject.MARKDOWN_OUTPUT_NAME
                first_stats = (json_path.stat(), markdown_path.stat())
                second = subject.diagnose_and_write()
                second_stats = (json_path.stat(), markdown_path.stat())

            self.assertEqual(first["status"], "published")
            self.assertEqual(second["status"], "already_complete")
            self.assertEqual(first["decision"], subject.PASSED_DECISION)
            self.assertEqual(
                [(item.st_ino, item.st_mtime_ns) for item in first_stats],
                [(item.st_ino, item.st_mtime_ns) for item in second_stats],
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(report["criteria"]), 5)
            self.assertTrue(report["all_five_criteria_passed"])
            self.assertEqual(
                report["criteria"]["median_false_pixels_per_epoch"][
                    "observed"
                ],
                40.0,
            )
            self.assertEqual(
                report["criteria"][
                    "mean_unmatched_predicted_objects_per_epoch"
                ]["observed"],
                7.5,
            )
            self.assertEqual(
                report["criteria"][
                    "epochs_with_matched_targets_ge_183_and_fa_le_5e_6"
                ]["observed"],
                25,
            )
            self.assertEqual(
                report["criteria"][
                    "mean_matched_targets_decrease_vs_v2"
                ]["observed_decrease"],
                1.0,
            )
            self.assertAlmostEqual(
                report["criteria"]["mean_miou_decrease_vs_v2"][
                    "observed_decrease"
                ],
                0.003,
            )
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("DIAGNOSTIC ONLY", markdown)
            self.assertIn(subject.PASSED_DECISION, markdown)

    def test_each_registered_gate_can_fail_the_and_decision(self) -> None:
        base_v3 = _metric_rows(subject.V3_VARIANT, v3=True)
        base_v2 = _metric_rows(subject.V2_VARIANT, v3=False)
        binding = {
            "run_id": "fixture",
            "run_dir": "/fixture",
            "split_sha256": "a" * 64,
            "data_sha256": "b" * 64,
        }
        cases = {}

        median = copy.deepcopy(base_v3)
        for index, row in enumerate(
            median[subject.WINDOW_START - 1 : subject.WINDOW_END]
        ):
            row["fa"] = (31 if index < 25 else 51) / 10_000_000
            row["matched_target_count"] = 184 if index < 25 else 182
            row["pd"] = row["matched_target_count"] / subject.TARGET_COUNT
        cases["median_false_pixels_per_epoch"] = median

        unmatched = copy.deepcopy(base_v3)
        for row in unmatched[subject.WINDOW_START - 1 : subject.WINDOW_END]:
            row["unmatched_predicted_object_count"] = 8
        cases["mean_unmatched_predicted_objects_per_epoch"] = unmatched

        joint = copy.deepcopy(base_v3)
        for index, row in enumerate(
            joint[subject.WINDOW_START - 1 : subject.WINDOW_END]
        ):
            row["matched_target_count"] = 182 if index < 26 else 185
            row["pd"] = row["matched_target_count"] / subject.TARGET_COUNT
        cases[
            "epochs_with_matched_targets_ge_183_and_fa_le_5e_6"
        ] = joint

        matched = copy.deepcopy(base_v3)
        for index, row in enumerate(
            matched[subject.WINDOW_START - 1 : subject.WINDOW_END]
        ):
            row["matched_target_count"] = 183 if index < 25 else 182
            row["pd"] = row["matched_target_count"] / subject.TARGET_COUNT
        cases["mean_matched_targets_decrease_vs_v2"] = matched

        miou = copy.deepcopy(base_v3)
        for row in miou[subject.WINDOW_START - 1 : subject.WINDOW_END]:
            row["miou"] = 0.896
        cases["mean_miou_decrease_vs_v2"] = miou

        for expected_failed, v3_rows in cases.items():
            with self.subTest(expected_failed=expected_failed):
                report = subject.build_report(
                    v3_rows,
                    base_v2,
                    v3_binding=binding,
                    v2_binding=binding,
                )
                failed = {
                    name
                    for name, criterion in report["criteria"].items()
                    if criterion["passed"] is False
                }
                self.assertEqual(failed, {expected_failed})
                self.assertEqual(report["decision"], subject.FAILED_DECISION)

    def test_rejects_noncontiguous_epoch_and_wrong_variant(self) -> None:
        for mutation, message in (
            ("gap", "not contiguous"),
            ("variant", "variant differs"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = _CanonicalFixture(Path(temporary))
                    if mutation == "gap":
                        del fixture.v3_rows[199]
                    else:
                        fixture.v3_rows[449]["variant"] = subject.V2_VARIANT
                    fixture.write_metrics()
                    with (
                        fixture.patch_paths(),
                        self.assertRaisesRegex(
                            subject.DiagnosticError,
                            message,
                        ),
                    ):
                        subject.diagnose_and_write()

    def test_rejects_seed_or_cross_run_split_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _CanonicalFixture(Path(temporary))
            fixture.v3_protocol["arguments"]["seed"] = 7
            _write_json(
                fixture.v3_dir / "protocol.json",
                fixture.v3_protocol,
            )
            with (
                fixture.patch_paths(),
                self.assertRaisesRegex(
                    subject.DiagnosticError,
                    "argument differs: seed",
                ),
            ):
                subject.diagnose_and_write()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = _CanonicalFixture(Path(temporary))
            different = copy.deepcopy(fixture.split)
            different["full_internal_train_ids"][0] = "different-train-id"
            different["used_train_ids"][0] = "different-train-id"
            fixture.rewrite_v2_split_and_binding(different)
            with (
                fixture.patch_paths(),
                self.assertRaisesRegex(
                    subject.DiagnosticError,
                    "split.json files differ",
                ),
            ):
                subject.diagnose_and_write()

    def test_conflicting_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _CanonicalFixture(Path(temporary))
            with fixture.patch_paths():
                subject.diagnose_and_write()
                markdown = fixture.v3_dir / subject.MARKDOWN_OUTPUT_NAME
                markdown.write_text("conflict\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    subject.DiagnosticError,
                    "output conflicts",
                ):
                    subject.diagnose_and_write()
            self.assertEqual(markdown.read_text(encoding="utf-8"), "conflict\n")

    def test_watcher_polls_active_service_then_analyzes_once(self) -> None:
        with (
            mock.patch.object(
                subject,
                "current_v3_epoch",
                side_effect=[83, 450],
            ),
            mock.patch.object(
                subject,
                "inspect_training_service",
                return_value={
                    "LoadState": "loaded",
                    "ActiveState": "active",
                    "SubState": "running",
                },
            ),
            mock.patch.object(subject.time, "sleep") as sleep,
            mock.patch.object(
                subject,
                "diagnose_and_write",
                return_value={"status": "published"},
            ) as diagnose,
        ):
            result = subject.watch_and_diagnose(poll_seconds=0.25)
        self.assertEqual(result, {"status": "published"})
        sleep.assert_called_once_with(0.25)
        diagnose.assert_called_once_with()

    def test_stopped_before_450_is_error_and_main_returns_nonzero(self) -> None:
        state = {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "exit-code",
            "ExecMainStatus": "7",
        }
        with (
            mock.patch.object(
                subject,
                "current_v3_epoch",
                side_effect=[83, 83],
            ),
            mock.patch.object(
                subject,
                "inspect_training_service",
                return_value=state,
            ),
            self.assertRaisesRegex(
                subject.DiagnosticError,
                "stopped before epoch 450",
            ),
        ):
            subject.watch_and_diagnose(poll_seconds=30)

        with mock.patch.object(
            subject,
            "watch_and_diagnose",
            side_effect=subject.DiagnosticError(
                "fixed V3 training service stopped before epoch 450"
            ),
        ):
            self.assertEqual(subject.main(["--poll-seconds", "30"]), 1)

    def test_poll_interval_above_thirty_fails_before_reading(self) -> None:
        with (
            mock.patch.object(subject, "current_v3_epoch") as current,
            self.assertRaisesRegex(
                subject.DiagnosticError,
                "no more than 30",
            ),
        ):
            subject.watch_and_diagnose(poll_seconds=30.01)
        current.assert_not_called()


if __name__ == "__main__":
    unittest.main()
