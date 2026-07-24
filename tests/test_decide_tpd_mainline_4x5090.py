import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import contextmanager

from experiments import decide_tpd_mainline_4x5090 as decision


FIXTURE_SPLIT_SHA256 = decision.hashlib.sha256(
    b'{"fixture":"split"}\n'
).hexdigest()


def point(pd, fa, tiny_pd=1.0, miou=0.8, threshold=0.5):
    return {
        "pd": pd,
        "fa": fa,
        "tiny_pd": tiny_pd,
        "miou": miou,
        "threshold": threshold,
    }


def operating(defaults):
    return {
        budget: {variant: defaults[variant](index) for variant in decision.VARIANTS}
        for index, budget in enumerate(decision.FORMAL_FA_BUDGET_KEYS)
    }


def pareto(exclusive_tpd=True):
    owners = ["tpd"] if exclusive_tpd else ["progressive"]
    return {
        "coordinates": [
            {
                "pd": 0.9,
                "fa": 1e-5,
                "owners": owners,
                "samples": [{"variant": owners[0], "threshold": 0.5}],
            }
        ]
    }


def full_point(matched, fa, tiny_matched=39, miou=0.8, threshold=0.5):
    unmatched = 1
    return {
        "val_loss": 0.1,
        "miou": miou,
        "niou": 0.8,
        "pixel_precision": 0.9,
        "pixel_recall": 0.9,
        "pixel_f1": 0.9,
        "pd": matched / 189,
        "tiny_pd": tiny_matched / 39,
        "fa": fa,
        "false_objects_per_image": unmatched / 133,
        "target_count": 189,
        "matched_target_count": matched,
        "tiny_target_count": 39,
        "matched_tiny_target_count": tiny_matched,
        "predicted_object_count": matched + unmatched,
        "unmatched_predicted_object_count": unmatched,
        "valid_pixel_count": 8716288,
        "threshold": threshold,
    }


def write_manifest(path, entries):
    path.write_text(
        "".join(
            decision.manifest_line(decision.sha256_file(payload), name)
            for payload, name in entries
        ),
        encoding="utf-8",
    )


@contextmanager
def contract_validation_mocks(root):
    runtime_state = root / "launch" / decision.RUNTIME_STATE_NAME
    extended_path = (
        root
        / decision.EXPECTED_DATASET
        / "comparison"
        / "extended_integrity_v1.json"
    )

    def recompute_extended(*_args, **_kwargs):
        raw = extended_path.read_bytes()
        return json.loads(raw), raw

    with (
        mock.patch.object(
            decision,
            "EXPECTED_SPLIT_ARTIFACT_SHA256",
            FIXTURE_SPLIT_SHA256,
        ),
        mock.patch.object(
            decision,
            "EXPECTED_RUNTIME_STATE_SHA256",
            decision.sha256_file(runtime_state),
        ),
        mock.patch.object(
            decision,
            "recompute_extended_audit",
            side_effect=recompute_extended,
        ),
        mock.patch.object(
            decision,
            "recompute_training_data_fingerprint",
            return_value=decision.EXPECTED_TRAINING_DATA_SHA256,
        ),
    ):
        yield


def build_contract_fixture(
    root,
    *,
    divergent_split_variant=None,
    official_test_isolation_evidence=None,
):
    dataset = decision.EXPECTED_DATASET
    run_name = decision.EXPECTED_RUN_NAME
    dataset_dir = root / dataset
    comparison_dir = dataset_dir / "comparison"
    comparison_dir.mkdir(parents=True)

    fixed = {
        "original": full_point(170, 8e-7, miou=0.80),
        "progressive": full_point(172, 8e-7, miou=0.81),
        "spd": full_point(171, 8e-7, miou=0.805),
        "tpd": full_point(175, 8e-7, miou=0.82),
    }
    threshold_configuration = {
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
        "tail_logit_step": 0.1,
        "fa_budgets": list(decision.FORMAL_FA_BUDGETS),
    }
    checkpoint_sha = {}
    source_artifacts = {}
    extended_artifacts = {}
    sweep_hashes = {}
    for variant in decision.VARIANTS:
        run_dir = dataset_dir / variant / run_name
        run_dir.mkdir(parents=True)
        raw_files = {
            "protocol.json": b'{"fixture":"protocol"}\n',
            "split.json": (
                f'{{"fixture":"split","variant":"{variant}"}}\n'.encode()
                if variant == divergent_split_variant
                else b'{"fixture":"split"}\n'
            ),
            "metrics.jsonl": b'{"fixture":"metrics"}\n',
            "summary.json": b'{"fixture":"summary"}\n',
            "best.pth.tar": f"{variant} best checkpoint\n".encode(),
            "best_miou.pth.tar": f"{variant} miou checkpoint\n".encode(),
            "last.pth.tar": f"{variant} last checkpoint\n".encode(),
        }
        for name, content in raw_files.items():
            (run_dir / name).write_bytes(content)
        launch_dir = root / "launch"
        logs_dir = root / "logs"
        launch_dir.mkdir(exist_ok=True)
        logs_dir.mkdir(exist_ok=True)
        launch_path = launch_dir / f"{variant}.json"
        log_path = logs_dir / f"{variant}.log"
        launch_path.write_text(f"{variant} launch\n", encoding="utf-8")
        log_path.write_text(f"{variant} log\n", encoding="utf-8")
        checkpoint_sha[variant] = {
            "best.pth.tar": decision.sha256_file(run_dir / "best.pth.tar"),
            "best_miou.pth.tar": decision.sha256_file(
                run_dir / "best_miou.pth.tar"
            ),
        }
        source_artifacts[variant] = {
            "protocol.json": decision.sha256_file(run_dir / "protocol.json"),
            "split.json": decision.sha256_file(run_dir / "split.json"),
            "summary.json": decision.sha256_file(run_dir / "summary.json"),
            "metrics.jsonl": decision.sha256_file(run_dir / "metrics.jsonl"),
            "checkpoint": checkpoint_sha[variant]["best.pth.tar"],
            "evaluator": decision.EXPECTED_EVALUATOR_SHA256,
        }
        extended_artifacts[variant] = {
            "protocol.json": source_artifacts[variant]["protocol.json"],
            "split.json": source_artifacts[variant]["split.json"],
            "metrics.jsonl": source_artifacts[variant]["metrics.jsonl"],
            "summary.json": source_artifacts[variant]["summary.json"],
            "best.pth.tar": checkpoint_sha[variant]["best.pth.tar"],
            "best_miou.pth.tar": checkpoint_sha[variant]["best_miou.pth.tar"],
            "last.pth.tar": decision.sha256_file(run_dir / "last.pth.tar"),
            "launch_manifest": decision.sha256_file(launch_path),
            "worker_log": decision.sha256_file(log_path),
        }
        sweep_payload = {
            "variant": variant,
            "dataset": dataset,
            "checkpoint_role": "best_validation_pd_primary",
            "checkpoint_sha256": checkpoint_sha[variant]["best.pth.tar"],
            "checkpoint_epoch": 10,
            "seed": 42,
            "split_seed": 20260722,
            "validation_count": 133,
            "validation_split_sha256": (
                decision.EXPECTED_VALIDATION_SPLIT_SHA256
            ),
            "official_test_accessed": False,
            "match_radius": 3.0,
            "tiny_area": 9,
            "threshold_configuration": threshold_configuration,
            "fixed_threshold_0_5": fixed[variant],
            "best_points_under_fa_budget": {
                budget: fixed[variant]
                for budget in decision.FORMAL_FA_BUDGET_KEYS
            },
            "points": [fixed[variant]],
        }
        sweep_path = run_dir / "pd_fa_sweep_best.pth.json"
        sweep_path.write_text(json.dumps(sweep_payload) + "\n", encoding="utf-8")
        sweep_hashes[variant] = decision.sha256_file(sweep_path)

    sweeps_path = comparison_dir / "SWEEPS.sha256"
    sweeps_path.write_text(
        "".join(
            decision.manifest_line(
                sweep_hashes[variant],
                str(
                    (
                        dataset_dir
                        / variant
                        / run_name
                        / "pd_fa_sweep_best.pth.json"
                    ).resolve()
                ),
            )
            for variant in decision.MANIFEST_VARIANTS
        ),
        encoding="utf-8",
    )
    split_hashes = {
        "full_internal_train_sha256": f"{31:064x}",
        "full_internal_val_sha256": f"{32:064x}",
        "used_train_sha256": f"{33:064x}",
        "used_val_sha256": decision.EXPECTED_VALIDATION_SPLIT_SHA256,
    }
    critical_arguments = {
        "dataset": dataset,
        "dataset_dir": "/fixture/datasets",
        "epochs": 800,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "seed": 42,
        "split_seed": 20260722,
        "val_fraction": 0.2,
        "eval_every": 1,
        "base_lr": 0.001,
        "min_lr": 0.00001,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
    }
    integrity_audit = {
        "seed": 42,
        "split_hashes": split_hashes,
        "shared_initialization_sha256": (
            decision.EXPECTED_SHARED_INITIALIZATION_SHA256
        ),
        "normalization": {"mean": 0.5, "std": 0.2},
        "critical_protocol_arguments": critical_arguments,
        "protocol_contract": {
            "primary_selection_rule": [],
            "secondary_selection_rule": [],
            "checkpoint_policy": {},
            "loss": "BCE",
            "optimizer": "Adam",
            "lr_schedule": {},
            "torch": "2.9.1+cu130",
            "cuda_runtime": "13.0",
            "device_name": "NVIDIA GeForce RTX 5090",
        },
        "split_counts": {
            "full_official_train_count": 663,
            "full_internal_train_count": 530,
            "full_internal_val_count": 133,
            "used_train_count": 530,
            "used_val_count": 133,
        },
    }
    rows = []
    for variant in decision.VARIANTS:
        p = fixed[variant]
        rows.append(
            {
                "variant": variant,
                "seed": 42,
                "pd_best_epoch": 10,
                "pd": p["pd"],
                "tiny_pd": p["tiny_pd"],
                "fa": p["fa"],
                "false_objects_per_image": p["false_objects_per_image"],
                "miou_at_pd_best": p["miou"],
                "niou_at_pd_best": p["niou"],
                "f1_at_pd_best": p["pixel_f1"],
                "miou_best_epoch": 11,
                "best_miou": p["miou"],
                "pd_at_miou_best": p["pd"],
                "fa_at_miou_best": p["fa"],
                "parameters": 100,
                "shallow_parameters": 10,
                "elapsed_seconds": 1.0,
                "best_checkpoint_sha256": checkpoint_sha[variant]["best.pth.tar"],
                "best_miou_checkpoint_sha256": checkpoint_sha[variant][
                    "best_miou.pth.tar"
                ],
                "run_dir": str((dataset_dir / variant / run_name).resolve()),
                "delta_pd_vs_original": p["pd"] - fixed["original"]["pd"],
                "delta_tiny_pd_vs_original": (
                    p["tiny_pd"] - fixed["original"]["tiny_pd"]
                ),
                "delta_fa_vs_original": p["fa"] - fixed["original"]["fa"],
                "delta_miou_at_pd_best_vs_original": (
                    p["miou"] - fixed["original"]["miou"]
                ),
            }
        )
    comparison = {
        "dataset": dataset,
        "run_name": run_name,
        "expected_epochs": 800,
        "report_title": "fixture",
        "variant_run_names": {
            variant: run_name for variant in decision.VARIANTS
        },
        "official_test_accessed": False,
        "validation_split_sha256": decision.EXPECTED_VALIDATION_SPLIT_SHA256,
        "training_split_sha256": split_hashes["used_train_sha256"],
        "shared_initialization_sha256": (
            decision.EXPECTED_SHARED_INITIALIZATION_SHA256
        ),
        "checkpoint_sha256": checkpoint_sha,
        "integrity_audit": integrity_audit,
        "rows": rows,
    }
    comparison_json = comparison_dir / f"{run_name}.json"
    comparison_json.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    comparison_md = comparison_dir / f"{run_name}.md"
    comparison_md.write_text("# fixture\n", encoding="utf-8")
    comparison_csv = comparison_dir / f"{run_name}.csv"
    comparison_csv.write_text("fixture\n", encoding="utf-8")
    complete = comparison_dir / "COMPLETE.sha256"
    write_manifest(
        complete,
        (
            (comparison_json, comparison_json.name),
            (comparison_md, comparison_md.name),
            (comparison_csv, comparison_csv.name),
            (sweeps_path, sweeps_path.name),
        ),
    )

    per_variant = {}
    for variant in decision.VARIANTS:
        per_variant[variant] = {
            "gpu_uuid": decision.GPU_UUIDS[variant],
            "invocation_id": decision.INVOCATION_IDS[variant],
            "event_audit": {
                "event_count": 800,
                "epoch_range": [1, 800],
                "processed_train_samples_each_epoch": 530,
                "learning_rate_schedule_exact": True,
                "online_best_flags_exact": True,
                "recomputed_best_pd_epoch": 10,
                "recomputed_best_miou_epoch": 11,
            },
            "best_pd_epoch": 10,
            "best_miou_epoch": 11,
            "last_checkpoint_audit": {
                "role": "last_evaluated_epoch",
                "epoch": 800,
                "sha256": extended_artifacts[variant]["last.pth.tar"],
                "strict_load": True,
                "all_checkpoint_tensors_finite": True,
                "metrics_equal_final_event": True,
            },
            "initialization_recomputed": {},
            "artifact_sha256": extended_artifacts[variant],
        }
    runtime_state = root / "launch" / decision.RUNTIME_STATE_NAME
    runtime_state.write_text('{"fixture":"runtime"}\n', encoding="utf-8")
    if official_test_isolation_evidence is None:
        official_test_isolation_evidence = {
            "official_train_index": str(
                decision.OFFICIAL_TRAIN_INDEX_PATH.resolve()
            ),
            "official_train_index_sha256": (
                decision.EXPECTED_OFFICIAL_TRAIN_INDEX_SHA256
            ),
            "official_train_count": 663,
            "internal_split_union_equals_official_train": True,
            "runner_code_path_reads_training_index_only": True,
            "official_test_code_path_isolation_verified": True,
            "syscall_level_trace_available": False,
        }
    extended = {
        "schema": "sctransnet_formal800_4x5090_extended_integrity_v1",
        "root": str(root.resolve()),
        "dataset": dataset,
        "run_name": run_name,
        "expected_epochs": 800,
        "official_test_accessed": False,
        "selection_source": "internal_validation_only",
        "training_data_sha256": decision.EXPECTED_TRAINING_DATA_SHA256,
        "runtime_binding": {
            "state_path": str(runtime_state.resolve()),
            "state_sha256": decision.sha256_file(runtime_state),
            "captured_at": "fixture",
            "units": [],
            "invocation_ids_bound": True,
            "exec_start_gpu_mapping_bound": True,
            "no_restarts_at_capture": True,
        },
        "per_variant": per_variant,
        "cross_variant_consistency": integrity_audit,
        "byte_identical_split_sha256": FIXTURE_SPLIT_SHA256,
        "independently_recomputed_shared_initialization_sha256": (
            decision.EXPECTED_SHARED_INITIALIZATION_SHA256
        ),
        "official_test_isolation_evidence": official_test_isolation_evidence,
        "frozen_sources_and_data": {
            "source_sha256": {
                label: decision.sha256_file(path)
                for label, path in decision.EXTENDED_SOURCE_PATHS.items()
            },
            "training_data_sha256": decision.EXPECTED_TRAINING_DATA_SHA256,
            "environment": {},
        },
        "checks_passed": {
            name: True for name in decision.EXTENDED_INTEGRITY_FLAGS
        },
        "limitations": {
            "single_dataset_single_seed_screening_only": True,
            "mainline_decision_not_made_by_this_audit": True,
        },
    }
    extended_json = comparison_dir / "extended_integrity_v1.json"
    extended_json.write_text(json.dumps(extended) + "\n", encoding="utf-8")
    extended_complete = comparison_dir / "EXTENDED_COMPLETE.sha256"
    write_manifest(
        extended_complete,
        (
            (complete, complete.name),
            (extended_json, extended_json.name),
        ),
    )

    source_sweeps = {}
    bindings = {}
    for variant in decision.VARIANTS:
        sweep_path = (
            dataset_dir / variant / run_name / "pd_fa_sweep_best.pth.json"
        ).resolve()
        source_sweeps[variant] = {
            "run_dir": str(sweep_path.parent),
            "sweep_path": str(sweep_path),
            "sweep_sha256": sweep_hashes[variant],
            "checkpoint_epoch": 10,
            "checkpoint_sha256": checkpoint_sha[variant]["best.pth.tar"],
            "artifact_sha256": source_artifacts[variant],
        }
        bindings[variant] = {
            "checkpoint_sha256": checkpoint_sha[variant]["best.pth.tar"],
            "checkpoint_epoch": 10,
            "row_metric_binding_passed": True,
        }
    source_seal = {
        "comparison_dir": str(comparison_dir.resolve()),
        "training_certificate_path": str(comparison_json.resolve()),
        "training_certificate_sha256": decision.sha256_file(comparison_json),
        "sweeps_manifest_path": str(sweeps_path.resolve()),
        "sweeps_manifest_sha256": decision.sha256_file(sweeps_path),
        "complete_manifest_path": str(complete.resolve()),
        "complete_manifest_sha256": decision.sha256_file(complete),
        "sealed_base_artifact_sha256": {
            comparison_json.name: decision.sha256_file(comparison_json),
            comparison_md.name: decision.sha256_file(comparison_md),
            comparison_csv.name: decision.sha256_file(comparison_csv),
            sweeps_path.name: decision.sha256_file(sweeps_path),
        },
        "sweep_paths": {
            variant: source_sweeps[variant]["sweep_path"]
            for variant in decision.VARIANTS
        },
        "sweep_sha256": {
            variant: source_sweeps[variant]["sweep_sha256"]
            for variant in decision.VARIANTS
        },
    }
    operating_points = {
        budget: {variant: fixed[variant] for variant in decision.VARIANTS}
        for budget in decision.FORMAL_FA_BUDGET_KEYS
    }
    aggregate = {
        "schema_version": "tpd-pd-fa-aggregate-v2",
        "dataset": dataset,
        "run_name": run_name,
        "expected_epochs": 800,
        "official_test_accessed": False,
        "selection_source": "internal_validation_only",
        "checkpoint_role": "best_validation_pd_primary",
        "operating_point_rule": list(decision.OPERATING_POINT_RULE),
        "auc_computed": False,
        "mainline_decision_made": False,
        "output_commit": {},
        "sealed_source_evidence": source_seal,
        "sealed_training_certificate": {
            "integrity_audit": integrity_audit,
            "training_to_sweep_binding": bindings,
            "hardware_note": "fixture",
        },
        "common_provenance": {
            "seed": 42,
            "split_seed": 20260722,
            "validation_count": 133,
            "validation_split_sha256": (
                decision.EXPECTED_VALIDATION_SPLIT_SHA256
            ),
            "match_radius": 3.0,
            "tiny_area": 9,
            "threshold_configuration": threshold_configuration,
            "evaluator_sha256": decision.EXPECTED_EVALUATOR_SHA256,
            "split_artifact_sha256": FIXTURE_SPLIT_SHA256,
            "metric_notes": {},
            "budgets": list(decision.FORMAL_FA_BUDGETS),
            "invariant_counts": {
                "target_count": 189,
                "tiny_target_count": 39,
                "valid_pixel_count": 8716288,
            },
        },
        "source_sweeps": source_sweeps,
        "fixed_threshold_0_5": fixed,
        "operating_points_by_fa_budget": operating_points,
        "global_pareto": {
            "scope": "joint_sampled_discrete_threshold_coordinates",
            "dominance_definition": (
                "coordinate A dominates B iff A.Fa <= B.Fa and A.Pd >= B.Pd, "
                "with at least one strict inequality; identical coordinates retain all owners"
            ),
            "unique_coordinate_count": 1,
            "owner_coordinate_counts": {
                "original": 0,
                "progressive": 0,
                "spd": 0,
                "tpd": 1,
            },
            "coordinates": [
                {
                    "pd": fixed["tpd"]["pd"],
                    "fa": fixed["tpd"]["fa"],
                    "owners": ["tpd"],
                    "samples": [
                        {
                            "variant": "tpd",
                            "threshold": 0.5,
                            "tiny_pd": 1.0,
                            "miou": fixed["tpd"]["miou"],
                        }
                    ],
                }
            ],
        },
        "aggregator_sha256": decision.EXPECTED_AGGREGATOR_SHA256,
        "integrity_checks_passed": {
            name: True for name in decision.AGGREGATE_INTEGRITY_FLAGS
        },
    }
    aggregate_stem = f"pd_fa_{run_name}"
    aggregate_json = comparison_dir / f"{aggregate_stem}.json"
    aggregate_md = comparison_dir / f"{aggregate_stem}.md"
    operating_csv = comparison_dir / f"{aggregate_stem}_operating_points.csv"
    curves_csv = comparison_dir / f"{aggregate_stem}_curves.csv"
    aggregate_marker = comparison_dir / f"{aggregate_stem}.COMPLETE.sha256"
    aggregate["output_commit"] = {
        "marker": str(aggregate_marker.resolve()),
        "semantics": "fixture",
    }
    aggregate_json.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
    aggregate_md.write_text("# aggregate fixture\n", encoding="utf-8")
    operating_csv.write_text("fixture\n", encoding="utf-8")
    curves_csv.write_text("fixture\n", encoding="utf-8")
    write_manifest(
        aggregate_marker,
        (
            (aggregate_json, aggregate_json.name),
            (aggregate_md, aggregate_md.name),
            (operating_csv, operating_csv.name),
            (curves_csv, curves_csv.name),
        ),
    )
    return root


class DecisionPolicyTests(unittest.TestCase):
    def test_advance_requires_unique_uncovered_noninferior_signal(self):
        points = operating(
            {
                "original": lambda _: point(0.80, 8e-7),
                "progressive": lambda _: point(0.85, 8e-7),
                "spd": lambda _: point(0.84, 8e-7),
                "tpd": lambda _: point(0.90, 8e-7),
            }
        )
        fixed = {
            "original": point(0.80, 2e-5),
            "progressive": point(0.85, 2e-5),
            "spd": point(0.84, 2e-5),
            "tpd": point(0.90, 2e-5),
        }
        result = decision.decide(fixed, points, pareto(True))
        self.assertEqual(result["decision"], "ADVANCE_TPD_TO_MULTI_SEED")

    def test_rejects_when_no_budget_beats_original(self):
        points = operating(
            {
                "original": lambda _: point(0.90, 8e-7),
                "progressive": lambda _: point(0.85, 8e-7),
                "spd": lambda _: point(0.84, 8e-7),
                "tpd": lambda _: point(0.80, 8e-7),
            }
        )
        fixed = {variant: points[decision.FORMAL_FA_BUDGET_KEYS[0]][variant] for variant in decision.VARIANTS}
        result = decision.decide(fixed, points, pareto(True))
        self.assertEqual(
            result["decision"], "DO_NOT_ESTABLISH_CURRENT_TPD_CORE"
        )

    def test_rejects_when_every_tpd_advantage_is_covered(self):
        points = operating(
            {
                "original": lambda _: point(0.80, 8e-7),
                "progressive": lambda _: point(0.95, 8e-7),
                "spd": lambda _: point(0.84, 8e-7),
                "tpd": lambda _: point(0.90, 8e-7),
            }
        )
        fixed = {variant: points[decision.FORMAL_FA_BUDGET_KEYS[0]][variant] for variant in decision.VARIANTS}
        result = decision.decide(fixed, points, pareto(True))
        self.assertEqual(
            result["decision"], "DO_NOT_ESTABLISH_CURRENT_TPD_CORE"
        )

    def test_crossing_uncovered_curves_are_inconclusive(self):
        points = operating(
            {
                "original": lambda _: point(0.80, 8e-7),
                "progressive": lambda index: point(
                    0.95 if index == 0 else 0.85, 8e-7
                ),
                "spd": lambda _: point(0.84, 8e-7),
                "tpd": lambda _: point(0.90, 8e-7),
            }
        )
        fixed = {
            "original": point(0.80, 2e-5),
            "progressive": point(0.85, 2e-5),
            "spd": point(0.84, 2e-5),
            "tpd": point(0.90, 2e-5),
        }
        result = decision.decide(fixed, points, pareto(True))
        self.assertEqual(result["decision"], "INCONCLUSIVE_MIXED_TRADEOFF")

    def test_missing_point_sorts_below_available_point(self):
        self.assertEqual(decision.comparison_label(None, None), "equal")
        self.assertEqual(decision.comparison_label(point(0.0, 0.0), None), "better")
        self.assertEqual(decision.comparison_label(None, point(0.0, 0.0)), "worse")

    def test_cross_method_comparison_ignores_threshold_distance(self):
        left = point(0.9, 8e-7, tiny_pd=1.0, miou=0.8, threshold=0.5)
        right = point(0.9, 8e-7, tiny_pd=1.0, miou=0.8, threshold=0.4)
        self.assertEqual(decision.comparison_label(left, right), "equal")

    def test_only_zero_detection_availability_advantage_is_inconclusive(self):
        points = {
            budget: {
                "original": None,
                "progressive": None,
                "spd": None,
                "tpd": point(0.0, 0.0, tiny_pd=0.0, miou=0.0, threshold=0.9999),
            }
            for budget in decision.FORMAL_FA_BUDGET_KEYS
        }
        fixed = {
            variant: point(0.0, 1e-4, tiny_pd=0.0, miou=0.0)
            for variant in decision.VARIANTS
        }
        result = decision.decide(fixed, points, pareto(True))
        self.assertEqual(result["decision"], "INCONCLUSIVE_MIXED_TRADEOFF")
        self.assertEqual(
            result["reason_codes"],
            ["ONLY_ZERO_DETECTION_AVAILABILITY_ADVANTAGE"],
        )

    def test_budget_recompute_uses_the_frozen_lexicographic_order(self):
        lower_fa = full_point(170, 5e-7, tiny_matched=38, miou=0.70, threshold=0.4)
        higher_pd = full_point(171, 9e-7, tiny_matched=37, miou=0.60, threshold=0.9)
        selected = decision.recompute_budget_point(
            [lower_fa, higher_pd], budget=1e-6
        )
        self.assertIs(selected, higher_pd)


class EvidenceAndOutputNegativeTests(unittest.TestCase):
    def test_contract_fixture_passes_with_independent_recompute_mock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(Path(temporary))
            with contract_validation_mocks(root):
                payload, inputs = decision.build_decision_payload(
                    root,
                    decision.EXPECTED_DATASET,
                    decision.EXPECTED_RUN_NAME,
                    decision.EXPECTED_EPOCHS,
                )
            self.assertEqual(
                payload["screening_decision"]["decision"],
                "ADVANCE_TPD_TO_MULTI_SEED",
            )
            self.assertFalse(payload["paper_core_established"])
            self.assertFalse(payload["stability_claim_supported"])
            self.assertGreater(len(inputs), 0)

    def test_divergent_current_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(
                Path(temporary), divergent_split_variant="tpd"
            )
            with contract_validation_mocks(root):
                with self.assertRaisesRegex(ValueError, "source split SHA"):
                    decision.build_decision_payload(
                        root,
                        decision.EXPECTED_DATASET,
                        decision.EXPECTED_RUN_NAME,
                        decision.EXPECTED_EPOCHS,
                    )

    def test_empty_official_test_isolation_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(
                Path(temporary), official_test_isolation_evidence={}
            )
            with contract_validation_mocks(root):
                with self.assertRaisesRegex(
                    ValueError, "official_test_isolation_evidence"
                ):
                    decision.build_decision_payload(
                        root,
                        decision.EXPECTED_DATASET,
                        decision.EXPECTED_RUN_NAME,
                        decision.EXPECTED_EPOCHS,
                    )

    def test_independent_extended_audit_byte_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(Path(temporary))
            with contract_validation_mocks(root), mock.patch.object(
                decision,
                "recompute_extended_audit",
                return_value=({"different": True}, b'{"different":true}\n'),
            ):
                with self.assertRaisesRegex(ValueError, "byte-identical"):
                    decision.build_decision_payload(
                        root,
                        decision.EXPECTED_DATASET,
                        decision.EXPECTED_RUN_NAME,
                        decision.EXPECTED_EPOCHS,
                    )

    def test_independent_auditor_uses_current_python_and_fixed_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_state = root / "runtime.json"
            runtime_state.write_text("{}\n", encoding="utf-8")

            def execute(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                output.write_text('{"recomputed":true}\n', encoding="utf-8")
                return mock.Mock(returncode=0, stdout="", stderr="")

            with (
                mock.patch.object(
                    decision,
                    "EXPECTED_RUNTIME_STATE_SHA256",
                    decision.sha256_file(runtime_state),
                ),
                mock.patch.object(
                    decision.subprocess, "run", side_effect=execute
                ) as run,
            ):
                parsed, raw = decision.recompute_extended_audit(
                    root,
                    decision.EXPECTED_DATASET,
                    decision.EXPECTED_RUN_NAME,
                    decision.EXPECTED_EPOCHS,
                    runtime_state,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[0], decision.sys.executable)
            self.assertEqual(run.call_args.kwargs["timeout"], 3600)
            self.assertEqual(parsed, {"recomputed": True})
            self.assertEqual(raw, b'{"recomputed":true}\n')

    def test_fingerprint_uses_current_python_and_fixed_timeout(self):
        completed = mock.Mock(
            returncode=0,
            stdout=decision.EXPECTED_TRAINING_DATA_SHA256 + "\n",
            stderr="",
        )
        with mock.patch.object(
            decision.subprocess, "run", return_value=completed
        ) as run:
            digest = decision.recompute_training_data_fingerprint(
                decision.EXPECTED_DATASET
            )
        command = run.call_args.args[0]
        self.assertEqual(command[0], decision.sys.executable)
        self.assertEqual(run.call_args.kwargs["timeout"], 300)
        self.assertEqual(digest, decision.EXPECTED_TRAINING_DATA_SHA256)

    def test_points_must_use_frozen_ground_truth_counts(self):
        bad = full_point(170, 8e-7)
        bad["target_count"] = 188
        with self.assertRaisesRegex(ValueError, "frozen value 189"):
            decision.validate_point(bad, "bad point", 133)

    def test_frozen_producer_source_sha_mismatch_is_rejected(self):
        altered = dict(decision.EXPECTED_PRODUCER_SOURCE_SHA256)
        altered["evaluator"] = "0" * 64
        with mock.patch.object(
            decision, "EXPECTED_PRODUCER_SOURCE_SHA256", altered
        ):
            with self.assertRaisesRegex(ValueError, "producer source SHA evaluator"):
                decision.validate_frozen_producer_sources()

    def test_wrong_formal_budget_is_rejected_even_when_marker_is_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(Path(temporary))
            comparison = root / decision.EXPECTED_DATASET / "comparison"
            stem = f"pd_fa_{decision.EXPECTED_RUN_NAME}"
            aggregate_json = comparison / f"{stem}.json"
            aggregate = json.loads(aggregate_json.read_text(encoding="utf-8"))
            aggregate["common_provenance"]["budgets"][-1] = 2e-4
            aggregate_json.write_text(
                json.dumps(aggregate) + "\n", encoding="utf-8"
            )
            write_manifest(
                comparison / f"{stem}.COMPLETE.sha256",
                (
                    (aggregate_json, aggregate_json.name),
                    (comparison / f"{stem}.md", f"{stem}.md"),
                    (
                        comparison / f"{stem}_operating_points.csv",
                        f"{stem}_operating_points.csv",
                    ),
                    (
                        comparison / f"{stem}_curves.csv",
                        f"{stem}_curves.csv",
                    ),
                ),
            )
            with contract_validation_mocks(root):
                with self.assertRaises(ValueError):
                    decision.build_decision_payload(
                        root,
                        decision.EXPECTED_DATASET,
                        decision.EXPECTED_RUN_NAME,
                        decision.EXPECTED_EPOCHS,
                    )

    def test_current_checkpoint_byte_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_contract_fixture(Path(temporary))
            checkpoint = (
                root
                / decision.EXPECTED_DATASET
                / "tpd"
                / decision.EXPECTED_RUN_NAME
                / "best.pth.tar"
            )
            checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
            with contract_validation_mocks(root):
                with self.assertRaises(ValueError):
                    decision.build_decision_payload(
                        root,
                        decision.EXPECTED_DATASET,
                        decision.EXPECTED_RUN_NAME,
                        decision.EXPECTED_EPOCHS,
                    )

    def test_manifest_must_match_exact_file_set_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            second = root / "second.md"
            marker = root / "COMPLETE.sha256"
            first.write_text("{}\n", encoding="utf-8")
            second.write_text("ok\n", encoding="utf-8")
            marker.write_text(
                decision.manifest_line(decision.sha256_file(first), first.name),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                decision.verify_exact_manifest(
                    marker,
                    ((first, first.name), (second, second.name)),
                    "test marker",
                )

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                decision.read_json(path, "duplicate fixture")

    def test_uncommitted_group_is_pending_not_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(decision.PendingEvidenceError):
                decision.require_committed_group(
                    root / "COMPLETE.sha256",
                    (root / "payload.json",),
                    "fixture",
                )

    def test_default_output_policy_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(json.dumps({"sealed": True}) + "\n", encoding="utf-8")
            snapshot = decision.collect_input_snapshot((source,))
            json_path = root / "decision.json"
            md_path = root / "decision.md"
            marker = root / "decision.COMPLETE.sha256"
            payloads = {json_path: "{}\n", md_path: "# decision\n"}
            decision.write_outputs_atomically(payloads, marker, snapshot)
            decision.verify_decision_outputs((json_path, md_path), marker)
            with self.assertRaises(FileExistsError):
                decision.write_outputs_atomically(payloads, marker, snapshot)

    def test_output_publication_uses_hard_links_and_marker_is_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}\n", encoding="utf-8")
            snapshot = decision.collect_input_snapshot((source,))
            json_path = root / "decision.json"
            md_path = root / "decision.md"
            marker = root / "decision.COMPLETE.sha256"
            with mock.patch.object(
                decision.os, "link", wraps=decision.os.link
            ) as link:
                decision.write_outputs_atomically(
                    {json_path: "{}\n", md_path: "# decision\n"},
                    marker,
                    snapshot,
                )
            destinations = [Path(call.args[1]) for call in link.call_args_list]
            self.assertEqual(destinations, [json_path, md_path, marker])

    def test_training_fingerprint_is_checked_before_and_after_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text("{}\n", encoding="utf-8")
            snapshot = decision.collect_input_snapshot((source,))
            with mock.patch.object(
                decision,
                "recompute_training_data_fingerprint",
                return_value=decision.EXPECTED_TRAINING_DATA_SHA256,
            ) as fingerprint:
                decision.write_outputs_atomically(
                    {
                        root / "decision.json": "{}\n",
                        root / "decision.md": "# decision\n",
                    },
                    root / "decision.COMPLETE.sha256",
                    snapshot,
                    decision.EXPECTED_TRAINING_DATA_SHA256,
                )
            self.assertEqual(fingerprint.call_count, 4)


if __name__ == "__main__":
    unittest.main()
