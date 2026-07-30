from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import collect_final_model_validation_statistics as cache_core
from analysis import run_final_qfg_six_mode_audit as bootstrap_contract
from experiments import (
    analyze_final_model_engineering_paired_screen as subject,
)
from experiments import (
    evaluate_final_model_engineering_replication_pd_fa as evaluator,
)
from experiments import final_model_replication_exact_core as replication_core
from experiments import final_model_replication_seed_contract as seed_contract


SOURCE_LOCK_SHA = "a" * 64
SEED_CONTRACT_SHA = "b" * 64
DATASET_SHA = "c" * 64
EVALUATION_SOURCE_BINDING = evaluator.frozen_evaluation_core_binding()
EVALUATOR_SHA = EVALUATION_SOURCE_BINDING[
    "checkpoint_local_adapter"
]["sha256"]
NORMALIZATION_SHA = "e" * 64
IMAGE_IDS = tuple(f"validation_{index:03d}" for index in range(133))
VALIDATION_IDS_SHA = cache_core.validation_identifier_sha256(IMAGE_IDS)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def recorded_assignment(arm: str) -> dict[str, object]:
    physical_gpu_index, physical_gpu_uuid = (
        evaluator._expected_arm_gpu_binding(arm)
    )
    return {
        "device": "cuda:0",
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": physical_gpu_uuid,
        "cuda_visible_devices": physical_gpu_uuid,
        "visible_cuda_device_count": 1,
        "device_name": "fixture RTX 5090",
    }


def engineering_request_identity(
    *,
    root: Path,
    label: str,
    arm: str,
    trajectory_seed: int,
    checkpoint_filename: str,
    checkpoint_epoch: int,
    checkpoint_sha256: str,
    threshold_domain_id: str,
) -> dict[str, object]:
    core = {
        "schema": evaluator.CACHE_REQUEST_IDENTITY_SCHEMA,
        "arm": arm,
        "variant": replication_core.arm_definition(arm).variant,
        "trajectory_seed": trajectory_seed,
        "run_id": f"fixture:{label}",
        "run_directory": str((root / "runs" / label).resolve()),
        "seed_contract_sha256": SEED_CONTRACT_SHA,
        "child_manifest_sha256": digest(f"child-manifest-{label}"),
        "source_lock_sha256": SOURCE_LOCK_SHA,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "threshold_domain_id": threshold_domain_id,
        "adapter_source_sha256": EVALUATOR_SHA,
    }
    request_sha256 = evaluator._canonical_digest(core)
    derivation = {
        "schema": evaluator.CACHE_EVALUATOR_DERIVATION_SCHEMA,
        "algorithm": "sha256_of_canonical_json",
        "adapter_source_sha256": EVALUATOR_SHA,
        "engineering_request_identity_sha256": request_sha256,
    }
    return {
        **core,
        "engineering_request_identity_sha256": request_sha256,
        "collector_evaluator_sha256": evaluator._canonical_digest(
            derivation
        ),
        "collector_evaluator_sha256_derivation": derivation,
    }


def target_for_image(
    image_index: int,
    *,
    shifted_first_target: bool = False,
) -> np.ndarray:
    target = np.zeros((16, 16), dtype=np.uint8)
    component_indices = (
        (2 * image_index, 2 * image_index + 1)
        if image_index < 56
        else (112 + image_index - 56,)
    )
    for slot, component_index in enumerate(component_indices):
        tiny = component_index < evaluator.EXPECTED_TINY_TARGET_COUNT
        if slot == 0:
            if tiny:
                column = (
                    2
                    if shifted_first_target and image_index == 0
                    else 1
                )
                target[1, column] = 1
            else:
                target[1:3, 1:6] = 1
        elif tiny:
            target[10, 10] = 1
        else:
            target[9:11, 9:14] = 1
    return target


def make_cache(
    root: Path,
    *,
    label: str,
    good: bool,
    arm: str = replication_core.ARM_B,
    trajectory_seed: int = 3407,
    checkpoint_filename: str = "best_miou.pth.tar",
    checkpoint_epoch: int = 17,
    threshold_domain_id: str | None = None,
    ordered_ids: tuple[str, ...] = IMAGE_IDS,
    shifted_first_target: bool = False,
) -> tuple[
    cache_core.PredictionCache,
    dict[str, object],
    dict[str, object],
]:
    checkpoint_sha = digest(f"checkpoint-{label}")
    domain_id = threshold_domain_id or digest(f"domain-{label}")
    engineering_identity = engineering_request_identity(
        root=root,
        label=label,
        arm=arm,
        trajectory_seed=trajectory_seed,
        checkpoint_filename=checkpoint_filename,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_sha256=checkpoint_sha,
        threshold_domain_id=domain_id,
    )
    identity = cache_core.build_cache_identity(
        checkpoint_sha256=checkpoint_sha,
        dataset_sha256=DATASET_SHA,
        evaluator_sha256=str(
            engineering_identity["collector_evaluator_sha256"]
        ),
        mode="full",
        normalization_sha256=NORMALIZATION_SHA,
        source_lock_sha256=SOURCE_LOCK_SHA,
        validation_ids_sha256=VALIDATION_IDS_SHA,
        validation_count=evaluator.EXPECTED_VALIDATION_COUNT,
        match_radius=evaluator.FORMAL_MATCH_RADIUS,
        tiny_area=evaluator.FORMAL_TINY_AREA,
    )
    collector = cache_core.PredictionCacheCollector(
        identity=identity,
        match_radius=evaluator.FORMAL_MATCH_RADIUS,
        tiny_area=evaluator.FORMAL_TINY_AREA,
    )
    for image_id in ordered_ids:
        image_index = int(image_id.rsplit("_", 1)[1])
        target = target_for_image(
            image_index,
            shifted_first_target=shifted_first_target,
        )
        if good:
            probability = np.where(target > 0, 0.9, 0.1).astype(
                np.float32
            )
            loss = 0.1
        else:
            probability = np.full(target.shape, 0.1, dtype=np.float32)
            loss = 1.0
        collector.append(
            image_id=image_id,
            probability=probability,
            target=target,
            loss=loss,
        )
    cache = collector.seal()
    cache_directory = root / label
    metadata_path = cache_core.write_prediction_cache(
        cache,
        cache_directory,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    arrays_path = metadata_path.parent / metadata["arrays"]["filename"]
    binding: dict[str, object] = {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": cache_core.sha256_file(metadata_path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": cache_core.sha256_file(arrays_path),
        "identity": copy.deepcopy(identity),
        "engineering_request_identity": copy.deepcopy(
            engineering_identity
        ),
        "prediction_content_sha256": cache.content_sha256,
        "image_count": len(cache.records),
        "image_ids_sha256": VALIDATION_IDS_SHA,
        "paired_image_statistics_available": True,
    }
    return cache, binding, identity


def point_with_threshold(
    cache: cache_core.PredictionCache,
    threshold: float,
) -> dict[str, object]:
    point = cache_core.recompute_metrics(cache, threshold=threshold)
    return evaluator.sweep_core.json_ready(
        {**point, "threshold": threshold}
    )


def build_complete_manifest(root: Path) -> tuple[Path, dict[str, object]]:
    results: list[dict[str, object]] = []
    result_by_key: dict[
        tuple[int, str, str],
        dict[str, object],
    ] = {}
    budget_points_by_quality: dict[bool, dict[str, object]] = {}
    for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS:
        for arm in replication_core.SUPPORTED_ARMS:
            for checkpoint_filename, selection_role, checkpoint_role in (
                evaluator.CHECKPOINT_SPECS
            ):
                label = (
                    f"seed_{trajectory_seed}_{arm}_{selection_role}"
                )
                threshold_domain_id = digest(f"domain-{label}")
                cache, binding, _ = make_cache(
                    root / "caches",
                    label=label,
                    good=arm == replication_core.ARM_D,
                    arm=arm,
                    trajectory_seed=trajectory_seed,
                    checkpoint_filename=checkpoint_filename,
                    checkpoint_epoch=17,
                    threshold_domain_id=threshold_domain_id,
                )
                fixed = point_with_threshold(
                    cache,
                    evaluator.FIXED_THRESHOLD,
                )
                if (arm == replication_core.ARM_D) not in (
                    budget_points_by_quality
                ):
                    scan = bootstrap_contract.fa_budget_scan(cache)
                    budget_points_by_quality[
                        arm == replication_core.ARM_D
                    ] = evaluator.sweep_core.json_ready(
                        scan["budget_points"]
                    )
                budget_points = copy.deepcopy(
                    budget_points_by_quality[
                        arm == replication_core.ARM_D
                    ]
                )
                result_path = (
                    root / "results" / f"{label}.json"
                ).resolve()
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_payload: dict[str, object] = {
                    "schema": evaluator.RESULT_SCHEMA,
                    "execution_complete": True,
                    "paired_image_statistics_available": True,
                    "threshold_selection_scope": "single_checkpoint_only",
                    "cross_checkpoint_point_pooling": False,
                    "official_test_accessed": False,
                    "checkpoint_epoch": 17,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_sha256": cache.identity[
                        "checkpoint_sha256"
                    ],
                    "variant": replication_core.arm_definition(arm).variant,
                    "seed": trajectory_seed,
                    "execution_device_assignment": recorded_assignment(arm),
                    "source_checkpoint_identity": {
                        "arm": arm,
                        "variant": replication_core.arm_definition(
                            arm
                        ).variant,
                        "trajectory_seed": trajectory_seed,
                        "selection_role": selection_role,
                        "checkpoint_filename": checkpoint_filename,
                        "checkpoint_role": checkpoint_role,
                        "checkpoint_epoch": 17,
                        "checkpoint_sha256": cache.identity[
                            "checkpoint_sha256"
                        ],
                        "threshold_domain_id": threshold_domain_id,
                    },
                    "source_run_identity": {
                        "variant": replication_core.arm_definition(
                            arm
                        ).variant,
                        "seed": trajectory_seed,
                    },
                    "replication_input_binding": {
                        "source_lock_sha256": SOURCE_LOCK_SHA,
                        "seed_contract_sha256": SEED_CONTRACT_SHA,
                    },
                    "evaluation_source_binding": copy.deepcopy(
                        EVALUATION_SOURCE_BINDING
                    ),
                    "fixed_threshold_0_5": copy.deepcopy(fixed),
                    "best_points_under_fa_budget": copy.deepcopy(
                        budget_points
                    ),
                    "prediction_cache": copy.deepcopy(binding),
                }
                result_path.write_bytes(
                    evaluator._result_json_bytes(result_payload)
                )
                result: dict[str, object] = {
                    "threshold_domain_id": threshold_domain_id,
                    "arm": arm,
                    "variant": replication_core.arm_definition(arm).variant,
                    "trajectory_seed": trajectory_seed,
                    "selection_role": selection_role,
                    "checkpoint_filename": checkpoint_filename,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_epoch": 17,
                    "checkpoint_sha256": cache.identity[
                        "checkpoint_sha256"
                    ],
                    "result_path": str(result_path),
                    "result_sha256": cache_core.sha256_file(result_path),
                    "execution_device_assignment": recorded_assignment(arm),
                    "fixed_threshold_0_5": fixed,
                    "best_points_under_fa_budget": budget_points,
                    "prediction_cache": binding,
                }
                results.append(result)
                result_by_key[
                    (trajectory_seed, arm, selection_role)
                ] = result
    paired_groups: list[dict[str, object]] = []
    for trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS:
        for selection_role in subject.SELECTION_ROLES:
            b_cache = result_by_key[
                (
                    trajectory_seed,
                    replication_core.ARM_B,
                    selection_role,
                )
            ]["prediction_cache"]
            d_cache = result_by_key[
                (
                    trajectory_seed,
                    replication_core.ARM_D,
                    selection_role,
                )
            ]["prediction_cache"]
            assert isinstance(b_cache, dict)
            assert isinstance(d_cache, dict)
            paired_groups.append(
                {
                    "trajectory_seed": trajectory_seed,
                    "selection_role": selection_role,
                    "validation_count": (
                        evaluator.EXPECTED_VALIDATION_COUNT
                    ),
                    "validation_ids_sha256": VALIDATION_IDS_SHA,
                    "arm_b_cache_metadata_path": b_cache[
                        "metadata_path"
                    ],
                    "arm_b_cache_metadata_sha256": b_cache[
                        "metadata_sha256"
                    ],
                    "arm_d_cache_metadata_path": d_cache[
                        "metadata_path"
                    ],
                    "arm_d_cache_metadata_sha256": d_cache[
                        "metadata_sha256"
                    ],
                    "image_level_pairing_ready": True,
                }
            )
    manifest: dict[str, object] = {
        "schema": evaluator.MANIFEST_SCHEMA,
        "status": "complete",
        "scope": "fixed_parent_engineering_b_d_only",
        "result_count": evaluator.EXPECTED_SWEEP_COUNT,
        "expected_result_count": evaluator.EXPECTED_SWEEP_COUNT,
        "all_checkpoint_local_results_valid": True,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "fixed_threshold": evaluator.FIXED_THRESHOLD,
        "fa_budgets": list(evaluator.FA_BUDGETS),
        "source_lock_sha256": SOURCE_LOCK_SHA,
        "seed_contract_sha256": SEED_CONTRACT_SHA,
        "validation_count": evaluator.EXPECTED_VALIDATION_COUNT,
        "validation_ids_sha256": VALIDATION_IDS_SHA,
        "formal_gpu_binding_policy": {
            "cpu_results_accepted": False,
            "arm_assignments": {
                arm: {
                    "physical_gpu_index": (
                        evaluator._expected_arm_gpu_binding(arm)[0]
                    ),
                    "physical_gpu_uuid": (
                        evaluator._expected_arm_gpu_binding(arm)[1]
                    ),
                    "logical_device": "cuda:0",
                }
                for arm in replication_core.SUPPORTED_ARMS
            },
        },
        "all_results_expected_physical_gpu_bound": True,
        "paired_checkpoint_group_count": 4,
        "paired_checkpoint_groups": paired_groups,
        "gate_m_train_image_level_inputs_ready": True,
        "paired_confidence_intervals_computed": False,
        "paired_confidence_intervals_claimed": False,
        "official_test_accessed": False,
        "evaluation_source_binding": copy.deepcopy(
            EVALUATION_SOURCE_BINDING
        ),
        "results": results,
    }
    ready = evaluator.sweep_core.json_ready(manifest)
    manifest_path = root / "eight-results.json"
    manifest_path.write_bytes(evaluator._manifest_json_bytes(ready))
    return manifest_path, ready


class EngineeringPairedScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.manifest_path, cls.manifest = build_complete_manifest(
            cls.root
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_missing_manifest_is_pending_without_metrics(self) -> None:
        payload = subject.analyze(
            manifest_path=self.root / "missing-manifest.json"
        )
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(
            payload["decision"],
            "ENGINEERING_PAIRED_SCREEN_PENDING",
        )
        self.assertIsNone(payload["engineering_paired_route_met"])
        self.assertNotIn(
            "per_seed_checkpoint_policy_results",
            payload,
        )
        self.assertFalse(payload["establishes_gate_m_train"])
        self.assertEqual(
            payload["gates"]["M-train"]["status"],
            "insufficient_evidence",
        )
        self.assertIsNone(payload["gates"]["M-train"]["passed"])

    def test_complete_screen_uses_fixed_contract_and_stays_descriptive(
        self,
    ) -> None:
        payload = subject.analyze(manifest_path=self.manifest_path)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(
            payload["decision"],
            "ENGINEERING_PAIRED_SCREEN_ROUTE_MET",
        )
        self.assertTrue(payload["engineering_paired_route_met"])
        self.assertFalse(payload["establishes_gate_m_train"])
        self.assertEqual(
            payload["gates"]["M-train"]["status"],
            "insufficient_evidence",
        )
        self.assertIsNone(payload["gates"]["M-train"]["passed"])
        self.assertFalse(
            payload["gates"]["M-train"]["establishes_gate_m_train"]
        )
        self.assertFalse(
            payload["claim_boundary"]["paper_core_established"]
        )
        self.assertFalse(
            payload["claim_boundary"]["stability_claim_supported"]
        )
        self.assertEqual(
            len(payload["per_seed_checkpoint_policy_results"]),
            4,
        )
        self.assertEqual(
            payload["cache_compatibility"][
                "lossless_prediction_cache_count"
            ],
            8,
        )
        self.assertEqual(
            payload["manifest"]["checkpoint_local_result_count"],
            8,
        )
        self.assertEqual(
            len(payload["manifest"]["checkpoint_local_results"]),
            8,
        )
        self.assertTrue(
            payload["cache_compatibility"][
                "all_eight_cache_targets_identical"
            ]
        )
        self.assertTrue(
            payload["cache_compatibility"][
                "all_eight_cache_image_ids_identical"
            ]
        )
        for result in payload[
            "per_seed_checkpoint_policy_results"
        ]:
            bootstrap = result["paired_image_bootstrap"]
            self.assertEqual(
                bootstrap["replicates"],
                10_000,
            )
            self.assertEqual(
                bootstrap["rng_seed"],
                20260730,
            )
            self.assertEqual(
                bootstrap["per_metric_two_sided_confidence"],
                0.99,
            )
            self.assertEqual(
                set(bootstrap["intervals"]),
                set(subject.METRIC_KEYS),
            )
            budgets = result["fa_budget_point_estimates"]
            self.assertTrue(
                budgets["recomputed_from_lossless_prediction_cache"]
            )
            self.assertEqual(
                set(budgets["points"]),
                set(evaluator.BUDGET_KEYS),
            )
        hierarchical = payload[
            "hierarchical_seed_image_bootstrap"
        ]
        self.assertTrue(hierarchical["descriptive_only"])
        self.assertEqual(
            set(hierarchical["policies"]),
            set(subject.SELECTION_ROLES),
        )
        primary = hierarchical["policies"][
            subject.PRIMARY_SELECTION_ROLE
        ]
        self.assertEqual(primary["unit"], "seed_then_paired_image")
        self.assertEqual(primary["seed_count"], 2)
        self.assertTrue(primary["shared_b_d_seed_indices"])
        self.assertTrue(primary["shared_b_d_image_indices"])
        self.assertFalse(primary["establishes_gate_m_train"])

    def test_missing_cache_after_manifest_is_pending(self) -> None:
        altered = copy.deepcopy(self.manifest)
        missing = self.root / "missing-cache.cache.json"
        altered["results"][0]["prediction_cache"][
            "metadata_path"
        ] = str(missing.resolve())
        for group in altered["paired_checkpoint_groups"]:
            if (
                group["trajectory_seed"]
                == altered["results"][0]["trajectory_seed"]
                and group["selection_role"]
                == altered["results"][0]["selection_role"]
            ):
                group["arm_b_cache_metadata_path"] = str(
                    missing.resolve()
                )
        path = self.root / "manifest-with-missing-cache.json"
        path.write_bytes(evaluator._manifest_json_bytes(altered))
        payload = subject.analyze(manifest_path=path)
        self.assertEqual(payload["status"], "pending")
        self.assertTrue(payload["missing_artifacts"])
        self.assertIsNone(payload["engineering_paired_route_met"])
        self.assertNotIn(
            "per_seed_checkpoint_policy_results",
            payload,
        )

    def test_checkpoint_epoch_bounds_are_one_through_800(self) -> None:
        epoch_zero = copy.deepcopy(self.manifest)
        epoch_zero["results"][0]["checkpoint_epoch"] = 0
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "checkpoint epoch is invalid",
        ):
            subject._validate_manifest(epoch_zero)

        epoch_800 = copy.deepcopy(self.manifest)
        first = epoch_800["results"][0]
        first["checkpoint_epoch"] = evaluator.EXPECTED_EPOCHS
        cache_binding = first["prediction_cache"]
        engineering_identity = cache_binding[
            "engineering_request_identity"
        ]
        engineering_identity["checkpoint_epoch"] = (
            evaluator.EXPECTED_EPOCHS
        )
        engineering_core = {
            name: value
            for name, value in engineering_identity.items()
            if name
            not in {
                "engineering_request_identity_sha256",
                "collector_evaluator_sha256",
                "collector_evaluator_sha256_derivation",
            }
        }
        request_sha256 = evaluator._canonical_digest(engineering_core)
        derivation = {
            "schema": evaluator.CACHE_EVALUATOR_DERIVATION_SCHEMA,
            "algorithm": "sha256_of_canonical_json",
            "adapter_source_sha256": EVALUATOR_SHA,
            "engineering_request_identity_sha256": request_sha256,
        }
        collector_evaluator_sha256 = evaluator._canonical_digest(
            derivation
        )
        engineering_identity.update(
            {
                "engineering_request_identity_sha256": request_sha256,
                "collector_evaluator_sha256": (
                    collector_evaluator_sha256
                ),
                "collector_evaluator_sha256_derivation": derivation,
            }
        )
        old_identity = cache_binding["identity"]
        cache_binding["identity"] = cache_core.build_cache_identity(
            checkpoint_sha256=old_identity["checkpoint_sha256"],
            dataset_sha256=old_identity["dataset_sha256"],
            evaluator_sha256=collector_evaluator_sha256,
            mode=old_identity["mode"]["name"],
            normalization_sha256=old_identity["normalization_sha256"],
            source_lock_sha256=old_identity["source_lock_sha256"],
            validation_ids_sha256=old_identity["validation_ids_sha256"],
            validation_count=old_identity["validation_count"],
            match_radius=old_identity["evaluation_contract"][
                "match_radius"
            ],
            tiny_area=old_identity["evaluation_contract"]["tiny_area"],
        )
        results, _ = subject._validate_manifest(epoch_800)
        key = (
            first["trajectory_seed"],
            first["arm"],
            first["selection_role"],
        )
        self.assertEqual(
            results[key]["checkpoint_epoch"],
            evaluator.EXPECTED_EPOCHS,
        )

    def test_manifest_requires_canonical_result_and_group_order(self) -> None:
        reversed_results = copy.deepcopy(self.manifest)
        reversed_results["results"].reverse()
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "canonical result order",
        ):
            subject._validate_manifest(reversed_results)

        reversed_groups = copy.deepcopy(self.manifest)
        reversed_groups["paired_checkpoint_groups"].reverse()
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "canonical paired-group order",
        ):
            subject._validate_manifest(reversed_groups)

    def test_manifest_rejects_gpu_policy_or_arm_assignment_drift(self) -> None:
        cpu_policy = copy.deepcopy(self.manifest)
        cpu_policy["formal_gpu_binding_policy"][
            "cpu_results_accepted"
        ] = True
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "formal GPU binding policy",
        ):
            subject._validate_manifest(cpu_policy)

        wrong_arm = copy.deepcopy(self.manifest)
        wrong_arm["results"][0]["execution_device_assignment"] = (
            recorded_assignment(replication_core.ARM_D)
        )
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "physical_gpu_index",
        ):
            subject._validate_manifest(wrong_arm)

        boolean_count = copy.deepcopy(self.manifest)
        boolean_count["results"][0]["execution_device_assignment"][
            "visible_cuda_device_count"
        ] = True
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "must be an integer",
        ):
            subject._validate_manifest(boolean_count)

    def test_manifest_rejects_cache_request_identity_tamper(self) -> None:
        altered = copy.deepcopy(self.manifest)
        altered["results"][0]["prediction_cache"][
            "engineering_request_identity"
        ]["child_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "engineering request SHA-256",
        ):
            subject._validate_manifest(altered)

    def test_missing_checkpoint_local_result_is_pending(self) -> None:
        altered = copy.deepcopy(self.manifest)
        missing = self.root / "missing-checkpoint-local-result.json"
        altered["results"][0]["result_path"] = str(missing.resolve())
        path = self.root / "manifest-with-missing-result.json"
        path.write_bytes(evaluator._manifest_json_bytes(altered))
        payload = subject.analyze(manifest_path=path)
        self.assertEqual(payload["status"], "pending")
        self.assertTrue(
            any(
                record["path"] == str(missing.resolve())
                for record in payload["missing_artifacts"]
            )
        )
        self.assertNotIn(
            "per_seed_checkpoint_policy_results",
            payload,
        )

    def test_checkpoint_local_result_hash_tamper_is_invalid(self) -> None:
        altered = copy.deepcopy(self.manifest)
        altered["results"][0]["result_sha256"] = "0" * 64
        path = self.root / "manifest-with-result-hash-tamper.json"
        path.write_bytes(evaluator._manifest_json_bytes(altered))
        payload = subject.analyze(manifest_path=path)
        self.assertEqual(payload["status"], "invalid")
        self.assertRegex(
            payload["errors"][0],
            "checkpoint-local result SHA-256",
        )

    def test_checkpoint_local_result_content_binding_is_enforced(
        self,
    ) -> None:
        altered = copy.deepcopy(self.manifest)
        original_path = Path(altered["results"][0]["result_path"])
        payload = evaluator._load_result_object(original_path)
        payload["source_checkpoint_identity"]["checkpoint_epoch"] = 18
        mismatched_path = self.root / "identity-mismatched-result.json"
        mismatched_path.write_bytes(
            evaluator._result_json_bytes(payload)
        )
        altered["results"][0]["result_path"] = str(
            mismatched_path.resolve()
        )
        altered["results"][0]["result_sha256"] = cache_core.sha256_file(
            mismatched_path
        )
        path = self.root / "manifest-with-result-identity-mismatch.json"
        path.write_bytes(evaluator._manifest_json_bytes(altered))
        screen = subject.analyze(manifest_path=path)
        self.assertEqual(screen["status"], "invalid")
        self.assertRegex(
            screen["errors"][0],
            "identity checkpoint_epoch",
        )

    def test_cache_pair_rejects_ordered_id_or_target_mismatch(self) -> None:
        mismatch_root = self.root / "mismatch"
        _, b_binding, _ = make_cache(
            mismatch_root,
            label="b",
            good=False,
        )
        _, reordered_binding, _ = make_cache(
            mismatch_root,
            label="d-reordered",
            good=True,
            ordered_ids=tuple(reversed(IMAGE_IDS)),
        )
        b_result = {
            "checkpoint_sha256": digest("checkpoint-b"),
            "prediction_cache": b_binding,
        }
        d_result = {
            "checkpoint_sha256": digest("checkpoint-d-reordered"),
            "prediction_cache": reordered_binding,
        }
        manifest = {
            "source_lock_sha256": SOURCE_LOCK_SHA,
            "validation_ids_sha256": VALIDATION_IDS_SHA,
            "evaluation_source_binding": copy.deepcopy(
                EVALUATION_SOURCE_BINDING
            ),
        }
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "ordered image IDs",
        ):
            subject._load_caches(
                manifest,
                {
                    (3407, replication_core.ARM_B, "primary"): b_result,
                    (3407, replication_core.ARM_D, "primary"): d_result,
                },
            )

        _, shifted_binding, _ = make_cache(
            mismatch_root,
            label="d-shifted-target",
            good=True,
            shifted_first_target=True,
        )
        shifted_result = {
            "checkpoint_sha256": digest(
                "checkpoint-d-shifted-target"
            ),
            "prediction_cache": shifted_binding,
        }
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "targets differ",
        ):
            subject._load_caches(
                manifest,
                {
                    (3407, replication_core.ARM_B, "primary"): b_result,
                    (
                        3407,
                        replication_core.ARM_D,
                        "primary",
                    ): shifted_result,
                },
            )

    def test_cache_binding_hash_tamper_is_rejected(self) -> None:
        results, _ = subject._validate_manifest(self.manifest)
        altered = copy.deepcopy(results)
        first_key = sorted(altered)[0]
        altered[first_key]["prediction_cache"][
            "metadata_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            subject.EngineeringPairedScreenError,
            "metadata SHA-256",
        ):
            subject._load_caches(
                {
                    "source_lock_sha256": SOURCE_LOCK_SHA,
                    "validation_ids_sha256": VALIDATION_IDS_SHA,
                    "evaluation_source_binding": copy.deepcopy(
                        EVALUATION_SOURCE_BINDING
                    ),
                },
                altered,
            )

    def test_miou_route_uses_strict_superiority_and_zero_margins(
        self,
    ) -> None:
        intervals = {
            name: {"lower": 0.0, "upper": 0.0}
            for name in subject.METRIC_KEYS
        }
        route = subject._miou_route(intervals)
        self.assertFalse(route["met"])
        self.assertFalse(route["criteria"]["miou_superior_delta_0"])
        intervals["miou"]["lower"] = 1e-12
        route = subject._miou_route(intervals)
        self.assertTrue(route["met"])
        intervals["fa"]["upper"] = 1e-12
        self.assertFalse(subject._miou_route(intervals)["met"])

    def test_write_once_refuses_existing_output(self) -> None:
        output = self.root / "screen-output.json"
        payload = {"schema": subject.SCHEMA, "status": "complete"}
        subject.write_once(output, payload)
        with self.assertRaises(FileExistsError):
            subject.write_once(output, payload)


if __name__ == "__main__":
    unittest.main()
