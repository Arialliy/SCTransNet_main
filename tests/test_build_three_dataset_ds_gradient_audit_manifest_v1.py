"""CPU tests for the train-only deep-supervision gradient-audit manifest."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter

import numpy as np
import torch

from analysis import build_three_dataset_ds_gradient_audit_manifest_v1 as subject


def _candidate(
    source_number: int,
    epoch: int,
    *,
    stratum: str = "tiny_positive",
    dataset_name: str = "NUAA-SIRST",
) -> subject.AuditCandidate:
    source_id = f"source_{source_number:03d}"
    digest = subject.stable_digest(
        subject.AUDIT_NAMESPACE,
        42,
        dataset_name,
        stratum,
        source_id,
        epoch,
    )
    return subject.AuditCandidate(
        dataset_name=dataset_name,
        dataset_index=source_number,
        source_id=source_id,
        namespaced_source_id=f"{dataset_name}::{source_id}",
        epoch=epoch,
        epoch_rank=epoch - 1,
        epoch_selection_sha256=subject.stable_digest(
            subject.AUDIT_NAMESPACE,
            42,
            dataset_name,
            epoch,
        ),
        candidate_selection_sha256=digest,
        augmentation_seed=source_number * 10_000 + epoch,
        transform_plan={
            "augmentation_seed": source_number * 10_000 + epoch,
            "crop_top": 0,
            "crop_left": 0,
            "crop_size": 256,
            "padded_height": 256,
            "padded_width": 256,
            "crop_attempts": 1,
            "flip_axis0": False,
            "flip_axis1": False,
            "transpose": False,
        },
        original_height=256,
        original_width=256,
        source_component_count=1,
        source_tiny_component_count=1 if stratum == "tiny_positive" else 0,
        source_normal_component_count=1 if stratum == "normal_positive" else 0,
        stratum=stratum,
        mixed_tiny=False,
        intersected_component_count=0 if stratum == "background_only" else 1,
        intersected_tiny_component_count=1 if stratum == "tiny_positive" else 0,
        intersected_normal_component_count=(
            1 if stratum == "normal_positive" else 0
        ),
        intersected_component_areas=(
            ()
            if stratum == "background_only"
            else ((4,) if stratum == "tiny_positive" else (10,))
        ),
    )


class StableSelectionAndTensorHashTests(unittest.TestCase):
    def test_stable_digest_matches_analyzer_contract_exactly(self) -> None:
        parts = [
            subject.AUDIT_NAMESPACE,
            42,
            "NUDT-SIRST",
            317,
        ]
        expected = hashlib.sha256(
            json.dumps(
                parts,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(subject.stable_digest(*parts), expected)

    def test_epoch_one_to_1000_hash_ranking_retains_exactly_first_32(self) -> None:
        dataset_name = "IRSTD-1K"
        observed = subject.ranked_candidate_epochs(dataset_name)
        expected = sorted(
            (
                subject.stable_digest(
                    subject.AUDIT_NAMESPACE,
                    42,
                    dataset_name,
                    epoch,
                ),
                epoch,
            )
            for epoch in range(1, 1001)
        )[:32]
        self.assertEqual(len(observed), 32)
        self.assertEqual(
            [(entry["selection_sha256"], entry["epoch"]) for entry in observed],
            expected,
        )
        self.assertEqual([entry["rank"] for entry in observed], list(range(32)))
        self.assertEqual(len({entry["epoch"] for entry in observed}), 32)

    def test_tensor_hash_is_raw_contiguous_uint8_view_for_chw_and_bchw(self) -> None:
        chw = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).transpose(1, 2)
        expected_chw = hashlib.sha256(
            chw.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(subject.tensor_sha256(chw), expected_chw)
        bchw = torch.stack((chw, chw + 1.0), dim=0)
        expected_bchw = hashlib.sha256(
            bchw.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        self.assertEqual(subject.tensor_sha256(bchw), expected_bchw)
        self.assertNotEqual(subject.tensor_sha256(chw), subject.tensor_sha256(bchw))


class OriginalGroundTruthStratificationTests(unittest.TestCase):
    @staticmethod
    def _mask() -> np.ndarray:
        mask = np.zeros((20, 20), dtype=np.bool_)
        # Diagonal contact is one 8-connected tiny component of area two.
        mask[1, 1] = True
        mask[2, 2] = True
        # A separate normal component with area ten.
        mask[10, 10:20] = True
        return mask

    def test_components_are_eight_connected_and_area_is_exact(self) -> None:
        components = subject.connected_components_8(self._mask())
        self.assertEqual([component.area for component in components], [2, 10])
        self.assertEqual(components[0].pixels, frozenset({(1, 1), (2, 2)}))

    def test_background_tiny_normal_and_normal_precedence(self) -> None:
        components = subject.connected_components_8(self._mask())
        background = subject.classify_crop(
            components,
            crop_top=5,
            crop_left=5,
            crop_size=2,
        )
        self.assertEqual(background["stratum"], "background_only")
        self.assertFalse(background["mixed_tiny"])

        tiny = subject.classify_crop(
            components,
            crop_top=0,
            crop_left=0,
            crop_size=4,
        )
        self.assertEqual(tiny["stratum"], "tiny_positive")
        self.assertEqual(tiny["intersected_component_areas"], [2])

        normal = subject.classify_crop(
            components,
            crop_top=9,
            crop_left=9,
            crop_size=11,
        )
        self.assertEqual(normal["stratum"], "normal_positive")
        self.assertFalse(normal["mixed_tiny"])

        mixed = subject.classify_crop(
            components,
            crop_top=0,
            crop_left=0,
            crop_size=20,
        )
        self.assertEqual(mixed["stratum"], "normal_positive")
        self.assertTrue(mixed["mixed_tiny"])
        self.assertEqual(mixed["intersected_tiny_component_count"], 1)
        self.assertEqual(mixed["intersected_normal_component_count"], 1)


class DiversityConstrainedSelectionTests(unittest.TestCase):
    @staticmethod
    def _pool(stratum: str = "tiny_positive") -> list[subject.AuditCandidate]:
        return [
            _candidate(source_number, epoch, stratum=stratum)
            for source_number in range(30)
            for epoch in range(1, 5)
        ]

    def test_available_stratum_is_64_records_with_cap_and_diversity(self) -> None:
        candidates = self._pool()
        first = subject.select_stratum_candidates(
            "NUAA-SIRST",
            "tiny_positive",
            candidates,
        )
        second = subject.select_stratum_candidates(
            "NUAA-SIRST",
            "tiny_positive",
            list(reversed(candidates)),
        )
        selected = first["selected_candidates"]
        self.assertEqual(len(selected), 64)
        histogram = Counter(candidate.source_id for candidate in selected)
        self.assertGreaterEqual(len(histogram), 24)
        self.assertLessEqual(max(histogram.values()), 3)
        self.assertEqual(first["selected_count"], 64)
        self.assertTrue(first["coverage_pass"])
        self.assertFalse(first["structurally_unavailable"])
        self.assertEqual(
            [candidate.identity for candidate in selected],
            [candidate.identity for candidate in second["selected_candidates"]],
        )

    def test_zero_candidate_background_is_descriptive_not_a_coverage_failure(self) -> None:
        observed = subject.select_stratum_candidates(
            "NUDT-SIRST",
            "background_only",
            [],
        )
        self.assertTrue(observed["structurally_unavailable"])
        self.assertEqual(observed["candidate_count"], 0)
        self.assertEqual(observed["selected_count"], 0)
        self.assertEqual(observed["selected_candidates"], [])
        self.assertTrue(observed["coverage_pass"])
        self.assertIn("no synthetic crop", observed["availability_reason"])

    def test_nonempty_background_and_positive_strata_still_fail_closed(self) -> None:
        insufficient_background = [
            _candidate(index, 1, stratum="background_only")
            for index in range(10)
        ]
        with self.assertRaisesRegex(
            subject.DSGradientAuditManifestError,
            "natural-availability proof is required",
        ):
            subject.select_stratum_candidates(
                "NUAA-SIRST",
                "background_only",
                insufficient_background,
            )
        with self.assertRaisesRegex(
            subject.DSGradientAuditManifestError,
            "has no formal crop candidates",
        ):
            subject.select_stratum_candidates(
                "NUAA-SIRST",
                "normal_positive",
                [],
            )

    def test_proven_natural_ceiling_21_uses_cap_four_and_balances_all_sources(
        self,
    ) -> None:
        candidates = [
            _candidate(source_number, epoch)
            for source_number in range(21)
            for epoch in range(1, 5)
        ]
        source_ids = [f"source_{source_number:03d}" for source_number in range(21)]
        proof = {
            "dataset_name": "NUAA-SIRST",
            "stratum": "tiny_positive",
            "full_epoch_range_covered_for_every_non_ruled_out_source": True,
            "distinct_matching_source_count": 21,
            "matching_source_ids": source_ids,
            "proof_sha256": "a" * 64,
        }
        observed = subject.select_stratum_candidates(
            "NUAA-SIRST",
            "tiny_positive",
            candidates,
            natural_availability_proof=proof,
        )
        histogram = Counter(
            candidate.source_id for candidate in observed["selected_candidates"]
        )
        self.assertEqual(observed["selected_count"], 64)
        self.assertEqual(observed["effective_min_distinct_sources"], 21)
        self.assertEqual(observed["effective_max_samples_per_source"], 4)
        self.assertTrue(
            observed["diversity_target_limited_by_natural_availability"]
        )
        self.assertEqual(set(histogram), set(source_ids))
        self.assertEqual(sorted(histogram.values()), [3] * 20 + [4])
        self.assertEqual(observed["repetition_count_range"], 1)

    def test_selection_rejects_duplicate_source_epoch_candidates(self) -> None:
        candidates = self._pool()
        candidates.append(candidates[0])
        with self.assertRaisesRegex(
            subject.DSGradientAuditManifestError,
            "identities are not unique",
        ):
            subject.select_stratum_candidates(
                "NUAA-SIRST",
                "tiny_positive",
                candidates,
            )


class PublicContractTests(unittest.TestCase):
    def test_schema_cli_defaults_and_formal_constants_are_frozen(self) -> None:
        self.assertEqual(
            subject.SCHEMA,
            "sctransnet_three_dataset_ds_gradient_audit_manifest_v1/v1",
        )
        self.assertEqual(subject.AUDIT_NAMESPACE, "sctransnet-ds-gradient-audit-v1")
        self.assertEqual(subject.SAMPLES_PER_AVAILABLE_STRATUM, 64)
        self.assertEqual(subject.BATCH_SIZE, 16)
        self.assertEqual(subject.BATCHES_PER_AVAILABLE_STRATUM, 4)
        self.assertEqual(subject.MAX_SAMPLES_PER_SOURCE_PER_STRATUM, 3)
        self.assertEqual(subject.MIN_DISTINCT_SOURCES_PER_AVAILABLE_STRATUM, 24)
        args = subject.parse_args([])
        self.assertEqual(args.output, subject.DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    unittest.main()
