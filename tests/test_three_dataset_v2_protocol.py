from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments import four_dataset_data_protocol_v1 as legacy
from experiments import prepare_nuaa_misc111_overlay_v2 as prepare
from experiments import three_dataset_v2_protocol as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DATASETS = REPO_ROOT / "datasets"


@pytest.fixture()
def prepared_dataset_root(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "datasets"
    for dataset_name in protocol.DATASETS:
        source = LIVE_DATASETS / dataset_name / "img_idx"
        destination = dataset_root / dataset_name / "img_idx"
        shutil.copytree(source, destination)

    nuaa = dataset_root / "NUAA-SIRST"
    (nuaa / "images").mkdir(parents=True)
    (nuaa / "masks").mkdir(parents=True)
    shutil.copyfile(
        LIVE_DATASETS / "NUAA-SIRST" / "images" / "Misc_111.png",
        nuaa / "images" / "Misc_111.png",
    )
    shutil.copyfile(
        LIVE_DATASETS / "NUAA-SIRST" / "masks" / "Misc_111.png",
        nuaa / "masks" / "Misc_111.png",
    )
    corrected_source = tmp_path / "verified_corrected_Misc_111.png"
    shutil.copyfile(
        LIVE_DATASETS
        / "NUAA-SIRST"
        / "masks_corrected"
        / "Misc_111.png",
        corrected_source,
    )
    prepare.prepare_overlay(
        dataset_root=dataset_root,
        corrected_source=corrected_source,
    )
    return dataset_root, corrected_source


def test_scope_is_exactly_three_datasets_and_rejects_sirst3() -> None:
    assert protocol.DATASETS == (
        "NUAA-SIRST",
        "NUDT-SIRST",
        "IRSTD-1K",
    )
    with pytest.raises(protocol.ThreeDatasetV2ProtocolError):
        protocol.require_dataset("SIRST3")
    with pytest.raises(protocol.ThreeDatasetV2ProtocolError):
        protocol.index_path(LIVE_DATASETS, "SIRST3", "train")


@pytest.mark.parametrize("dataset_name", protocol.DATASETS)
@pytest.mark.parametrize("split", protocol.SPLITS)
def test_live_img_idx_is_bound_by_path_count_bytes_and_order(
    dataset_name: str, split: str
) -> None:
    expected = protocol.EXPECTED_SPLITS[dataset_name][split]
    path = protocol.index_path(LIVE_DATASETS, dataset_name, split)
    identifiers = protocol.load_index(LIVE_DATASETS, dataset_name, split)
    assert path.relative_to(LIVE_DATASETS).as_posix() == expected[
        "index_relpath"
    ]
    assert len(identifiers) == expected["count"]
    assert protocol.sha256_file(path) == expected["file_sha256"]
    assert protocol.ordered_ids_sha256(identifiers) == expected[
        "ordered_ids_sha256"
    ]


def test_changed_index_bytes_are_rejected(
    prepared_dataset_root: tuple[Path, Path]
) -> None:
    dataset_root, _ = prepared_dataset_root
    path = protocol.index_path(dataset_root, "NUDT-SIRST", "train")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(
        protocol.ThreeDatasetV2ProtocolError, match="file SHA-256 mismatch"
    ):
        protocol.load_index(dataset_root, "NUDT-SIRST", "train")


def test_overlay_creation_is_internal_idempotent_and_preserves_raw_mask(
    prepared_dataset_root: tuple[Path, Path]
) -> None:
    dataset_root, corrected_source = prepared_dataset_root
    raw = dataset_root / protocol.NUAA_MISC111_PATHS["raw_mask_relpath"]
    corrected = (
        dataset_root
        / protocol.NUAA_MISC111_PATHS["corrected_mask_relpath"]
    )
    raw_before = raw.read_bytes()
    assert corrected.is_file()
    assert corrected.parent.name == "masks_corrected"

    second = prepare.prepare_overlay(
        dataset_root=dataset_root,
        corrected_source=corrected_source,
    )
    assert second["action"] == "verified_existing"
    assert second["raw_mask_preserved"] is True
    assert raw.read_bytes() == raw_before

    overlay = protocol.validate_nuaa_misc111_overlay(dataset_root)
    assert overlay["image_sha256"] == protocol.EXPECTED_NUAA_MISC111[
        "image_sha256"
    ]
    assert overlay["raw_mask_sha256"] == protocol.EXPECTED_NUAA_MISC111[
        "raw_mask_sha256"
    ]
    assert overlay[
        "corrected_mask_sha256"
    ] == protocol.EXPECTED_NUAA_MISC111["corrected_mask_sha256"]
    assert overlay["raw_mask_preserved"] is True


def test_resolver_uses_only_nuaa_internal_overlay(
    prepared_dataset_root: tuple[Path, Path]
) -> None:
    dataset_root, _ = prepared_dataset_root
    sample = protocol.resolve_sample(
        dataset_root,
        "NUAA-SIRST",
        "Misc_111",
        split="test",
    )
    assert sample.correction_applied is True
    assert sample.mask_path == (
        dataset_root
        / "NUAA-SIRST"
        / "masks_corrected"
        / "Misc_111.png"
    ).resolve()
    assert sample.raw_mask_path != sample.mask_path
    pair = protocol.validate_sample_pair(sample, include_hashes=True)
    assert pair["image_size_width_height"] == [325, 220]
    assert pair["raw_mask_size_width_height"] == [592, 400]
    assert pair["effective_mask_size_width_height"] == [325, 220]


def test_resolver_fast_path_requires_complete_frozen_split_identity(
    prepared_dataset_root: tuple[Path, Path]
) -> None:
    dataset_root, _ = prepared_dataset_root
    test_ids = protocol.load_index(
        dataset_root, "NUAA-SIRST", "test"
    )
    with pytest.raises(
        protocol.ThreeDatasetV2ProtocolError,
        match="frozen verified ID set",
    ):
        protocol.resolve_sample(
            dataset_root,
            "NUAA-SIRST",
            "Misc_111",
            split="test",
            known_ids=set(test_ids),
        )
    with pytest.raises(
        protocol.ThreeDatasetV2ProtocolError,
        match="known_ids count differs",
    ):
        protocol.resolve_sample(
            dataset_root,
            "NUAA-SIRST",
            "Misc_111",
            split="test",
            known_ids=frozenset({"Misc_111"}),
        )


def test_resolver_fast_path_does_not_reload_index(
    prepared_dataset_root: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root, _ = prepared_dataset_root
    frozen_ids = frozenset(
        protocol.load_index(dataset_root, "NUAA-SIRST", "test")
    )

    def forbidden_reload(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("hot-path resolve_sample reloaded img_idx")

    monkeypatch.setattr(protocol, "load_index", forbidden_reload)
    sample = protocol.resolve_sample(
        dataset_root,
        "NUAA-SIRST",
        "Misc_111",
        split="test",
        known_ids=frozen_ids,
    )
    assert sample.mask_path.parent.name == "masks_corrected"


def test_manifest_exposes_roles_and_contains_no_out_of_scope_paths(
    prepared_dataset_root: tuple[Path, Path], tmp_path: Path
) -> None:
    dataset_root, _ = prepared_dataset_root
    output = tmp_path / "three_dataset_protocol.json"
    payload = protocol.build_protocol_manifest(
        dataset_root=dataset_root,
        output_path=output,
    )
    assert payload["dataset_order"] == list(protocol.DATASETS)
    assert payload["split_roles"]["train"] == {
        "role": "optimization_and_train_statistics",
        "model_optimization": True,
        "train_statistics": True,
        "checkpoint_selection": False,
        "formal_evaluation": False,
    }
    assert payload["split_roles"]["test"] == {
        "role": "checkpoint_selection_and_formal_evaluation",
        "model_optimization": False,
        "train_statistics": False,
        "checkpoint_selection": True,
        "formal_evaluation": True,
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "SIRST3" not in serialized
    assert "masks_corrected/Misc_111.png" in serialized

    assert protocol.load_protocol_manifest(
        output, dataset_root=dataset_root
    ) == payload
    assert protocol.load_frozen_index(
        dataset_root, "IRSTD-1K", "test", output
    ) == protocol.load_index(dataset_root, "IRSTD-1K", "test")


def test_manifest_tampering_is_rejected(
    prepared_dataset_root: tuple[Path, Path]
) -> None:
    dataset_root, _ = prepared_dataset_root
    payload = protocol.build_protocol_manifest(dataset_root=dataset_root)
    payload["datasets"]["NUAA-SIRST"]["splits"]["train"]["ids"][
        0
    ] = "tampered"
    with pytest.raises(
        protocol.ThreeDatasetV2ProtocolError,
        match="differs from the frozen live data contract",
    ):
        protocol.validate_protocol_manifest(
            payload, dataset_root=dataset_root
        )


def test_seed_and_transform_plan_remain_legacy_compatible() -> None:
    components = (
        protocol.PROTOCOL_SEED,
        "NUAA-SIRST",
        17,
        "NUAA-SIRST::Misc_1",
    )
    assert protocol.stable_sha256_uint64(
        *components
    ) == legacy.stable_sha256_uint64(*components)
    assert protocol.dataloader_seed(
        "IRSTD-1K"
    ) == legacy.dataloader_seed("IRSTD-1K")

    callback = lambda top, left, size: top <= 20 and left <= 30
    new_plan = protocol.derive_stateless_transform_plan(
        protocol_seed=42,
        dataset_name="NUDT-SIRST",
        epoch=9,
        namespaced_id="NUDT-SIRST::000001",
        image_height=300,
        image_width=340,
        has_positive_in_crop=callback,
    )
    old_plan = legacy.derive_stateless_transform_plan(
        protocol_seed=42,
        dataset_name="NUDT-SIRST",
        epoch=9,
        namespaced_id="NUDT-SIRST::000001",
        image_height=300,
        image_width=340,
        has_positive_in_crop=callback,
    )
    assert new_plan == protocol.StatelessTransformPlan(**old_plan.__dict__)


def test_normalization_is_limited_to_three_frozen_entries() -> None:
    assert set(protocol.LEGACY_NORMALIZATION) == set(protocol.DATASETS)
    assert protocol.get_legacy_normalization("NUDT-SIRST") == {
        "mean": 107.80905151367188,
        "std": 33.02274703979492,
    }
    with pytest.raises(protocol.ThreeDatasetV2ProtocolError):
        protocol.get_legacy_normalization("SIRST3")
