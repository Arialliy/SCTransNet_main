from __future__ import annotations

import json
import math

import numpy as np
import pytest

from analysis import collect_final_model_validation_statistics as subject
from experiments.evaluate_pd_fa_sweep import ValidationMetrics


def _image_ids() -> tuple[str, ...]:
    return (
        "synthetic/one",
        "synthetic/two",
        *tuple(
            f"synthetic/padding_{index:03d}"
            for index in range(
                subject.EXPECTED_VALIDATION_COUNT - 2
            )
        ),
    )


def _identity(mode: str = "full", *, evaluator: str = "c") -> dict[str, object]:
    return subject.build_cache_identity(
        checkpoint_sha256="a" * 64,
        dataset_sha256="b" * 64,
        evaluator_sha256=evaluator * 64,
        normalization_sha256="d" * 64,
        source_lock_sha256="e" * 64,
        validation_ids_sha256=subject.validation_identifier_sha256(
            _image_ids()
        ),
        validation_count=subject.EXPECTED_VALIDATION_COUNT,
        match_radius=3.0,
        tiny_area=9,
        mode=mode,
    )


def _records(*, probability_delta: float = 0.0) -> tuple[subject.PredictionRecord, ...]:
    first_probability = np.asarray(
        [
            [0.01, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.90, 0.90, 0.01, 0.01],
            [0.01, 0.90, 0.90, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.01],
            [0.80, 0.01, 0.01, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    first_probability = np.clip(
        first_probability + np.float32(probability_delta),
        0.0,
        1.0,
    ).astype(np.float32)
    first_target = np.zeros((5, 5), dtype=np.uint8)
    first_target[1:3, 1:3] = 1

    second_probability = np.asarray(
        [
            [0.01, 0.01, 0.01, 0.01],
            [0.01, 0.70, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01],
        ],
        dtype=np.float32,
    )
    second_probability = np.clip(
        second_probability + np.float32(probability_delta),
        0.0,
        1.0,
    ).astype(np.float32)
    second_target = np.zeros((3, 4), dtype=np.uint8)
    primary = (
        subject.PredictionRecord(
            image_id="synthetic/one",
            probability=first_probability,
            target=first_target,
            loss=0.125,
        ),
        subject.PredictionRecord(
            image_id="synthetic/two",
            probability=second_probability,
            target=second_target,
            loss=0.375,
        ),
    )
    padding = tuple(
        subject.PredictionRecord(
            image_id=image_id,
            probability=np.zeros((1, 1), dtype=np.float32),
            target=np.zeros((1, 1), dtype=np.uint8),
            loss=0.0,
        )
        for image_id in _image_ids()[2:]
    )
    return (*primary, *padding)


def _cache(
    mode: str = "full",
    *,
    evaluator: str = "c",
    probability_delta: float = 0.0,
) -> subject.PredictionCache:
    return subject.create_prediction_cache(
        _records(probability_delta=probability_delta),
        identity=_identity(mode, evaluator=evaluator),
        match_radius=3.0,
        tiny_area=9,
    )


def _assert_metric_equal(actual: object, expected: object) -> None:
    if isinstance(expected, float):
        if math.isnan(expected):
            assert isinstance(actual, float) and math.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
    else:
        assert actual == expected


def test_identity_key_binds_checkpoint_dataset_evaluator_and_mode() -> None:
    full = _identity("full")
    modes = [
        _identity("all_off"),
        *[_identity(f"level_{level}_off") for level in range(1, 5)],
    ]

    assert len({full["cache_key_sha256"], *(item["cache_key_sha256"] for item in modes)}) == 6
    assert all(
        item["compatibility_sha256"] == full["compatibility_sha256"]
        for item in modes
    )
    assert modes[0]["mode"]["knockout_level_indices_zero_based"] == [0, 1, 2, 3]
    for level, item in enumerate(modes[1:]):
        assert item["mode"]["knockout_level_indices_zero_based"] == [level]

    changed_evaluator = _identity("full", evaluator="e")
    assert changed_evaluator["cache_key_sha256"] != full["cache_key_sha256"]
    assert changed_evaluator["compatibility_sha256"] != full["compatibility_sha256"]
    for field, digest in (
        ("checkpoint_sha256", "e" * 64),
        ("dataset_sha256", "f" * 64),
        ("evaluator_sha256", "0" * 64),
    ):
        arguments = {
            "checkpoint_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "evaluator_sha256": "c" * 64,
            "normalization_sha256": "d" * 64,
            "source_lock_sha256": "e" * 64,
            "validation_ids_sha256": (
                subject.validation_identifier_sha256(_image_ids())
            ),
            "validation_count": subject.EXPECTED_VALIDATION_COUNT,
            "match_radius": 3.0,
            "tiny_area": 9,
            "mode": "full",
        }
        arguments[field] = digest
        changed = subject.build_cache_identity(**arguments)
        assert changed["cache_key_sha256"] != full["cache_key_sha256"]
        assert changed["compatibility_sha256"] != full["compatibility_sha256"]
    with pytest.raises(ValueError, match="mode must"):
        _identity("level_0_off")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        subject.build_cache_identity(
            checkpoint_sha256="A" * 64,
            dataset_sha256="b" * 64,
            evaluator_sha256="c" * 64,
            normalization_sha256="d" * 64,
            source_lock_sha256="e" * 64,
            validation_ids_sha256=(
                subject.validation_identifier_sha256(_image_ids())
            ),
            validation_count=subject.EXPECTED_VALIDATION_COUNT,
            match_radius=3.0,
            tiny_area=9,
            mode="full",
        )


def test_identity_key_binds_normalization_source_split_and_metric_contract() -> None:
    baseline = _identity()
    arguments = {
        "checkpoint_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "evaluator_sha256": "c" * 64,
        "normalization_sha256": "d" * 64,
        "source_lock_sha256": "e" * 64,
        "validation_ids_sha256": (
            subject.validation_identifier_sha256(_image_ids())
        ),
        "validation_count": subject.EXPECTED_VALIDATION_COUNT,
        "match_radius": 3.0,
        "tiny_area": 9,
        "mode": "full",
    }
    changes = (
        ("normalization_sha256", "f" * 64),
        ("source_lock_sha256", "0" * 64),
        ("validation_ids_sha256", "1" * 64),
        ("match_radius", 4.0),
        ("tiny_area", 10),
    )
    for field, value in changes:
        changed_arguments = dict(arguments)
        changed_arguments[field] = value
        changed = subject.build_cache_identity(**changed_arguments)
        assert changed["cache_key_sha256"] != baseline["cache_key_sha256"]
        assert (
            changed["compatibility_sha256"]
            != baseline["compatibility_sha256"]
        )
    with pytest.raises(ValueError, match="validation_count must equal"):
        subject.build_cache_identity(
            **{**arguments, "validation_count": 132}
        )


def test_round_trip_is_variable_shape_write_once_and_internal_only(tmp_path) -> None:
    cache = _cache()
    metadata_path = subject.write_prediction_cache(cache, tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    loaded = subject.load_prediction_cache(
        metadata_path,
        expected_identity=cache.identity,
    )

    assert metadata["data_scope"] == "internal_validation"
    assert metadata["official_test_accessed"] is False
    assert metadata["arrays"]["allow_pickle"] is False
    assert metadata["write_once"] is True
    assert loaded.identity == cache.identity
    assert loaded.content_sha256 == cache.content_sha256
    assert len(loaded.records) == subject.EXPECTED_VALIDATION_COUNT
    assert [record.probability.shape for record in loaded.records[:2]] == [
        (5, 5),
        (3, 4),
    ]
    for expected, actual in zip(cache.records, loaded.records):
        assert actual.image_id == expected.image_id
        assert actual.loss == expected.loss
        np.testing.assert_array_equal(actual.probability, expected.probability)
        np.testing.assert_array_equal(actual.target, expected.target)

    with pytest.raises(FileExistsError, match="replace existing"):
        subject.write_prediction_cache(cache, tmp_path)


def test_tampered_array_payload_is_rejected(tmp_path) -> None:
    cache = _cache()
    metadata_path = subject.write_prediction_cache(cache, tmp_path)
    _, arrays_path = subject.cache_paths(tmp_path, cache.identity)
    with arrays_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="arrays SHA differs"):
        subject.load_prediction_cache(metadata_path)


def test_metadata_requires_canonical_exact_non_symlink_contract(tmp_path) -> None:
    canonical_dir = tmp_path / "noncanonical"
    metadata_path = subject.write_prediction_cache(_cache(), canonical_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        subject.load_prediction_cache(metadata_path)

    extra_dir = tmp_path / "extra"
    metadata_path = subject.write_prediction_cache(_cache(), extra_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["arrays"]["unexpected"] = True
    metadata_path.write_bytes(subject.canonical_json_bytes(metadata))
    with pytest.raises(ValueError, match="arrays binding fields differ"):
        subject.load_prediction_cache(metadata_path)

    symlink_dir = tmp_path / "symlink"
    metadata_path = subject.write_prediction_cache(_cache(), symlink_dir)
    alias = tmp_path / "cache-alias.json"
    alias.symlink_to(metadata_path)
    with pytest.raises(ValueError, match="must not be a symlink"):
        subject.load_prediction_cache(alias)


def test_failed_metadata_write_removes_only_locally_created_arrays(
    tmp_path,
    monkeypatch,
) -> None:
    cache = _cache()
    metadata_path, arrays_path = subject.cache_paths(tmp_path, cache.identity)

    def fail_metadata(*_args, **_kwargs):
        raise RuntimeError("synthetic metadata failure")

    monkeypatch.setattr(subject, "_atomic_create_bytes", fail_metadata)
    with pytest.raises(RuntimeError, match="synthetic metadata failure"):
        subject.write_prediction_cache(cache, tmp_path)
    assert not metadata_path.exists()
    assert not arrays_path.exists()

    def peer_array_wins(path, **_kwargs):
        path.write_bytes(b"peer-created-array")
        raise FileExistsError("synthetic concurrent array")

    monkeypatch.setattr(subject, "_atomic_create_arrays", peer_array_wins)
    with pytest.raises(FileExistsError, match="concurrent array"):
        subject.write_prediction_cache(cache, tmp_path)
    assert arrays_path.read_bytes() == b"peer-created-array"


def test_cached_metrics_match_formal_accumulator_and_bootstrap_resample() -> None:
    cache = _cache()
    actual = subject.recompute_metrics(cache, threshold=0.5)

    reference = ValidationMetrics(0.5, cache.match_radius, cache.tiny_area)
    for record in cache.records:
        reference.update(record.probability, record.target, float(record.loss))
    expected = reference.compute()
    for key, value in expected.items():
        _assert_metric_equal(actual[key], value)

    rows = subject.image_sufficient_statistics(cache, threshold=0.5)
    resampled = subject.aggregate_sufficient_statistics(
        rows,
        sample_indices=[0, 0, 1],
    )
    bootstrap_reference = ValidationMetrics(
        0.5,
        cache.match_radius,
        cache.tiny_area,
    )
    for index in (0, 0, 1):
        record = cache.records[index]
        bootstrap_reference.update(
            record.probability,
            record.target,
            float(record.loss),
        )
    for key, value in bootstrap_reference.compute().items():
        _assert_metric_equal(resampled[key], value)
    assert resampled["image_count"] == 3
    assert resampled["intersection"] == 8
    assert resampled["union"] == 11


def test_record_validation_prevents_probability_or_identity_aliases() -> None:
    invalid_probability = np.asarray([[0.1, 0.2]], dtype=np.float64)
    with pytest.raises(ValueError, match="probability must be FP32"):
        subject.create_prediction_cache(
            [
                subject.PredictionRecord(
                    "bad",
                    invalid_probability,
                    np.zeros_like(invalid_probability),
                )
            ],
            identity=_identity(),
        )

    changed = _identity("full", evaluator="e")
    cache = _cache()
    assert changed["cache_key_sha256"] != cache.identity["cache_key_sha256"]

    with pytest.raises(ValueError, match="complete validation set"):
        subject.create_prediction_cache(
            [_records()[0]],
            identity=_identity(),
        )
    changed_ids = list(_records())
    last = changed_ids[-1]
    changed_ids[-1] = subject.PredictionRecord(
        image_id="synthetic/unregistered",
        probability=last.probability,
        target=last.target,
        loss=last.loss,
    )
    with pytest.raises(ValueError, match="validation image IDs differ"):
        subject.create_prediction_cache(
            changed_ids,
            identity=_identity(),
        )
    with pytest.raises(ValueError, match="metric contract differs"):
        subject.create_prediction_cache(
            _records(),
            identity=_identity(),
            match_radius=4.0,
        )
    missing_loss = list(_records())
    first = missing_loss[0]
    missing_loss[0] = subject.PredictionRecord(
        image_id=first.image_id,
        probability=first.probability,
        target=first.target,
        loss=None,
    )
    with pytest.raises(ValueError, match="formal FP32 BCELoss"):
        subject.create_prediction_cache(
            missing_loss,
            identity=_identity(),
        )
    negative_loss = list(_records())
    first = negative_loss[0]
    negative_loss[0] = subject.PredictionRecord(
        image_id=first.image_id,
        probability=first.probability,
        target=first.target,
        loss=-0.1,
    )
    with pytest.raises(ValueError, match="non-negative"):
        subject.create_prediction_cache(
            negative_loss,
            identity=_identity(),
        )


def test_incremental_collector_seals_once_and_rejects_duplicate_images() -> None:
    records = _records()
    collector = subject.PredictionCacheCollector(identity=_identity())
    collector.append(
        image_id=records[0].image_id,
        probability=records[0].probability,
        target=records[0].target,
        loss=records[0].loss,
    )
    assert collector.image_count == 1
    with pytest.raises(ValueError, match="duplicate prediction image ID"):
        collector.append(
            image_id=records[0].image_id,
            probability=records[0].probability,
            target=records[0].target,
            loss=records[0].loss,
        )
    collector.append(
        image_id=records[1].image_id,
        probability=records[1].probability,
        target=records[1].target,
        loss=records[1].loss,
    )
    for record in records[2:]:
        collector.append(
            image_id=record.image_id,
            probability=record.probability,
            target=record.target,
            loss=record.loss,
        )
    cache = collector.seal()
    assert len(cache.records) == subject.EXPECTED_VALIDATION_COUNT
    with pytest.raises(RuntimeError, match="already sealed"):
        collector.seal()
    with pytest.raises(RuntimeError, match="already sealed"):
        collector.append(
            image_id="synthetic/three",
            probability=np.zeros((1, 1), dtype=np.float32),
            target=np.zeros((1, 1), dtype=np.uint8),
        )


def test_collector_rejects_metric_contract_mismatch_at_construction() -> None:
    with pytest.raises(ValueError, match="collector metric contract differs"):
        subject.PredictionCacheCollector(
            identity=_identity(),
            match_radius=4.0,
            tiny_area=9,
        )
