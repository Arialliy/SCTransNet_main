from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from experiments import tpd_exact_runner as exact_runner
from experiments import tpd_group_scaled_exact_runner as scaled_runner
from experiments import tpd_training_loss
from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_exact as single_dlr,
)
from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as entry,
)


def require(condition: bool, message: str = "") -> None:
    """Assertion helper that remains active under ``python -O``."""

    if not condition:
        raise AssertionError(message or "required condition is false")


class StateScaler:
    def __init__(self) -> None:
        self.updates = 0

    def state_dict(self) -> dict[str, int]:
        return {"updates": self.updates}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.updates = int(state["updates"])


class TinyValidationSet(Dataset):
    def __init__(self, image: torch.Tensor, mask: torch.Tensor) -> None:
        self.image = image[0].detach().clone()
        self.mask = mask[0].detach().clone()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return (
            self.image,
            self.mask,
            torch.tensor([self.image.shape[-2], self.image.shape[-1]]),
            "paired-validation",
        )


def parse(
    variant: str,
    trailing: list[str] | None = None,
    *,
    initialization: str = "--fresh",
):
    return entry.parse_args(
        [
            "--variant",
            variant,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            initialization,
            *(trailing or []),
        ]
    )


def extension_provenance() -> dict:
    return {
        "schema": entry.v2.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(
            entry.PARENT_CHECKPOINT_PATH.resolve()
        ),
        "parent_checkpoint_sha256": entry.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": list(entry.v2.PARENT_STATE_DICT_PATH),
        "parent_state_key_count": entry.v2.FORMAL_PARENT_STATE_KEY_COUNT,
        "preserved_new_state_key_count": (
            len(entry.v2.SURVIVAL_STATE_KEYS)
            + len(entry.v2.QFG_STATE_KEYS)
        ),
        "new_module_prefixes": list(entry.v2.QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(entry.v2.QFG_ZERO_INIT_PREFIXES),
    }


def validation_metrics() -> dict[str, int | float]:
    count_fields = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    return {
        name: 1 if name in count_fields else 0.5
        for name in entry.STORED_VALIDATION_METRICS
    }


def fingerprints():
    return (
        {
            "train": exact_runner.OrderedFingerprint.from_values(
                "train", ("a", "b")
            ),
            "validation": exact_runner.OrderedFingerprint.from_values(
                "validation", ("c",)
            ),
        },
        {
            "train_samples": exact_runner.OrderedFingerprint.from_values(
                "train_samples", ("a:image", "b:image")
            ),
            "validation_samples": exact_runner.OrderedFingerprint.from_values(
                "validation_samples", ("c:image",)
            ),
        },
    )


def source_locks(
    paired_digest: str = "1" * 64,
) -> dict[str, str]:
    statistics = entry.load_survival_target_statistics()
    return {
        entry.SOURCE_LOCK_KEY: paired_digest,
        entry.UPSTREAM_SOURCE_LOCK_KEY: (
            entry.UPSTREAM_SOURCE_LOCK_SHA256
        ),
        "training_data": "2" * 64,
        "survival_target_statistics": statistics["sha256"],
        "parent_checkpoint": entry.PARENT_CHECKPOINT_SHA256,
    }


def make_spec(
    *,
    args,
    model,
    metadata,
    optimizer,
    scaler,
    initial_sha: str,
    initial_rng: dict,
):
    split_records, data_records = fingerprints()
    initialization_contract = (
        exact_runner.extension_parent_initialization_contract(
            extension_provenance(),
            loaded_child_model_state_sha256=initial_sha,
        )
    )
    return entry.make_exact_run_spec(
        args,
        model=model,
        model_metadata=metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=initialization_contract,
        initial_model_state_sha256=initial_sha,
        initial_rng=initial_rng,
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ).normalized(),
        source_locks=source_locks(),
        split_records=split_records,
        data_records=data_records,
        environment={"name": "cpu-paired-ramp100-fixture"},
    )


def event_fields(
    variant: str,
    *,
    epoch: int,
    survival_loss: float = 0.0,
    emb1: float | None = None,
    emb2: float | None = None,
) -> dict:
    candidate = entry.candidate_contract(variant)
    weight = entry.survival_weight_for_epoch(variant, epoch)
    segmentation = 0.2
    total = segmentation + weight * survival_loss
    return {
        "variant": variant,
        "candidate_variant": variant,
        "base_model_variant": candidate["base_model_variant"],
        "qfg_variant": entry.v2.QFG_VARIANT,
        "tss_variant": candidate["tss_variant"],
        "family_recipe": entry.FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "train_total_loss": total,
        "train_segmentation_loss": segmentation,
        "train_survival_loss": survival_loss,
        "train_survival_emb1_loss": emb1,
        "train_survival_emb2_loss": emb2,
        "train_loss": total,
        "survival_pos_weight": 1.0,
        entry.BATCHNORM_EVENT_FIELD: 26,
        "processed_train_samples": 16,
        "epoch_seconds": 1.0,
        "skipped_singleton_batches": 0,
        **validation_metrics(),
    }


def state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            state_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def test_parser_and_formal_contract_are_strictly_paired():
    control = parse(entry.QFG_DLR_VARIANT)
    treatment = parse(entry.TSS_QFG_DLR_VARIANT)
    require(
        entry.supported_candidate_variants()
        == ("qfg_dlr", "tss_qfg_dlr")
    )
    require(control.survival_weight_max == 0.0)
    require(treatment.survival_weight_max == 0.005)
    require(control.tss_variant == entry.v2.TSS_CONTROL_VARIANT)
    require(treatment.tss_variant == entry.v2.TSS_ON_VARIANT)
    require(control.run_tag == entry.FORMAL_RUN_TAGS["qfg_dlr"])
    require(
        treatment.run_tag
        == entry.FORMAL_RUN_TAGS["tss_qfg_dlr"]
    )
    contract = entry.formal_contract()
    require(
        contract["candidate_variants"]
        == ["qfg_dlr", "tss_qfg_dlr"]
    )
    require(contract["preselected_ramp_epochs"] == 100)
    require(contract["paired_epoch1_exact_zero_tss"] is True)
    for variant, override in (
        (entry.QFG_DLR_VARIANT, ["--survival-weight-max", "0.005"]),
        (
            entry.TSS_QFG_DLR_VARIANT,
            ["--survival-weight-max", "0.001"],
        ),
        (entry.QFG_DLR_VARIANT, ["--run-tag", "changed"]),
        (entry.QFG_DLR_VARIANT, ["--seed", "7"]),
    ):
        with pytest.raises((ValueError, SystemExit)):
            parse(variant, override)


@pytest.mark.parametrize(
    ("epoch", "fraction", "weight"),
    (
        (1, 0.0, 0.0),
        (2, 1.0 / 99.0, 0.005 * (1.0 / 99.0)),
        (10, 9.0 / 99.0, 0.005 * 9.0 / 99.0),
        (50, 49.0 / 99.0, 0.005 * (49.0 / 99.0)),
        (99, 98.0 / 99.0, 0.005 * (98.0 / 99.0)),
        (100, 1.0, 0.005),
        (101, 1.0, 0.005),
        (800, 1.0, 0.005),
    ),
)
def test_ramp100_boundaries_are_fixed(epoch, fraction, weight):
    require(entry.tss_ramp_fraction(epoch) == fraction)
    require(
        entry.survival_weight_for_epoch(
            entry.TSS_QFG_DLR_VARIANT,
            epoch,
        )
        == weight
    )
    require(
        entry.survival_weight_for_epoch(
            entry.QFG_DLR_VARIANT,
            epoch,
        )
        == 0.0
    )


def test_two_builders_and_parent_warm_starts_have_identical_initial_state(
    tmp_path,
):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        arms = {}
        for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
            args = parse(variant)
            model, metadata = entry.build_selected_model(variant, 42)
            plan = entry.initialization_plan(
                args,
                tmp_path / variant,
                model,
            )
            arms[variant] = (args, model, metadata, plan)

        control_model = arms[entry.QFG_DLR_VARIANT][1]
        treatment_model = arms[entry.TSS_QFG_DLR_VARIANT][1]
        control_state = control_model.state_dict()
        treatment_state = treatment_model.state_dict()
        require(list(control_state) == list(treatment_state))
        for name in control_state:
            require(
                torch.equal(control_state[name], treatment_state[name]),
                f"warm-start tensor differs: {name}",
            )
        control_sha = exact_runner.initial_model_state_sha256(control_model)
        treatment_sha = exact_runner.initial_model_state_sha256(
            treatment_model
        )
        require(control_sha == treatment_sha)
        require(
            arms[entry.QFG_DLR_VARIANT][3].initial_model_state_sha256
            == arms[entry.TSS_QFG_DLR_VARIANT][
                3
            ].initial_model_state_sha256
            == control_sha
        )
        require(len(control_state) == 568)
        require(
            sum(parameter.numel() for parameter in control_model.parameters())
            == 10_870_228
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_epoch1_zero_path_never_builds_or_reads_tss():
    target = torch.zeros(2, 1, 32, 32)
    probability = torch.full(
        (2, 1, 32, 32),
        0.25,
        requires_grad=True,
    )
    legacy_output = tuple(probability for _ in range(6))
    with mock.patch.object(
        tpd_training_loss,
        "build_survival_target",
        side_effect=AssertionError("epoch1 must not build Y16"),
    ):
        losses = [
            entry.compute_stage_loss(
                legacy_output,
                target,
                nn.BCELoss(),
                survival_weight=entry.survival_weight_for_epoch(
                    variant,
                    1,
                ),
                survival_pos_weight=1.0,
            )
            for variant in entry.SUPPORTED_CANDIDATE_VARIANTS
        ]
    require(losses[0].survival_terms == losses[1].survival_terms == ())
    require(torch.equal(losses[0].total, losses[1].total))
    for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
        evidence = entry.validate_epoch_loss_fields(
            variant,
            1,
            event_fields(variant, epoch=1),
        )
        require(evidence[entry.SURVIVAL_WEIGHT_FIELD] == 0.0)
        require(evidence[entry.SURVIVAL_ENABLED_FIELD] is False)


def test_epoch1_full_optimizer_step_and_validation_are_bitwise_equal():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        models = {}
        optimizers = {}
        for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
            model, _ = entry.build_selected_model(variant, 42)
            models[variant] = model
            optimizers[variant] = entry.build_optimizer(model)
        image = (
            torch.arange(2 * 64 * 64, dtype=torch.float32)
            .reshape(2, 1, 64, 64)
            .div(float(2 * 64 * 64))
        )
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 20:23, 25:28] = 1.0
        criterion = nn.BCELoss()
        rng = torch.get_rng_state()
        losses = {}
        for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
            torch.set_rng_state(rng)
            model = models[variant]
            optimizer = optimizers[variant]
            model.train()
            require(
                entry.freeze_formal_batchnorm_running_stats(model) == 26
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss = entry.compute_stage_loss(
                outputs,
                mask,
                criterion,
                survival_weight=entry.survival_weight_for_epoch(
                    variant,
                    1,
                ),
                survival_pos_weight=1.0,
            )
            loss.total.backward()
            optimizer.step()
            losses[variant] = loss

        control = entry.QFG_DLR_VARIANT
        treatment = entry.TSS_QFG_DLR_VARIANT
        require(torch.equal(losses[control].total, losses[treatment].total))
        require(losses[control].survival_terms == ())
        require(losses[treatment].survival_terms == ())
        for name, value in models[control].state_dict().items():
            require(
                torch.equal(value, models[treatment].state_dict()[name]),
                f"epoch1 model state differs: {name}",
            )
        require(
            state_equal(
                optimizers[control].state_dict(),
                optimizers[treatment].state_dict(),
            )
        )
        for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
            tss_parameters = {
                parameter
                for name, parameter in models[variant].named_parameters()
                if name.startswith("target_survival.")
            }
            require(not (set(optimizers[variant].state) & tss_parameters))

        val_loader = DataLoader(
            TinyValidationSet(image[:1], mask[:1]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        metrics = {}
        for variant in entry.SUPPORTED_CANDIDATE_VARIANTS:
            metrics[variant] = entry.v2.base.validate(
                models[variant],
                val_loader,
                torch.device("cpu"),
                criterion,
                entry.FORMAL_THRESHOLD,
                entry.FORMAL_MATCH_RADIUS,
                entry.FORMAL_TINY_AREA,
                False,
            )
        require(metrics[control] == metrics[treatment])
    finally:
        torch.set_num_threads(previous_threads)


def test_run_identity_binds_candidate_schedule_optimizer_and_bn():
    args = parse(entry.TSS_QFG_DLR_VARIANT)
    model, metadata = entry.build_selected_model(args.variant, 42)
    optimizer = entry.build_optimizer(model)
    scaler = StateScaler()
    initial_sha = exact_runner.initial_model_state_sha256(model)
    spec = make_spec(
        args=args,
        model=model,
        metadata=metadata,
        optimizer=optimizer,
        scaler=scaler,
        initial_sha=initial_sha,
        initial_rng=exact_runner.initial_rng_contract(),
    )
    identity = exact_runner.build_run_identity(model, spec)
    validated = entry.require_paired_run_identity(
        identity,
        label="paired fixture",
        expected_variant=args.variant,
    )
    determinism = validated["training_contract"]["determinism"]
    require(
        determinism["paired_run_identity_schema"]
        == entry.PAIRED_RUN_IDENTITY_SCHEMA
    )
    require(
        determinism["survival_weight_schedule"]["schedule_id"]
        == entry.TSS_WEIGHT_SCHEDULE_ID
    )
    require(
        determinism["optimizer_recipe"]["group_parameter_numel"]
        == {"parent": 10_854_446, "qfg": 15_684, "tss": 98}
    )
    require(
        determinism["batchnorm_recipe"]["expected_module_count"] == 26
    )
    tampered = copy.deepcopy(identity)
    tampered["training_contract"]["determinism"][
        "survival_weight_schedule"
    ]["end_epoch"] = 50
    with pytest.raises(ValueError, match="determinism contract"):
        entry.require_paired_run_identity(tampered, label="ramp50")


def test_runner_owns_epoch_weight_evidence_and_exact_resume(tmp_path):
    variant = entry.TSS_QFG_DLR_VARIANT
    args = parse(variant)
    model, metadata = entry.build_selected_model(variant, 42)
    optimizer = entry.build_optimizer(model)
    scaler = StateScaler()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    initial_sha = exact_runner.initial_model_state_sha256(model)
    initial_rng = exact_runner.initial_rng_contract()
    spec = make_spec(
        args=args,
        model=model,
        metadata=metadata,
        optimizer=optimizer,
        scaler=scaler,
        initial_sha=initial_sha,
        initial_rng=initial_rng,
    )
    request = exact_runner.InitializationRequest.extension_parent(
        extension_provenance(),
        loaded_child_model_state_sha256=initial_sha,
    )
    run_root = tmp_path / "paired-treatment"
    runner = entry.PairedRamp100ExactRunner(
        run_root,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        spec=spec,
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ),
        compatibility_payload_factory=entry.EvaluatorCheckpointAdapter(
            model_metadata=metadata,
            split_hashes={"train": "a" * 64},
        ),
    )
    runner.startup(request)
    control = runner.next_epoch_control()
    require(control.epoch == 1)
    require(control.survival_weight_effective == 0.0)
    require(control.tss_ramp_fraction == 0.0)
    require(
        [group["lr"] for group in optimizer.param_groups]
        == [
            control.learning_rate * 0.1,
            control.learning_rate,
            control.learning_rate,
        ]
    )
    with pytest.raises(
        exact_runner.ExactRunnerError,
        match="ramp-runner-owned",
    ):
        runner.commit_epoch(
            {
                **event_fields(variant, epoch=1),
                entry.SURVIVAL_WEIGHT_FIELD: 0.0,
            }
        )
    runner.commit_epoch(event_fields(variant, epoch=1))
    event = json.loads(
        (run_root / exact_runner.METRICS_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    entry.validate_epoch_event(event, variant=variant)
    require(event[entry.SURVIVAL_WEIGHT_FIELD] == 0.0)
    require(event["optimizer_group_names"] == ["parent", "qfg", "tss"])
    checkpoint = torch.load(
        run_root / exact_runner.LAST_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    require(checkpoint["schema"] == entry.CHECKPOINT_SCHEMA)
    require(checkpoint["variant"] == variant)
    require(
        checkpoint["scheduler"]["checkpoint_group_lr"]
        == "manual_cosine_lr(completed_epoch)"
    )

    resumed_model, resumed_metadata = entry.build_selected_model(variant, 42)
    resumed_optimizer = entry.build_optimizer(resumed_model)
    resumed_scaler = StateScaler()
    resumed_generator = torch.Generator(device="cpu")
    resumed_generator.manual_seed(42)
    resumed_spec = make_spec(
        args=args,
        model=resumed_model,
        metadata=resumed_metadata,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        initial_sha=initial_sha,
        initial_rng=initial_rng,
    )
    resumed_runner = entry.PairedRamp100ExactRunner(
        run_root,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scaler=resumed_scaler,
        loader_generator=resumed_generator,
        spec=resumed_spec,
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ),
        compatibility_payload_factory=entry.EvaluatorCheckpointAdapter(
            model_metadata=resumed_metadata,
            split_hashes={"train": "a" * 64},
        ),
    )
    resumed = resumed_runner.startup(
        exact_runner.InitializationRequest.exact()
    )
    require(resumed.completed_epoch == 1)
    next_control = resumed_runner.next_epoch_control()
    require(next_control.epoch == 2)
    require(
        next_control.survival_weight_effective
        == entry.TSS_MAX_WEIGHT * (1.0 / 99.0)
    )
    require(
        [group["lr"] for group in resumed_optimizer.param_groups]
        == [
            next_control.learning_rate * 0.1,
            next_control.learning_rate,
            next_control.learning_rate,
        ]
    )

    cross_args = parse(entry.QFG_DLR_VARIANT)
    cross_model, cross_metadata = entry.build_selected_model(
        cross_args.variant,
        42,
    )
    cross_optimizer = entry.build_optimizer(cross_model)
    cross_scaler = StateScaler()
    cross_generator = torch.Generator(device="cpu")
    cross_generator.manual_seed(42)
    cross_spec = make_spec(
        args=cross_args,
        model=cross_model,
        metadata=cross_metadata,
        optimizer=cross_optimizer,
        scaler=cross_scaler,
        initial_sha=initial_sha,
        initial_rng=initial_rng,
    )
    cross_runner = entry.PairedRamp100ExactRunner(
        run_root,
        model=cross_model,
        optimizer=cross_optimizer,
        scaler=cross_scaler,
        loader_generator=cross_generator,
        spec=cross_spec,
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ),
    )
    with pytest.raises(ValueError, match="variant"):
        cross_runner.startup(exact_runner.InitializationRequest.exact())


def test_event_checker_rejects_weight_fraction_lr_loss_bn_and_aux_forgery(
    tmp_path,
):
    variant = entry.TSS_QFG_DLR_VARIANT
    epoch = 2
    base_lr = exact_runner.ManualCosineSchedule(
        total_epochs=800,
        base_lr=1e-4,
        min_lr=1e-6,
        warmup_epochs=10,
    ).learning_rate(epoch)
    survival_loss = 0.7
    event = {
        "epoch": epoch,
        "learning_rate": base_lr,
        "optimizer_group_names": ["parent", "qfg", "tss"],
        "schedule_multipliers": [0.1, 1.0, 1.0],
        "group_learning_rates": [
            base_lr * 0.1,
            base_lr,
            base_lr,
        ],
        **event_fields(
            variant,
            epoch=epoch,
            survival_loss=survival_loss,
            emb1=0.3,
            emb2=0.4,
        ),
    }
    event.update(
        entry.validate_epoch_loss_fields(variant, epoch, event)
    )
    entry.validate_epoch_event(event, variant=variant)
    for field, forged in (
        (entry.SURVIVAL_WEIGHT_FIELD, 0.001),
        (entry.TSS_RAMP_FRACTION_FIELD, 0.5),
        ("group_learning_rates", [base_lr] * 3),
        ("train_total_loss", 0.2),
        (entry.BATCHNORM_EVENT_FIELD, 25),
        ("train_survival_emb1_loss", None),
    ):
        changed = copy.deepcopy(event)
        changed[field] = forged
        with pytest.raises((ValueError, TypeError)):
            entry.validate_epoch_event(changed, variant=variant)

    epoch1_base_lr = exact_runner.ManualCosineSchedule(
        total_epochs=800,
        base_lr=1e-4,
        min_lr=1e-6,
        warmup_epochs=10,
    ).learning_rate(1)
    epoch1 = {
        "epoch": 1,
        "learning_rate": epoch1_base_lr,
        "optimizer_group_names": ["parent", "qfg", "tss"],
        "schedule_multipliers": [0.1, 1.0, 1.0],
        "group_learning_rates": [
            epoch1_base_lr * 0.1,
            epoch1_base_lr,
            epoch1_base_lr,
        ],
        **event_fields(variant, epoch=1),
    }
    epoch1.update(
        entry.validate_epoch_loss_fields(variant, 1, epoch1)
    )
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps(item, sort_keys=True)
            for item in (epoch1, event)
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = entry._load_complete_events(
        metrics_path,
        2,
        variant=variant,
    )
    require(loaded == [epoch1, event])


def test_runtime_closure_and_frozen_predecessor_files_remain_distinct():
    relative = {
        str(path.relative_to(entry.REPO_ROOT))
        for path in entry.RUNTIME_SOURCE_PATHS
    }
    require(
        "experiments/"
        "train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact.py"
        in relative
    )
    require(
        {
            str(path.relative_to(entry.REPO_ROOT))
            for path in single_dlr.RUNTIME_SOURCE_PATHS
        }
        < relative
    )
    require(
        entry.DEFAULT_EXACT_SOURCE_LOCK_PATH
        != single_dlr.DEFAULT_EXACT_SOURCE_LOCK_PATH
    )
    require(
        entry.file_sha256(
            entry.REPO_ROOT
            / "experiments/"
            "train_tpd_ner_v4_qfg_v2_croa_dlr_exact.py"
        )
        == "67cee6d59740af7ccaa9d7d1abfe62fa3f0c31a728deb8d7c400a675fb7c7190"
    )
    require(
        entry.file_sha256(single_dlr.DEFAULT_EXACT_SOURCE_LOCK_PATH)
        == "048558ee8b751847bd3f27afa4376be4a08bd158e0b73cdf6872185bfd406f88"
    )
    require(
        entry.file_sha256(entry.UPSTREAM_SOURCE_LOCK_PATH)
        == entry.UPSTREAM_SOURCE_LOCK_SHA256
    )
    training_data_sha256 = (
        "39ce329032b7d6e70dcf16e7cd6a0624"
        "f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
    )
    locks = entry.source_lock_contract(
        training_data_sha256,
        entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
    )
    require(
        set(locks)
        == {
            entry.SOURCE_LOCK_KEY,
            entry.UPSTREAM_SOURCE_LOCK_KEY,
            "training_data",
            "survival_target_statistics",
            "parent_checkpoint",
        }
    )
    payload = json.loads(
        entry.DEFAULT_EXACT_SOURCE_LOCK_PATH.read_text(encoding="utf-8")
    )
    require(payload["source_count"] == 51)
    require(payload["ramp_selection"]["preselected_epochs"] == 100)


def test_main_dispatches_cpu_cli_without_starting_cuda_or_a_run(tmp_path):
    argv = [
        "--variant",
        entry.TSS_QFG_DLR_VARIANT,
        "--device",
        "cpu",
        "--allow-cpu-smoke",
        "--fresh",
        "--output-root",
        str(tmp_path),
    ]
    with (
        mock.patch.object(
            entry,
            "run_training",
            return_value=tmp_path / "not-started",
        ) as run_training,
        mock.patch.object(
            torch.cuda,
            "is_available",
            side_effect=AssertionError("CLI dispatch queried CUDA"),
        ),
    ):
        entry.main(argv)
    require(run_training.call_count == 1)
    parsed = run_training.call_args.args[0]
    require(parsed.variant == entry.TSS_QFG_DLR_VARIANT)
    require(not (tmp_path / "NUDT-SIRST").exists())
