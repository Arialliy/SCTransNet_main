from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.nn as nn

from experiments import tpd_exact_runner as exact_runner
from experiments import tpd_group_scaled_exact_runner as scaled_runner
from experiments import train_tpd_ner_v4_qfg_v2_croa_dlr_exact as entry
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as v2


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


def parse(
    trailing: list[str] | None = None,
    *,
    initialization: str = "--fresh",
):
    return entry.parse_args(
        [
            "--variant",
            entry.QFG_DLR_VARIANT,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            initialization,
            *(trailing or []),
        ]
    )


def extension_provenance() -> dict:
    return {
        "schema": v2.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(
            entry.PARENT_CHECKPOINT_PATH.resolve()
        ),
        "parent_checkpoint_sha256": entry.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": list(v2.PARENT_STATE_DICT_PATH),
        "parent_state_key_count": v2.FORMAL_PARENT_STATE_KEY_COUNT,
        "preserved_new_state_key_count": (
            len(v2.SURVIVAL_STATE_KEYS) + len(v2.QFG_STATE_KEYS)
        ),
        "new_module_prefixes": list(v2.QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(v2.QFG_ZERO_INIT_PREFIXES),
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


def source_locks() -> dict[str, str]:
    statistics = entry.load_survival_target_statistics()
    return {
        entry.SOURCE_LOCK_KEY: "1" * 64,
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
        environment={"name": "cpu-dlr-fixture"},
    )


@pytest.fixture(scope="module")
def formal_components():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    args = parse()
    model, metadata = entry.build_selected_model(
        entry.QFG_DLR_VARIANT,
        entry.TRAINING_SEED,
    )
    optimizer = entry.build_optimizer(model)
    try:
        yield {
            "args": args,
            "model": model,
            "metadata": metadata,
            "optimizer": optimizer,
        }
    finally:
        torch.set_num_threads(previous_threads)


def test_parser_exposes_only_the_c_style_dlr_recipe():
    args = parse()
    require(entry.supported_candidate_variants() == ("qfg_dlr",))
    require(args.variant == "qfg_dlr")
    require(args.qfg_variant == v2.QFG_VARIANT)
    require(args.tss_variant == v2.TSS_CONTROL_VARIANT)
    require(args.survival_weight == 0.0)
    require(args.training_recipe == entry.TRAINING_RECIPE)
    require(args.run_tag == entry.FORMAL_RUN_TAG)
    require(args.output_root == entry.DEFAULT_OUTPUT_ROOT)
    for name, expected in (
        ("seed", 42),
        ("epochs", 800),
        ("base_lr", 1e-4),
        ("min_lr", 1e-6),
        ("warmup_epochs", 10),
        ("workers", 0),
        ("amp", False),
    ):
        require(getattr(args, name) == expected, name)

    with pytest.raises(SystemExit):
        entry.parse_args(
            [
                "--variant",
                v2.QFG_ONLY_VARIANT,
                "--device",
                "cpu",
                "--allow-cpu-smoke",
                "--fresh",
            ]
        )
    for override in (
        ["--run-tag", "changed"],
        ["--survival-weight", "0.005"],
        ["--base-lr", "0.001"],
        ["--seed", "7"],
    ):
        with pytest.raises((ValueError, SystemExit)):
            parse(override)


def test_builder_is_tensor_identical_to_the_old_qfg_only_model():
    dlr_model, dlr_metadata = entry.build_selected_model(
        entry.QFG_DLR_VARIANT,
        42,
    )
    old_model, old_metadata = v2.build_selected_model(
        v2.QFG_ONLY_VARIANT,
        42,
    )
    dlr_state = dlr_model.state_dict()
    old_state = old_model.state_dict()
    require(list(dlr_state) == list(old_state))
    for name in dlr_state:
        require(
            torch.equal(dlr_state[name], old_state[name]),
            f"initial tensor differs: {name}",
        )
    require(
        dlr_metadata["architecture_manifest"]
        == old_metadata["architecture_manifest"]
    )
    require(
        dlr_metadata["architecture_id"]
        == old_metadata["architecture_id"]
    )
    require(dlr_metadata["variant"] == entry.QFG_DLR_VARIANT)
    require(dlr_metadata["model_variant"] == v2.QFG_ONLY_VARIANT)

    value = (
        torch.arange(64 * 64, dtype=torch.float32)
        .reshape(1, 1, 64, 64)
        .div(4095.0)
    )
    dlr_model.eval()
    old_model.eval()
    with torch.inference_mode():
        dlr_outputs = dlr_model(value)
        old_outputs = old_model(value)
    require(len(dlr_outputs) == len(old_outputs) == 6)
    for index, (dlr_output, old_output) in enumerate(
        zip(dlr_outputs, old_outputs)
    ):
        require(
            torch.equal(dlr_output, old_output),
            f"initial output {index} differs",
        )


def test_optimizer_partition_is_ordered_mutually_exclusive_and_complete(
    formal_components,
):
    model = formal_components["model"]
    optimizer = formal_components["optimizer"]
    manifest = entry.optimizer_group_manifest(model)
    require(manifest["group_order"] == ["parent", "qfg", "tss"])
    require(manifest["parameter_numel"] == 10_870_228)
    require(manifest["parameter_tensor_count"] == 486)
    require(
        [record["parameter_numel"] for record in manifest["groups"]]
        == [10_854_446, 15_684, 98]
    )
    require(
        [record["parameter_tensor_count"] for record in manifest["groups"]]
        == [466, 16, 4]
    )
    grouped_names = [
        name
        for record in manifest["groups"]
        for name in record["parameter_names"]
    ]
    require(len(grouped_names) == len(set(grouped_names)))
    require(set(grouped_names) == set(dict(model.named_parameters())))
    require(
        [group["group_name"] for group in optimizer.param_groups]
        == ["parent", "qfg", "tss"]
    )
    require(
        [group["schedule_multiplier"] for group in optimizer.param_groups]
        == [0.1, 1.0, 1.0]
    )
    require(
        [group["lr"] for group in optimizer.param_groups]
        == [1e-4, 1e-4, 1e-4]
    )


def test_batchnorm_freeze_is_reapplied_without_freezing_affine_parameters(
    formal_components,
):
    model = formal_components["model"]
    model.train()
    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    require(len(batchnorm_modules) == entry.FORMAL_BATCHNORM_MODULE_COUNT)
    require(all(module.training for module in batchnorm_modules))
    count = entry.freeze_formal_batchnorm_running_stats(model)
    require(count == 26)
    require(all(not module.training for module in batchnorm_modules))
    for module in batchnorm_modules:
        for parameter in (module.weight, module.bias):
            if parameter is not None:
                require(parameter.requires_grad)
    model.train()
    require(all(module.training for module in batchnorm_modules))
    require(entry.freeze_formal_batchnorm_running_stats(model) == 26)


def test_exact_run_spec_binds_group_formula_counts_and_batchnorm_policy(
    formal_components,
):
    model = formal_components["model"]
    metadata = formal_components["metadata"]
    optimizer = formal_components["optimizer"]
    args = formal_components["args"]
    scaler = StateScaler()
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
    identity = exact_runner.build_run_identity(model, spec)
    validated = entry.require_dlr_run_identity(
        identity,
        label="DLR fixture",
    )
    determinism = validated["training_contract"]["determinism"]
    require(determinism["dlr_run_identity_schema"] == entry.DLR_RUN_IDENTITY_SCHEMA)
    require(determinism["group_lr_formula"] == scaled_runner.GROUP_LR_FORMULA)
    require(
        determinism["optimizer_recipe"]["group_parameter_numel"]
        == {"parent": 10_854_446, "qfg": 15_684, "tss": 98}
    )
    require(
        determinism["batchnorm_recipe"]["expected_module_count"] == 26
    )
    require(
        validated["training_contract"]["loss"]["survival"][
            "survival_weight"
        ]
        == 0.0
    )

    tampered = copy.deepcopy(identity)
    tampered["training_contract"]["determinism"][
        "group_lr_formula"
    ] = "forged"
    with pytest.raises(ValueError, match="determinism contract"):
        entry.require_dlr_run_identity(tampered, label="tampered")


def test_group_scaled_epoch_evidence_and_exact_resume_are_dlr_owned(tmp_path):
    args = parse()
    model, metadata = entry.build_selected_model(
        entry.QFG_DLR_VARIANT,
        42,
    )
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
    adapter = entry.EvaluatorCheckpointAdapter(
        model_metadata=metadata,
        split_hashes={"train": "a" * 64},
    )
    run_root = tmp_path / "dlr"
    runner = entry.DLRExactRunner(
        run_root,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=generator,
        spec=spec,
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=entry.STORED_VALIDATION_METRICS
        ),
        compatibility_payload_factory=adapter,
    )
    runner.startup(request)
    control = runner.next_epoch_control()
    require(control.epoch == 1)
    require(
        [group["lr"] for group in optimizer.param_groups]
        == [
            control.learning_rate * 0.1,
            control.learning_rate,
            control.learning_rate,
        ]
    )
    metrics = validation_metrics()
    runner.commit_epoch(
        {
            "variant": entry.QFG_DLR_VARIANT,
            "candidate_variant": entry.QFG_DLR_VARIANT,
            "model_variant": entry.MODEL_VARIANT,
            "qfg_variant": entry.QFG_VARIANT,
            "tss_variant": entry.TSS_VARIANT,
            "training_recipe": entry.TRAINING_RECIPE,
            "train_total_loss": 0.2,
            "train_segmentation_loss": 0.2,
            "train_survival_loss": 0.0,
            "train_survival_emb1_loss": None,
            "train_survival_emb2_loss": None,
            "train_loss": 0.2,
            "survival_weight": 0.0,
            "survival_pos_weight": 1.0,
            entry.BATCHNORM_EVENT_FIELD: 26,
            "processed_train_samples": 16,
            "epoch_seconds": 1.0,
            "skipped_singleton_batches": 0,
            **metrics,
        }
    )
    events = entry._load_complete_events(
        run_root / exact_runner.METRICS_FILENAME,
        1,
    )
    require(events[0]["optimizer_group_names"] == ["parent", "qfg", "tss"])
    require(events[0]["schedule_multipliers"] == [0.1, 1.0, 1.0])
    require(
        events[0]["group_learning_rates"]
        == [
            control.learning_rate * 0.1,
            control.learning_rate,
            control.learning_rate,
        ]
    )
    require(
        [group["lr"] for group in optimizer.param_groups]
        == [control.learning_rate] * 3
    )
    checkpoint = torch.load(
        run_root / exact_runner.LAST_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    require(checkpoint["schema"] == entry.CHECKPOINT_SCHEMA)
    require(checkpoint["variant"] == entry.QFG_DLR_VARIANT)
    require(checkpoint["model_variant"] == v2.QFG_ONLY_VARIANT)
    require(
        checkpoint["scheduler"]["kind"]
        == "identity_bound_manual_group_scaled_schedule"
    )

    resumed_model, resumed_metadata = entry.build_selected_model(
        entry.QFG_DLR_VARIANT,
        42,
    )
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
    resumed_runner = entry.DLRExactRunner(
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
        [group["lr"] for group in resumed_optimizer.param_groups]
        == [
            next_control.learning_rate * 0.1,
            next_control.learning_rate,
            next_control.learning_rate,
        ]
    )


def test_runtime_source_closure_is_independent_of_the_old_48_source_lock():
    relative = {
        str(path.relative_to(entry.REPO_ROOT))
        for path in entry.RUNTIME_SOURCE_PATHS
    }
    require(
        "experiments/train_tpd_ner_v4_qfg_v2_croa_dlr_exact.py"
        in relative
    )
    require("experiments/tpd_group_scaled_exact_runner.py" in relative)
    require(
        {
            str(path.relative_to(entry.REPO_ROOT))
            for path in v2.RUNTIME_SOURCE_PATHS
        }
        < relative
    )
    require(entry.DEFAULT_EXACT_SOURCE_LOCK_PATH != v2.DEFAULT_EXACT_SOURCE_LOCK_PATH)
    training_data_sha256 = (
        "39ce329032b7d6e70dcf16e7cd6a0624"
        "f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
    )
    locks = entry.source_lock_contract(
        training_data_sha256,
        entry.DEFAULT_EXACT_SOURCE_LOCK_PATH,
    )
    require(set(locks) == {
        entry.SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    })
    lock_payload = json.loads(
        entry.DEFAULT_EXACT_SOURCE_LOCK_PATH.read_text(encoding="utf-8")
    )
    require(lock_payload["source_count"] == 50)

    active_v2_lock = (
        entry.REPO_ROOT
        / "experiments/"
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
    )
    require(
        entry.file_sha256(active_v2_lock)
        == "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478"
    )
    v2.source_lock_contract(
        training_data_sha256,
        active_v2_lock,
        v2.DEFAULT_TARGET_STATISTICS_PATH,
    )


def test_main_dispatches_the_fixed_cpu_cli_without_touching_cuda(tmp_path):
    argv = [
        "--variant",
        entry.QFG_DLR_VARIANT,
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
    require(parsed.variant == entry.QFG_DLR_VARIANT)
    require(parsed.device == "cpu")
    require(not (tmp_path / "NUDT-SIRST").exists())
