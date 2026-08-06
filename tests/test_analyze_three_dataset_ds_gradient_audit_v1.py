from __future__ import annotations

import inspect
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from analysis import analyze_three_dataset_ds_gradient_audit_v1 as subject
from analysis import build_three_dataset_ds_gradient_audit_manifest_v1 as manifest_builder


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.weight


class _TinyMTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_1 = _Scale(0.99)
        self.embeddings_2 = _Scale(1.01)
        self.embeddings_3 = _Scale(0.98)
        self.embeddings_4 = _Scale(1.02)
        self.encoder = _Scale(0.97)
        self.reconstruct_1 = _Scale(1.03)
        self.reconstruct_2 = _Scale(0.96)
        self.reconstruct_3 = _Scale(1.04)
        self.reconstruct_4 = _Scale(0.95)


class _TinySixHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(1, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Dropout2d(p=0.25),
        )
        self.down_encoder1 = _Scale(1.01)
        self.down_encoder2 = _Scale(0.99)
        self.down_encoder3 = _Scale(1.02)
        self.down_encoder4 = _Scale(0.98)
        self.mtc = _TinyMTC()
        self.tpd_qfg = _Scale(1.01)
        self.tpd_ner = _Scale(0.99)
        self.up_decoder4 = _Scale(1.02)
        self.up_decoder3 = _Scale(0.98)
        self.up_decoder2 = _Scale(1.03)
        self.up_decoder1 = _Scale(0.97)
        self.gt_conv5 = nn.Conv2d(1, 1, 1)
        self.gt_conv4 = nn.Conv2d(1, 1, 1)
        self.gt_conv3 = nn.Conv2d(1, 1, 1)
        self.gt_conv2 = nn.Conv2d(1, 1, 1)
        self.outconv = nn.Conv2d(1, 1, 1)
        self.outc = nn.Conv2d(1, 1, 1)
        self.target_survival = nn.ModuleDict(
            {"emb1": nn.Conv2d(1, 1, 1), "emb2": nn.Conv2d(1, 1, 1)}
        )

    def forward(self, value: torch.Tensor):
        value = self.inc(value)
        for name in (
            "down_encoder1",
            "down_encoder2",
            "down_encoder3",
            "down_encoder4",
        ):
            value = getattr(self, name)(value)
        for name in (
            "embeddings_1",
            "embeddings_2",
            "embeddings_3",
            "embeddings_4",
            "encoder",
            "reconstruct_1",
            "reconstruct_2",
            "reconstruct_3",
            "reconstruct_4",
        ):
            value = getattr(self.mtc, name)(value)
        value = self.tpd_qfg(value)
        value = self.tpd_ner(value)
        for name in (
            "up_decoder4",
            "up_decoder3",
            "up_decoder2",
            "up_decoder1",
        ):
            value = getattr(self, name)(value)
        return (
            torch.sigmoid(self.gt_conv5(value)),
            torch.sigmoid(self.gt_conv4(value)),
            torch.sigmoid(self.gt_conv3(value)),
            torch.sigmoid(self.gt_conv2(value)),
            torch.sigmoid(self.outconv(value)),
            torch.sigmoid(self.outc(value)),
        )


def _batch() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = torch.linspace(-1.0, 1.0, 16 * 4 * 4).reshape(16, 1, 4, 4)
    masks = torch.zeros_like(images)
    masks[:, :, 1:3, 1:3] = 1.0
    identifiers = [f"tiny::{index}" for index in range(16)]
    return images, masks, identifiers


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_digest_and_tensor_hash_match_manifest_builder() -> None:
    assert subject.NAMESPACE == manifest_builder.AUDIT_NAMESPACE
    assert subject.stable_digest("a", 42, "b") == manifest_builder.stable_digest(
        "a", 42, "b"
    )
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert subject.tensor_sha256(tensor) == manifest_builder.tensor_sha256(tensor)
    for dataset in subject.data_protocol.DATASETS:
        assert subject.expected_audit_epochs(dataset) == [
            row["epoch"] for row in manifest_builder.ranked_candidate_epochs(dataset)
        ]


def test_partition_is_exhaustive_and_shared_groups_are_disjoint() -> None:
    model = _TinySixHeadModel()
    partition = subject.build_parameter_partition(
        model, require_formal_counts=False
    )
    assert partition["all_trainable_parameters_assigned_once"] is True
    assert partition["shared_groups_mutually_exclusive"] is True
    assert set(partition["shared_groups"]) == set(subject.SHARED_ATOMIC_MEMBERS)
    atomic_indices = [
        index
        for group in partition["atomic_groups"].values()
        for index in group["indices"]
    ]
    assert sorted(atomic_indices) == list(
        range(partition["trainable_parameter_tensor_count"])
    )
    assert len(atomic_indices) == len(set(atomic_indices))
    assert set(partition["head_local_groups"]) == set(subject.HEAD_ORDER)


def test_device_gram_matches_manual_cpu_and_cuda_when_available() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260805)
    parameters = (
        nn.Parameter(torch.empty(2, 3)),
        nn.Parameter(torch.empty(5)),
        nn.Parameter(torch.empty(2, 2)),
    )
    rows: list[tuple[torch.Tensor | None, ...]] = []
    for head_index in range(6):
        rows.append(
            (
                torch.randn(2, 3, generator=generator),
                None if head_index in (1, 4) else torch.randn(5, generator=generator),
                torch.randn(2, 2, generator=generator),
            )
        )
    observed_cpu = torch.tensor(
        subject._group_gram(rows, parameters, (0, 1, 2)), dtype=torch.float64
    )
    packed = torch.zeros((6, 11), dtype=torch.float32)
    for head_index, row in enumerate(rows):
        packed[head_index, :6] = row[0].reshape(-1)
        if row[1] is not None:
            packed[head_index, 6:11] = row[1]
    expected = torch.empty((6, 6), dtype=torch.float64)
    for left_index in range(6):
        for right_index in range(left_index, 6):
            value = torch.sum(
                packed[left_index] * packed[right_index], dtype=torch.float64
            )
            expected[left_index, right_index] = value
            expected[right_index, left_index] = value
    # Parameter index 2 was deliberately excluded from this manual packing.
    observed_first_two = torch.tensor(
        subject._group_gram(rows, parameters, (0, 1)), dtype=torch.float64
    )
    assert torch.equal(observed_first_two, expected)
    assert torch.isfinite(observed_cpu).all()
    assert torch.equal(observed_cpu, observed_cpu.transpose(0, 1))

    if not torch.cuda.is_available():
        return
    cuda_parameters = tuple(
        nn.Parameter(parameter.detach().to("cuda:0")) for parameter in parameters
    )
    cuda_rows = tuple(
        tuple(None if value is None else value.to("cuda:0") for value in row)
        for row in rows
    )
    observed_cuda = torch.tensor(
        subject._group_gram(cuda_rows, cuda_parameters, (0, 1, 2)),
        dtype=torch.float64,
    )
    repeated_cuda = torch.tensor(
        subject._group_gram(cuda_rows, cuda_parameters, (0, 1, 2)),
        dtype=torch.float64,
    )
    assert torch.equal(repeated_cuda, observed_cuda)
    assert torch.allclose(observed_cuda, observed_cpu, rtol=2e-6, atol=2e-6)


def test_one_batch_uses_six_vjps_and_restores_every_mutable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(123)
    model = _TinySixHeadModel().eval()
    partition = subject.build_parameter_partition(
        model, require_formal_counts=False
    )
    images, masks, identifiers = _batch()
    state_before = subject.checkpoint_adapter.module_state_sha256(model)
    flags_before = [module.training for module in model.modules()]
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    calls: list[tuple[int, tuple[int, ...]]] = []
    original = torch.autograd.grad

    def observed_grad(*args, **kwargs):
        parameters = args[1]
        calls.append((id(parameters), tuple(id(value) for value in parameters)))
        return original(*args, **kwargs)

    monkeypatch.setattr(subject.torch.autograd, "grad", observed_grad)
    result = subject.audit_one_loaded_batch(
        model,
        images,
        masks,
        partition,
        dataset="NUDT-SIRST",
        stratum="tiny_positive",
        batch_index=0,
        expected_sample_ids=identifiers,
    )
    assert len(calls) == 6
    assert len({call[0] for call in calls}) == 1
    assert len({call[1] for call in calls}) == 1
    assert result["vjp_call_count"] == 6
    assert result["leaf_grad_all_none_during_audit"] is True
    assert result["restoration"]["model_state_unchanged"] is True
    assert subject.checkpoint_adapter.module_state_sha256(model) == state_before
    assert [module.training for module in model.modules()] == flags_before
    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert result["tss_control_all_unused"] is True
    assert set(result["shared_groups"]) == set(subject.SHARED_ATOMIC_MEMBERS)
    for group in result["shared_groups"].values():
        gram = group["gram_6x6"]
        assert len(gram) == 6
        assert all(len(row) == 6 for row in gram)
        for left in range(6):
            for right in range(6):
                assert gram[left][right] == pytest.approx(gram[right][left])
        assert set(group["heads"]) == set(subject.HEAD_ORDER)
        assert set(group["aux_total"]) >= {
            "cosine_aux_final",
            "aux_to_final_norm_ratio",
            "cancellation",
        }
    for values in result["losses"]["per_sample"].values():
        assert len(values) == 16


def test_sentinel_replay_summary_is_exact() -> None:
    torch.manual_seed(321)
    model = _TinySixHeadModel().eval()
    partition = subject.build_parameter_partition(
        model, require_formal_counts=False
    )
    images, masks, identifiers = _batch()
    first = subject.audit_one_loaded_batch(
        model,
        images,
        masks,
        partition,
        dataset="NUAA-SIRST",
        stratum="tiny_positive",
        batch_index=0,
        expected_sample_ids=identifiers,
    )
    second = subject.audit_one_loaded_batch(
        model,
        images,
        masks,
        partition,
        dataset="NUAA-SIRST",
        stratum="tiny_positive",
        batch_index=0,
        expected_sample_ids=identifiers,
    )
    first_summary = subject._summary_for_replay(first)
    second_summary = subject._summary_for_replay(second)
    assert first_summary == second_summary
    assert subject.canonical_sha256(first_summary) == subject.canonical_sha256(
        second_summary
    )


def test_background_batches_do_not_displace_tiny_local_zero_sentinel() -> None:
    normalized_manifest = {
        "strata": {
            "background_only": {
                "available": True,
                "batches": [
                    {
                        "batch_index": index,
                        "manifest_global_batch_index": index,
                    }
                    for index in range(4)
                ],
            },
            "tiny_positive": {
                "available": True,
                "batches": [
                    {
                        "batch_index": index,
                        "manifest_global_batch_index": index + 4,
                    }
                    for index in range(4)
                ],
            },
            "normal_positive": {
                "available": True,
                "batches": [
                    {
                        "batch_index": index,
                        "manifest_global_batch_index": index + 8,
                    }
                    for index in range(4)
                ],
            },
        }
    }
    binding = subject.sentinel_batch_binding(normalized_manifest)
    assert binding == {
        "stratum": "tiny_positive",
        "stratum_batch_index": 0,
        "manifest_global_batch_index": 4,
    }
    assert binding["manifest_global_batch_index"] != binding[
        "stratum_batch_index"
    ]


def test_registered_backward_hook_is_rejected() -> None:
    model = _TinySixHeadModel()
    handle = model.inc.register_full_backward_hook(lambda module, inputs, outputs: None)
    try:
        with pytest.raises(subject.DSGradientAuditError, match="backward hook"):
            subject._validate_no_backward_hooks(model)
    finally:
        handle.remove()


def test_aggregate_has_four_raw_values_median_iqr_and_sign_counts() -> None:
    torch.manual_seed(456)
    model = _TinySixHeadModel().eval()
    partition = subject.build_parameter_partition(
        model, require_formal_counts=False
    )
    images, masks, identifiers = _batch()
    batches = [
        subject.audit_one_loaded_batch(
            model,
            images + index * 0.01,
            masks,
            partition,
            dataset="IRSTD-1K",
            stratum="normal_positive",
            batch_index=index,
            expected_sample_ids=identifiers,
        )
        for index in range(4)
    ]
    aggregate = subject.aggregate_stratum_batches(batches)
    assert aggregate["batch_count"] == 4
    for group in aggregate["shared_groups"].values():
        for head in group["heads"].values():
            for metric in head.values():
                assert len(metric["values"]) == 4
                assert metric["valid_count"] <= 4
                assert "median" in metric and "iqr" in metric
        cosine = group["aux_total"]["cosine_aux_final"]
        assert cosine["negative_count"] + cosine["positive_count"] + cosine[
            "zero_count"
        ] == cosine["valid_count"]


def test_cli_defaults_and_source_forbid_training_mutations() -> None:
    args = subject.parse_args(
        [
            "--dataset",
            "NUDT-SIRST",
            "--checkpoint-role",
            "best_pd",
            "--device",
            "cpu",
        ]
    )
    assert args.run_dir == subject._default_run_dir("NUDT-SIRST")
    assert args.output == subject._default_output("NUDT-SIRST", "best_pd")
    assert args.output.as_posix().endswith(
        "results/three_dataset_ds_gradient_audit_v1/runs/NUDT-SIRST/best_pd/audit.json"
    )
    source = inspect.getsource(subject)
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert source.count("torch.autograd.grad(") == 1


def test_output_validator_accepts_minimal_complete_shape() -> None:
    # Reuse one real synthetic batch four times.  The validator intentionally
    # checks structural engineering gates, while the comparator recomputes the
    # scientific PC/AC/PA predicates from these raw values.
    torch.manual_seed(789)
    model = _TinySixHeadModel().eval()
    partition = subject.build_parameter_partition(
        model, require_formal_counts=False
    )
    images, masks, identifiers = _batch()
    batch = subject.audit_one_loaded_batch(
        model,
        images,
        masks,
        partition,
        dataset="NUDT-SIRST",
        stratum="tiny_positive",
        batch_index=0,
        expected_sample_ids=identifiers,
    )
    batches = [json.loads(json.dumps(batch)) for _ in range(4)]
    for index, item in enumerate(batches):
        item["batch_index"] = index
    replay_sha = "a" * 64
    payload = {
        "schema": subject.SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "checkpoint_role": "best_miou",
        "seed": 42,
        "head_order": list(subject.HEAD_ORDER),
        "sentinel_replay": {
            "stratum": "tiny_positive",
            "batch_index": 0,
            "stratum_batch_index": 0,
            "manifest_global_batch_index": 4,
            "repeat_count": 2,
            "first_summary_sha256": replay_sha,
            "second_summary_sha256": replay_sha,
            "replay_exact": True,
        },
        "parameter_partition": subject._public_partition(partition),
        "strata": {
            "tiny_positive": {
                "required_or_conditional": "required",
                "available": True,
                "sample_count": 64,
                "distinct_source_count": 24,
                "batch_count": 4,
                "batches": batches,
            },
            "normal_positive": {
                "required_or_conditional": "required",
                "available": True,
                "sample_count": 64,
                "distinct_source_count": 24,
                "batch_count": 4,
                "batches": batches,
            },
            "background_only": {
                "required_or_conditional": "conditional",
                "available": False,
                "structurally_unavailable": True,
                "candidate_count": 0,
                "reason": "structurally absent",
                "batches": [],
                "aggregate": None,
            },
        },
        "restoration_audit": {
            "model_state_unchanged": True,
            "leaf_grad_always_none": True,
        },
    }
    subject.validate_output_payload(payload)
