#!/usr/bin/env python3
"""Train-only six-head gradient audit for the frozen seed-42 TSS-off model.

The analyzer replays batches selected by the separately frozen DS audit
manifest.  Every batch receives one train-mode forward and six VJPs, one for
each production BCE term.  It never changes model parameters and restores
BatchNorm buffers, module training flags, and all captured RNG streams before
publishing any result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as checkpoint_adapter  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as tss_off_adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as evaluator  # noqa: E402
from experiments import four_dataset_models_seed42_v1 as model_builder  # noqa: E402
from experiments import paper_three_dataset_v2 as paper_data  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as engine  # noqa: E402
from experiments import train_three_dataset_tss_off_seed42_v1 as tss_off_runner  # noqa: E402
from experiments.tpd_training_loss import compute_tpd_training_loss  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_v1/v1"
MANIFEST_SCHEMA = "sctransnet_three_dataset_ds_gradient_audit_manifest_v1/v1"
NAMESPACE = "sctransnet-ds-gradient-audit-v1"
SEED = 42
HEAD_ORDER = ("gt5", "gt4", "gt3", "gt2", "d0", "final")
AUXILIARY_HEADS = HEAD_ORDER[:-1]
CHECKPOINT_ROLES = ("best_miou", "best_pd")
STRATUM_ORDER = ("tiny_positive", "normal_positive", "background_only")
REQUIRED_STRATA = frozenset(STRATUM_ORDER[:2])
SAMPLES_PER_STRATUM = 64
DISTINCT_SOURCES_PER_STRATUM = 24
MAX_CROPS_PER_SOURCE = 3
BATCH_SIZE = 16
BATCHES_PER_STRATUM = 4
AUDIT_EPOCH_COUNT = 32
EPSILON = 1e-30

DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "three_dataset_ds_gradient_audit_v1"
DEFAULT_MANIFEST = (
    data_protocol.DEFAULT_RESULTS_ROOT
    / "manifests"
    / "three_dataset_ds_gradient_audit_manifest_v1.json"
)


# These seven groups are an exhaustive, mutually exclusive partition of the
# formal training model.  The four paper-facing shared groups below are unions
# of these atoms and deliberately exclude every prediction head and TSS head.
ATOMIC_PREFIXES: dict[str, tuple[str, ...]] = {
    "encoder_shared": (
        "inc.",
        "down_encoder1.",
        "down_encoder2.",
        "down_encoder3.",
        "down_encoder4.",
    ),
    "tpd_shallow": ("mtc.embeddings_1.", "mtc.embeddings_2."),
    "sctb_qfg_transform": (
        "mtc.embeddings_3.",
        "mtc.embeddings_4.",
        "mtc.encoder.",
        "mtc.reconstruct_1.",
        "mtc.reconstruct_2.",
        "mtc.reconstruct_3.",
        "mtc.reconstruct_4.",
        "tpd_qfg.",
    ),
    "ner_shared": ("tpd_ner.",),
    "decoder_trunk_shared": (
        "up_decoder4.",
        "up_decoder3.",
        "up_decoder2.",
        "up_decoder1.",
    ),
    "head_local": (
        "gt_conv5.",
        "gt_conv4.",
        "gt_conv3.",
        "gt_conv2.",
        "outconv.",
        "outc.",
    ),
    "tss_control": ("target_survival.",),
}

SHARED_ATOMIC_MEMBERS: dict[str, tuple[str, ...]] = {
    "encoder_shared": ("encoder_shared",),
    "tpd_qfg_sctb_shared": ("tpd_shallow", "sctb_qfg_transform"),
    "ner_shared": ("ner_shared",),
    "decoder_trunk_shared": ("decoder_trunk_shared",),
}

HEAD_LOCAL_PREFIX: dict[str, str] = {
    "gt5": "gt_conv5.",
    "gt4": "gt_conv4.",
    "gt3": "gt_conv3.",
    "gt2": "gt_conv2.",
    "d0": "outconv.",
    "final": "outc.",
}

FORMAL_ATOMIC_COUNTS: dict[str, tuple[int, int]] = {
    "encoder_shared": (56, 2_399_264),
    "tpd_shallow": (21, 66_176),
    "sctb_qfg_transform": (326, 7_220_972),
    "ner_shared": (19, 11_291),
    "decoder_trunk_shared": (48, 1_171_904),
    "head_local": (12, 523),
    "tss_control": (4, 98),
}

FORMAL_ATOMIC_NAME_SHA256: dict[str, str] = {
    "encoder_shared": "3c313d812c03a3187165b273fcec8340c11b7eb0ff2749c27df0cb88529b22d8",
    "tpd_shallow": "a8817d954c8a28961ec5fbca3eb8f5b978a342f7380d9cb4794f0d162b006988",
    "sctb_qfg_transform": "bac3db7fb4849e9c590abec58f54e8fdc02de205767475cce0d9395eb88d2755",
    "ner_shared": "095c9c17663331dee6b42438cb2f0327adacb2152f2d4430c57b49200cb2edc1",
    "decoder_trunk_shared": "d2a73b9c1ed51bf788317dbc5e026ffaabe480d201d7ac818d6cced8ed3133ba",
    "head_local": "f4086a0a67468629328dfcc3e369b4b47417a700981b69257cd4f71f4a9533da",
    "tss_control": "0a37be9630360059d3a7b0efeaa47810d9fc127f3e79f0902afcc2cd7e974414",
}

FORMAL_SHARED_NAME_SHA256: dict[str, str] = {
    "encoder_shared": FORMAL_ATOMIC_NAME_SHA256["encoder_shared"],
    "tpd_qfg_sctb_shared": "79ae78dd4755604dc26d45afa5f7eeb4ae869c57c3f2fd2cc0fa6268beac0a4d",
    "ner_shared": FORMAL_ATOMIC_NAME_SHA256["ner_shared"],
    "decoder_trunk_shared": FORMAL_ATOMIC_NAME_SHA256["decoder_trunk_shared"],
}

FORMAL_HEAD_LOCAL_NAME_SHA256: dict[str, str] = {
    "gt5": "e7dd4f801ee54078bc1d0b470aa9b3e9cae98f9d583ac8df4d4a8531404c4e7b",
    "gt4": "c2b233ccb917ea28214db75f2bff3201300d60da567e5ad35b9259c1c00f0721",
    "gt3": "4f1c4166e317f6a0d09334f6d4dc257b036c9750faa506505e169f9cda2581f6",
    "gt2": "45b6aa2663ae6876c3cf76bd73bf2394fc84d02e9bcf68d50c96dff9c35b3fb1",
    "d0": "19ddad3069c3084ed51e8f8c41546b8b4a4c7ed1d424d6ccea347e66831e5f18",
    "final": "eb88d9f9ceeb05556d5fe0bac67e86b4a51fb7ebbf6021042d8477d7ac9cf1e8",
}


class DSGradientAuditError(ValueError):
    """The requested run differs from the frozen DS audit contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DSGradientAuditError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact contiguous tensor bytes (the manifest tensors are FP32)."""

    _require(isinstance(value, torch.Tensor), "tensor SHA input is not a Tensor")
    ready = value.detach().cpu().contiguous()
    return hashlib.sha256(ready.view(torch.uint8).numpy().tobytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stable_digest(*parts: Any) -> str:
    return canonical_sha256(list(parts))


def expected_audit_epochs(dataset: str) -> list[int]:
    data_protocol.require_dataset(dataset)
    ranked = sorted(
        range(1, 1001),
        key=lambda epoch: (stable_digest(NAMESPACE, SEED, dataset, epoch), epoch),
    )
    return ranked[:AUDIT_EPOCH_COUNT]


def forward_seed(dataset: str, stratum: str, batch_index: int) -> int:
    digest = stable_digest(NAMESPACE, SEED, dataset, stratum, batch_index)
    return int(digest[:16], 16) & ((1 << 63) - 1)


def _json_object(path: Path, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DSGradientAuditError(f"cannot read {label}: {candidate}: {exc}")
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ordered_names_sha256(names: Sequence[str]) -> str:
    return canonical_sha256(list(names))


def build_parameter_partition(
    model: nn.Module,
    *,
    require_formal_counts: bool = True,
) -> dict[str, Any]:
    """Build the exhaustive atomic partition and four shared unions."""

    named = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    _require(bool(named), "model has no trainable parameters")
    _require(
        len({name for name, _ in named}) == len(named),
        "trainable parameter names are duplicated",
    )
    _require(
        len({id(parameter) for _, parameter in named}) == len(named),
        "trainable parameter objects are aliased",
    )
    atomic_indices: dict[str, list[int]] = {name: [] for name in ATOMIC_PREFIXES}
    for index, (name, _) in enumerate(named):
        matches = [
            group
            for group, prefixes in ATOMIC_PREFIXES.items()
            if name.startswith(prefixes)
        ]
        _require(len(matches) == 1, f"parameter has {len(matches)} atomic groups: {name}")
        atomic_indices[matches[0]].append(index)

    atoms: dict[str, Any] = {}
    for group, indices in atomic_indices.items():
        names = [named[index][0] for index in indices]
        numel = sum(int(named[index][1].numel()) for index in indices)
        if require_formal_counts:
            expected_tensors, expected_numel = FORMAL_ATOMIC_COUNTS[group]
            _require(len(indices) == expected_tensors, f"{group} tensor count differs")
            _require(numel == expected_numel, f"{group} numel differs")
            _require(
                _ordered_names_sha256(names) == FORMAL_ATOMIC_NAME_SHA256[group],
                f"{group} ordered parameter-name SHA differs",
            )
        atoms[group] = {
            "parameter_tensor_count": len(indices),
            "parameter_numel": numel,
            "ordered_parameter_names": names,
            "ordered_parameter_names_sha256": _ordered_names_sha256(names),
            "indices": indices,
        }

    shared: dict[str, Any] = {}
    seen_shared: set[int] = set()
    for group, members in SHARED_ATOMIC_MEMBERS.items():
        indices = sorted(
            index for member in members for index in atoms[member]["indices"]
        )
        _require(not (seen_shared & set(indices)), "shared parameter groups overlap")
        seen_shared.update(indices)
        names = [named[index][0] for index in indices]
        shared[group] = {
            "atomic_members": list(members),
            "parameter_tensor_count": len(indices),
            "parameter_numel": sum(int(named[index][1].numel()) for index in indices),
            "ordered_parameter_names": names,
            "ordered_parameter_names_sha256": _ordered_names_sha256(names),
            "indices": indices,
        }
        if require_formal_counts:
            _require(
                shared[group]["ordered_parameter_names_sha256"]
                == FORMAL_SHARED_NAME_SHA256[group],
                f"{group} shared ordered parameter-name SHA differs",
            )

    local: dict[str, Any] = {}
    for head, prefix in HEAD_LOCAL_PREFIX.items():
        indices = [index for index, (name, _) in enumerate(named) if name.startswith(prefix)]
        _require(len(indices) == 2, f"{head} local head must contain weight and bias")
        names = [named[index][0] for index in indices]
        local[head] = {
            "prefix": prefix,
            "parameter_tensor_count": len(indices),
            "parameter_numel": sum(int(named[index][1].numel()) for index in indices),
            "ordered_parameter_names": names,
            "ordered_parameter_names_sha256": _ordered_names_sha256(names),
            "indices": indices,
        }
        if require_formal_counts:
            _require(
                local[head]["ordered_parameter_names_sha256"]
                == FORMAL_HEAD_LOCAL_NAME_SHA256[head],
                f"{head} local ordered parameter-name SHA differs",
            )
    return {
        "named_parameters": named,
        "atomic_groups": atoms,
        "shared_groups": shared,
        "head_local_groups": local,
        "trainable_parameter_tensor_count": len(named),
        "trainable_parameter_numel": sum(int(parameter.numel()) for _, parameter in named),
        "all_trainable_parameters_assigned_once": sum(len(v["indices"]) for v in atoms.values())
        == len(named),
        "shared_groups_mutually_exclusive": len(seen_shared)
        == sum(len(v["indices"]) for v in shared.values()),
    }


def _public_partition(partition: Mapping[str, Any]) -> dict[str, Any]:
    def strip_indices(groups: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: {key: value for key, value in group.items() if key != "indices"}
            for name, group in groups.items()
        }

    return {
        "atomic_groups": strip_indices(partition["atomic_groups"]),
        "shared_groups": strip_indices(partition["shared_groups"]),
        "head_local_groups": strip_indices(partition["head_local_groups"]),
        "trainable_parameter_tensor_count": partition["trainable_parameter_tensor_count"],
        "trainable_parameter_numel": partition["trainable_parameter_numel"],
        "all_trainable_parameters_assigned_once": partition[
            "all_trainable_parameters_assigned_once"
        ],
        "shared_groups_mutually_exclusive": partition[
            "shared_groups_mutually_exclusive"
        ],
    }


def _validate_no_backward_hooks(model: nn.Module) -> None:
    for name, module in model.named_modules():
        for attribute in ("_backward_hooks", "_backward_pre_hooks"):
            hooks = getattr(module, attribute, None)
            _require(not hooks, f"backward hook is registered on module {name!r}")
    for name, parameter in model.named_parameters():
        hooks = getattr(parameter, "_backward_hooks", None)
        _require(not hooks, f"backward hook is registered on parameter {name!r}")


def _snapshot_rng() -> dict[str, Any]:
    cuda_states: list[torch.Tensor] | None = None
    if torch.cuda.is_available() and getattr(torch.cuda, "_initialized", False):
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            np.array(numpy_state[1], copy=True),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_states,
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    cuda_states = snapshot["torch_cuda"]
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def _set_forward_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available() and getattr(torch.cuda, "_initialized", False):
        torch.cuda.manual_seed_all(seed)


def _snapshot_batchnorm(model: nn.Module) -> dict[str, dict[str, torch.Tensor | None]]:
    snapshot: dict[str, dict[str, torch.Tensor | None]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            snapshot[name] = {
                field: (
                    None
                    if getattr(module, field) is None
                    else getattr(module, field).detach().clone()
                )
                for field in ("running_mean", "running_var", "num_batches_tracked")
            }
    return snapshot


def _restore_batchnorm(
    model: nn.Module,
    snapshot: Mapping[str, Mapping[str, torch.Tensor | None]],
) -> None:
    modules = dict(model.named_modules())
    _require(set(snapshot) <= set(modules), "BatchNorm module set changed")
    with torch.no_grad():
        for name, fields in snapshot.items():
            module = modules[name]
            for field, value in fields.items():
                current = getattr(module, field)
                if value is None:
                    _require(current is None, f"BatchNorm {name}.{field} appeared")
                else:
                    _require(isinstance(current, torch.Tensor), f"BatchNorm {name}.{field} vanished")
                    current.copy_(value)


def _module_training_flags(model: nn.Module) -> tuple[tuple[nn.Module, bool], ...]:
    return tuple((module, bool(module.training)) for module in model.modules())


def _restore_training_flags(flags: Sequence[tuple[nn.Module, bool]]) -> None:
    for module, training in flags:
        module.training = training


def _finite_float(value: torch.Tensor, label: str) -> float:
    _require(isinstance(value, torch.Tensor) and value.ndim == 0, f"{label} is not scalar")
    number = float(value.detach().cpu().item())
    _require(math.isfinite(number), f"{label} is non-finite")
    return number


def _group_gram(
    gradients: Sequence[Sequence[torch.Tensor | None]],
    parameters: Sequence[nn.Parameter],
    indices: Sequence[int],
) -> list[list[float]]:
    """Compute one six-head Gram on the parameter device.

    Unused entries are represented by the zero-initialized slices of the
    6-by-N matrix.  Only the final 36 scalar values cross the device boundary;
    full gradient tensors are never copied to CPU.
    """

    _require(len(gradients) == len(HEAD_ORDER), "six gradient tuples are required")
    _require(bool(indices), "shared gradient group is empty")
    _require(
        all(len(row) == len(parameters) for row in gradients),
        "gradient/parameter tuple length differs",
    )
    reference = parameters[indices[0]]
    device = reference.device
    group_numel = sum(int(parameters[index].numel()) for index in indices)
    matrix = torch.zeros(
        (len(HEAD_ORDER), group_numel),
        dtype=torch.float32,
        device=device,
    )
    offset = 0
    for index in indices:
        parameter = parameters[index]
        _require(parameter.device == device, "shared group spans multiple devices")
        count = int(parameter.numel())
        for head_index, row in enumerate(gradients):
            gradient = row[index]
            if gradient is None:
                continue
            _require(gradient.device == device, "gradient/parameter device differs")
            _require(gradient.dtype == torch.float32, "formal gradient is not FP32")
            matrix[head_index, offset : offset + count].copy_(
                gradient.reshape(-1)
            )
        offset += count
    _require(offset == group_numel, "shared gradient packing length differs")
    # Products remain FP32, matching the historical audit, while reduction is
    # FP64.  On RTX 5090 this 21-reduction path is both faster and more accurate
    # than a skinny 6xN FP32 GEMM; it also constructs only the upper triangle.
    gram_tensor = torch.empty(
        (len(HEAD_ORDER), len(HEAD_ORDER)),
        dtype=torch.float64,
        device=device,
    )
    for left_index in range(len(HEAD_ORDER)):
        for right_index in range(left_index, len(HEAD_ORDER)):
            value = torch.sum(
                matrix[left_index] * matrix[right_index],
                dtype=torch.float64,
            )
            gram_tensor[left_index, right_index] = value
            gram_tensor[right_index, left_index] = value
    _require(
        bool(torch.isfinite(gram_tensor).all().item()),
        "group Gram entry is non-finite",
    )
    gram = gram_tensor.detach().cpu().tolist()
    del gram_tensor, matrix
    return [[float(value) for value in row] for row in gram]


def _safe_cosine(dot: float, left_norm: float, right_norm: float) -> float | None:
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = dot / (left_norm * right_norm)
    _require(math.isfinite(value), "cosine is non-finite")
    return max(-1.0, min(1.0, value))


def _gram_summary(
    gram: Sequence[Sequence[float]],
    *,
    group_numel: int,
    active_numel: Sequence[int],
) -> dict[str, Any]:
    _require(len(gram) == len(HEAD_ORDER), "Gram row count differs")
    norms = [math.sqrt(max(0.0, float(gram[index][index]))) for index in range(6)]
    final_norm = norms[-1]
    heads: dict[str, Any] = {}
    for index, head in enumerate(HEAD_ORDER):
        dot_final = float(gram[index][-1])
        heads[head] = {
            "raw_l2_norm": norms[index],
            "gradient_rms": norms[index] / math.sqrt(group_numel),
            "active_parameter_numel": int(active_numel[index]),
            "unused": int(active_numel[index]) == 0,
            "partially_unused": 0 < int(active_numel[index]) < group_numel,
            "zero_gradient": norms[index] == 0.0,
            "norm_ratio_to_final": (
                None if final_norm == 0.0 else norms[index] / final_norm
            ),
            "cosine_to_final": _safe_cosine(dot_final, norms[index], final_norm),
            "dot_with_final": dot_final,
            "projection_onto_final": (
                None if final_norm == 0.0 else dot_final / (final_norm * final_norm)
            ),
        }

    aux_square = sum(float(gram[i][j]) for i in range(5) for j in range(5))
    aux_final_dot = sum(float(gram[i][5]) for i in range(5))
    total_square = aux_square + float(gram[5][5]) + 2.0 * aux_final_dot
    aux_norm = math.sqrt(max(0.0, aux_square))
    total_norm = math.sqrt(max(0.0, total_square))
    individual_norm_sum = sum(norms)
    return {
        "gram_head_order": list(HEAD_ORDER),
        "gram_6x6": [[float(value) for value in row] for row in gram],
        "parameter_numel": group_numel,
        "heads": heads,
        "aux_total": {
            "aux_l2_norm": aux_norm,
            "final_l2_norm": final_norm,
            "total_l2_norm": total_norm,
            "aux_final_dot": aux_final_dot,
            "cosine_aux_final": _safe_cosine(aux_final_dot, aux_norm, final_norm),
            "aux_to_final_norm_ratio": (
                None if final_norm == 0.0 else aux_norm / final_norm
            ),
            # The document's frozen cancellation statistic: lower means that
            # the vector sum cancels more of the six individual vectors.
            "cancellation": (
                None if individual_norm_sum == 0.0 else total_norm / individual_norm_sum
            ),
            "individual_head_norm_sum": individual_norm_sum,
        },
    }


def _per_sample_bce(probability: torch.Tensor, target: torch.Tensor) -> list[float]:
    detached = F.binary_cross_entropy(
        probability.detach().float(), target.detach().float(), reduction="none"
    )
    values = detached.flatten(1).mean(dim=1).cpu().tolist()
    ready = [float(value) for value in values]
    _require(all(math.isfinite(value) for value in ready), "per-sample BCE is non-finite")
    return ready


def _local_gradient_summary(
    gradient: Sequence[torch.Tensor | None],
    indices: Sequence[int],
    numel: int,
) -> dict[str, Any]:
    used = [gradient[index] for index in indices if gradient[index] is not None]
    if used:
        device = used[0].device
        _require(all(value.device == device for value in used), "local gradients span devices")
        square_tensor = torch.stack(
            [torch.sum(value * value, dtype=torch.float32) for value in used]
        ).sum()
        square = float(square_tensor.item())
        _require(math.isfinite(square), "local gradient norm is non-finite")
    else:
        square = 0.0
    norm = math.sqrt(max(0.0, square))
    active = sum(
        int(gradient[index].numel())
        for index in indices
        if gradient[index] is not None
    )
    return {
        "raw_l2_norm": norm,
        "gradient_rms": norm / math.sqrt(numel),
        "parameter_numel": numel,
        "active_parameter_numel": active,
        "unused": active == 0,
        "zero_gradient": norm == 0.0,
    }


def audit_one_loaded_batch(
    model: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    partition: Mapping[str, Any],
    *,
    dataset: str,
    stratum: str,
    batch_index: int,
    expected_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Run one isolated forward and six VJPs, then restore every mutation."""

    _require(images.ndim == masks.ndim == 4, "audit tensors must be BCHW")
    _require(images.shape == masks.shape, "audit image/mask shapes differ")
    _require(int(images.shape[0]) == BATCH_SIZE, "formal audit batch must contain 16 samples")
    _require(len(expected_sample_ids) == BATCH_SIZE, "sample ID count differs")
    _require(images.dtype == masks.dtype == torch.float32, "audit tensors must be FP32")
    _require(bool(torch.isfinite(images).all()), "audit images are non-finite")
    _require(bool(torch.isfinite(masks).all()), "audit masks are non-finite")
    _require(bool(torch.all((masks >= 0) & (masks <= 1))), "audit masks leave [0,1]")
    _validate_no_backward_hooks(model)

    named = partition["named_parameters"]
    parameters = tuple(parameter for _, parameter in named)
    _require(all(parameter.grad is None for parameter in parameters), "leaf .grad must be None before audit")
    state_before = checkpoint_adapter.module_state_sha256(model)
    rng_snapshot = _snapshot_rng()
    flags = _module_training_flags(model)
    batchnorm_snapshot = _snapshot_batchnorm(model)
    seed = forward_seed(dataset, stratum, batch_index)
    result: dict[str, Any] | None = None
    caught: BaseException | None = None
    try:
        _set_forward_rng(seed)
        model.train()
        output = model(images)
        criterion = nn.BCELoss(reduction="mean")
        formal = compute_tpd_training_loss(
            output,
            masks,
            criterion,
            survival_weight=0.0,
        )
        terms = tuple(formal.segmentation_terms)
        _require(len(terms) == len(HEAD_ORDER), "formal segmentation term count differs")
        _require(len(formal.survival_terms) == 0, "TSS-off produced survival terms")
        _require(float(formal.effective_survival_weight.item()) == 0.0, "TSS weight is nonzero")
        _require(float(formal.weighted_survival.item()) == 0.0, "weighted TSS is nonzero")

        segmentation = (
            output.legacy_output()
            if hasattr(output, "legacy_output")
            else output
        )
        _require(isinstance(segmentation, (tuple, list)), "training output is not six maps")
        maps = tuple(segmentation)
        _require(len(maps) == len(HEAD_ORDER), "training output does not contain six maps")
        direct_terms = tuple(criterion(probability.float(), masks.float()) for probability in maps)
        _require(
            all(torch.equal(left, right) for left, right in zip(terms, direct_terms)),
            "formal and direct production BCE terms differ",
        )

        gradient_rows: list[tuple[torch.Tensor | None, ...]] = []
        for head_index, term in enumerate(terms):
            row = torch.autograd.grad(
                term,
                parameters,
                retain_graph=head_index < len(terms) - 1,
                allow_unused=True,
            )
            _require(len(row) == len(parameters), "VJP parameter tuple differs")
            retained: list[torch.Tensor | None] = []
            finite_checks: list[torch.Tensor] = []
            for parameter, gradient in zip(parameters, row):
                if gradient is None:
                    retained.append(None)
                    continue
                _require(gradient.shape == parameter.shape, "gradient shape differs")
                _require(gradient.dtype == torch.float32, "formal gradient is not FP32")
                finite_checks.append(torch.isfinite(gradient).all())
                retained.append(gradient.detach())
            if finite_checks:
                _require(
                    bool(torch.stack(finite_checks).all().item()),
                    "gradient is non-finite",
                )
            gradient_rows.append(tuple(retained))
            del row

        shared_groups: dict[str, Any] = {}
        for group_name, group in partition["shared_groups"].items():
            indices = group["indices"]
            gram = _group_gram(gradient_rows, parameters, indices)
            active_numel = [
                sum(
                    int(gradient_rows[head_index][index].numel())
                    for index in indices
                    if gradient_rows[head_index][index] is not None
                )
                for head_index in range(len(HEAD_ORDER))
            ]
            shared_groups[group_name] = _gram_summary(
                gram,
                group_numel=int(group["parameter_numel"]),
                active_numel=active_numel,
            )

        head_local: dict[str, Any] = {}
        for head_index, head in enumerate(HEAD_ORDER):
            group = partition["head_local_groups"][head]
            head_local[head] = _local_gradient_summary(
                gradient_rows[head_index],
                group["indices"],
                int(group["parameter_numel"]),
            )

        tss_indices = partition["atomic_groups"]["tss_control"]["indices"]
        _require(
            all(gradient_rows[h][index] is None for h in range(6) for index in tss_indices),
            "TSS control received a segmentation gradient under weight zero",
        )
        losses = {
            "head_scalars": {
                head: _finite_float(term, f"loss[{head}]")
                for head, term in zip(HEAD_ORDER, terms)
            },
            "segmentation_total": _finite_float(formal.segmentation, "segmentation total"),
            "formal_total": _finite_float(formal.total, "formal total"),
            "per_sample": {
                head: _per_sample_bce(probability, masks)
                for head, probability in zip(HEAD_ORDER, maps)
            },
        }
        _require(
            losses["segmentation_total"] == losses["formal_total"],
            "formal total differs from TSS-off segmentation total",
        )
        result = {
            "batch_index": int(batch_index),
            "forward_seed": seed,
            "sample_ids": list(expected_sample_ids),
            "images_sha256": tensor_sha256(images),
            "masks_sha256": tensor_sha256(masks),
            "losses": losses,
            "shared_groups": shared_groups,
            "head_local_gradients": head_local,
            "tss_control_all_unused": True,
            "vjp_call_count": 6,
            "same_ordered_parameter_tuple_for_all_vjps": True,
            "leaf_grad_all_none_during_audit": all(parameter.grad is None for parameter in parameters),
        }
        _require(result["leaf_grad_all_none_during_audit"] is True, "leaf .grad was populated")
        del gradient_rows, output, maps, terms, direct_terms, formal
    except BaseException as exc:
        caught = exc
    finally:
        _restore_batchnorm(model, batchnorm_snapshot)
        _restore_training_flags(flags)
        _restore_rng(rng_snapshot)

    state_after = checkpoint_adapter.module_state_sha256(model)
    state_unchanged = state_after == state_before
    leaves_clear = all(parameter.grad is None for parameter in parameters)
    if caught is not None:
        raise caught
    _require(state_unchanged, "model state SHA changed after batch restoration")
    _require(leaves_clear, "leaf .grad changed after batch restoration")
    _require(result is not None, "batch audit did not produce a result")
    result["restoration"] = {
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": state_unchanged,
        "batchnorm_buffers_restored": True,
        "module_training_flags_restored": True,
        "python_numpy_cpu_cuda_rng_restored": True,
        "leaf_grad_all_none_after": leaves_clear,
    }
    return result


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    _require(bool(sorted_values), "quantile requires values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_values(values: Sequence[float | None]) -> dict[str, Any]:
    ready = [float(value) for value in values if value is not None]
    _require(all(math.isfinite(value) for value in ready), "aggregate value is non-finite")
    if not ready:
        return {
            "values": list(values),
            "valid_count": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "negative_count": 0,
            "positive_count": 0,
            "zero_count": 0,
        }
    ordered = sorted(ready)
    q1 = _linear_quantile(ordered, 0.25)
    median = _linear_quantile(ordered, 0.5)
    q3 = _linear_quantile(ordered, 0.75)
    return {
        "values": list(values),
        "valid_count": len(ready),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "negative_count": sum(value < 0.0 for value in ready),
        "positive_count": sum(value > 0.0 for value in ready),
        "zero_count": sum(value == 0.0 for value in ready),
    }


def aggregate_stratum_batches(batches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(batches) == BATCHES_PER_STRATUM, "stratum requires four batches")
    losses = {
        head: summarize_values([batch["losses"]["head_scalars"][head] for batch in batches])
        for head in HEAD_ORDER
    }
    shared: dict[str, Any] = {}
    for group in SHARED_ATOMIC_MEMBERS:
        head_rows: dict[str, Any] = {}
        for head in HEAD_ORDER:
            metrics = (
                "raw_l2_norm",
                "gradient_rms",
                "norm_ratio_to_final",
                "cosine_to_final",
                "dot_with_final",
                "projection_onto_final",
            )
            head_rows[head] = {
                metric: summarize_values(
                    [batch["shared_groups"][group]["heads"][head][metric] for batch in batches]
                )
                for metric in metrics
            }
        aux_metrics = (
            "aux_l2_norm",
            "final_l2_norm",
            "total_l2_norm",
            "aux_final_dot",
            "cosine_aux_final",
            "aux_to_final_norm_ratio",
            "cancellation",
            "individual_head_norm_sum",
        )
        shared[group] = {
            "heads": head_rows,
            "aux_total": {
                metric: summarize_values(
                    [batch["shared_groups"][group]["aux_total"][metric] for batch in batches]
                )
                for metric in aux_metrics
            },
        }
    local = {
        head: {
            metric: summarize_values(
                [batch["head_local_gradients"][head][metric] for batch in batches]
            )
            for metric in ("raw_l2_norm", "gradient_rms")
        }
        for head in HEAD_ORDER
    }
    return {
        "quartile_method": "linear_interpolation_positions_(n-1)*p",
        "batch_count": len(batches),
        "losses": losses,
        "shared_groups": shared,
        "head_local_gradients": local,
    }


def _validate_source_map(source_map: Any) -> dict[str, str]:
    _require(isinstance(source_map, Mapping) and bool(source_map), "manifest source_sha256 is missing")
    ready: dict[str, str] = {}
    for raw_name, raw_sha in source_map.items():
        _require(isinstance(raw_name, str) and raw_name, "source SHA key is invalid")
        _require(_is_sha256(raw_sha), f"source SHA is invalid: {raw_name}")
        ready[raw_name] = str(raw_sha)
        candidate = Path(raw_name)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        if candidate.is_file() and not candidate.is_symlink():
            _require(file_sha256(candidate) == raw_sha, f"manifest source SHA differs: {raw_name}")
    return ready


def _field_sha(binding: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = binding.get(name)
        if value is not None:
            _require(_is_sha256(value), f"{name} is not a SHA256")
            return str(value)
    return None


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    protocol_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require(payload.get("schema") == MANIFEST_SCHEMA, "DS manifest schema differs")
    _require(payload.get("status") == "complete", "DS manifest is incomplete")
    _require(payload.get("audit_namespace") == NAMESPACE, "DS manifest namespace differs")
    _require(payload.get("training_seed") == SEED, "DS manifest seed differs")
    _require(payload.get("split") == "train", "DS manifest split differs")
    _require(payload.get("uses_only_img_idx_train") is True, "DS manifest data scope differs")
    _require(payload.get("uses_model_outputs") is False, "DS manifest used model outputs")
    _require(payload.get("uses_checkpoints") is False, "DS manifest used checkpoints")
    _require(payload.get("dataset_order") == list(data_protocol.DATASETS), "DS manifest dataset order differs")
    bindings = payload.get("bindings")
    _require(isinstance(bindings, Mapping), "DS manifest bindings are missing")
    _validate_source_map(bindings.get("source_sha256"))
    expected_protocol_sha = bindings.get("protocol_manifest_file_sha256")
    _require(_is_sha256(expected_protocol_sha), "protocol manifest SHA binding is invalid")
    datasets = payload.get("datasets")
    _require(isinstance(datasets, Mapping), "DS manifest datasets are missing")
    _require(set(datasets) == set(data_protocol.DATASETS), "DS manifest dataset scope differs")
    entry = datasets.get(dataset)
    _require(isinstance(entry, Mapping), "selected dataset manifest entry is missing")

    epoch_selection = entry.get("candidate_epoch_selection")
    _require(isinstance(epoch_selection, Mapping), "candidate epoch selection is missing")
    epochs = epoch_selection.get("epochs")
    _require(isinstance(epochs, list), "candidate epoch list is missing")
    _require(epochs == expected_audit_epochs(dataset), "audit epoch candidates differ")
    _require(epoch_selection.get("epoch_range_inclusive") == [1, 1000], "audit epoch range differs")
    _require(epoch_selection.get("retained_epoch_count") == 32, "audit epoch count differs")

    expected_train = protocol_manifest["datasets"][dataset]["splits"]["train"]
    train_binding = entry.get("train_index_binding")
    _require(isinstance(train_binding, Mapping), "train index binding is missing")
    file_digest = _field_sha(train_binding, "file_sha256", "sha256")
    ordered_digest = _field_sha(train_binding, "ordered_ids_sha256", "ordered_id_sha256")
    _require(file_digest == expected_train["file_sha256"], "train index file SHA differs")
    _require(ordered_digest == expected_train["ordered_ids_sha256"], "train ordered-ID SHA differs")
    if "count" in train_binding:
        _require(train_binding["count"] == expected_train["count"], "train index count differs")

    strata = entry.get("strata")
    _require(isinstance(strata, Mapping), "strata are missing")
    _require(set(strata) == set(STRATUM_ORDER), "stratum scope differs")
    records = entry.get("records")
    raw_batches = entry.get("batches")
    _require(isinstance(records, list), "selected records are missing")
    _require(isinstance(raw_batches, list), "selected batches are missing")
    global_identities: set[tuple[int, str]] = set()
    normalized_strata: dict[str, Any] = {}
    for stratum in STRATUM_ORDER:
        record = strata[stratum]
        _require(isinstance(record, Mapping), f"{stratum} is malformed")
        structurally_unavailable = record.get("structurally_unavailable")
        _require(type(structurally_unavailable) is bool, f"{stratum}.structurally_unavailable must be bool")
        available = not structurally_unavailable
        if stratum in REQUIRED_STRATA:
            _require(available is True, f"required stratum is unavailable: {stratum}")
        batch_indices = record.get("batch_indices")
        _require(isinstance(batch_indices, list), f"{stratum}.batch_indices is missing")
        if not available:
            _require(stratum == "background_only", "only background may be unavailable")
            _require(record.get("candidate_count") == 0, "unavailable background candidate_count must be zero")
            reason = record.get("availability_reason")
            _require(isinstance(reason, str) and reason, "unavailable background lacks reason")
            _require(batch_indices == [], "unavailable background must have no partial batches")
            normalized_strata[stratum] = {
                "available": False,
                "structurally_unavailable": True,
                "candidate_count": 0,
                "observed_natural_candidate_count": 0,
                "reason": reason,
                "batches": [],
            }
            continue
        _require(len(batch_indices) == BATCHES_PER_STRATUM, f"{stratum} batch count differs")
        normalized_batches: list[dict[str, Any]] = []
        source_counts: dict[str, int] = {}
        sample_count = 0
        for local_batch_index, global_batch_index in enumerate(batch_indices):
            _require(type(global_batch_index) is int and 0 <= global_batch_index < len(raw_batches), "global batch index is invalid")
            batch = raw_batches[global_batch_index]
            _require(isinstance(batch, Mapping), "manifest batch is malformed")
            _require(batch.get("batch_index") == global_batch_index, "manifest global batch index differs")
            _require(batch.get("stratum") == stratum, "manifest batch stratum differs")
            _require(batch.get("stratum_batch_index") == local_batch_index, "manifest stratum batch index differs")
            _require(batch.get("batch_size") == BATCH_SIZE, "manifest batch size differs")
            record_indices = batch.get("record_indices")
            _require(isinstance(record_indices, list) and len(record_indices) == BATCH_SIZE, "manifest record indices differ")
            samples: list[Mapping[str, Any]] = []
            for record_index in record_indices:
                _require(type(record_index) is int and 0 <= record_index < len(records), "manifest record index is invalid")
                sample = records[record_index]
                _require(isinstance(sample, Mapping), "manifest sample is malformed")
                _require(sample.get("record_index") == record_index, "manifest record index binding differs")
                _require(sample.get("batch_index") == global_batch_index, "sample batch binding differs")
                _require(sample.get("stratum_batch_index") == local_batch_index, "sample stratum-batch binding differs")
                samples.append(sample)
            image_sha = batch.get("images_tensor_sha256")
            mask_sha = batch.get("masks_tensor_sha256")
            _require(_is_sha256(image_sha), "batch image SHA is invalid")
            _require(_is_sha256(mask_sha), "batch mask SHA is invalid")
            for sample in samples:
                sample_id = sample.get("source_id")
                namespaced = sample.get("namespaced_source_id")
                epoch = sample.get("epoch")
                _require(isinstance(sample_id, str) and sample_id, "sample_id is invalid")
                _require(namespaced == f"{dataset}::{sample_id}", "namespaced sample ID differs")
                _require(type(epoch) is int and epoch in epochs, "sample audit epoch differs")
                identity = (epoch, namespaced)
                _require(identity not in global_identities, "manifest crop identity is duplicated")
                global_identities.add(identity)
                _require(sample.get("stratum") == stratum, "sample stratum differs")
                _require(type(sample.get("mixed_tiny")) is bool, "sample mixed_tiny is not bool")
                _require(type(sample.get("dataset_index")) is int, "sample dataset_index is invalid")
                _require(type(sample.get("augmentation_seed")) is int, "sample augmentation seed is invalid")
                _require(isinstance(sample.get("transform_plan"), Mapping), "sample transform plan is missing")
                _require(_is_sha256(sample.get("image_tensor_sha256")), "sample image tensor SHA is invalid")
                _require(_is_sha256(sample.get("mask_tensor_sha256")), "sample mask tensor SHA is invalid")
                source_counts[sample_id] = source_counts.get(sample_id, 0) + 1
                sample_count += 1
            normalized_batches.append(
                {
                    "batch_index": local_batch_index,
                    "manifest_global_batch_index": global_batch_index,
                    "samples": samples,
                    "images_sha256": image_sha,
                    "masks_sha256": mask_sha,
                }
            )
        _require(sample_count == SAMPLES_PER_STRATUM, f"{stratum} sample count differs")
        effective_target = record.get("effective_min_distinct_sources")
        effective_cap = record.get("effective_max_samples_per_source")
        _require(
            type(effective_target) is int
            and type(effective_cap) is int
            and effective_target >= 16
            and effective_cap >= MAX_CROPS_PER_SOURCE,
            f"{stratum} effective diversity contract is invalid",
        )
        diversity_limited = record.get(
            "diversity_target_limited_by_natural_availability"
        )
        _require(type(diversity_limited) is bool, f"{stratum} diversity-limited flag differs")
        natural_ceiling = record.get("natural_distinct_source_ceiling")
        _require(type(natural_ceiling) is int, f"{stratum} natural ceiling is invalid")
        proof_sha: str | None = None
        if diversity_limited:
            _require(
                16 <= natural_ceiling < DISTINCT_SOURCES_PER_STRATUM
                and effective_target == natural_ceiling,
                f"{stratum} natural-availability exception differs",
            )
            proof = record.get("exhaustive_natural_availability_proof")
            _require(isinstance(proof, Mapping), f"{stratum} exhaustive proof is missing")
            proof_sha = proof.get("proof_sha256")
            _require(_is_sha256(proof_sha), f"{stratum} exhaustive proof SHA is invalid")
            _require(
                proof.get("full_epoch_range_covered_for_every_non_ruled_out_source")
                is True,
                f"{stratum} exhaustive proof range differs",
            )
            proof_sources = proof.get("matching_source_ids")
            _require(
                isinstance(proof_sources, list)
                and set(proof_sources) == set(source_counts),
                f"{stratum} exhaustive proof source universe differs",
            )
        else:
            _require(
                effective_target == DISTINCT_SOURCES_PER_STRATUM
                and effective_cap == MAX_CROPS_PER_SOURCE
                and natural_ceiling >= DISTINCT_SOURCES_PER_STRATUM,
                f"{stratum} default diversity contract differs",
            )
            _require(
                record.get("exhaustive_natural_availability_proof") is None,
                f"{stratum} has an unnecessary availability proof",
            )
        _require(len(source_counts) == effective_target, f"{stratum} distinct-source coverage differs")
        _require(max(source_counts.values()) <= effective_cap, f"{stratum} source crop cap differs")
        _require(
            max(source_counts.values()) - min(source_counts.values()) <= 1,
            f"{stratum} source repetition is not balanced",
        )
        _require(record.get("selected_count") == sample_count, f"{stratum} declared sample count differs")
        _require(record.get("selected_distinct_source_count") == len(source_counts), f"{stratum} declared source count differs")
        _require(record.get("coverage_pass") is True, f"{stratum} coverage flag differs")
        normalized_strata[stratum] = {
            "available": True,
            "structurally_unavailable": False,
            "candidate_count": record.get("candidate_count"),
            "observed_natural_candidate_count": record.get("candidate_count"),
            "reason": None,
            "sample_count": sample_count,
            "distinct_source_count": len(source_counts),
            "natural_distinct_source_ceiling": natural_ceiling,
            "diversity_target": effective_target,
            "max_repeat_cap": effective_cap,
            "diversity_target_limited_by_natural_availability": diversity_limited,
            "exhaustive_natural_availability_proof_sha256": proof_sha,
            "batches": normalized_batches,
        }
    return {
        "train_index_binding": dict(train_binding),
        "normalization": data_protocol.get_legacy_normalization(dataset),
        "candidate_epoch_selection": dict(epoch_selection),
        "strata": normalized_strata,
        "manifest_source_sha256": dict(bindings["source_sha256"]),
        "protocol_manifest_file_sha256": expected_protocol_sha,
    }


def reconstruct_manifest_batch(
    dataset_object: paper_data.ThreeDatasetV2TrainDataset,
    *,
    dataset: str,
    stratum: str,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    samples = batch["samples"]
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    identifiers: list[str] = []
    for manifest_sample in samples:
        epoch = int(manifest_sample["epoch"])
        dataset_object.set_epoch(epoch)
        observed = dataset_object[int(manifest_sample["dataset_index"])]
        _require(isinstance(observed, Mapping), "metadata-enabled dataset returned a non-mapping")
        _require(observed["sample_id"] == manifest_sample["source_id"], "replayed sample ID differs")
        _require(observed["namespaced_sample_id"] == manifest_sample["namespaced_source_id"], "replayed namespaced ID differs")
        _require(observed["epoch"] == epoch, "replayed epoch differs")
        _require(observed["augmentation_seed"] == manifest_sample["augmentation_seed"], "replayed augmentation seed differs")
        _require(
            json.loads(json.dumps(observed["transform_plan"], sort_keys=True))
            == json.loads(json.dumps(manifest_sample["transform_plan"], sort_keys=True)),
            "replayed transform plan differs",
        )
        image = observed["image"]
        mask = observed["mask"]
        _require(tensor_sha256(image) == manifest_sample["image_tensor_sha256"], "replayed sample image SHA differs")
        _require(tensor_sha256(mask) == manifest_sample["mask_tensor_sha256"], "replayed sample mask SHA differs")
        images.append(image)
        masks.append(mask)
        identifiers.append(str(observed["namespaced_sample_id"]))
    image_batch = torch.stack(images, dim=0)
    mask_batch = torch.stack(masks, dim=0)
    _require(tensor_sha256(image_batch) == batch["images_sha256"], "replayed batch image SHA differs")
    _require(tensor_sha256(mask_batch) == batch["masks_sha256"], "replayed batch mask SHA differs")
    _require(all(sample["stratum"] == stratum for sample in samples), "batch stratum differs")
    return image_batch, mask_batch, identifiers


def _runtime_source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_ds_gradient_audit_v1.py": Path(__file__),
        "experiments/tpd_training_loss.py": REPO_ROOT / "experiments" / "tpd_training_loss.py",
        "experiments/paper_three_dataset_v2.py": Path(paper_data.__file__),
        "experiments/three_dataset_v2_protocol.py": Path(data_protocol.__file__),
        "experiments/train_three_dataset_tss_off_seed42_v1.py": Path(tss_off_runner.__file__),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(tss_off_adapter.__file__),
        "experiments/four_dataset_models_seed42_v1.py": Path(model_builder.__file__),
    }
    return {name: file_sha256(path.resolve(strict=True)) for name, path in sources.items()}


def _load_training_checkpoint(
    *,
    dataset: str,
    checkpoint_role: str,
    run_dir: Path,
    protocol_manifest_path: Path,
    protocol_manifest: Mapping[str, Any],
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    tss_off_adapter.configure_core()
    request = evaluator.EvaluationRequest(
        dataset=dataset,
        method="final",
        checkpoint_role=checkpoint_role,
        requested_tss_weight=0.0,
    )
    request.validate()
    payload, binding = checkpoint_adapter._load_checkpoint_allowing_added_sources(
        request,
        Path(run_dir),
        protocol_manifest_path,
        protocol_manifest,
    )
    model, metadata = tss_off_runner._build_method_model("final", SEED, dataset_name=dataset)
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "strict training checkpoint load differs")
    state_sha = model_builder.state_dict_sha256(model.state_dict())
    _require(state_sha == model_builder.state_dict_sha256(payload["state_dict"]), "loaded training state SHA differs")
    binding = dict(binding)
    binding["training_state_dict_sha256"] = state_sha
    binding["checkpoint_payload_schema"] = payload.get("schema")
    binding["checkpoint_payload_recipe"] = payload.get("recipe")
    return model, metadata, binding


def _summary_for_replay(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "losses": batch["losses"],
        "shared_groups": batch["shared_groups"],
        "head_local_gradients": batch["head_local_gradients"],
        "tss_control_all_unused": batch["tss_control_all_unused"],
        "vjp_call_count": batch["vjp_call_count"],
        "leaf_grad_all_none_during_audit": batch["leaf_grad_all_none_during_audit"],
    }


def sentinel_batch_binding(dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve tiny batch zero by its stratum-local index, never global order."""

    strata = dataset_manifest.get("strata")
    _require(isinstance(strata, Mapping), "normalized manifest strata are missing")
    tiny = strata.get("tiny_positive")
    _require(isinstance(tiny, Mapping), "normalized tiny stratum is missing")
    _require(tiny.get("available") is True, "tiny sentinel stratum is unavailable")
    batches = tiny.get("batches")
    _require(isinstance(batches, list), "tiny sentinel batches are missing")
    matches = [batch for batch in batches if batch.get("batch_index") == 0]
    _require(len(matches) == 1, "tiny sentinel requires one local batch zero")
    selected = matches[0]
    global_index = selected.get("manifest_global_batch_index")
    _require(type(global_index) is int and global_index >= 0, "sentinel global batch index is invalid")
    return {
        "stratum": "tiny_positive",
        "stratum_batch_index": 0,
        "manifest_global_batch_index": global_index,
    }


def analyze_run(
    *,
    dataset: str,
    checkpoint_role: str,
    manifest_path: Path,
    run_dir: Path,
    dataset_root: Path,
    data_protocol_manifest_path: Path,
    device_name: str,
) -> dict[str, Any]:
    data_protocol.require_dataset(dataset)
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(SEED == data_protocol.PROTOCOL_SEED, "formal seed differs")
    engine.configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    frozen_sources = _runtime_source_sha256()

    protocol_path = Path(data_protocol_manifest_path).resolve(strict=True)
    protocol_payload = data_protocol.load_protocol_manifest(
        protocol_path, dataset_root=dataset_root
    )
    audit_manifest_path = Path(manifest_path).resolve(strict=True)
    audit_manifest = _json_object(audit_manifest_path, "DS gradient audit manifest")
    dataset_manifest = validate_manifest(
        audit_manifest,
        dataset=dataset,
        protocol_manifest=protocol_payload,
    )
    _require(
        dataset_manifest["protocol_manifest_file_sha256"]
        == file_sha256(protocol_path),
        "audit manifest protocol-file SHA differs",
    )
    model, model_metadata, checkpoint_binding = _load_training_checkpoint(
        dataset=dataset,
        checkpoint_role=checkpoint_role,
        run_dir=run_dir,
        protocol_manifest_path=protocol_path,
        protocol_manifest=protocol_payload,
    )
    model.to(device)
    partition = build_parameter_partition(model, require_formal_counts=True)
    _validate_no_backward_hooks(model)
    _require(all(parameter.grad is None for parameter in model.parameters()), "loaded model has populated leaf .grad")
    checkpoint_state_sha = checkpoint_adapter.module_state_sha256(model)

    train_dataset = paper_data.ThreeDatasetV2TrainDataset(
        dataset,
        dataset_root=dataset_root,
        protocol_manifest=protocol_path,
        patch_size=data_protocol.PATCH_SIZE,
        seed=SEED,
        return_metadata=True,
    )
    strata_output: dict[str, Any] = {}
    sentinel_replay: dict[str, Any] | None = None
    sentinel_binding = sentinel_batch_binding(dataset_manifest)
    batch_model_state_shas: set[str] = set()
    for stratum in STRATUM_ORDER:
        stratum_manifest = dataset_manifest["strata"][stratum]
        if stratum_manifest["available"] is not True:
            strata_output[stratum] = {
                "required_or_conditional": "conditional",
                "available": False,
                "structurally_unavailable": True,
                "candidate_count": 0,
                "observed_natural_candidate_count": stratum_manifest.get(
                    "observed_natural_candidate_count", 0
                ),
                "reason": stratum_manifest["reason"],
                "sample_count": 0,
                "distinct_source_count": 0,
                "batch_count": 0,
                "batches": [],
                "aggregate": None,
            }
            continue
        ready_batches: list[dict[str, Any]] = []
        source_ids: set[str] = set()
        for stratum_batch_index, manifest_batch in enumerate(
            stratum_manifest["batches"]
        ):
            _require(
                manifest_batch.get("batch_index") == stratum_batch_index,
                "normalized stratum-local batch index differs",
            )
            manifest_global_batch_index = int(
                manifest_batch["manifest_global_batch_index"]
            )
            images_cpu, masks_cpu, identifiers = reconstruct_manifest_batch(
                train_dataset,
                dataset=dataset,
                stratum=stratum,
                batch=manifest_batch,
            )
            source_ids.update(
                sample["source_id"] for sample in manifest_batch["samples"]
            )
            images = images_cpu.to(device, non_blocking=device.type == "cuda")
            masks = masks_cpu.to(device, non_blocking=device.type == "cuda")
            audited = audit_one_loaded_batch(
                model,
                images,
                masks,
                partition,
                dataset=dataset,
                stratum=stratum,
                batch_index=stratum_batch_index,
                expected_sample_ids=identifiers,
            )
            audited["stratum_batch_index"] = stratum_batch_index
            audited["manifest_global_batch_index"] = manifest_global_batch_index
            _require(audited["images_sha256"] == manifest_batch["images_sha256"], "device-roundtrip image SHA differs")
            _require(audited["masks_sha256"] == manifest_batch["masks_sha256"], "device-roundtrip mask SHA differs")
            batch_model_state_shas.add(audited["restoration"]["model_state_sha256_after"])
            is_sentinel = (
                stratum == sentinel_binding["stratum"]
                and stratum_batch_index
                == sentinel_binding["stratum_batch_index"]
                and manifest_global_batch_index
                == sentinel_binding["manifest_global_batch_index"]
            )
            if is_sentinel:
                replay = audit_one_loaded_batch(
                    model,
                    images,
                    masks,
                    partition,
                    dataset=dataset,
                    stratum=stratum,
                    batch_index=stratum_batch_index,
                    expected_sample_ids=identifiers,
                )
                first_summary = _summary_for_replay(audited)
                second_summary = _summary_for_replay(replay)
                first_sha = canonical_sha256(first_summary)
                second_sha = canonical_sha256(second_summary)
                _require(first_summary == second_summary, "sentinel replay gradient summary differs")
                sentinel_replay = {
                    "stratum": stratum,
                    "batch_index": stratum_batch_index,
                    "stratum_batch_index": stratum_batch_index,
                    "manifest_global_batch_index": manifest_global_batch_index,
                    "repeat_count": 2,
                    "same_checkpoint_state_and_forward_rng_start": True,
                    "first_summary_sha256": first_sha,
                    "second_summary_sha256": second_sha,
                    "replay_exact": first_sha == second_sha,
                }
            ready_batches.append(audited)
            print(
                "PROGRESS "
                f"dataset={dataset} role={checkpoint_role} "
                f"stratum={stratum} batch={stratum_batch_index + 1}/4 "
                f"manifest_global_batch={manifest_global_batch_index} "
                f"sentinel_replay={str(is_sentinel).lower()}",
                flush=True,
            )
            del images, masks, images_cpu, masks_cpu
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _require(len(ready_batches) == BATCHES_PER_STRATUM, "available stratum did not produce four batches")
        strata_output[stratum] = {
            "required_or_conditional": (
                "required" if stratum in REQUIRED_STRATA else "conditional"
            ),
            "available": True,
            "structurally_unavailable": False,
            "candidate_count": stratum_manifest.get("candidate_count"),
            "observed_natural_candidate_count": stratum_manifest.get(
                "observed_natural_candidate_count"
            ),
            "reason": None,
            "sample_count": sum(len(batch["sample_ids"]) for batch in ready_batches),
            "distinct_source_count": len(source_ids),
            "natural_distinct_source_ceiling": stratum_manifest[
                "natural_distinct_source_ceiling"
            ],
            "diversity_target": stratum_manifest["diversity_target"],
            "max_repeat_cap": stratum_manifest["max_repeat_cap"],
            "max_repeats_per_source": stratum_manifest["max_repeat_cap"],
            "diversity_target_limited_by_natural_availability": stratum_manifest[
                "diversity_target_limited_by_natural_availability"
            ],
            "exhaustive_natural_availability_proof_sha256": stratum_manifest[
                "exhaustive_natural_availability_proof_sha256"
            ],
            "natural_diversity_exhaustive_proof_sha256": stratum_manifest[
                "exhaustive_natural_availability_proof_sha256"
            ],
            "batch_count": len(ready_batches),
            "batches": ready_batches,
            "aggregate": aggregate_stratum_batches(ready_batches),
        }

    _require(sentinel_replay is not None and sentinel_replay["replay_exact"] is True, "sentinel replay gate did not pass")
    final_state_sha = checkpoint_adapter.module_state_sha256(model)
    _require(final_state_sha == checkpoint_state_sha, "complete audit changed model state")
    _require(batch_model_state_shas == {checkpoint_state_sha}, "per-batch restored state SHA differs")
    _require(_runtime_source_sha256() == frozen_sources, "runtime source changed during audit")
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "purpose": "train_only_six_head_gradient_conflict_audit",
        "dataset": dataset,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "head_order": list(HEAD_ORDER),
        "auxiliary_heads": list(AUXILIARY_HEADS),
        "checkpoint_binding": checkpoint_binding,
        "manifest_binding": {
            "path": str(audit_manifest_path),
            "sha256": file_sha256(audit_manifest_path),
            "schema": audit_manifest["schema"],
            "namespace": audit_manifest["audit_namespace"],
            "source_sha256": dataset_manifest["manifest_source_sha256"],
        },
        "data_protocol_binding": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
            "schema": protocol_payload["schema"],
            "split": "img_idx/train",
            "train_index_binding": dataset_manifest["train_index_binding"],
            "normalization": data_protocol.get_legacy_normalization(dataset),
            "workers": 0,
        },
        "model": {
            "method": "final_tss_off",
            "training_graph": True,
            "training_model_metadata": model_metadata,
            "model_state_sha256_before": checkpoint_state_sha,
            "model_state_sha256_after": final_state_sha,
        },
        "objective": {
            "formula": "BCE(gt5)+BCE(gt4)+BCE(gt3)+BCE(gt2)+BCE(d0)+BCE(final)",
            "criterion": "torch.nn.BCELoss(reduction=mean)",
            "segmentation_probability_maps": True,
            "tss_weight": 0.0,
            "loss_formula_changed": False,
        },
        "parameter_partition": _public_partition(partition),
        "strata": strata_output,
        "sentinel_replay": sentinel_replay,
        "execution_audit": {
            "train_mode_forward_per_regular_batch": 1,
            "sentinel_additional_replay_forward": 1,
            "vjp_calls_per_forward": 6,
            "ordered_parameter_tuple_reused": True,
            "optimizer_constructed": False,
            "parameter_update_performed": False,
            "backward_hooks_allowed": False,
            "workers": 0,
            "fp32": True,
        },
        "restoration_audit": {
            "all_batches_restored": True,
            "all_parameter_grads_none": True,
            "rng_restored": True,
            "batchnorm_buffers_restored_each_batch": True,
            "module_training_flags_restored_each_batch": True,
            "python_numpy_cpu_cuda_rng_restored_each_batch": True,
            "leaf_grad_always_none": True,
            "model_state_sha256_before": checkpoint_state_sha,
            "model_state_sha256_after": final_state_sha,
            "model_state_unchanged": final_state_sha == checkpoint_state_sha,
        },
        "source_sha256": frozen_sources,
        "interpretation_scope": {
            "train_only": True,
            "test_data_read": False,
            "checkpoint_reselected": False,
            "model_performance_claim_supported": False,
            "ds_v2_training_authorized_by_this_file": False,
        },
    }
    validate_output_payload(output)
    return output


def validate_output_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "output schema differs")
    _require(payload.get("status") == "complete", "output is incomplete")
    _require(payload.get("dataset") in data_protocol.DATASETS, "output dataset differs")
    _require(payload.get("checkpoint_role") in CHECKPOINT_ROLES, "output role differs")
    _require(payload.get("seed") == SEED, "output seed differs")
    _require(payload.get("head_order") == list(HEAD_ORDER), "output head order differs")
    replay = payload.get("sentinel_replay")
    _require(
        isinstance(replay, Mapping)
        and replay.get("stratum") == "tiny_positive"
        and replay.get("batch_index") == 0
        and replay.get("stratum_batch_index") == 0
        and type(replay.get("manifest_global_batch_index")) is int
        and replay.get("manifest_global_batch_index") >= 0
        and replay.get("repeat_count") == 2
        and replay.get("replay_exact") is True
        and replay.get("first_summary_sha256") == replay.get("second_summary_sha256")
        and _is_sha256(replay.get("first_summary_sha256")),
        "sentinel replay contract differs",
    )
    partition = payload.get("parameter_partition")
    _require(isinstance(partition, Mapping), "parameter partition is missing")
    _require(partition.get("all_trainable_parameters_assigned_once") is True, "atomic partition is not exhaustive")
    _require(partition.get("shared_groups_mutually_exclusive") is True, "shared groups overlap")
    _require(set(partition.get("shared_groups", {})) == set(SHARED_ATOMIC_MEMBERS), "shared group scope differs")
    strata = payload.get("strata")
    _require(isinstance(strata, Mapping) and set(strata) == set(STRATUM_ORDER), "output strata differ")
    for stratum in REQUIRED_STRATA:
        record = strata[stratum]
        _require(record.get("available") is True, f"required output stratum unavailable: {stratum}")
        _require(record.get("sample_count") == 64, f"{stratum} output sample count differs")
        _require(record.get("batch_count") == 4, f"{stratum} output batch count differs")
        diversity_target = record.get("diversity_target", 24)
        _require(
            type(diversity_target) is int
            and int(record.get("distinct_source_count", 0)) == diversity_target
            and diversity_target >= 16,
            f"{stratum} output source coverage differs",
        )
    for stratum, record in strata.items():
        if record.get("available") is not True:
            _require(stratum == "background_only" and record.get("aggregate") is None, "unavailable output stratum differs")
            continue
        batches = record.get("batches")
        _require(isinstance(batches, list) and len(batches) == 4, "output stratum batches differ")
        for batch in batches:
            _require(batch.get("vjp_call_count") == 6, "output VJP count differs")
            _require(batch.get("leaf_grad_all_none_during_audit") is True, "output leaf grad flag differs")
            restoration = batch.get("restoration")
            _require(
                isinstance(restoration, Mapping)
                and restoration.get("model_state_unchanged") is True
                and restoration.get("model_state_sha256_before")
                == restoration.get("model_state_sha256_after"),
                "batch restoration differs",
            )
            groups = batch.get("shared_groups")
            _require(isinstance(groups, Mapping) and set(groups) == set(SHARED_ATOMIC_MEMBERS), "batch shared groups differ")
            for group in groups.values():
                gram = group.get("gram_6x6")
                _require(isinstance(gram, list) and len(gram) == 6, "batch Gram differs")
                _require(all(isinstance(row, list) and len(row) == 6 for row in gram), "batch Gram shape differs")
                _require(all(math.isfinite(float(value)) for row in gram for value in row), "batch Gram is non-finite")
    restoration = payload.get("restoration_audit")
    _require(
        isinstance(restoration, Mapping)
        and restoration.get("model_state_unchanged") is True
        and restoration.get("leaf_grad_always_none") is True,
        "global restoration audit differs",
    )


def _default_run_dir(dataset: str) -> Path:
    return DEFAULT_CHECKPOINT_ROOT / "runs" / dataset / "final_tss_off" / "seed_42"


def _default_output(dataset: str, checkpoint_role: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / "runs" / dataset / checkpoint_role / "audit.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument("--checkpoint-role", choices=CHECKPOINT_ROLES, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--data-protocol-manifest",
        type=Path,
        default=data_protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    args.run_dir = args.run_dir or _default_run_dir(args.dataset)
    args.output = args.output or _default_output(args.dataset, args.checkpoint_role)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output before audit: {args.output}")
    output = analyze_run(
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        manifest_path=args.manifest,
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest_path=args.data_protocol_manifest,
        device_name=args.device,
    )
    atomic_create_json(args.output, output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "dataset": args.dataset,
                "checkpoint_role": args.checkpoint_role,
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
