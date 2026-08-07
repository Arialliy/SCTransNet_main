"""Audited, frozen independent-Baseline teacher for IRSTD-BGCR.

Only the schedule-defined epoch-1000 checkpoint may be constructed as a
teacher.  The epoch-713 checkpoint is retained in the audit ledger because it
is the user's operational best-mIoU reference, but it was selected by repeated
official-test evaluation and is therefore fail-closed for every training API.

This module contains no dataset or index access.  Raw final logits are captured
from ``SCTransNet.outc`` during one ordinary evaluation forward and are checked
against the model's public sigmoid probability before being returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from experiments.pbdr_v4_state_contract import state_semantic_sha256
from model.SCTransNet import SCTransNet, get_CTranS_config


SCHEMA = "sctransnet_irstd_baseline_teacher/v1"
CHECKPOINT_WRAPPER_PREFIX = "model."
EXPECTED_STATE_KEY_COUNT = 510

BASELINE_REPOSITORY = Path("/home/ly/SCTransNet")
BASELINE_MODEL_SOURCE = BASELINE_REPOSITORY / "model/SCTransNet.py"
LOCAL_MODEL_SOURCE = Path("/home/ly/SCTransNet_main/model/SCTransNet.py")
MODEL_SOURCE_BYTES = 28_663
MODEL_SOURCE_SHA256 = (
    "5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb"
)


class IRSTDBaselineTeacherError(RuntimeError):
    """A Baseline artifact or teacher operation violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBaselineTeacherError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineCheckpointBinding:
    """Immutable identity and provenance for one audited Baseline checkpoint."""

    name: str
    path: Path
    file_bytes: int
    file_sha256: str
    raw_state_semantic_sha256: str
    normalized_state_semantic_sha256: str
    epoch: int
    state_key_count: int
    historical_official_test_evaluated: bool
    historical_official_test_selected: bool
    selection_provenance: str
    teacher_allowed: bool
    training_use: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["checkpoint_wrapper_prefix"] = CHECKPOINT_WRAPPER_PREFIX
        return payload


OPERATIONAL_BEST = BaselineCheckpointBinding(
    name="epoch713_operational_best_miou",
    path=(
        BASELINE_REPOSITORY
        / "checkpoints/IRSTD-1K/SCTransNet_best_mIoU.pth.tar"
    ),
    file_bytes=45_536_899,
    file_sha256=(
        "5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b"
    ),
    raw_state_semantic_sha256=(
        "8d314d45f68de9b6747c5ada4ea7efc4f62a423a4bdf50db2f2d60bd8509d022"
    ),
    normalized_state_semantic_sha256=(
        "5ecf6f812f00e323ab5f8cec55d0ca86ea9f7db2225080bbc0ea947f44e181a4"
    ),
    epoch=713,
    state_key_count=EXPECTED_STATE_KEY_COUNT,
    historical_official_test_evaluated=True,
    historical_official_test_selected=True,
    selection_provenance=(
        "operational best selected by official-test mIoU after repeated "
        "official-test evaluation from epoch 500"
    ),
    teacher_allowed=False,
    training_use="prohibited_test_selected_contamination",
)

FORMAL_TEACHER = BaselineCheckpointBinding(
    name="epoch1000_schedule_defined_formal_teacher",
    path=(BASELINE_REPOSITORY / "checkpoints/IRSTD-1K/SCTransNet_1000.pth.tar"),
    file_bytes=45_535_091,
    file_sha256=(
        "b4cb66be6e4a410dfd902ba050da82d0b666dd071bfb2c5477a7c3173ff07bc5"
    ),
    raw_state_semantic_sha256=(
        "972e7c15f8da8142da85112f535fb555a86293e12d7341d7c5be653fb4076d9b"
    ),
    normalized_state_semantic_sha256=(
        "1961ed8ee278fde09508145fe537324172599bfa704c181dc53f756578070b5c"
    ),
    epoch=1000,
    state_key_count=EXPECTED_STATE_KEY_COUNT,
    historical_official_test_evaluated=True,
    historical_official_test_selected=False,
    selection_provenance="fixed terminal epoch from the 1000-epoch schedule",
    teacher_allowed=True,
    training_use="frozen_bgcr_teacher_only",
)

BASELINE_CHECKPOINTS = (OPERATIONAL_BEST, FORMAL_TEACHER)


def declared_baseline_audit_manifest() -> dict[str, object]:
    """Return the immutable ledger without opening either checkpoint."""

    return {
        "schema": SCHEMA,
        "checkpoints": [binding.as_dict() for binding in BASELINE_CHECKPOINTS],
        "formal_teacher_name": FORMAL_TEACHER.name,
        "operational_reference_name": OPERATIONAL_BEST.name,
        "operational_reference_is_test_selected": True,
        "operational_reference_training_use_prohibited": True,
        "only_epoch1000_teacher_allowed": True,
        "model_source": {
            "audited_path": str(BASELINE_MODEL_SOURCE),
            "local_path": str(LOCAL_MODEL_SOURCE),
            "bytes": MODEL_SOURCE_BYTES,
            "sha256": MODEL_SOURCE_SHA256,
        },
        "checkpoint_wrapper_prefix": CHECKPOINT_WRAPPER_PREFIX,
        "strict_state_key_count": EXPECTED_STATE_KEY_COUNT,
        "current_run_official_test_accessed": False,
        "current_run_official_test_index_opened": False,
        "current_run_official_test_loader_built": False,
    }


def _require_known_binding(binding: BaselineCheckpointBinding) -> None:
    _require(
        binding in BASELINE_CHECKPOINTS,
        "checkpoint binding is not one of the two audited Baseline artifacts",
    )


def require_teacher_eligible(binding: BaselineCheckpointBinding) -> None:
    """Fail closed unless ``binding`` is exactly the epoch-1000 teacher."""

    _require_known_binding(binding)
    _require(
        binding == FORMAL_TEACHER and binding.teacher_allowed,
        "epoch-713 operational best is official-test-selected and cannot be a "
        "BGCR training teacher",
    )


def _audit_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    _require(path.is_absolute(), "audited artifact path must be absolute")
    _require(path.is_file(), f"audited artifact is missing: {path}")
    _require(path.stat().st_size == expected_bytes, f"artifact byte size differs: {path}")
    _require(_file_sha256(path) == expected_sha256, f"artifact file SHA differs: {path}")


def audit_model_sources() -> dict[str, object]:
    """Verify that the audited and local SCTransNet definitions remain identical."""

    for path in (BASELINE_MODEL_SOURCE, LOCAL_MODEL_SOURCE):
        _audit_file(
            path,
            expected_bytes=MODEL_SOURCE_BYTES,
            expected_sha256=MODEL_SOURCE_SHA256,
        )
    return {
        "audited_path": str(BASELINE_MODEL_SOURCE),
        "local_path": str(LOCAL_MODEL_SOURCE),
        "bytes": MODEL_SOURCE_BYTES,
        "sha256": MODEL_SOURCE_SHA256,
        "identical": True,
    }


def strip_exact_checkpoint_wrapper_prefix(
    state: Mapping[str, torch.Tensor],
    *,
    expected_key_count: int = EXPECTED_STATE_KEY_COUNT,
) -> dict[str, torch.Tensor]:
    """Strip only the observed ``model.`` wrapper; reject mixed/generic stripping."""

    _require(
        isinstance(expected_key_count, int)
        and not isinstance(expected_key_count, bool)
        and expected_key_count > 0,
        "expected_key_count must be a positive integer",
    )
    _require(isinstance(state, Mapping), "checkpoint state_dict must be a mapping")
    _require(len(state) == expected_key_count, "checkpoint state key count differs")
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        _require(isinstance(key, str), "checkpoint state key is not a string")
        _require(
            key.startswith(CHECKPOINT_WRAPPER_PREFIX),
            "every checkpoint key must begin with the exact 'model.' wrapper",
        )
        bare = key.removeprefix(CHECKPOINT_WRAPPER_PREFIX)
        _require(bool(bare), "checkpoint wrapper produced an empty state key")
        _require(isinstance(value, torch.Tensor), f"state value is not a tensor: {key}")
        _require(bare not in stripped, f"state key collision after wrapper strip: {bare}")
        stripped[bare] = value
    _require(len(stripped) == expected_key_count, "stripped state key count differs")
    return stripped


def _load_and_validate_bound_state(
    binding: BaselineCheckpointBinding,
) -> dict[str, torch.Tensor]:
    _require_known_binding(binding)
    _audit_file(
        binding.path,
        expected_bytes=binding.file_bytes,
        expected_sha256=binding.file_sha256,
    )
    payload = torch.load(binding.path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, Mapping), "checkpoint payload must be a mapping")
    epoch = payload.get("epoch")
    _require(
        isinstance(epoch, int) and not isinstance(epoch, bool) and epoch == binding.epoch,
        "checkpoint epoch differs from the audited binding",
    )
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping), "checkpoint lacks a state_dict mapping")
    _require(
        state_semantic_sha256(state) == binding.raw_state_semantic_sha256,
        "checkpoint wrapped semantic state SHA differs",
    )
    stripped = strip_exact_checkpoint_wrapper_prefix(
        state,  # type: ignore[arg-type]
        expected_key_count=binding.state_key_count,
    )
    _require(
        state_semantic_sha256(stripped)
        == binding.normalized_state_semantic_sha256,
        "checkpoint normalized semantic state SHA differs",
    )
    return stripped


def audit_checkpoint_binding(
    binding: BaselineCheckpointBinding,
    *,
    verify_state: bool = True,
) -> dict[str, object]:
    """Audit a declared artifact; this function never constructs a teacher."""

    _require_known_binding(binding)
    if verify_state:
        _load_and_validate_bound_state(binding)
    else:
        _audit_file(
            binding.path,
            expected_bytes=binding.file_bytes,
            expected_sha256=binding.file_sha256,
        )
    result = binding.as_dict()
    result.update(
        {
            "file_verified": True,
            "state_verified": verify_state,
            "teacher_constructed": False,
            "current_run_official_test_accessed": False,
        }
    )
    return result


@torch.inference_mode()
def capture_outc_raw_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return raw ``outc`` logits from one frozen evaluation forward."""

    _require(isinstance(model, nn.Module), "teacher model must be an nn.Module")
    _require(not model.training, "teacher model must remain in evaluation mode")
    _require(
        all(not parameter.requires_grad for parameter in model.parameters()),
        "teacher parameters must all be frozen",
    )
    _require(
        isinstance(image, torch.Tensor)
        and image.ndim == 4
        and image.shape[1] == 1
        and image.dtype.is_floating_point,
        "teacher input must be floating BCHW with one channel",
    )
    _require(bool(torch.isfinite(image).all()), "teacher input contains non-finite values")
    outc = getattr(model, "outc", None)
    _require(isinstance(outc, nn.Module), "teacher model lacks the outc hook point")
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        _require(isinstance(output, torch.Tensor), "outc output is not a tensor")
        captured.append(output)

    handle = outc.register_forward_hook(capture)
    try:
        probability = model(image)
    finally:
        handle.remove()
    _require(len(captured) == 1, "outc hook must fire exactly once")
    raw = captured[0]
    _require(isinstance(probability, torch.Tensor), "teacher public output is not a tensor")
    _require(
        probability.shape == raw.shape
        and probability.dtype == raw.dtype
        and probability.device == raw.device,
        "teacher probability metadata differs from raw logits",
    )
    _require(
        torch.equal(torch.sigmoid(raw), probability),
        "teacher public output is not bitwise sigmoid(outc raw logits)",
    )
    return raw


class FrozenIRSTDBaselineTeacher(nn.Module):
    """Inference-only wrapper whose forward result is the audited raw logit map."""

    def __init__(self, model: SCTransNet) -> None:
        super().__init__()
        self.model = model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> "FrozenIRSTDBaselineTeacher":
        # A caller cannot accidentally put the teacher or its BatchNorm layers
        # into training mode.  ``mode`` is intentionally ignored.
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return capture_outc_raw_logits(self.model, image)

    def audit(self) -> dict[str, object]:
        _require(not self.training and not self.model.training, "teacher is not frozen-eval")
        _require(
            all(not parameter.requires_grad for parameter in self.parameters()),
            "teacher contains a trainable parameter",
        )
        return {
            "schema": SCHEMA,
            "binding": FORMAL_TEACHER.as_dict(),
            "all_parameters_frozen": True,
            "teacher_training": False,
            "backbone_training": False,
            "raw_logit_source": "single_outc_forward_hook",
            "public_probability_crosscheck": "bitwise_sigmoid",
            "current_run_official_test_accessed": False,
        }


def build_formal_teacher() -> FrozenIRSTDBaselineTeacher:
    """Construct the only permitted teacher with a 510-key strict state load."""

    require_teacher_eligible(FORMAL_TEACHER)
    audit_model_sources()
    state = _load_and_validate_bound_state(FORMAL_TEACHER)
    model = SCTransNet(
        get_CTranS_config(),
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="test",
        deepsuper=True,
    )
    expected = model.state_dict()
    _require(len(expected) == EXPECTED_STATE_KEY_COUNT, "local model key count differs")
    _require(set(state) == set(expected), "formal teacher and local model state keys differ")
    incompatible = model.load_state_dict(state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict formal teacher load reported incompatible keys",
    )
    teacher = FrozenIRSTDBaselineTeacher(model)
    teacher.audit()
    return teacher


__all__ = [
    "BASELINE_CHECKPOINTS",
    "BASELINE_MODEL_SOURCE",
    "CHECKPOINT_WRAPPER_PREFIX",
    "EXPECTED_STATE_KEY_COUNT",
    "FORMAL_TEACHER",
    "FrozenIRSTDBaselineTeacher",
    "IRSTDBaselineTeacherError",
    "LOCAL_MODEL_SOURCE",
    "MODEL_SOURCE_SHA256",
    "OPERATIONAL_BEST",
    "audit_checkpoint_binding",
    "audit_model_sources",
    "build_formal_teacher",
    "capture_outc_raw_logits",
    "declared_baseline_audit_manifest",
    "require_teacher_eligible",
    "strip_exact_checkpoint_wrapper_prefix",
]
