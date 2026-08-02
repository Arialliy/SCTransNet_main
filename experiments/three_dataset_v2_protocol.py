"""Frozen data contract for the three-dataset V2 experiments.

This module is the only data-protocol entry point for the new V2 runner and
evaluator.  It deliberately does not import the legacy four-regime protocol.
The existing ``img_idx/train_*.txt`` and ``img_idx/test_*.txt`` files are
authoritative: their relative paths, byte hashes, parsed order, and counts are
all frozen below.

The split roles are also explicit.  ``train`` is used for optimization and
training-set statistics; ``test`` is used for checkpoint selection and formal
evaluation under the repository's declared test-selected protocol.  This
module never creates a new split.

``NUAA-SIRST::Misc_111`` is resolved non-destructively through an internal
``NUAA-SIRST/masks_corrected`` overlay.  The original raw mask remains in
``NUAA-SIRST/masks`` and is fingerprinted independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image


PROTOCOL_VERSION = 2
PROTOCOL_SEED = 42
PATCH_SIZE = 256
PAD_MULTIPLE = 32
TSS_DOWNSAMPLE = 16
TRAIN_POSITIVE_CROP_PROBABILITY = 0.5
SCHEMA = "sctransnet_three_dataset_v2_protocol/v1"
MANIFEST_ID = "three_dataset_v2_img_idx_and_overlay_v1"

DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
SPLITS = ("train", "test")
SPLIT_ROLES: dict[str, dict[str, Any]] = {
    "train": {
        "role": "optimization_and_train_statistics",
        "model_optimization": True,
        "train_statistics": True,
        "checkpoint_selection": False,
        "formal_evaluation": False,
    },
    "test": {
        "role": "checkpoint_selection_and_formal_evaluation",
        "model_optimization": False,
        "train_statistics": False,
        "checkpoint_selection": True,
        "formal_evaluation": True,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "three_dataset_v2"
DEFAULT_MANIFEST_PATH = (
    DEFAULT_RESULTS_ROOT / "manifests" / "three_dataset_v2_protocol.json"
)

# Both hashes are frozen.  ``file_sha256`` binds exact bytes, while
# ``ordered_ids_sha256`` binds the parsed sequence independently of line
# ending representation.
EXPECTED_SPLITS: dict[str, dict[str, dict[str, Any]]] = {
    "NUAA-SIRST": {
        "train": {
            "index_relpath": "NUAA-SIRST/img_idx/train_NUAA-SIRST.txt",
            "count": 213,
            "file_sha256": (
                "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4"
            ),
            "ordered_ids_sha256": (
                "5cc9a267490af5f5230f34dd64fa872484761956fa730679406208b6ac7253fb"
            ),
        },
        "test": {
            "index_relpath": "NUAA-SIRST/img_idx/test_NUAA-SIRST.txt",
            "count": 214,
            "file_sha256": (
                "e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e"
            ),
            "ordered_ids_sha256": (
                "b8a1b96c74306247d56da4abbaf7871619bc2210c710be51816f40002dc5ad1d"
            ),
        },
    },
    "NUDT-SIRST": {
        "train": {
            "index_relpath": "NUDT-SIRST/img_idx/train_NUDT-SIRST.txt",
            "count": 663,
            "file_sha256": (
                "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3"
            ),
            "ordered_ids_sha256": (
                "4cf3882265e4f0a55e80d58e5e53e5f9a12ed721b6995a42a5e8320ad6f51c75"
            ),
        },
        "test": {
            "index_relpath": "NUDT-SIRST/img_idx/test_NUDT-SIRST.txt",
            "count": 664,
            "file_sha256": (
                "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e"
            ),
            "ordered_ids_sha256": (
                "c1e75c9342a138dfcb90173bae4485a6ac1a172312217f291e1fcead1e9f9f66"
            ),
        },
    },
    "IRSTD-1K": {
        "train": {
            "index_relpath": "IRSTD-1K/img_idx/train_IRSTD-1K.txt",
            "count": 800,
            "file_sha256": (
                "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744"
            ),
            "ordered_ids_sha256": (
                "681e4d741fb857703471d6555faa0d86e931aa790567c28f4254331ea9ba3d95"
            ),
        },
        "test": {
            "index_relpath": "IRSTD-1K/img_idx/test_IRSTD-1K.txt",
            "count": 201,
            "file_sha256": (
                "8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579"
            ),
            "ordered_ids_sha256": (
                "48e0661ba187561d1031b2fa22d4b157fd31e00570775b483b51bb21e1def38b"
            ),
        },
    },
}

# Frozen legacy values used by the already completed three source-dataset
# runs.  They must not be recomputed during V2 recipe comparison.
LEGACY_NORMALIZATION: dict[str, dict[str, float]] = {
    "NUAA-SIRST": {
        "mean": 101.06385040283203,
        "std": 34.619606018066406,
    },
    "NUDT-SIRST": {
        "mean": 107.80905151367188,
        "std": 33.02274703979492,
    },
    "IRSTD-1K": {
        "mean": 87.4661865234375,
        "std": 39.71953201293945,
    },
}

NUAA_MISC111_KEY = "NUAA-SIRST::Misc_111"
NUAA_MISC111_CORRECTION_ID = "nuaa_misc111_internal_overlay_v2"
NUAA_MISC111_PATHS = {
    "image_relpath": "NUAA-SIRST/images/Misc_111.png",
    "raw_mask_relpath": "NUAA-SIRST/masks/Misc_111.png",
    "corrected_mask_relpath": (
        "NUAA-SIRST/masks_corrected/Misc_111.png"
    ),
}
EXPECTED_NUAA_MISC111 = {
    "image_sha256": (
        "72561a22b2d1e09a167563f1f3dab7ee04153aabd87579df749ca15ecf3e60b1"
    ),
    "raw_mask_sha256": (
        "1bec16e5b0413d08f5b01c70faac97c72454586b03d10129fde778db4194a4aa"
    ),
    "corrected_mask_sha256": (
        "7e20ff7267737f367d2ea0545289152710225fe871d7c34c34b2d97c66b06fff"
    ),
    "image_size_width_height": [325, 220],
    "raw_mask_size_width_height": [592, 400],
    "corrected_mask_size_width_height": [325, 220],
}

_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_SUFFIXES = (".png", ".bmp")


class ThreeDatasetV2ProtocolError(ValueError):
    """An operation violates the frozen three-dataset V2 contract."""


def _fail(message: str) -> None:
    raise ThreeDatasetV2ProtocolError(message)


def require_dataset(dataset_name: str) -> str:
    if dataset_name not in DATASETS:
        _fail(f"dataset must be one of {DATASETS}, got {dataset_name!r}")
    return dataset_name


def require_split(split: str) -> str:
    if split not in SPLITS:
        _fail(f"split must be one of {SPLITS}, got {split!r}")
    return split


def require_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        _fail("seed must be an integer")
    if seed != PROTOCOL_SEED:
        _fail(f"formal protocol has one seed only: {PROTOCOL_SEED}")
    return seed


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"expected a regular, non-symlink file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"value cannot be encoded as canonical JSON: {exc}")
    return f"{encoded}\n".encode("utf-8")


def compact_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value cannot be encoded as compact JSON: {exc}")


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    return sha256_bytes(compact_json_bytes(list(ids)))


def get_legacy_normalization(dataset_name: str) -> dict[str, float]:
    dataset_name = require_dataset(dataset_name)
    return dict(LEGACY_NORMALIZATION[dataset_name])


def stable_sha256_uint64(*components: Any) -> int:
    """Reproduce the existing three source-run seed derivation exactly.

    The namespace string intentionally retains its historical spelling for
    compatibility.  Changing it would change every crop, flip, transpose,
    DataLoader seed, and the associated frozen TSS crop statistics.
    """

    digest = hashlib.sha256()
    digest.update(b"sctransnet-four-dataset-seed-v1\0")
    for component in components:
        encoded = compact_json_bytes(component)
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class StatelessTransformPlan:
    augmentation_seed: int
    crop_top: int
    crop_left: int
    crop_size: int
    padded_height: int
    padded_width: int
    crop_attempts: int
    flip_axis0: bool
    flip_axis1: bool
    transpose: bool


def derive_stateless_transform_plan(
    *,
    protocol_seed: int,
    dataset_name: str,
    epoch: int,
    namespaced_id: str,
    image_height: int,
    image_width: int,
    has_positive_in_crop: Callable[[int, int, int], bool],
    patch_size: int = PATCH_SIZE,
    pos_prob: float = TRAIN_POSITIVE_CROP_PROBABILITY,
) -> StatelessTransformPlan:
    """Reproduce the legacy sample-local crop and augmentation draws."""

    require_seed(protocol_seed)
    require_dataset(dataset_name)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        _fail("epoch must be a non-negative integer")
    if not isinstance(namespaced_id, str) or "::" not in namespaced_id:
        _fail("namespaced_id must have the form dataset::sample_id")
    expected_prefix = f"{dataset_name}::"
    if not namespaced_id.startswith(expected_prefix):
        _fail(
            f"namespaced_id must start with {expected_prefix!r} for this run"
        )
    if image_height < 1 or image_width < 1:
        _fail("image dimensions must be positive")
    if isinstance(patch_size, bool) or patch_size != PATCH_SIZE:
        _fail(f"formal protocol patch_size must be {PATCH_SIZE}")
    if (
        isinstance(pos_prob, bool)
        or not isinstance(pos_prob, (int, float))
        or not 0 <= pos_prob <= 1
    ):
        _fail("pos_prob must be in [0, 1]")
    padded_height = max(image_height, patch_size)
    padded_width = max(image_width, patch_size)
    augmentation_seed = stable_sha256_uint64(
        protocol_seed, dataset_name, epoch, namespaced_id
    )
    rng = random.Random(augmentation_seed)
    attempts = 0
    while True:
        attempts += 1
        crop_top = rng.randint(0, padded_height - patch_size)
        crop_left = rng.randint(0, padded_width - patch_size)
        unbiased_accept = rng.random() > pos_prob
        if unbiased_accept or has_positive_in_crop(
            crop_top, crop_left, patch_size
        ):
            break
        if attempts >= 1_000_000:
            _fail("stateless positive-biased crop exceeded safety limit")
    return StatelessTransformPlan(
        augmentation_seed=augmentation_seed,
        crop_top=crop_top,
        crop_left=crop_left,
        crop_size=patch_size,
        padded_height=padded_height,
        padded_width=padded_width,
        crop_attempts=attempts,
        flip_axis0=rng.random() < 0.5,
        flip_axis1=rng.random() < 0.5,
        transpose=rng.random() < 0.5,
    )


def dataloader_seed(
    dataset_name: str, seed: int = PROTOCOL_SEED
) -> int:
    require_dataset(dataset_name)
    require_seed(seed)
    return stable_sha256_uint64(seed, dataset_name, "dataloader")


def make_dataloader_generator(
    dataset_name: str, seed: int = PROTOCOL_SEED
) -> Any:
    try:
        import torch
    except ImportError as exc:
        _fail(f"PyTorch is required for a DataLoader generator: {exc}")
    generator = torch.Generator()
    generator.manual_seed(dataloader_seed(dataset_name, seed))
    return generator


def write_canonical_json(
    path: str | os.PathLike[str], payload: Mapping[str, Any]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(dict(payload))
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def index_path(
    dataset_root: str | os.PathLike[str], dataset_name: str, split: str
) -> Path:
    dataset_name = require_dataset(dataset_name)
    split = require_split(split)
    expected = EXPECTED_SPLITS[dataset_name][split]
    return Path(dataset_root) / str(expected["index_relpath"])


def load_index(
    dataset_root: str | os.PathLike[str], dataset_name: str, split: str
) -> list[str]:
    """Load one frozen index exactly as written, without sorting or sampling."""

    dataset_name = require_dataset(dataset_name)
    split = require_split(split)
    expected = EXPECTED_SPLITS[dataset_name][split]
    path = index_path(dataset_root, dataset_name, split)
    if path.is_symlink() or not path.is_file():
        _fail(f"index must be a regular, non-symlink file: {path}")
    content = path.read_bytes()
    observed_file_sha = sha256_bytes(content)
    if observed_file_sha != expected["file_sha256"]:
        _fail(
            f"{dataset_name} {split} index file SHA-256 mismatch: "
            f"{observed_file_sha} != {expected['file_sha256']}"
        )
    try:
        identifiers = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        _fail(f"index is not UTF-8: {path}: {exc}")
    if len(identifiers) != expected["count"]:
        _fail(
            f"{dataset_name} {split} index count mismatch: "
            f"{len(identifiers)} != {expected['count']}"
        )
    for position, identifier in enumerate(identifiers, start=1):
        if (
            identifier != identifier.strip()
            or _SAMPLE_ID_RE.fullmatch(identifier) is None
        ):
            _fail(
                f"unsafe sample ID at {path}:{position}: {identifier!r}"
            )
    if len(identifiers) != len(set(identifiers)):
        _fail(f"index contains duplicate IDs: {path}")
    observed_order_sha = ordered_ids_sha256(identifiers)
    if observed_order_sha != expected["ordered_ids_sha256"]:
        _fail(
            f"{dataset_name} {split} ordered-ID SHA-256 mismatch: "
            f"{observed_order_sha} != {expected['ordered_ids_sha256']}"
        )
    return identifiers


def _regular_relative_path(root: Path, relative: str, *, label: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        _fail(f"{label} must stay inside dataset_root: {relative!r}")
    root_resolved = root.resolve(strict=True)
    lexical_candidate = root_resolved / rel
    if lexical_candidate.is_symlink() or not lexical_candidate.is_file():
        _fail(
            f"{label} must be a regular, non-symlink file: "
            f"{lexical_candidate}"
        )
    candidate = lexical_candidate.resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        _fail(f"{label} escapes dataset_root: {relative!r}")
    return candidate


def _image_size(path: Path, *, label: str) -> list[int]:
    try:
        with Image.open(path) as image:
            image.load()
            return list(image.size)
    except OSError as exc:
        _fail(f"{label} is not a readable image: {path}: {exc}")


def validate_nuaa_misc111_overlay(
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Verify the internal overlay, raw source, dimensions, and three hashes."""

    root = Path(dataset_root).resolve(strict=True)
    paths = {
        field: _regular_relative_path(root, relpath, label=field)
        for field, relpath in NUAA_MISC111_PATHS.items()
    }
    observed_hashes = {
        "image_sha256": sha256_file(paths["image_relpath"]),
        "raw_mask_sha256": sha256_file(paths["raw_mask_relpath"]),
        "corrected_mask_sha256": sha256_file(
            paths["corrected_mask_relpath"]
        ),
    }
    observed_sizes = {
        "image_size_width_height": _image_size(
            paths["image_relpath"], label="Misc_111 image"
        ),
        "raw_mask_size_width_height": _image_size(
            paths["raw_mask_relpath"], label="Misc_111 raw mask"
        ),
        "corrected_mask_size_width_height": _image_size(
            paths["corrected_mask_relpath"],
            label="Misc_111 corrected mask",
        ),
    }
    for field, observed in {**observed_hashes, **observed_sizes}.items():
        expected = EXPECTED_NUAA_MISC111[field]
        if observed != expected:
            _fail(
                f"NUAA Misc_111 {field} mismatch: {observed!r} != "
                f"{expected!r}"
            )
    if observed_sizes["image_size_width_height"] != observed_sizes[
        "corrected_mask_size_width_height"
    ]:
        _fail("NUAA Misc_111 corrected mask is not aligned to the image")
    if paths["raw_mask_relpath"] == paths["corrected_mask_relpath"]:
        _fail("NUAA Misc_111 raw and corrected masks must be distinct files")
    return {
        "correction_id": NUAA_MISC111_CORRECTION_ID,
        "dataset": "NUAA-SIRST",
        "sample_id": "Misc_111",
        "applies_to_splits": ["test"],
        **NUAA_MISC111_PATHS,
        **observed_hashes,
        **observed_sizes,
        "raw_mask_preserved": True,
        "operation": "internal_path_overlay_only",
    }


def _find_unique_file(directory: Path, sample_id: str) -> Path:
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in _SUPPORTED_SUFFIXES
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        _fail(
            f"expected exactly one data file for {sample_id!r} in "
            f"{directory}, found {len(candidates)}"
        )
    path = candidates[0]
    if path.is_symlink():
        _fail(f"dataset files must not be symlinks: {path}")
    return path.resolve(strict=True)


@dataclass(frozen=True)
class ResolvedSample:
    dataset_name: str
    split: str
    sample_id: str
    image_path: Path
    raw_mask_path: Path
    mask_path: Path
    correction_id: str | None

    @property
    def correction_applied(self) -> bool:
        return self.correction_id is not None


def resolve_sample(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    sample_id: str,
    *,
    split: str,
    known_ids: frozenset[str] | None = None,
) -> ResolvedSample:
    """Resolve one indexed image/mask pair through the frozen V2 overlay.

    A trainer should call :func:`load_frozen_index` once during Dataset
    construction, retain a ``frozenset`` of those IDs, and pass it through
    ``known_ids``.  That is the O(1) hot path.  Omitting ``known_ids`` keeps a
    strict standalone path that revalidates the complete index before use.
    """

    dataset_name = require_dataset(dataset_name)
    split = require_split(split)
    if _SAMPLE_ID_RE.fullmatch(sample_id) is None:
        _fail(f"invalid sample ID: {sample_id!r}")
    membership: Sequence[str] | frozenset[str]
    if known_ids is None:
        membership = load_index(dataset_root, dataset_name, split)
    else:
        if not isinstance(known_ids, frozenset):
            _fail("known_ids fast path requires a frozen verified ID set")
        expected_count = int(EXPECTED_SPLITS[dataset_name][split]["count"])
        if len(known_ids) != expected_count:
            _fail(
                f"known_ids count differs for {dataset_name} {split}: "
                f"{len(known_ids)} != {expected_count}"
            )
        membership = known_ids
    if sample_id not in membership:
        _fail(f"{dataset_name}::{sample_id} is not in frozen {split} index")
    root = Path(dataset_root).resolve(strict=True)
    directory = root / dataset_name
    image_path = _find_unique_file(directory / "images", sample_id)
    raw_mask_path = _find_unique_file(directory / "masks", sample_id)
    mask_path = raw_mask_path
    correction_id: str | None = None
    if dataset_name == "NUAA-SIRST" and sample_id == "Misc_111":
        if split != "test":
            _fail("NUAA Misc_111 correction is valid only in img_idx/test")
        overlay = validate_nuaa_misc111_overlay(root)
        mask_path = _regular_relative_path(
            root,
            str(overlay["corrected_mask_relpath"]),
            label="corrected_mask_relpath",
        )
        correction_id = NUAA_MISC111_CORRECTION_ID
    return ResolvedSample(
        dataset_name=dataset_name,
        split=split,
        sample_id=sample_id,
        image_path=image_path,
        raw_mask_path=raw_mask_path,
        mask_path=mask_path,
        correction_id=correction_id,
    )


def validate_sample_pair(
    sample: ResolvedSample, *, include_hashes: bool = False
) -> dict[str, Any]:
    image_size = _image_size(sample.image_path, label="sample image")
    raw_mask_size = _image_size(sample.raw_mask_path, label="raw mask")
    effective_mask_size = _image_size(sample.mask_path, label="effective mask")
    if image_size != effective_mask_size:
        _fail(
            f"effective image/mask size mismatch for "
            f"{sample.dataset_name}::{sample.sample_id}: "
            f"{image_size} != {effective_mask_size}"
        )
    record: dict[str, Any] = {
        "dataset": sample.dataset_name,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "image_size_width_height": image_size,
        "raw_mask_size_width_height": raw_mask_size,
        "effective_mask_size_width_height": effective_mask_size,
        "correction_applied": sample.correction_applied,
        "correction_id": sample.correction_id,
    }
    if include_hashes:
        record.update(
            {
                "image_sha256": sha256_file(sample.image_path),
                "raw_mask_sha256": sha256_file(sample.raw_mask_path),
                "effective_mask_sha256": sha256_file(sample.mask_path),
            }
        )
    return record


def _build_protocol_payload(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    records: dict[str, Any] = {}
    for dataset_name in DATASETS:
        split_records: dict[str, Any] = {}
        loaded: dict[str, list[str]] = {}
        for split in SPLITS:
            identifiers = load_index(root, dataset_name, split)
            loaded[split] = identifiers
            expected = EXPECTED_SPLITS[dataset_name][split]
            split_records[split] = {
                **expected,
                "role": SPLIT_ROLES[split]["role"],
                "ids": identifiers,
            }
        overlap = sorted(set(loaded["train"]) & set(loaded["test"]))
        if overlap:
            _fail(
                f"{dataset_name} train/test overlap contains "
                f"{len(overlap)} IDs"
            )
        records[dataset_name] = {
            "splits": split_records,
            "train_test_disjoint": True,
        }
    nuaa_test = records["NUAA-SIRST"]["splits"]["test"]["ids"]
    nuaa_train = records["NUAA-SIRST"]["splits"]["train"]["ids"]
    if "Misc_111" not in nuaa_test or "Misc_111" in nuaa_train:
        _fail("NUAA Misc_111 must occur only in the frozen test index")
    overlay = validate_nuaa_misc111_overlay(root)
    return {
        "schema": SCHEMA,
        "manifest_id": MANIFEST_ID,
        "protocol_version": PROTOCOL_VERSION,
        "training_seed": PROTOCOL_SEED,
        "dataset_root": str(root),
        "dataset_scope_closed": True,
        "dataset_order": list(DATASETS),
        "split_order": list(SPLITS),
        "split_roles": SPLIT_ROLES,
        "split_policy": (
            "use existing img_idx/train for optimization and train "
            "statistics; use existing img_idx/test for checkpoint selection "
            "and evaluation; do not resplit, sort, sample, or rewrite IDs"
        ),
        "evaluation_protocol": "img_idx_test_selected",
        "normalization": {
            dataset_name: {
                **LEGACY_NORMALIZATION[dataset_name],
                "source": "frozen_legacy_training_configuration",
                "recomputed_for_v2": False,
            }
            for dataset_name in DATASETS
        },
        "datasets": records,
        "corrections": {NUAA_MISC111_KEY: overlay},
        "raw_masks_modified": False,
    }


def build_protocol_manifest(
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
    output_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build and optionally persist the complete source-bound data lock."""

    payload = _build_protocol_payload(Path(dataset_root))
    if output_path is not None:
        write_canonical_json(output_path, payload)
    return payload


def _load_json_object(
    source: str | os.PathLike[str] | Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular, non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {label}: {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label} must contain one JSON object")
    return payload


def validate_protocol_manifest(
    payload: Mapping[str, Any],
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    """Bind a stored manifest to the complete current frozen data state."""

    observed = dict(payload)
    if observed.get("schema") != SCHEMA:
        _fail("unsupported three-dataset V2 protocol schema")
    if observed.get("manifest_id") != MANIFEST_ID:
        _fail("unexpected three-dataset V2 manifest_id")
    expected = _build_protocol_payload(Path(dataset_root))
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        _fail("protocol manifest differs from the frozen live data contract")
    return observed


def load_protocol_manifest(
    source: str | os.PathLike[str] | Mapping[str, Any] = (
        DEFAULT_MANIFEST_PATH
    ),
    *,
    dataset_root: str | os.PathLike[str] = DEFAULT_DATASET_ROOT,
) -> dict[str, Any]:
    payload = _load_json_object(source, label="three-dataset V2 manifest")
    return validate_protocol_manifest(payload, dataset_root=dataset_root)


def load_frozen_index(
    dataset_root: str | os.PathLike[str],
    dataset_name: str,
    split: str,
    manifest: str | os.PathLike[str] | Mapping[str, Any],
) -> list[str]:
    """Load an index only after the entire three-dataset lock is verified."""

    dataset_name = require_dataset(dataset_name)
    split = require_split(split)
    payload = load_protocol_manifest(manifest, dataset_root=dataset_root)
    identifiers = load_index(dataset_root, dataset_name, split)
    stored = payload["datasets"][dataset_name]["splits"][split]["ids"]
    if stored != identifiers:
        _fail(f"{dataset_name} {split} IDs differ from protocol manifest")
    return identifiers


__all__ = [
    "DATASETS",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_RESULTS_ROOT",
    "EXPECTED_NUAA_MISC111",
    "EXPECTED_SPLITS",
    "LEGACY_NORMALIZATION",
    "MANIFEST_ID",
    "NUAA_MISC111_CORRECTION_ID",
    "NUAA_MISC111_KEY",
    "NUAA_MISC111_PATHS",
    "PROTOCOL_SEED",
    "PROTOCOL_VERSION",
    "PAD_MULTIPLE",
    "PATCH_SIZE",
    "ResolvedSample",
    "SCHEMA",
    "SPLITS",
    "SPLIT_ROLES",
    "StatelessTransformPlan",
    "TRAIN_POSITIVE_CROP_PROBABILITY",
    "TSS_DOWNSAMPLE",
    "ThreeDatasetV2ProtocolError",
    "build_protocol_manifest",
    "canonical_json_bytes",
    "dataloader_seed",
    "derive_stateless_transform_plan",
    "get_legacy_normalization",
    "index_path",
    "load_frozen_index",
    "load_index",
    "load_protocol_manifest",
    "make_dataloader_generator",
    "ordered_ids_sha256",
    "require_dataset",
    "require_split",
    "require_seed",
    "resolve_sample",
    "sha256_file",
    "stable_sha256_uint64",
    "validate_nuaa_misc111_overlay",
    "validate_protocol_manifest",
    "validate_sample_pair",
    "write_canonical_json",
]
