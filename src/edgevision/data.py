"""File: Generates, validates, corrupts, hashes, and persists the deterministic vision dataset.

Functions: generate_dataset, apply_corruption, dataset_digest, save_dataset, and load_dataset own
the complete data contract. Classes: VisionDataset groups stratified splits. Variables: local arrays
remain bounded by constants.py; exact declaration locations are in ``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from edgevision.constants import (
    CLASS_COUNT,
    DEFAULT_SEED,
    DIAGONAL_LABEL,
    IMAGE_SIDE,
    INPUT_SIZE,
    MATRIX_RANK,
    MAX_DATASET_BYTES,
    MAX_DATASET_SAMPLES_PER_CLASS,
    MIN_DATASET_SAMPLES_PER_CLASS,
)

FloatArray = NDArray[np.float32]
LabelArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class VisionDataset:
    """Hold train, validation, and test splits with defensive shape checks."""

    train_x: FloatArray
    train_y: LabelArray
    validation_x: FloatArray
    validation_y: LabelArray
    test_x: FloatArray
    test_y: LabelArray

    def __post_init__(self) -> None:
        """Reject malformed, non-finite, out-of-range, or label-misaligned arrays."""

        for name in ("train", "validation", "test"):
            features = getattr(self, f"{name}_x")
            labels = getattr(self, f"{name}_y")
            if features.ndim != MATRIX_RANK or features.shape[1] != INPUT_SIZE:
                raise ValueError(f"{name} features must have shape [N, {INPUT_SIZE}]")
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError(f"{name} labels must align with features")
            if not np.all(np.isfinite(features)) or np.any((features < 0.0) | (features > 1.0)):
                raise ValueError(f"{name} features must be finite values in [0, 1]")
            if np.any((labels < 0) | (labels >= CLASS_COUNT)):
                raise ValueError(f"{name} labels are outside the supported class range")


def _render_pattern(label: int, offset: int, thickness: int) -> FloatArray:
    """Render one bounded 8x8 line-orientation prototype without external image assets."""

    image = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float32)
    center = IMAGE_SIDE // 2 + offset
    for width_offset in range(thickness):
        displacement = width_offset - (thickness // 2)
        if label == 0:
            column = int(np.clip(center + displacement, 0, IMAGE_SIDE - 1))
            image[1:-1, column] = 1.0
        elif label == 1:
            row = int(np.clip(center + displacement, 0, IMAGE_SIDE - 1))
            image[row, 1:-1] = 1.0
        elif label == DIAGONAL_LABEL:
            for row in range(1, IMAGE_SIDE - 1):
                column = row + offset + displacement
                if 0 <= column < IMAGE_SIDE:
                    image[row, column] = 1.0
        else:
            raise ValueError("unsupported class label")
    return image.reshape(INPUT_SIZE)


def _stratified_indices(
    labels: LabelArray, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Create deterministic 60/20/20 indexes while preserving every class in every split."""

    train_parts: list[NDArray[np.int64]] = []
    validation_parts: list[NDArray[np.int64]] = []
    test_parts: list[NDArray[np.int64]] = []
    for label in range(CLASS_COUNT):
        indexes = np.flatnonzero(labels == label).astype(np.int64)
        rng.shuffle(indexes)
        train_end = max(1, int(indexes.size * 0.6))
        validation_end = max(train_end + 1, int(indexes.size * 0.8))
        train_parts.append(indexes[:train_end])
        validation_parts.append(indexes[train_end:validation_end])
        test_parts.append(indexes[validation_end:])
    outputs = tuple(np.concatenate(parts) for parts in (train_parts, validation_parts, test_parts))
    for output in outputs:
        rng.shuffle(output)
    return outputs  # type: ignore[return-value]


def generate_dataset(samples_per_class: int = 80, seed: int = DEFAULT_SEED) -> VisionDataset:
    """Generate a balanced synthetic dataset with deterministic geometry and sensor noise."""

    if not MIN_DATASET_SAMPLES_PER_CLASS <= samples_per_class <= MAX_DATASET_SAMPLES_PER_CLASS:
        raise ValueError("samples_per_class must be between 10 and the documented safety limit")
    rng = np.random.default_rng(seed)
    features: list[FloatArray] = []
    labels: list[int] = []
    for label in range(CLASS_COUNT):
        for _ in range(samples_per_class):
            base = _render_pattern(label, int(rng.integers(-1, 2)), int(rng.integers(1, 3)))
            noisy = np.clip(base + rng.normal(0.0, 0.055, INPUT_SIZE), 0.0, 1.0)
            features.append(noisy.astype(np.float32))
            labels.append(label)
    all_x = np.stack(features).astype(np.float32)
    all_y = np.asarray(labels, dtype=np.int64)
    train, validation, test = _stratified_indices(all_y, rng)
    return VisionDataset(
        all_x[train],
        all_y[train],
        all_x[validation],
        all_y[validation],
        all_x[test],
        all_y[test],
    )


def apply_corruption(
    features: FloatArray,
    kind: Literal["gaussian", "occlusion", "dim"],
    *,
    seed: int = DEFAULT_SEED,
) -> FloatArray:
    """Apply a bounded deployment corruption without mutating caller-owned input."""

    if features.ndim != MATRIX_RANK or features.shape[1] != INPUT_SIZE:
        raise ValueError(f"features must have shape [N, {INPUT_SIZE}]")
    output = features.copy()
    if kind == "gaussian":
        noise = np.random.default_rng(seed).normal(0.0, 0.16, output.shape)
        output = np.clip(output + noise, 0.0, 1.0).astype(np.float32)
    elif kind == "occlusion":
        images = output.reshape((-1, IMAGE_SIDE, IMAGE_SIDE))
        images[:, 3:5, 3:5] = 0.0
    elif kind == "dim":
        output *= np.float32(0.45)
    else:
        raise ValueError(f"unsupported corruption: {kind}")
    return output


def dataset_digest(dataset: VisionDataset) -> str:
    """Return a stable SHA-256 digest over typed split names, shapes, and array bytes."""

    digest = hashlib.sha256()
    for name in ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y"):
        value = np.ascontiguousarray(getattr(dataset, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_dataset(dataset: VisionDataset, path: Path) -> dict[str, object]:
    """Persist a compressed dataset and adjacent integrity manifest with stable JSON fields."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_x=dataset.train_x,
        train_y=dataset.train_y,
        validation_x=dataset.validation_x,
        validation_y=dataset.validation_y,
        test_x=dataset.test_x,
        test_y=dataset.test_y,
    )
    manifest: dict[str, object] = {
        "dataset_sha256": dataset_digest(dataset),
        "format": "numpy-npz",
        "image_shape": [IMAGE_SIDE, IMAGE_SIDE],
        "split_counts": {
            "train": int(dataset.train_y.size),
            "validation": int(dataset.validation_y.size),
            "test": int(dataset.test_y.size),
        },
    }
    with path.with_suffix(".manifest.json").open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_dataset(path: Path) -> VisionDataset:
    """Load only non-pickle arrays and reapply the full data contract."""

    if not path.is_file() or path.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("dataset file is missing or exceeds the 50 MB safety limit")
    with np.load(path, allow_pickle=False) as archive:
        required = ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y")
        if set(archive.files) != set(required):
            raise ValueError("dataset archive does not match the required split contract")
        return VisionDataset(*(archive[name] for name in required))
