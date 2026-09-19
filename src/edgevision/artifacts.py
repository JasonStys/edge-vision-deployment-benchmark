"""File: Creates and verifies portable checksums, CSV vectors, predictions, and manifests.

Functions: sha256_file, write/read_vectors, write/read_predictions, write_manifest, and
verify_manifest define artifact exchange. Variables: row counts and file sizes are bounded by
constants.py; exact declaration lines are in ``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from edgevision.constants import (
    CLASS_COUNT,
    CLASS_NAMES,
    INPUT_SIZE,
    MATRIX_RANK,
    MAX_PREDICTION_FILE_BYTES,
    MAX_VECTOR_FILE_BYTES,
    MAX_VECTOR_ROWS,
)

FloatArray = NDArray[np.float32]
LabelArray = NDArray[np.int64]


def sha256_file(path: Path) -> str:
    """Hash one bounded repository artifact in streaming chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(65_536):
            digest.update(chunk)
    return digest.hexdigest()


def write_vectors(path: Path, features: FloatArray, labels: LabelArray) -> None:
    """Write native-runtime test vectors with stable identifiers and fixed feature columns."""

    if (
        features.ndim != MATRIX_RANK
        or features.shape[1] != INPUT_SIZE
        or not 1 <= features.shape[0] <= MAX_VECTOR_ROWS
    ):
        raise ValueError("test vectors do not satisfy the bounded input contract")
    if labels.shape != (features.shape[0],):
        raise ValueError("test vector labels must align with features")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["id", "label", *(f"f{index}" for index in range(INPUT_SIZE))])
        for index, (row, label) in enumerate(zip(features, labels, strict=True)):
            writer.writerow(
                [
                    f"case-{index:03d}",
                    CLASS_NAMES[int(label)],
                    *(f"{float(value):.9g}" for value in row),
                ]
            )


def read_vectors(path: Path) -> tuple[list[str], LabelArray, FloatArray]:
    """Read bounded test vectors while rejecting schema drift and invalid numeric values."""

    if not path.is_file() or path.stat().st_size > MAX_VECTOR_FILE_BYTES:
        raise ValueError("test-vector file is missing or exceeds the 5 MB limit")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    expected_header = ["id", "label", *(f"f{index}" for index in range(INPUT_SIZE))]
    if not rows or rows[0] != expected_header or not 1 <= len(rows) - 1 <= MAX_VECTOR_ROWS:
        raise ValueError("test-vector CSV does not match the required schema")
    identifiers: list[str] = []
    labels: list[int] = []
    features: list[list[float]] = []
    for row in rows[1:]:
        if len(row) != 2 + INPUT_SIZE or row[1] not in CLASS_NAMES or not row[0]:
            raise ValueError("test-vector row does not match the required schema")
        identifiers.append(row[0])
        labels.append(CLASS_NAMES.index(row[1]))
        features.append([float(value) for value in row[2:]])
    feature_array = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(feature_array)) or np.any(
        (feature_array < 0.0) | (feature_array > 1.0)
    ):
        raise ValueError("test-vector features must be finite values in [0, 1]")
    return identifiers, np.asarray(labels, dtype=np.int64), feature_array


def write_predictions(path: Path, identifiers: list[str], probabilities: FloatArray) -> None:
    """Write runtime-neutral probability rows for cross-language agreement checks."""

    if probabilities.shape != (len(identifiers), CLASS_COUNT):
        raise ValueError("prediction rows must align with identifiers and supported classes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(["id", *CLASS_NAMES])
        for identifier, row in zip(identifiers, probabilities, strict=True):
            writer.writerow([identifier, *(f"{float(value):.9g}" for value in row)])


def read_predictions(path: Path) -> tuple[list[str], FloatArray]:
    """Read bounded native predictions and reject missing, malformed, or non-finite output."""

    if not path.is_file() or path.stat().st_size > MAX_PREDICTION_FILE_BYTES:
        raise ValueError("prediction file is missing or exceeds the 1 MB limit")
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if not rows or rows[0] != ["id", *CLASS_NAMES] or not 1 <= len(rows) - 1 <= MAX_VECTOR_ROWS:
        raise ValueError("prediction CSV does not match the required schema")
    identifiers = [row[0] for row in rows[1:]]
    if any(len(row) != CLASS_COUNT + 1 or not row[0] for row in rows[1:]):
        raise ValueError("prediction row does not match the required schema")
    probabilities = np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]], dtype=np.float32
    )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("prediction probabilities must be finite")
    return identifiers, probabilities


def write_manifest(root: Path, relative_paths: list[Path], destination: Path) -> None:
    """Write sorted SHA-256 and size evidence for generated repository artifacts."""

    entries = {
        path.as_posix(): {"bytes": (root / path).stat().st_size, "sha256": sha256_file(root / path)}
        for path in sorted(relative_paths)
    }
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        output.write(
            json.dumps({"format_version": 1, "files": entries}, indent=2, sort_keys=True) + "\n"
        )


def verify_manifest(root: Path, manifest_path: Path) -> None:
    """Verify declared paths remain inside the root and match recorded sizes and hashes."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("files"), dict):
        raise ValueError("artifact manifest has an unsupported schema")
    resolved_root = root.resolve()
    for relative, expected in payload["files"].items():
        path = (root / relative).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise ValueError(f"manifest path is unsafe or missing: {relative}")
        if path.stat().st_size != expected.get("bytes") or sha256_file(path) != expected.get(
            "sha256"
        ):
            raise ValueError(f"artifact integrity check failed: {relative}")
