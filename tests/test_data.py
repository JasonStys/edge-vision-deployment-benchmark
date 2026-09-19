"""File: Tests generation, stratification, corruptions, persistence, and data rejection.

Functions: test cases cover the public data contract. Variables: datasets and temporary paths are
test-local; exact declaration lines are generated in docs/CODE_INDEX.md.
"""

from pathlib import Path

import numpy as np
import pytest

from edgevision.constants import CLASS_COUNT, INPUT_SIZE
from edgevision.data import (
    VisionDataset,
    apply_corruption,
    dataset_digest,
    generate_dataset,
    load_dataset,
    save_dataset,
)


def test_generation_is_deterministic_balanced_and_stratified() -> None:
    first = generate_dataset(samples_per_class=20, seed=7)
    second = generate_dataset(samples_per_class=20, seed=7)
    assert dataset_digest(first) == dataset_digest(second)
    assert first.train_x.shape[1] == INPUT_SIZE
    for labels in (first.train_y, first.validation_y, first.test_y):
        assert set(labels.tolist()) == set(range(CLASS_COUNT))
        counts = np.bincount(labels, minlength=CLASS_COUNT)
        assert len(set(counts.tolist())) == 1


@pytest.mark.parametrize("kind", ["gaussian", "occlusion", "dim"])
def test_corruptions_are_bounded_and_non_mutating(kind: str) -> None:
    dataset = generate_dataset(samples_per_class=10)
    original = dataset.test_x.copy()
    corrupted = apply_corruption(dataset.test_x, kind)  # type: ignore[arg-type]
    assert np.array_equal(dataset.test_x, original)
    assert corrupted.shape == original.shape
    assert np.all((corrupted >= 0.0) & (corrupted <= 1.0))
    assert not np.array_equal(corrupted, original)


def test_dataset_round_trip_and_archive_contract(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=12)
    path = tmp_path / "dataset.npz"
    manifest = save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert dataset_digest(loaded) == manifest["dataset_sha256"]
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, train_x=dataset.train_x)
    with pytest.raises(ValueError, match="split contract"):
        load_dataset(invalid)


def test_dataset_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="samples_per_class"):
        generate_dataset(samples_per_class=2)
    dataset = generate_dataset(samples_per_class=10)
    with pytest.raises(ValueError, match="unsupported corruption"):
        apply_corruption(dataset.test_x, "blur")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        VisionDataset(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            dataset.validation_x,
            dataset.validation_y,
            dataset.test_x,
            dataset.test_y,
        )
