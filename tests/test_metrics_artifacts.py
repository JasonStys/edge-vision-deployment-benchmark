"""File: Tests classification metrics, benchmark evidence, CSV contracts, and manifest integrity.

Functions: test cases cover metrics and artifact boundaries. Variables: arrays and temporary files
are test-local; exact declaration lines are generated in docs/CODE_INDEX.md.
"""

from pathlib import Path

import numpy as np
import pytest

from edgevision.artifacts import (
    read_predictions,
    read_vectors,
    verify_manifest,
    write_manifest,
    write_predictions,
    write_vectors,
)
from edgevision.data import generate_dataset
from edgevision.metrics import benchmark_predictor, classification_metrics


def test_classification_and_benchmark_metrics() -> None:
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    probabilities = np.eye(3, dtype=np.float32) * 0.9 + np.float32(1.0 / 30.0)
    report = classification_metrics(labels, probabilities)
    assert report["accuracy"] == 1.0
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    features = np.zeros((4, 64), dtype=np.float32)
    benchmark = benchmark_predictor(
        lambda rows: np.tile(probabilities[:1], (rows.shape[0], 1)), features
    )
    assert benchmark["throughput_images_per_second"] > 0.0
    assert "not a hardware reading" in str(benchmark["energy_proxy_method"])


def test_vector_prediction_and_manifest_round_trip(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=10)
    vector_path = tmp_path / "vectors.csv"
    prediction_path = tmp_path / "predictions.csv"
    write_vectors(vector_path, dataset.test_x, dataset.test_y)
    identifiers, labels, features = read_vectors(vector_path)
    assert np.array_equal(labels, dataset.test_y)
    assert np.allclose(features, dataset.test_x)
    probabilities = np.full((len(identifiers), 3), 1.0 / 3.0, dtype=np.float32)
    write_predictions(prediction_path, identifiers, probabilities)
    restored_ids, restored = read_predictions(prediction_path)
    assert restored_ids == identifiers
    assert np.allclose(restored, probabilities)
    manifest = tmp_path / "manifest.json"
    write_manifest(tmp_path, [Path("vectors.csv"), Path("predictions.csv")], manifest)
    verify_manifest(tmp_path, manifest)
    vector_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_manifest(tmp_path, manifest)


def test_metrics_reject_malformed_probabilities() -> None:
    with pytest.raises(ValueError, match="one supported-class row"):
        classification_metrics(np.asarray([0]), np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="iteration"):
        benchmark_predictor(lambda rows: rows, np.zeros((1, 64), dtype=np.float32), iterations=1)
