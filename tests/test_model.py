"""File: Tests training, calibration, baselines, portable round trips, and rejection.

Functions: test cases validate learning and fail-closed parsing. Variables: models, probabilities,
and temporary artifacts are test-local; exact lines are in docs/CODE_INDEX.md.
"""

from pathlib import Path

import numpy as np
import pytest

from edgevision.data import generate_dataset
from edgevision.model import (
    centroid_baseline,
    fit_temperature,
    read_model,
    train_model,
    write_model,
)


def test_training_converges_and_calibration_is_bounded() -> None:
    dataset = generate_dataset(samples_per_class=30)
    model, summary = train_model(dataset.train_x, dataset.train_y, epochs=220)
    assert summary.final_loss < summary.initial_loss * 0.4
    temperature = fit_temperature(model, dataset.validation_x, dataset.validation_y)
    calibrated = model.calibrated(temperature)
    probabilities = calibrated.predict_proba(dataset.test_x)
    assert np.mean(np.argmax(probabilities, axis=1) == dataset.test_y) >= 0.95
    assert np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6)


def test_portable_model_round_trip_and_trailing_data_rejection(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=12)
    model, _ = train_model(dataset.train_x, dataset.train_y, epochs=40)
    path = tmp_path / "model.evm"
    write_model(model, path)
    restored = read_model(path)
    assert np.allclose(model.predict_proba(dataset.test_x), restored.predict_proba(dataset.test_x))
    path.write_text(path.read_text(encoding="utf-8") + "trailing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trailing"):
        read_model(path)


def test_baseline_and_model_input_validation() -> None:
    dataset = generate_dataset(samples_per_class=10)
    baseline = centroid_baseline(dataset.train_x, dataset.train_y, dataset.test_x)
    assert baseline.shape == (dataset.test_y.size, 3)
    model, _ = train_model(dataset.train_x, dataset.train_y, epochs=10)
    invalid = dataset.test_x.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict_proba(invalid)
    with pytest.raises(ValueError, match="hyperparameters"):
        train_model(dataset.train_x, dataset.train_y, epochs=0)
