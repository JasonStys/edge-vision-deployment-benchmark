"""File: Tests ONNX export, quantization, CPU inference, and numerical agreement thresholds.

Functions: test cases exercise deployment conversions and runtime guards. Variables: temporary model
paths and probability arrays are test-local; exact lines are in docs/CODE_INDEX.md.
"""

from pathlib import Path

import numpy as np
import pytest

from edgevision.data import generate_dataset
from edgevision.export import export_onnx, quantize_onnx, validate_onnx
from edgevision.model import train_model
from edgevision.runtime import OnnxPredictor, maximum_absolute_error


def test_export_quantization_and_runtime_agreement(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=14)
    model, _ = train_model(dataset.train_x, dataset.train_y, epochs=80)
    float_path = tmp_path / "model.onnx"
    quantized_path = tmp_path / "model.int8.onnx"
    export_onnx(model, float_path)
    quantize_onnx(float_path, quantized_path)
    validate_onnx(float_path)
    reference = model.predict_proba(dataset.test_x)
    deployed = OnnxPredictor(float_path).predict(dataset.test_x)
    quantized = OnnxPredictor(quantized_path).predict(dataset.test_x)
    assert maximum_absolute_error(reference, deployed) <= 1e-5
    assert np.mean(np.argmax(reference, axis=1) == np.argmax(quantized, axis=1)) >= 0.95


def test_runtime_and_agreement_reject_bad_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        OnnxPredictor(tmp_path / "missing.onnx")
    with pytest.raises(ValueError, match="identical"):
        maximum_absolute_error(
            np.zeros((1, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
        )
