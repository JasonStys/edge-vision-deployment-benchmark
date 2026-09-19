"""File: Wraps ONNX Runtime with bounded input validation and deterministic CPU session settings.

Functions: OnnxPredictor.predict and maximum_absolute_error provide deployment inference and
agreement checks. Classes: OnnxPredictor owns one CPU session. Variables: input arrays and outputs
are batch-bounded; exact declaration lines are in ``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from edgevision.constants import (
    CLASS_COUNT,
    INPUT_SIZE,
    MATRIX_RANK,
    MAX_INFERENCE_ROWS,
    MAX_ONNX_BYTES,
)

FloatArray = NDArray[np.float32]


class OnnxPredictor:
    """Run a locally generated ONNX graph through the CPU execution provider."""

    def __init__(self, model_path: Path) -> None:
        """Create a sequential single-thread session after artifact-size validation."""

        if not model_path.is_file() or model_path.stat().st_size > MAX_ONNX_BYTES:
            raise ValueError("ONNX artifact is missing or exceeds the 2 MB safety limit")
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def predict(self, features: FloatArray) -> FloatArray:
        """Validate and infer at most 100,000 images per call."""

        checked = np.asarray(features, dtype=np.float32)
        if (
            checked.ndim != MATRIX_RANK
            or checked.shape[1] != INPUT_SIZE
            or checked.shape[0] > MAX_INFERENCE_ROWS
        ):
            raise ValueError(f"features must have bounded shape [N, {INPUT_SIZE}]")
        if not np.all(np.isfinite(checked)) or np.any((checked < 0.0) | (checked > 1.0)):
            raise ValueError("features must contain finite values in [0, 1]")
        outputs = self._session.run(["probabilities"], {"input": checked})[0]
        probabilities = np.asarray(outputs, dtype=np.float32)
        if probabilities.shape != (checked.shape[0], CLASS_COUNT):
            raise RuntimeError("runtime output did not match the model contract")
        return probabilities


def maximum_absolute_error(reference: FloatArray, candidate: FloatArray) -> float:
    """Return the largest element-wise probability difference after shape checks."""

    if reference.shape != candidate.shape:
        raise ValueError("agreement arrays must have identical shapes")
    if reference.size == 0:
        return 0.0
    return float(np.max(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
