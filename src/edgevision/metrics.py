"""File: Computes accuracy, calibration, robustness, latency, throughput, and energy proxies.

Functions: classification_metrics and benchmark_predictor produce JSON-safe evidence. Variables:
Confusion matrices, confidence bins, and latency samples are locally bounded; exact declaration
lines are generated in ``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from edgevision.constants import (
    CLASS_COUNT,
    CLASS_NAMES,
    MAX_BENCHMARK_ITERATIONS,
    MAX_BENCHMARK_WARMUPS,
    MAX_POWER_PROXY_WATTS,
    MIN_BENCHMARK_ITERATIONS,
    MIN_POWER_PROXY_WATTS,
)

FloatArray = NDArray[np.float32]
LabelArray = NDArray[np.int64]


def classification_metrics(labels: LabelArray, probabilities: FloatArray) -> dict[str, object]:
    """Compute multiclass quality and expected calibration error without external ML packages."""

    y = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.shape != (y.size, CLASS_COUNT) or y.size == 0:
        raise ValueError("probabilities must have one supported-class row per non-empty label")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    predictions = np.argmax(scores, axis=1)
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for expected, predicted in zip(y, predictions, strict=True):
        confusion[int(expected), int(predicted)] += 1
    per_class: dict[str, object] = {}
    for label, name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[label, label])
        false_positive = int(np.sum(confusion[:, label]) - true_positive)
        false_negative = int(np.sum(confusion[label, :]) - true_positive)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}

    confidences = np.max(scores, axis=1)
    correct = predictions == y
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        members = (confidences > lower) & (confidences <= upper)
        if np.any(members):
            gap = abs(float(np.mean(correct[members])) - float(np.mean(confidences[members])))
            calibration_error += float(np.mean(members)) * gap
    row_indexes = np.arange(y.size)
    negative_log_likelihood = -float(np.mean(np.log(scores[row_indexes, y] + 1e-12)))
    return {
        "accuracy": float(np.mean(correct)),
        "confusion_matrix": confusion.tolist(),
        "ece_10_bin": calibration_error,
        "negative_log_likelihood": negative_log_likelihood,
        "per_class": per_class,
        "sample_count": int(y.size),
    }


def benchmark_predictor(
    predictor: Callable[[FloatArray], FloatArray],
    features: FloatArray,
    *,
    warmups: int = 5,
    iterations: int = 30,
    assumed_watts: float = 15.0,
) -> dict[str, float | int | str]:
    """Measure warm latency distributions and a clearly labeled power-based energy proxy."""

    if not 1 <= warmups <= MAX_BENCHMARK_WARMUPS or not (
        MIN_BENCHMARK_ITERATIONS <= iterations <= MAX_BENCHMARK_ITERATIONS
    ):
        raise ValueError("benchmark iteration counts exceed documented bounds")
    if (
        features.shape[0] == 0
        or not MIN_POWER_PROXY_WATTS <= assumed_watts <= MAX_POWER_PROXY_WATTS
    ):
        raise ValueError("benchmark inputs and power proxy must be non-empty and bounded")
    for _ in range(warmups):
        predictor(features)
    latencies_ms: list[float] = []
    tracemalloc.start()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        predictor(features)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        latencies_ms.append(elapsed_ms / features.shape[0])
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_seconds = sum(latencies_ms) * features.shape[0] / 1_000.0
    total_samples = iterations * features.shape[0]
    return {
        "batch_size": int(features.shape[0]),
        "iterations": iterations,
        "latency_ms_per_image_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_per_image_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_per_image_p99": float(np.percentile(latencies_ms, 99)),
        "throughput_images_per_second": total_samples / max(total_seconds, 1e-12),
        "python_tracemalloc_peak_bytes": peak_bytes,
        "energy_proxy_joules_per_1000_images": assumed_watts
        * total_seconds
        / total_samples
        * 1_000,
        "energy_proxy_method": "elapsed_seconds_per_image * assumed_watts; not a hardware reading",
    }
