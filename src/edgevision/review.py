"""File: Compare deployment candidates on a shared held-out quality/robustness envelope.

Author: Jason Stys. Functions: review_candidate and deployment_review produce read-only evidence.
Variables: per-class recalls, quality thresholds and same-run CPU measurements stay explicit.
Exact declaration locations are generated in docs/CODE_INDEX.md.
"""

from __future__ import annotations

import math
import platform
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from edgevision.artifacts import sha256_file, verify_manifest
from edgevision.data import apply_corruption, dataset_digest, load_dataset
from edgevision.metrics import FloatArray, LabelArray, benchmark_predictor, classification_metrics
from edgevision.model import read_model
from edgevision.runtime import OnnxPredictor, maximum_absolute_error


@dataclass(frozen=True)
class ReviewLimits:
    """User-selected laboratory gates; changing them never mutates release thresholds/models."""

    minimum_accuracy: float = 0.90
    minimum_class_recall: float = 0.75
    maximum_ece: float = 0.25
    maximum_runtime_error: float = 0.01

    def __post_init__(self) -> None:
        """Reject NaN, infinity and values outside probability bounds before inference."""
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in asdict(self).values()):
            raise ValueError("review limits must be finite numbers in [0,1]")


def review_candidate(
    predictor: Callable[[FloatArray], FloatArray],
    features: FloatArray,
    labels: LabelArray,
    reference: dict[str, FloatArray],
    limits: ReviewLimits,
) -> dict[str, Any]:
    """Gate every clean/corrupted class, not only an average that can hide a weak class."""
    results: dict[str, Any] = {}
    reasons: list[str] = []
    for condition, expected in reference.items():
        if condition not in ("clean", "gaussian", "occlusion", "dim"):
            raise ValueError("unsupported review corruption")
        inputs = (
            features
            if condition == "clean"
            else apply_corruption(
                features, cast("Literal['gaussian', 'occlusion', 'dim']", condition)
            )
        )
        probabilities = predictor(inputs)
        metrics = classification_metrics(labels, probabilities)
        error = maximum_absolute_error(expected, probabilities)
        per_class = metrics["per_class"]
        assert isinstance(per_class, dict)
        worst_recall = min(float(value["recall"]) for value in per_class.values())
        gates = {
            "accuracy": float(str(metrics["accuracy"])) >= limits.minimum_accuracy,
            "class_recall": worst_recall >= limits.minimum_class_recall,
            "calibration": float(str(metrics["ece_10_bin"])) <= limits.maximum_ece,
            "runtime_agreement": error <= limits.maximum_runtime_error,
        }
        reasons.extend(f"{condition}/{gate}" for gate, passed in gates.items() if not passed)
        results[condition] = {"metrics": metrics, "reference_max_abs": error, "gates": gates}
    return {"eligible": not reasons, "failed_gates": reasons, "conditions": results}


def deployment_review(root: Path, limits: ReviewLimits | None = None) -> dict[str, Any]:
    """Verify checked artifacts, run the same test split/CPU session and expose tradeoffs.

    No source/artifact/report is changed. Timing is host-specific, tracemalloc is Python-only,
    and the wattage-derived proxy is never a measured energy or device qualification claim.
    """
    limits = limits or ReviewLimits()
    manifest = root / "artifacts/manifest.json"
    verify_manifest(root, manifest)
    dataset_path = root / "artifacts/dataset/shapes-v1.npz"
    dataset = load_dataset(dataset_path)
    model_path = root / "artifacts/model/compact-mlp.evm"
    predictors = {
        "portable_numpy": (model_path, read_model(model_path).predict_proba),
        "onnx_float32": (root / "artifacts/model/compact-mlp.onnx", None),
        "onnx_int8_dynamic_weights": (root / "artifacts/model/compact-mlp.int8.onnx", None),
    }
    reference_predictor = read_model(model_path).predict_proba
    conditions: tuple[Literal["clean", "gaussian", "occlusion", "dim"], ...] = (
        "clean",
        "gaussian",
        "occlusion",
        "dim",
    )
    reference: dict[str, FloatArray] = {
        condition: reference_predictor(
            dataset.test_x if condition == "clean" else apply_corruption(dataset.test_x, condition)
        )
        for condition in conditions
    }
    candidates: dict[str, Any] = {}
    for name, (path, portable_predictor) in predictors.items():
        predictor = portable_predictor or OnnxPredictor(path).predict
        result = review_candidate(predictor, dataset.test_x, dataset.test_y, reference, limits)
        result.update(
            {
                "model_sha256": sha256_file(path),
                "model_bytes": path.stat().st_size,
                "benchmark": benchmark_predictor(predictor, dataset.test_x),
            }
        )
        candidates[name] = result
    return {
        "schema_version": 1,
        "review_id": str(uuid.uuid4()),
        "manifest_sha256": sha256_file(manifest),
        "dataset_digest": dataset_digest(dataset),
        "test_samples": int(dataset.test_y.size),
        "limits": asdict(limits),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "candidates": candidates,
        "notice": "Synthetic 8x8 shapes only. Same-run CPU timings, Python-only traced memory, "
        "assumed-wattage proxy. Eligibility is a configurable review gate, "
        "not automatic promotion. No camera/people/safety deployment qualification. "
        "Native C++/Rust agreement remains a separate gate.",
    }
