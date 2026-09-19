"""File: Orchestrates data, training, export, evaluation, benchmarking, and validation.

Functions: build_artifacts, validate_artifacts, run_benchmarks, build_parser, and main expose the
end-to-end workflow. Variables: paths and reports remain repository-relative and JSON-safe; exact
declaration locations are generated in ``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from edgevision.artifacts import (
    read_predictions,
    read_vectors,
    verify_manifest,
    write_manifest,
    write_predictions,
    write_vectors,
)
from edgevision.constants import (
    CLASS_NAMES,
    DEFAULT_SEED,
    MAX_PORTABLE_ERROR,
    MAX_QUANTIZED_ACCURACY_LOSS,
    MAX_RUNTIME_ERROR,
    MIN_RELEASE_ACCURACY,
)
from edgevision.data import (
    apply_corruption,
    dataset_digest,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from edgevision.export import export_onnx, quantize_onnx, validate_onnx
from edgevision.metrics import benchmark_predictor, classification_metrics
from edgevision.model import (
    centroid_baseline,
    fit_temperature,
    read_model,
    train_model,
    write_model,
)
from edgevision.runtime import OnnxPredictor, maximum_absolute_error


def _write_json(path: Path, payload: object) -> None:
    """Write stable human-readable JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _paths(root: Path) -> dict[str, Path]:
    """Return the canonical generated-artifact locations beneath one repository root."""

    return {
        "dataset": root / "artifacts/dataset/shapes-v1.npz",
        "model": root / "artifacts/model/compact-mlp.evm",
        "onnx": root / "artifacts/model/compact-mlp.onnx",
        "quantized": root / "artifacts/model/compact-mlp.int8.onnx",
        "vectors": root / "artifacts/test-vectors.csv",
        "reference": root / "artifacts/reference-predictions.csv",
        "manifest": root / "artifacts/manifest.json",
        "metrics": root / "docs/reports/generated/metrics.json",
        "benchmark": root / "docs/reports/generated/benchmark.json",
        "validation": root / "docs/reports/generated/validation.json",
        "model_card": root / "docs/reports/generated/MODEL_CARD.md",
    }


def build_artifacts(root: Path) -> dict[str, object]:
    """Build every small artifact and enforce quality thresholds before commit."""

    paths = _paths(root)
    dataset = generate_dataset(samples_per_class=80, seed=DEFAULT_SEED)
    dataset_manifest = save_dataset(dataset, paths["dataset"])
    model, summary = train_model(dataset.train_x, dataset.train_y)
    temperature = fit_temperature(model, dataset.validation_x, dataset.validation_y)
    model = model.calibrated(temperature)
    summary = replace(summary, temperature=temperature)
    write_model(model, paths["model"])
    export_onnx(model, paths["onnx"])
    quantize_onnx(paths["onnx"], paths["quantized"])

    vector_count = min(18, dataset.test_y.size)
    write_vectors(paths["vectors"], dataset.test_x[:vector_count], dataset.test_y[:vector_count])
    identifiers, _, vector_features = read_vectors(paths["vectors"])
    write_predictions(paths["reference"], identifiers, model.predict_proba(vector_features))

    portable_probabilities = model.predict_proba(dataset.test_x)
    onnx_probabilities = OnnxPredictor(paths["onnx"]).predict(dataset.test_x)
    quantized_probabilities = OnnxPredictor(paths["quantized"]).predict(dataset.test_x)
    baseline_probabilities = centroid_baseline(dataset.train_x, dataset.train_y, dataset.test_x)
    corruption_metrics = {
        kind: classification_metrics(
            dataset.test_y, model.predict_proba(apply_corruption(dataset.test_x, kind))
        )
        for kind in ("gaussian", "occlusion", "dim")
    }
    metrics: dict[str, Any] = {
        "dataset": dataset_manifest,
        "training": asdict(summary),
        "baseline": classification_metrics(dataset.test_y, baseline_probabilities),
        "portable_float32": classification_metrics(dataset.test_y, portable_probabilities),
        "onnx_float32": classification_metrics(dataset.test_y, onnx_probabilities),
        "onnx_int8_dynamic_weights": classification_metrics(
            dataset.test_y, quantized_probabilities
        ),
        "corruptions": corruption_metrics,
        "agreement": {
            "portable_to_onnx_max_abs": maximum_absolute_error(
                portable_probabilities, onnx_probabilities
            ),
            "float_to_quantized_max_abs": maximum_absolute_error(
                onnx_probabilities, quantized_probabilities
            ),
        },
    }
    float_accuracy = float(metrics["portable_float32"]["accuracy"])
    quantized_accuracy = float(metrics["onnx_int8_dynamic_weights"]["accuracy"])
    agreement = float(metrics["agreement"]["portable_to_onnx_max_abs"])
    if (
        float_accuracy < MIN_RELEASE_ACCURACY
        or float_accuracy - quantized_accuracy > MAX_QUANTIZED_ACCURACY_LOSS
        or agreement > MAX_RUNTIME_ERROR
    ):
        raise RuntimeError("generated artifacts did not satisfy the documented release thresholds")
    _write_json(paths["metrics"], metrics)
    benchmarks = run_benchmarks(root, paths["benchmark"])
    model_card = _render_model_card(metrics, benchmarks)
    paths["model_card"].parent.mkdir(parents=True, exist_ok=True)
    with paths["model_card"].open("w", encoding="utf-8", newline="\n") as output:
        output.write(model_card)

    tracked = [
        Path("artifacts/dataset/shapes-v1.npz"),
        Path("artifacts/dataset/shapes-v1.manifest.json"),
        Path("artifacts/model/compact-mlp.evm"),
        Path("artifacts/model/compact-mlp.onnx"),
        Path("artifacts/model/compact-mlp.int8.onnx"),
        Path("artifacts/test-vectors.csv"),
        Path("artifacts/reference-predictions.csv"),
        Path("docs/reports/generated/metrics.json"),
        Path("docs/reports/generated/benchmark.json"),
        Path("docs/reports/generated/MODEL_CARD.md"),
    ]
    write_manifest(root, tracked, paths["manifest"])
    validation = validate_artifacts(root)
    return {"metrics": metrics, "benchmarks": benchmarks, "validation": validation}


def _render_model_card(metrics: dict[str, Any], benchmarks: dict[str, object]) -> str:
    """Render an evidence-linked model card from generated measurements."""

    float_metrics = metrics["portable_float32"]
    quantized_metrics = metrics["onnx_int8_dynamic_weights"]
    return f"""# Generated Model Card

## Intended use

This small classifier distinguishes generated vertical, horizontal, and diagonal 8x8 patterns. It is
an engineering benchmark for portable deployment, not a production perception system.

## Measured results

- Float32 test accuracy: {float_metrics["accuracy"]:.4f}
- Dynamically weight-quantized ONNX test accuracy: {quantized_metrics["accuracy"]:.4f}
- Float32 10-bin expected calibration error: {float_metrics["ece_10_bin"]:.6f}
- Model classes: {", ".join(CLASS_NAMES)}
- Benchmark runtimes: {", ".join(sorted(benchmarks))}

## Data and evaluation

Data is generated from documented geometric rules with deterministic splits. Evaluation covers clean
data, Gaussian noise, center occlusion, dimming, calibration, native-runtime agreement, and warm CPU
latency. The generated metrics JSON is the source of truth for detailed results.

## Limitations

The data is synthetic, grayscale, low resolution, balanced, and intentionally simple. Results do not
generalize to camera imagery, people, safety decisions, or uncontrolled environments. The energy
number is an elapsed-time proxy based on an assumed wattage, not a physical power measurement.
"""


def validate_artifacts(root: Path) -> dict[str, object]:
    """Verify checksums, schemas, model loading, ONNX validity, and reference agreement."""

    paths = _paths(root)
    verify_manifest(root, paths["manifest"])
    dataset = load_dataset(paths["dataset"])
    dataset_manifest = json.loads(paths["dataset"].with_suffix(".manifest.json").read_text("utf-8"))
    if dataset_digest(dataset) != dataset_manifest.get("dataset_sha256"):
        raise ValueError("dataset digest does not match its manifest")
    model = read_model(paths["model"])
    validate_onnx(paths["onnx"])
    validate_onnx(paths["quantized"])
    identifiers, _, features = read_vectors(paths["vectors"])
    reference_identifiers, reference = read_predictions(paths["reference"])
    if identifiers != reference_identifiers:
        raise ValueError("reference identifiers do not align with test vectors")
    portable_error = maximum_absolute_error(reference, model.predict_proba(features))
    onnx_error = maximum_absolute_error(reference, OnnxPredictor(paths["onnx"]).predict(features))
    report: dict[str, object] = {
        "status": "passed",
        "checks": [
            "artifact SHA-256 and size manifest",
            "dataset schema and deterministic content digest",
            "portable model bounds and tensor shapes",
            "ONNX checker and strict shape inference",
            "portable and ONNX reference-vector agreement",
        ],
        "portable_reference_max_abs": portable_error,
        "onnx_reference_max_abs": onnx_error,
    }
    if portable_error > MAX_PORTABLE_ERROR or onnx_error > MAX_RUNTIME_ERROR:
        raise RuntimeError("runtime reference agreement exceeded its release tolerance")
    _write_json(paths["validation"], report)
    return report


def run_benchmarks(root: Path, destination: Path) -> dict[str, Any]:
    """Benchmark portable and ONNX paths with shared inputs, warmups, and percentile reporting."""

    paths = _paths(root)
    dataset = load_dataset(paths["dataset"])
    model = read_model(paths["model"])
    sample = dataset.test_x[: min(48, dataset.test_x.shape[0])]
    results: dict[str, Any] = {
        "methodology": {
            "device": "current CPU; compare runs only on the same host",
            "warmups": 5,
            "iterations": 30,
            "batch_size": int(sample.shape[0]),
        },
        "portable_numpy": benchmark_predictor(model.predict_proba, sample),
        "onnx_float32": benchmark_predictor(OnnxPredictor(paths["onnx"]).predict, sample),
        "onnx_int8_dynamic_weights": benchmark_predictor(
            OnnxPredictor(paths["quantized"]).predict, sample
        ),
    }
    _write_json(destination, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build a small explicit CLI with repository-root and report-path controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate", "benchmark"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one workflow command and return a process-compatible status code."""

    arguments = build_parser().parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == "build":
        result: object = build_artifacts(root)
    elif arguments.command == "validate":
        result = validate_artifacts(root)
    else:
        destination = arguments.report or root / ".runtime/benchmark.json"
        result = run_benchmarks(root, destination)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
