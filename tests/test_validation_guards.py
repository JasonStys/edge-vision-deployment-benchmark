"""File: Exercises fail-closed guards for artifacts, data, models, metrics, and runtimes.

Functions: focused negative tests cover malformed shapes, schemas, values, and model tokens.
Variables: invalid arrays and temporary files are test-local; exact lines are in docs/CODE_INDEX.md.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from edgevision.artifacts import (
    read_predictions,
    read_vectors,
    verify_manifest,
    write_predictions,
    write_vectors,
)
from edgevision.constants import INPUT_SIZE
from edgevision.data import VisionDataset, apply_corruption, generate_dataset, load_dataset
from edgevision.metrics import benchmark_predictor, classification_metrics
from edgevision.model import (
    PortableModel,
    centroid_baseline,
    fit_temperature,
    read_model,
    train_model,
)
from edgevision.runtime import OnnxPredictor, maximum_absolute_error


def test_vector_guards_reject_shapes_schemas_and_nonfinite_values(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=10)
    with pytest.raises(ValueError, match="bounded input"):
        write_vectors(
            tmp_path / "bad.csv", np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.int64)
        )
    with pytest.raises(ValueError, match="align"):
        write_vectors(tmp_path / "bad.csv", dataset.test_x, np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError, match="missing"):
        read_vectors(tmp_path / "missing.csv")

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_vectors(malformed)
    header = "id,label," + ",".join(f"f{index}" for index in range(INPUT_SIZE))
    malformed.write_text(
        header + "\ncase-0,unknown," + ",".join(["0"] * INPUT_SIZE) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="row"):
        read_vectors(malformed)
    malformed.write_text(
        header + "\ncase-0,vertical,nan," + ",".join(["0"] * (INPUT_SIZE - 1)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        read_vectors(malformed)


def test_prediction_and_manifest_guards_reject_malformed_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="align"):
        write_predictions(tmp_path / "bad.csv", ["one"], np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing"):
        read_predictions(tmp_path / "missing.csv")
    malformed = tmp_path / "predictions.csv"
    malformed.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        read_predictions(malformed)
    malformed.write_text("id,vertical,horizontal,diagonal\n,0,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row"):
        read_predictions(malformed)
    malformed.write_text("id,vertical,horizontal,diagonal\none,nan,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_predictions(malformed)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"format_version": 9, "files": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        verify_manifest(tmp_path, manifest)
    manifest.write_text(
        '{"format_version": 1, "files": {"../escape": {"bytes": 0, "sha256": ""}}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        verify_manifest(tmp_path, manifest)


def test_dataset_guards_reject_alignment_ranges_and_missing_files(tmp_path: Path) -> None:
    dataset = generate_dataset(samples_per_class=10)
    with pytest.raises(ValueError, match="align"):
        VisionDataset(
            dataset.train_x,
            dataset.train_y[:-1],
            dataset.validation_x,
            dataset.validation_y,
            dataset.test_x,
            dataset.test_y,
        )
    invalid_features = dataset.train_x.copy()
    invalid_features[0, 0] = 2.0
    with pytest.raises(ValueError, match="finite values"):
        VisionDataset(
            invalid_features,
            dataset.train_y,
            dataset.validation_x,
            dataset.validation_y,
            dataset.test_x,
            dataset.test_y,
        )
    invalid_labels = dataset.train_y.copy()
    invalid_labels[0] = 99
    with pytest.raises(ValueError, match="class range"):
        VisionDataset(
            dataset.train_x,
            invalid_labels,
            dataset.validation_x,
            dataset.validation_y,
            dataset.test_x,
            dataset.test_y,
        )
    with pytest.raises(ValueError, match="shape"):
        apply_corruption(np.zeros((1, 2), dtype=np.float32), "dim")
    with pytest.raises(ValueError, match="missing"):
        load_dataset(tmp_path / "missing.npz")


def test_model_guards_reject_bad_parameters_and_training_contracts() -> None:
    dataset = generate_dataset(samples_per_class=10)
    model, _ = train_model(dataset.train_x, dataset.train_y, epochs=5)
    with pytest.raises(ValueError, match="weights_one"):
        PortableModel(
            np.zeros((1, 1), dtype=np.float32),
            model.bias_one,
            model.weights_two,
            model.bias_two,
            model.weights_three,
            model.bias_three,
        )
    with pytest.raises(ValueError, match="temperature"):
        replace(model, temperature=0.0)
    with pytest.raises(ValueError, match="bounded shape"):
        model.predict_proba(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="labels"):
        train_model(dataset.train_x, dataset.train_y[:-1], epochs=5)
    with pytest.raises(ValueError, match="calibration labels"):
        fit_temperature(model, dataset.validation_x, dataset.validation_y[:-1])
    with pytest.raises(ValueError, match="baseline labels"):
        centroid_baseline(dataset.train_x, dataset.train_y[:-1], dataset.test_x)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda text: text.replace("EDGEVISION_MLP 1", "EDGEVISION_MLP 2", 1), "version"),
        (lambda text: text.replace("dims 64", "dims 63", 1), "dimensions"),
        (lambda text: text.replace("vertical", "unknown", 1), "class order"),
        (lambda text: text.replace("weights_one 1536", "weights_one 1", 1), "count"),
        (
            lambda text: text.replace("weights_one 1536\n", "weights_one 1536\ninvalid ", 1),
            "non-numeric",
        ),
        (lambda text: text.replace("end\n", ""), "unexpectedly"),
    ],
)
def test_portable_parser_rejects_contract_mutations(
    tmp_path: Path, mutation: object, message: str
) -> None:
    source = Path(__file__).resolve().parents[1] / "artifacts/model/compact-mlp.evm"
    destination = tmp_path / "mutated.evm"
    destination.write_text(mutation(source.read_text(encoding="utf-8")), encoding="utf-8")  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        read_model(destination)


def test_runtime_and_metric_guards_reject_invalid_values() -> None:
    root = Path(__file__).resolve().parents[1]
    predictor = OnnxPredictor(root / "artifacts/model/compact-mlp.onnx")
    with pytest.raises(ValueError, match="bounded shape"):
        predictor.predict(np.zeros((1, 2), dtype=np.float32))
    invalid = np.zeros((1, INPUT_SIZE), dtype=np.float32)
    invalid[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(invalid)
    assert (
        maximum_absolute_error(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="non-negative"):
        classification_metrics(np.asarray([0]), np.asarray([[-1.0, 1.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="power proxy"):
        benchmark_predictor(lambda rows: rows, np.zeros((0, INPUT_SIZE), dtype=np.float32))
