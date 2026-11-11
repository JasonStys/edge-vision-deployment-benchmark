"""File: tests/test_desktop_project.py
Purpose: Independently validate custom authoring and actual NumPy/ONNX owner-local workflows.
Symbols: see docs/CODE_INDEX.md. Temporary models/data are copied from checked public resources.
"""

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from desktop.project import MAX_ROWS, Workspace, parse_vectors, verify_bundle
from desktop.storage import read_regular
from edgevision.constants import MAX_RUNTIME_ERROR, MIN_RELEASE_ACCURACY
from edgevision.review import ReviewLimits

ROOT = Path(__file__).resolve().parents[1]


def geometric_vectors() -> str:
    """Author independent patterns and ground-truth classes, not target outputs."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["id", "label", *(f"f{i}" for i in range(64))])
    for name, condition in (
        ("vertical", lambda _row, column: column == 3),
        ("horizontal", lambda row, _column: row == 3),
        ("diagonal", lambda row, column: row == column),
    ):
        writer.writerow(
            [name, name, *(int(condition(row, column)) for row in range(8) for column in range(8))]
        )
    return output.getvalue()


def test_actual_three_runtime_inference_and_durable_reopen(tmp_path):
    """Real engines must agree on independently authored vertical/horizontal/diagonal inputs."""
    workspace = Workspace(tmp_path / "profile", ROOT)
    text = geometric_vectors()
    result = workspace.infer(text, include_native=False)
    assert result["kind"] == "inference"
    assert result["result"]["ids"] == ["vertical", "horizontal", "diagonal"]
    assert set(result["result"]["runtimes"]) == {
        "portable_numpy",
        "onnx_float32",
        "onnx_int8_dynamic_weights",
    }
    for runtime in result["result"]["runtimes"].values():
        probabilities = np.asarray(runtime["probabilities"])
        np.testing.assert_array_equal(np.argmax(probabilities, axis=1), [0, 1, 2])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert result["result"]["runtimes"]["onnx_float32"]["reference_max_abs"] < MAX_RUNTIME_ERROR
    generation = workspace.state["active"]
    workspace.close()
    reopened = Workspace(tmp_path / "profile", ROOT)
    assert reopened.state["active"] == generation
    assert reopened.last_report() == result
    reopened.close()


@pytest.mark.parametrize("bad", ["nan", "inf", "-0.1", "1.01", "not-a-number"])
def test_invalid_pixel_values_never_reach_inference(bad):
    """Malformed user input is independently constructed at a real data position."""
    rows = list(csv.reader(io.StringIO(geometric_vectors())))
    rows[1][2] = bad
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    with pytest.raises(ValueError):
        parse_vectors(output.getvalue())


def test_duplicate_id_extra_rows_wrong_header_and_formula_identifier_are_refused():
    """CSV admission prevents ambiguous identities, schema drift and over-cap rows."""
    rows = list(csv.reader(io.StringIO(geometric_vectors())))
    for malformed in (
        [rows[0], rows[1], rows[1]],
        [rows[0], *[rows[1]] * (MAX_ROWS + 1)],
        [["wrong"], rows[1]],
        [rows[0], ["=cmd()", *rows[1][1:]]],
    ):
        output = io.StringIO()
        csv.writer(output).writerows(malformed)
        with pytest.raises(ValueError):
            parse_vectors(output.getvalue())


def test_real_csv_parser_field_error_becomes_actionable_validation():
    """A genuine parser limit error remains an ordinary visible input refusal."""
    original = csv.field_size_limit()
    try:
        csv.field_size_limit(16)
        with pytest.raises(ValueError, match="CSV field"):
            parse_vectors(
                geometric_vectors().replace("vertical,vertical", "x" * 32 + ",vertical", 1)
            )
    finally:
        csv.field_size_limit(original)


def test_complete_manifest_required_and_corruption_is_preserved(tmp_path):
    """An omitted model cannot pass a partial manifest; corrupt data is not silently regenerated."""
    workspace = Workspace(tmp_path / "profile", ROOT)
    root = workspace.active
    manifest = root / "artifacts/manifest.json"
    original = manifest.read_bytes()
    value = json.loads(original)
    value["files"].pop("artifacts/model/compact-mlp.int8.onnx")
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete"):
        verify_bundle(root)
    assert "compact-mlp.int8.onnx" not in manifest.read_text()
    manifest.write_bytes(original)
    model = root / "artifacts/model/compact-mlp.evm"
    model.write_bytes(b"Retain invalid original bytes rather than replacing them")
    workspace.close()
    with pytest.raises(ValueError, match="fingerprint"):
        Workspace(tmp_path / "profile", ROOT)
    assert model.read_bytes() == b"Retain invalid original bytes rather than replacing them"


def test_missing_state_and_failed_rebuild_preserve_the_original_generation(tmp_path, monkeypatch):
    """Missing state and build failure cannot reset an existing profile."""
    workspace = Workspace(tmp_path / "profile", ROOT)
    before = read_regular(workspace.state_path, 4096)

    def fail(_):
        raise RuntimeError("Synthetic quality-gate failure")

    monkeypatch.setattr("desktop.project.build_artifacts", fail)
    with pytest.raises(RuntimeError, match="quality-gate"):
        workspace.rebuild()
    assert read_regular(workspace.state_path, 4096) == before
    generation = workspace.active
    workspace.close()
    state = tmp_path / "profile/state.json"
    state.unlink()
    with pytest.raises(ValueError, match="no state pointer"):
        Workspace(tmp_path / "profile", ROOT)
    assert generation.is_dir() and not state.exists()


def test_actual_review_exposes_each_condition_and_preserves_release_thresholds(tmp_path):
    """Strict advisory gates expose failures; release constants and artifacts stay unchanged."""
    workspace = Workspace(tmp_path / "profile", ROOT)
    before = (workspace.active / "artifacts/model/compact-mlp.evm").read_bytes()
    report = workspace.review(ReviewLimits(1.0, 1.0, 0.0, 0.0))["result"]
    assert report["test_samples"] == 48
    for result in report["candidates"].values():
        assert set(result["conditions"]) == {"clean", "gaussian", "occlusion", "dim"}
        assert not result["eligible"] and result["failed_gates"]
        for condition in result["conditions"].values():
            assert sum(sum(row) for row in condition["metrics"]["confusion_matrix"]) == 48
    assert MIN_RELEASE_ACCURACY == 0.95
    assert (workspace.active / "artifacts/model/compact-mlp.evm").read_bytes() == before
    workspace.close()
