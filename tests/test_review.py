"""File: Verify deployment review with hand-calculated weak-class gates and real runtimes.

Author: Jason Stys. Variables: synthetic three-class scores and disposable reports stay isolated.
Functions: tests cover threshold boundaries, tampering, no overwrite and complete shared evidence.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from edgevision.pipeline import main
from edgevision.review import ReviewLimits, deployment_review, review_candidate


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_bad_limits(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ReviewLimits(minimum_accuracy=value)


def test_weak_class_cannot_hide_in_average() -> None:
    features = np.zeros((4, 64), dtype=np.float32)
    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    scores = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    reference = np.eye(3, dtype=np.float32)[labels]
    report = review_candidate(
        lambda _: scores, features, labels, {"clean": reference}, ReviewLimits(0.75, 0.1, 0.25, 1.0)
    )
    assert report["eligible"] is False
    assert report["failed_gates"] == ["clean/class_recall"]
    gates = report["conditions"]["clean"]["gates"]
    assert gates == {
        "accuracy": True,
        "class_recall": False,
        "calibration": True,
        "runtime_agreement": True,
    }
    strict = review_candidate(
        lambda _: scores, features, labels, {"clean": reference}, ReviewLimits(1, 1, 0, 0)
    )
    assert len(strict["failed_gates"]) == 4
    with pytest.raises(ValueError, match="unsupported"):
        review_candidate(
            lambda _: scores, features, labels, {"unexpected": reference}, ReviewLimits()
        )


def test_actual_candidates_share_split_and_all_conditions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).resolve().parents[1]
    before = (root / "artifacts/manifest.json").read_bytes()
    report = deployment_review(root, ReviewLimits(0, 0, 1, 1))
    assert report["test_samples"] == 48
    assert len(report["candidates"]) == 3
    for candidate in report["candidates"].values():
        assert candidate["eligible"]
        assert set(candidate["conditions"]) == {"clean", "gaussian", "occlusion", "dim"}
        assert candidate["model_bytes"] > 0
        assert len(candidate["model_sha256"]) == 64
        assert candidate["benchmark"]["batch_size"] == 48
        for condition in candidate["conditions"].values():
            assert condition["metrics"]["sample_count"] == 48
            assert sum(map(sum, condition["metrics"]["confusion_matrix"])) == 48
    destination = tmp_path / "review.json"
    assert main(["review", "--root", str(root), "--report", str(destination)]) == 0
    assert json.loads(destination.read_text())["dataset_digest"] == report["dataset_digest"]
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    assert main(["review", "--root", str(root)]) == 0
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        main(["review", "--root", str(root), "--report", str(destination)])
    assert (root / "artifacts/manifest.json").read_bytes() == before


def test_tampered_artifacts_fail_before_review(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "artifacts", tmp_path / "artifacts")
    shutil.copytree(root / "docs/reports/generated", tmp_path / "docs/reports/generated")
    (tmp_path / "artifacts/model/compact-mlp.evm").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        deployment_review(tmp_path)
