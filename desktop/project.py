"""File: desktop/project.py
Purpose: Invoke real model rebuilding, runtime review and custom-vector inference.
Symbols/variables: see docs/CODE_INDEX.md; Workspace owns immutable generations and reports.
No camera, remote provider, executable model import or automatic deployment is supported.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from desktop.storage import Home, atomic_json, new_json, read_regular, real_ancestors, strict_json
from edgevision.artifacts import read_predictions, verify_manifest, write_vectors
from edgevision.constants import CLASS_NAMES, INPUT_SIZE, MAX_RUNTIME_ERROR
from edgevision.data import load_dataset
from edgevision.model import read_model
from edgevision.pipeline import build_artifacts
from edgevision.review import ReviewLimits, deployment_review
from edgevision.runtime import OnnxPredictor, maximum_absolute_error

FILES = (
    "artifacts/dataset/shapes-v1.npz",
    "artifacts/dataset/shapes-v1.manifest.json",
    "artifacts/model/compact-mlp.evm",
    "artifacts/model/compact-mlp.onnx",
    "artifacts/model/compact-mlp.int8.onnx",
    "artifacts/test-vectors.csv",
    "artifacts/reference-predictions.csv",
    "docs/reports/generated/metrics.json",
    "docs/reports/generated/benchmark.json",
    "docs/reports/generated/MODEL_CARD.md",
)
MAX_CSV = 1024 * 1024
MAX_ROWS = 256
MAX_HISTORY = 100
MAX_ARTIFACT_BYTES = 5_000_000
MAX_PROCESS_OUTPUT = 8192


def verify_bundle(root: Path) -> None:
    """Require every fixed artifact, not a caller-defined incomplete manifest, before model
    loading."""
    real_ancestors(root)
    manifest = strict_json(read_regular(root / "artifacts/manifest.json", 16384))
    if not isinstance(manifest, dict) or set(manifest) != {"format_version", "files"}:
        raise ValueError("Candidate manifest schema differs")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("Candidate manifest version differs")
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(FILES):
        raise ValueError("Candidate must contain the complete fixed artifact set")
    for name, expected in manifest["files"].items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"bytes", "sha256"}
            or type(expected["bytes"]) is not int
            or not 0 <= expected["bytes"] <= MAX_ARTIFACT_BYTES
            or not isinstance(expected["sha256"], str)
            or re.fullmatch(r"[a-f0-9]{64}", expected["sha256"]) is None
        ):
            raise ValueError("Candidate fingerprint is invalid")
        path = root / name
        real_ancestors(path.parent)
        raw = read_regular(path, MAX_ARTIFACT_BYTES)
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("Candidate artifact fingerprint differs")
    verify_manifest(root, root / "artifacts/manifest.json")


def parse_vectors(text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read independently bounded authorable CSV: unique IDs, explicit labels and64 finite
    pixels."""
    if len(text.encode("utf-8")) > MAX_CSV:
        raise ValueError("Custom vectors exceed1MiB")
    try:
        rows = list(itertools.islice(csv.reader(io.StringIO(text)), MAX_ROWS + 2))
    except csv.Error as error:
        raise ValueError("CSV field syntax or size exceeds the parser's supported input") from error
    expected = ["id", "label", *(f"f{index}" for index in range(INPUT_SIZE))]
    if not rows or rows[0] != expected or not 1 <= len(rows) - 1 <= MAX_ROWS:
        raise ValueError("Use the exact 66-column CSV header and 1-256 rows")
    identifiers, labels, features = [], [], []
    for row in rows[1:]:
        if (
            len(row) != len(expected)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row[0]) is None
            or row[0] in identifiers
            or row[1] not in CLASS_NAMES
        ):
            raise ValueError("Each row needs a unique safe ID and a supported shape label")
        values = [float(value) for value in row[2:]]
        if not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values):
            raise ValueError("Pixels must be finite numbers in[0,1]")
        identifiers.append(row[0])
        labels.append(CLASS_NAMES.index(row[1]))
        features.append(values)
    return identifiers, np.asarray(labels, dtype=np.int64), np.asarray(features, dtype=np.float32)


def sample_text(root: Path) -> str:
    """Prepare editable real held-out examples from the active checksummed synthetic dataset."""
    verify_bundle(root)
    dataset = load_dataset(root / "artifacts/dataset/shapes-v1.npz")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["id", "label", *(f"f{index}" for index in range(INPUT_SIZE))])
    for index, (values, label) in enumerate(
        zip(dataset.test_x[:6], dataset.test_y[:6], strict=True)
    ):
        writer.writerow([f"sample-{index}", CLASS_NAMES[int(label)], *map(float, values)])
    return output.getvalue()


class Workspace:
    """A complete durable owner-local workspace over the repository's actual deployed model
    formats."""

    def __init__(self, profile: Path, seed: Path, native: Path | None = None):
        """Retain a profile lock and admit only valid complete generations; unknown state
        is preserved."""
        self.home = Home(profile)
        self.path, self.seed, self.native = self.home.path, seed, native
        try:
            self.candidates = self.path / "candidates"
            self.reports = self.path / "reports"
            self.analyses = self.path / "analyses"
            for directory in (self.candidates, self.reports, self.analyses):
                real_ancestors(directory)
                directory.mkdir(exist_ok=True)
            self.state_path = self.path / "state.json"
            if not self.state_path.exists():
                if (
                    any(self.candidates.iterdir())
                    or any(self.reports.iterdir())
                    or any(self.analyses.iterdir())
                ):
                    raise ValueError("Existing work has no state pointer; restore it, do not reset")
                candidate = self._fresh_candidate()
                verify_bundle(seed)
                for name in (*FILES, "artifacts/manifest.json"):
                    destination = candidate / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as output:
                        output.write(read_regular(seed / name, MAX_ARTIFACT_BYTES))
                verify_bundle(candidate)
                atomic_json(
                    self.state_path, {"schema": 1, "active": candidate.name, "last_report": None}
                )
            self.state = strict_json(read_regular(self.state_path, 4096))
            self._validate_state()
            verify_bundle(self.active)
        except BaseException:
            self.home.close()
            raise

    def _validate_state(self) -> None:
        """Confine restored pointers to authored generation/report names, without trusting
        paths in JSON."""
        value = self.state
        if (
            not isinstance(value, dict)
            or set(value) != {"schema", "active", "last_report"}
            or type(value["schema"]) is not int
            or value["schema"] != 1
            or not isinstance(value["active"], str)
            or re.fullmatch(r"candidate-[a-f0-9]{32}", value["active"]) is None
            or (
                value["last_report"] is not None
                and (
                    not isinstance(value["last_report"], str)
                    or re.fullmatch(r"report-[a-f0-9]{32}[.]json", value["last_report"]) is None
                )
            )
        ):
            raise ValueError("Workspace state is corrupt; no reset attempted")

    @property
    def active(self) -> Path:
        """Resolve only the already validated generation identifier."""
        return self.candidates / self.state["active"]

    def _fresh_candidate(self) -> Path:
        """Keep failed/previous generations untouched and enforce a visible finite local
        capacity."""
        if sum(1 for _ in itertools.islice(self.candidates.iterdir(), MAX_HISTORY)) >= MAX_HISTORY:
            raise ValueError("Candidate capacity100 reached; preserve/export the closed profile")
        path = self.candidates / ("candidate-" + secrets.token_hex(16))
        path.mkdir()
        return path

    def _record(
        self, kind: str, result: dict[str, Any], *, generation: str | None = None
    ) -> dict[str, Any]:
        """Save complete new evidence, then point state at it; existing evidence is never
        overwritten."""
        if sum(1 for _ in itertools.islice(self.reports.iterdir(), MAX_HISTORY)) >= MAX_HISTORY:
            raise ValueError("Report capacity100 reached; preserve/export the closed profile")
        chosen = self.state["active"] if generation is None else generation
        payload = {"schema": 1, "kind": kind, "generation": chosen, "result": result}
        name = "report-" + secrets.token_hex(16) + ".json"
        new_json(self.reports / name, payload)
        next_state = {**self.state, "active": chosen, "last_report": name}
        atomic_json(self.state_path, next_state)
        self.state = next_state
        return payload

    def last_report(self) -> dict[str, Any] | None:
        """Restore the captured last result; replaying saved metrics is not a fresh benchmark."""
        name = self.state["last_report"]
        if name is None:
            return None
        return strict_json(read_regular(self.reports / name, 8 * 1024 * 1024))

    def rebuild(self) -> dict[str, Any]:
        """Run the original deterministic training/export/release gates in a new durable
        generation."""
        candidate = self._fresh_candidate()
        result = build_artifacts(candidate)
        verify_bundle(candidate)
        return self._record("rebuild", result, generation=candidate.name)

    def review(self, limits: ReviewLimits) -> dict[str, Any]:
        """Run all three actual runtime variants and four held-out conditions with explicit
        user gates."""
        verify_bundle(self.active)
        return self._record("review", deployment_review(self.active, limits))

    def infer(self, text: str, *, include_native: bool) -> dict[str, Any]:
        """Predict custom vectors with real NumPy/ONNX and optionally verified C++/Rust
        executables."""
        identifiers, labels, features = parse_vectors(text)
        verify_bundle(self.active)
        predictors = {
            "portable_numpy": read_model(
                self.active / "artifacts/model/compact-mlp.evm"
            ).predict_proba,
            "onnx_float32": OnnxPredictor(self.active / "artifacts/model/compact-mlp.onnx").predict,
            "onnx_int8_dynamic_weights": OnnxPredictor(
                self.active / "artifacts/model/compact-mlp.int8.onnx"
            ).predict,
        }
        results = {}
        reference = None
        for name, predictor in predictors.items():
            start = time.perf_counter()
            values = predictor(features)
            elapsed = time.perf_counter() - start
            if reference is None:
                reference = values
            results[name] = {
                "probabilities": values.tolist(),
                "batch_seconds": elapsed,
                "reference_max_abs": maximum_absolute_error(reference, values),
            }
        if include_native:
            results.update(self._native_infer(identifiers, labels, features, reference))
        return self._record(
            "inference",
            {
                "ids": identifiers,
                "expected_labels": [CLASS_NAMES[int(label)] for label in labels],
                "classes": list(CLASS_NAMES),
                "vector_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "runtimes": results,
                ("notice"): (
                    "Synthetic 8x8 shape models only. Python timings cover predict calls; "
                    "native timings include process startup and file I/O. These are "
                    "different timing scopes, not comparable kernel benchmarks."
                ),
            },
        )

    def _native_infer(self, identifiers, labels, features, reference) -> dict[str, Any]:
        """Use fixed packaged executables and fresh fixed paths; target code never comes
        from user inputs."""
        if self.native is None:
            raise ValueError(
                "Build/include the actual C++ and Rust tools before requesting native agreement"
            )
        metadata = strict_json(read_regular(self.native / "KERNELS.json", 4096))
        names = (
            ("edgevision_cpp.exe", "edgevision-rust.exe")
            if sys.platform == "win32"
            else ("edgevision_cpp", "edgevision-rust")
        )
        if not isinstance(metadata, dict) or set(metadata) != set(names):
            raise ValueError("The fixed native executable inventory differs")
        if sum(1 for _ in itertools.islice(self.analyses.iterdir(), MAX_HISTORY)) >= MAX_HISTORY:
            raise ValueError(
                "Native analysis capacity 100 reached; preserve/export the closed profile"
            )
        for name in names:
            if (
                hashlib.sha256(read_regular(self.native / name, 32 * 1024 * 1024)).hexdigest()
                != metadata[name]
            ):
                raise ValueError("Packaged native executable differs from its recorded digest")
        analysis = self.analyses / ("analysis-" + secrets.token_hex(16))
        (analysis / "artifacts/model").mkdir(parents=True)
        (analysis / ".runtime").mkdir()
        shutil.copyfile(
            self.active / "artifacts/model/compact-mlp.evm",
            analysis / "artifacts/model/compact-mlp.evm",
        )
        vectors = analysis / "artifacts/test-vectors.csv"
        write_vectors(vectors, features, labels)
        # Retain explicit IDs rather than the core writer's generated identifiers.
        rows = list(csv.reader(io.StringIO(vectors.read_text(encoding="utf-8"))))
        for row, identifier in zip(rows[1:], identifiers, strict=True):
            row[0] = identifier
        with vectors.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output, lineterminator="\n").writerows(rows)
        results = {}
        for name, label, destination in (
            (names[0], "cpp", ".runtime/cpp.csv"),
            (names[1], "rust", ".runtime/rust.csv"),
        ):
            executable = self.native / name
            if (
                hashlib.sha256(read_regular(executable, 32 * 1024 * 1024)).hexdigest()
                != metadata[name]
            ):
                raise ValueError("Packaged native executable differs from its recorded digest")
            arguments = [str(executable)]
            if label == "rust":
                arguments += [
                    "artifacts/model/compact-mlp.evm",
                    "artifacts/test-vectors.csv",
                    destination,
                ]
            start = time.perf_counter()
            # Fixed verified packaged code and generated arguments, never user executables.
            process = subprocess.run(  # noqa: S603 - fixed hash-verified resource, no user code
                arguments, cwd=analysis, capture_output=True, check=False, timeout=30
            )
            elapsed = time.perf_counter() - start
            if (
                process.returncode != 0
                or len(process.stdout) + len(process.stderr) > MAX_PROCESS_OUTPUT
            ):
                raise ValueError(
                    "The actual native inference process failed; no agreement is claimed"
                )
            found_ids, probabilities = read_predictions(analysis / destination)
            if found_ids != identifiers:
                raise ValueError("Native prediction identities/count/order differ")
            error = maximum_absolute_error(reference, probabilities)
            if error > MAX_RUNTIME_ERROR:
                raise ValueError(
                    "Actual native predictions exceed the unchanged1e-5 agreement bound"
                )
            results[label] = {
                "probabilities": probabilities.tolist(),
                "batch_seconds": elapsed,
                "reference_max_abs": error,
                "exit_code": process.returncode,
            }
        return results

    def close(self) -> None:
        """Release the actual OS profile lock only after the application retained task completes."""
        self.home.close()
