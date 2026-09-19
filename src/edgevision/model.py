"""File: Trains, calibrates, serializes, and evaluates the bounded portable neural model.

Functions: train_model, fit_temperature, centroid_baseline, read_model, and write_model provide the
learning and portable inference path. Classes: PortableModel and TrainingSummary. Variables:
weights, biases, logits, and gradients are shape-bounded; exact locations are in the code index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from edgevision.constants import (
    CLASS_COUNT,
    CLASS_NAMES,
    DEFAULT_SEED,
    HIDDEN_ONE,
    HIDDEN_TWO,
    INPUT_SIZE,
    MATRIX_RANK,
    MAX_EPOCHS,
    MAX_INFERENCE_ROWS,
    MAX_MODEL_BYTES,
    MAX_TEMPERATURE,
    MIN_LEARNING_RATE,
    MIN_TEMPERATURE,
    MODEL_FORMAT_VERSION,
    MODEL_MAGIC,
)

FloatArray = NDArray[np.float32]
LabelArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Describe optimization convergence without retaining per-epoch mutable history."""

    epochs: int
    initial_loss: float
    final_loss: float
    temperature: float


@dataclass(frozen=True, slots=True)
class PortableModel:
    """Represent an inspectable 64-24-12-3 ReLU MLP shared by all runtime harnesses."""

    weights_one: FloatArray
    bias_one: FloatArray
    weights_two: FloatArray
    bias_two: FloatArray
    weights_three: FloatArray
    bias_three: FloatArray
    temperature: float = 1.0

    def __post_init__(self) -> None:
        """Enforce exact tensor shapes, finite parameters, and a safe calibration range."""

        shapes = {
            "weights_one": (INPUT_SIZE, HIDDEN_ONE),
            "bias_one": (HIDDEN_ONE,),
            "weights_two": (HIDDEN_ONE, HIDDEN_TWO),
            "bias_two": (HIDDEN_TWO,),
            "weights_three": (HIDDEN_TWO, CLASS_COUNT),
            "bias_three": (CLASS_COUNT,),
        }
        for name, expected in shapes.items():
            value = getattr(self, name)
            if value.shape != expected or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {expected}")
        if (
            not math.isfinite(self.temperature)
            or not MIN_TEMPERATURE <= self.temperature <= MAX_TEMPERATURE
        ):
            raise ValueError("temperature must be finite and within [0.05, 10.0]")

    def predict_logits(self, features: FloatArray) -> FloatArray:
        """Run bounded dense inference in O(N * parameter_count) time."""

        checked = _validated_features(features)
        hidden_one = np.maximum(checked @ self.weights_one + self.bias_one, 0.0)
        hidden_two = np.maximum(hidden_one @ self.weights_two + self.bias_two, 0.0)
        return np.asarray(hidden_two @ self.weights_three + self.bias_three, dtype=np.float32)

    def predict_proba(self, features: FloatArray) -> FloatArray:
        """Convert calibrated logits to stable class probabilities."""

        logits = self.predict_logits(features).astype(np.float64) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return (exponentials / np.sum(exponentials, axis=1, keepdims=True)).astype(np.float32)

    def calibrated(self, temperature: float) -> Self:
        """Return a validated immutable copy with an updated calibration temperature."""

        return replace(self, temperature=temperature)


def _validated_features(features: FloatArray) -> FloatArray:
    """Normalize array ownership and reject malformed or non-finite inference input."""

    checked = np.asarray(features, dtype=np.float32)
    if (
        checked.ndim != MATRIX_RANK
        or checked.shape[1] != INPUT_SIZE
        or checked.shape[0] > MAX_INFERENCE_ROWS
    ):
        raise ValueError(f"features must have bounded shape [N, {INPUT_SIZE}]")
    if not np.all(np.isfinite(checked)) or np.any((checked < 0.0) | (checked > 1.0)):
        raise ValueError("features must be finite values in [0, 1]")
    return checked


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute stable row-wise probabilities for optimization and calibration."""

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def train_model(
    features: FloatArray,
    labels: LabelArray,
    *,
    epochs: int = 260,
    learning_rate: float = 0.09,
    seed: int = DEFAULT_SEED,
) -> tuple[PortableModel, TrainingSummary]:
    """Train a deterministic compact MLP with full-batch gradient descent and L2 decay."""

    x = _validated_features(features).astype(np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (x.shape[0],) or np.any((y < 0) | (y >= CLASS_COUNT)):
        raise ValueError("labels must align with features and use supported classes")
    if not 1 <= epochs <= MAX_EPOCHS or not MIN_LEARNING_RATE <= learning_rate <= 1.0:
        raise ValueError("training hyperparameters exceed documented bounds")

    rng = np.random.default_rng(seed)
    weights_one = rng.normal(0.0, math.sqrt(2.0 / INPUT_SIZE), (INPUT_SIZE, HIDDEN_ONE))
    bias_one = np.zeros(HIDDEN_ONE, dtype=np.float64)
    weights_two = rng.normal(0.0, math.sqrt(2.0 / HIDDEN_ONE), (HIDDEN_ONE, HIDDEN_TWO))
    bias_two = np.zeros(HIDDEN_TWO, dtype=np.float64)
    weights_three = rng.normal(0.0, math.sqrt(2.0 / HIDDEN_TWO), (HIDDEN_TWO, CLASS_COUNT))
    bias_three = np.zeros(CLASS_COUNT, dtype=np.float64)
    targets = np.eye(CLASS_COUNT, dtype=np.float64)[y]
    weight_decay = 0.0005
    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(epochs):
        hidden_one_pre = x @ weights_one + bias_one
        hidden_one = np.maximum(hidden_one_pre, 0.0)
        hidden_two_pre = hidden_one @ weights_two + bias_two
        hidden_two = np.maximum(hidden_two_pre, 0.0)
        logits = hidden_two @ weights_three + bias_three
        probabilities = _softmax(logits)
        loss = -float(np.mean(np.log(probabilities[np.arange(y.size), y] + 1e-12)))
        if epoch == 0:
            initial_loss = loss
        final_loss = loss

        gradient_logits = (probabilities - targets) / y.size
        gradient_weights_three = hidden_two.T @ gradient_logits + weight_decay * weights_three
        gradient_bias_three = np.sum(gradient_logits, axis=0)
        gradient_hidden_two = gradient_logits @ weights_three.T
        gradient_hidden_two[hidden_two_pre <= 0.0] = 0.0
        gradient_weights_two = hidden_one.T @ gradient_hidden_two + weight_decay * weights_two
        gradient_bias_two = np.sum(gradient_hidden_two, axis=0)
        gradient_hidden_one = gradient_hidden_two @ weights_two.T
        gradient_hidden_one[hidden_one_pre <= 0.0] = 0.0
        gradient_weights_one = x.T @ gradient_hidden_one + weight_decay * weights_one
        gradient_bias_one = np.sum(gradient_hidden_one, axis=0)

        weights_three -= learning_rate * gradient_weights_three
        bias_three -= learning_rate * gradient_bias_three
        weights_two -= learning_rate * gradient_weights_two
        bias_two -= learning_rate * gradient_bias_two
        weights_one -= learning_rate * gradient_weights_one
        bias_one -= learning_rate * gradient_bias_one

    model = PortableModel(
        weights_one.astype(np.float32),
        bias_one.astype(np.float32),
        weights_two.astype(np.float32),
        bias_two.astype(np.float32),
        weights_three.astype(np.float32),
        bias_three.astype(np.float32),
    )
    return model, TrainingSummary(epochs, initial_loss, final_loss, model.temperature)


def fit_temperature(model: PortableModel, features: FloatArray, labels: LabelArray) -> float:
    """Select a scalar temperature on held-out data by bounded grid search over NLL."""

    y = np.asarray(labels, dtype=np.int64)
    logits = model.predict_logits(features).astype(np.float64)
    if y.shape != (logits.shape[0],):
        raise ValueError("calibration labels must align with features")
    best_temperature = 1.0
    best_loss = math.inf
    for temperature in np.linspace(0.35, 3.0, 107):
        probabilities = _softmax(logits / float(temperature))
        loss = -float(np.mean(np.log(probabilities[np.arange(y.size), y] + 1e-12)))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return best_temperature


def centroid_baseline(train_x: FloatArray, train_y: LabelArray, features: FloatArray) -> FloatArray:
    """Produce a transparent nearest-centroid probability baseline for comparison."""

    checked_train = _validated_features(train_x)
    checked_features = _validated_features(features)
    labels = np.asarray(train_y, dtype=np.int64)
    if labels.shape != (checked_train.shape[0],):
        raise ValueError("baseline labels must align with training features")
    centroids = np.stack(
        [np.mean(checked_train[labels == label], axis=0) for label in range(CLASS_COUNT)]
    )
    distances = np.sum((checked_features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return _softmax(-distances.astype(np.float64)).astype(np.float32)


def _section(name: str, values: FloatArray) -> str:
    """Encode one named, counted tensor section with round-trip-safe decimal values."""

    flattened = values.reshape(-1)
    return f"{name} {flattened.size}\n" + " ".join(f"{float(value):.9g}" for value in flattened)


def write_model(model: PortableModel, path: Path) -> None:
    """Write a human-inspectable, count-delimited portable model artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"{MODEL_MAGIC} {MODEL_FORMAT_VERSION}",
            f"dims {INPUT_SIZE} {HIDDEN_ONE} {HIDDEN_TWO} {CLASS_COUNT}",
            "classes " + " ".join(CLASS_NAMES),
            f"temperature {model.temperature:.9g}",
            _section("weights_one", model.weights_one),
            _section("bias_one", model.bias_one),
            _section("weights_two", model.weights_two),
            _section("bias_two", model.bias_two),
            _section("weights_three", model.weights_three),
            _section("bias_three", model.bias_three),
            "end",
        ]
    )
    if len(content.encode("utf-8")) > MAX_MODEL_BYTES:
        raise ValueError("serialized model exceeds the documented size limit")
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(content + "\n")


class _TokenReader:
    """Read the portable format with explicit token, count, and trailing-data checks."""

    def __init__(self, content: str) -> None:
        """Split a bounded, already decoded artifact into a forward-only token stream."""

        self._tokens = content.split()
        self._offset = 0

    def take(self, expected: str | None = None) -> str:
        """Consume one token and optionally require an exact marker."""

        if self._offset >= len(self._tokens):
            raise ValueError("portable model ended unexpectedly")
        value = self._tokens[self._offset]
        self._offset += 1
        if expected is not None and value != expected:
            raise ValueError(f"expected {expected!r}, received {value!r}")
        return value

    def floats(self, name: str, shape: tuple[int, ...]) -> FloatArray:
        """Consume one named tensor after checking its declared element count."""

        self.take(name)
        expected_count = math.prod(shape)
        if int(self.take()) != expected_count:
            raise ValueError(f"{name} element count does not match the model contract")
        try:
            values = np.asarray(
                [float(self.take()) for _ in range(expected_count)], dtype=np.float32
            )
        except ValueError as error:
            raise ValueError(f"{name} contains a non-numeric value") from error
        return values.reshape(shape)

    def finished(self) -> bool:
        """Return true only when every token has been consumed."""

        return self._offset == len(self._tokens)


def read_model(path: Path) -> PortableModel:
    """Parse an untrusted portable artifact after size, format, dimension, and value checks."""

    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("portable model is missing or exceeds the size limit")
    reader = _TokenReader(path.read_text(encoding="utf-8", errors="strict"))
    reader.take(MODEL_MAGIC)
    if int(reader.take()) != MODEL_FORMAT_VERSION:
        raise ValueError("unsupported portable model version")
    reader.take("dims")
    dimensions = tuple(int(reader.take()) for _ in range(4))
    if dimensions != (INPUT_SIZE, HIDDEN_ONE, HIDDEN_TWO, CLASS_COUNT):
        raise ValueError("portable model dimensions do not match the runtime")
    reader.take("classes")
    if tuple(reader.take() for _ in range(CLASS_COUNT)) != CLASS_NAMES:
        raise ValueError("portable model class order does not match the runtime")
    reader.take("temperature")
    temperature = float(reader.take())
    model = PortableModel(
        reader.floats("weights_one", (INPUT_SIZE, HIDDEN_ONE)),
        reader.floats("bias_one", (HIDDEN_ONE,)),
        reader.floats("weights_two", (HIDDEN_ONE, HIDDEN_TWO)),
        reader.floats("bias_two", (HIDDEN_TWO,)),
        reader.floats("weights_three", (HIDDEN_TWO, CLASS_COUNT)),
        reader.floats("bias_three", (CLASS_COUNT,)),
        temperature,
    )
    reader.take("end")
    if not reader.finished():
        raise ValueError("portable model contains trailing data")
    return model
