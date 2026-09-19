"""File: Centralizes bounded model, dataset, and artifact constants used by every runtime.

Functions: This module intentionally contains no functions.
Variables: Image dimensions, layer widths, class names, seeds, and safety limits are declared here;
exact declaration lines are generated in ``docs/CODE_INDEX.md``.
"""

from typing import Final

IMAGE_SIDE: Final = 8
INPUT_SIZE: Final = IMAGE_SIDE * IMAGE_SIDE
HIDDEN_ONE: Final = 24
HIDDEN_TWO: Final = 12
CLASS_NAMES: Final = ("vertical", "horizontal", "diagonal")
CLASS_COUNT: Final = len(CLASS_NAMES)
MODEL_MAGIC: Final = "EDGEVISION_MLP"
MODEL_FORMAT_VERSION: Final = 1
DEFAULT_SEED: Final = 41
MAX_MODEL_BYTES: Final = 1_000_000
MAX_ONNX_BYTES: Final = 2_000_000
MAX_VECTOR_FILE_BYTES: Final = 5_000_000
MAX_PREDICTION_FILE_BYTES: Final = 1_000_000
MAX_DATASET_BYTES: Final = 50_000_000
MAX_VECTOR_ROWS: Final = 4_096
MAX_DATASET_SAMPLES_PER_CLASS: Final = 10_000
MIN_DATASET_SAMPLES_PER_CLASS: Final = 10
MAX_INFERENCE_ROWS: Final = 100_000
MATRIX_RANK: Final = 2
DIAGONAL_LABEL: Final = 2
MIN_TEMPERATURE: Final = 0.05
MAX_TEMPERATURE: Final = 10.0
MAX_EPOCHS: Final = 5_000
MIN_LEARNING_RATE: Final = 0.0001
MAX_BENCHMARK_WARMUPS: Final = 100
MIN_BENCHMARK_ITERATIONS: Final = 5
MAX_BENCHMARK_ITERATIONS: Final = 10_000
MIN_POWER_PROXY_WATTS: Final = 0.1
MAX_POWER_PROXY_WATTS: Final = 1_000.0
MIN_RELEASE_ACCURACY: Final = 0.95
MAX_QUANTIZED_ACCURACY_LOSS: Final = 0.02
MAX_PORTABLE_ERROR: Final = 1e-6
MAX_RUNTIME_ERROR: Final = 1e-5
