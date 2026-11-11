"""File: Exposes the stable public API for dataset, model, export, and evaluation workflows.

Functions: Public functions are imported from focused modules below.
Variables: ``__version__`` identifies the repository artifact version; exact declaration lines are
indexed in ``docs/CODE_INDEX.md``.
"""

import os

# Process-only vendor opt-out must precede every optional ONNX Runtime initialization.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

from edgevision.data import VisionDataset, generate_dataset
from edgevision.model import PortableModel, train_model

__all__ = ["PortableModel", "VisionDataset", "generate_dataset", "train_model"]
__version__ = "1.1.0"
