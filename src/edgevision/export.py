"""File: Exports the portable MLP to ONNX, validates it, and creates a weight-quantized variant.

Functions: export_onnx, quantize_onnx, and validate_onnx form the deployment conversion boundary.
Variables: graph nodes and initializers mirror PortableModel tensors; exact lines are indexed in
``docs/CODE_INDEX.md``.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from edgevision.constants import CLASS_COUNT, HIDDEN_ONE, HIDDEN_TWO, INPUT_SIZE, MAX_ONNX_BYTES
from edgevision.model import PortableModel

ONNX_OPSET = 18


def export_onnx(model: PortableModel, path: Path) -> None:
    """Translate the fixed dense graph to an explicit, checked ONNX model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    initializers = [
        numpy_helper.from_array(model.weights_one, "weights_one"),
        numpy_helper.from_array(model.bias_one, "bias_one"),
        numpy_helper.from_array(model.weights_two, "weights_two"),
        numpy_helper.from_array(model.bias_two, "bias_two"),
        numpy_helper.from_array(model.weights_three, "weights_three"),
        numpy_helper.from_array(model.bias_three, "bias_three"),
        numpy_helper.from_array(np.asarray(model.temperature, dtype=np.float32), "temperature"),
    ]
    nodes = [
        helper.make_node("MatMul", ["input", "weights_one"], ["hidden_one_linear"]),
        helper.make_node("Add", ["hidden_one_linear", "bias_one"], ["hidden_one_biased"]),
        helper.make_node("Relu", ["hidden_one_biased"], ["hidden_one"]),
        helper.make_node("MatMul", ["hidden_one", "weights_two"], ["hidden_two_linear"]),
        helper.make_node("Add", ["hidden_two_linear", "bias_two"], ["hidden_two_biased"]),
        helper.make_node("Relu", ["hidden_two_biased"], ["hidden_two"]),
        helper.make_node("MatMul", ["hidden_two", "weights_three"], ["logits_linear"]),
        helper.make_node("Add", ["logits_linear", "bias_three"], ["logits"]),
        helper.make_node("Div", ["logits", "temperature"], ["calibrated_logits"]),
        helper.make_node("Softmax", ["calibrated_logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "edgevision_compact_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, INPUT_SIZE])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, CLASS_COUNT])],
        initializer=initializers,
        value_info=[
            helper.make_tensor_value_info("hidden_one", TensorProto.FLOAT, [None, HIDDEN_ONE]),
            helper.make_tensor_value_info("hidden_two", TensorProto.FLOAT, [None, HIDDEN_TWO]),
        ],
    )
    exported = helper.make_model(
        graph,
        producer_name="edge-vision-deployment-benchmark",
        producer_version="1.0.0",
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    # ONNX 1.23 defaults to IR 14 while ONNX Runtime 1.30 supports through IR 13.
    # This graph uses no IR-14-only feature, so the compatibility ceiling is explicit.
    exported.ir_version = 13
    exported.doc_string = "Calibrated compact line-orientation classifier generated from source."
    exported.metadata_props.add(key="portable_format", value="EDGEVISION_MLP/1")
    onnx.checker.check_model(exported, full_check=True)
    onnx.save_model(exported, path)


def quantize_onnx(source: Path, destination: Path) -> None:
    """Apply deterministic signed 8-bit dynamic weight quantization to dense operators."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="edgevision-quantize-") as temporary_directory:
        preprocessed = Path(temporary_directory) / "preprocessed.onnx"
        quant_pre_process(
            str(source),
            str(preprocessed),
            skip_symbolic_shape=True,
            skip_optimization=False,
            skip_onnx_shape=False,
        )
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(destination),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
    validate_onnx(destination)


def validate_onnx(path: Path) -> None:
    """Reject missing, oversized, structurally invalid, or shape-inference-invalid ONNX files."""

    if not path.is_file() or path.stat().st_size > MAX_ONNX_BYTES:
        raise ValueError("ONNX artifact is missing or exceeds the 2 MB project safety limit")
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
