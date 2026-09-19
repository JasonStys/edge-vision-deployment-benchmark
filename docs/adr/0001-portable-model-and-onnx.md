# ADR 0001: Pair an inspectable native format with ONNX

- Status: Accepted
- Date: 2026-09-18

## Context

The project must demonstrate neural-model training, standard export, quantization, and equivalent
Python/C++/Rust inference while remaining fast and reliable on CPU-only CI. Pulling full native ONNX
runtimes into two language builds would make the example substantially larger and would blur the code
that employers should be able to review.

## Decision drivers

- Numerical equivalence across runtimes must be executable, not asserted.
- ONNX must remain the ecosystem deployment artifact.
- Native parsers must be bounded and easy to audit.
- Artifacts must be small, reproducible, and independent of remote datasets.
- CI duration and dependency surface should remain proportionate to a portfolio repository.

## Options considered

### PyTorch training plus native ONNX Runtime in every language

Strong production familiarity and direct ONNX execution everywhere, but substantially larger Python,
C++, and Rust dependency trees; slower clean CI; and more build-system code than inference logic.

### NumPy training, explicit ONNX export, and a narrow native weight format

Keeps every training/export operation visible, still validates ONNX with ONNX Runtime, and lets C++
and Rust independently implement the exact graph with no packages. The cost is maintaining two small
serialization representations and proving their transitive agreement.

### TensorFlow Lite as the only deployment representation

Attractive for mobile targets, but weaker for the requested cross-language ONNX demonstration and
still requires native runtime distribution.

## Decision

Choose NumPy training, explicit ONNX construction, and `EDGEVISION_MLP/1` for native inference. The
pipeline exports both formats from one immutable `PortableModel`. Python proves portable-to-ONNX
agreement; reference vectors then prove C++ and Rust-to-portable agreement. SHA-256 manifests bind the
committed artifacts.

## Consequences

Positive consequences are small builds, readable numerical code, independent native implementations,
and deterministic offline CI. Negative consequences are that native binaries do not parse arbitrary
ONNX graphs and the portable format must be versioned if architecture changes. The limitation is
explicit: ONNX is the interoperability artifact; the text format is an embedded-runtime teaching and
verification format, not a proposed general standard.

