# Research and Standards Notes

Sources were selected from primary project and standards documentation and reviewed on 2026-09-18.

## ONNX and runtime deployment

- [ONNX versioning](https://onnx.ai/onnx/repo-docs/Versioning.html) separates IR, operator-set, and
  model versions. The exported graph therefore declares an explicit opset and the native artifact has
  its own independent version.
- [ONNX Runtime model validation](https://onnxruntime.ai/docs/) states that developers remain
  responsible for accuracy, performance, and suitability. This repository adds agreement, accuracy,
  corruption, and parser gates rather than treating successful load as validation.
- [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/tune-performance/) identifies
  latency, throughput, memory, and application size as important dimensions. The benchmark reports all
  four where practical and fixes thread settings for lower variance.
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
  explains dynamic/static methods, accuracy trade-offs, operator formats, and hardware dependence.
  This project labels its dynamic weight quantization and rejects excessive accuracy loss.

## AI evaluation

- [NIST AI Risk Management Framework 1.0](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)
  places test, evaluation, verification, and validation throughout the lifecycle. Generated data/model
  cards, integrity evidence, robustness metrics, and known limitations apply that principle at this
  project's scale.

## Secure automation and native code

- [GitHub secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
  recommends pinning third-party actions to full commit SHAs; all workflow actions are pinned and
  checked by repository policy.
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) inform use of
  RAII containers, explicit bounds, narrow interfaces, and warnings/sanitizers.
- [The Rust Book](https://doc.rust-lang.org/book/) informs ownership, error propagation, immutable
  models, iterator use, and dependency-free parsing in the native Rust path.

