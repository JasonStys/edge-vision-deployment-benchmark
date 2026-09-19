# Validation Report

- Release: 1.0.0
- Validation date: 2026-09-18
- Status: passed locally and on GitHub-hosted Ubuntu runners

The release validator passed SHA-256 and byte-size checks, dataset schema/content identity, portable
model format/dimensions/finiteness, float and quantized ONNX structure/shape inference, and Python
portable/ONNX reference-vector agreement. Portable reference error was exactly `0`; ONNX reference
maximum absolute error was `1.1920929e-7`, below the `1e-5` gate. The dataset digest was
`314cd9d7e7ff699791baeb57b9dec6f64f004403a107bfe5099bb7e3d238fd22`.

Machine-readable results are in `docs/reports/generated/validation.json`. GitHub Actions commit
`48dc8d2` passed Python quality/tests/audits, C++ sanitizers and CTest, Rust format/Clippy/tests/audit,
cross-runtime agreement, repository policy, the non-root container build, and CodeQL for Python,
C++, and Rust. The latest workflow run remains the authority after later maintenance commits.
