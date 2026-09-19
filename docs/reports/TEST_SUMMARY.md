# Test Summary

The suite covers deterministic data, split invariants, corruption bounds, learning convergence,
temperature calibration, baseline behavior, serialization round trips, malformed artifacts, ONNX
export, dynamic weight quantization, runtime agreement, classification/calibration metrics, benchmark
methodology, CSV schemas, manifest tampering, pipeline validation, C++ normalized inference, Rust
normalized inference, native invalid-input rejection, and cross-runtime reference comparison.

## Local release results

- Python: 29 tests passed on Python 3.14.6.
- Coverage: 97.34% combined statement/branch coverage; the enforced minimum is 94%.
- C++: MSVC 19.51 warning-as-error build passed; one CTest executable passed.
- Rust: formatting, warning-free Clippy, three library tests, binary tests, and doc tests passed.
- Cross-runtime: C++ and Rust each differed from Python reference probabilities by at most
  `1.1920929e-7`, below the `1e-5` gate.

Remote Ubuntu sanitizers, dependency audits, the container build, and CodeQL remain independently
enforced by the published GitHub Actions workflows.
