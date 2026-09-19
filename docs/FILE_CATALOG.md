# File Catalog

Every tracked file has one primary responsibility. Generated binaries are called out explicitly so
reviewers can distinguish source, configuration, evidence, and deployment artifacts.

## Root configuration and policy

- `.dockerignore`: removes Git, caches, virtual environments, and build output from container context.
- `.gitattributes`: normalizes source line endings and marks ONNX/NPZ files as binary.
- `.gitignore`: excludes environments, caches, coverage output, compilers, and transient evidence.
- `README.md`: project outcome, architecture, commands, features, file map, and release gates.
- `CONTRIBUTING.md`: branch, testing, artifact, documentation, and commit expectations.
- `SECURITY.md`: supported version, private reporting path, and trust-boundary summary.
- `LICENSE`: MIT license terms.
- `pyproject.toml`: Python metadata, exact dependencies, CLI, lint, typing, tests, and coverage.
- `requirements.lock`: exact full Python quality, audit, build, and runtime environment.
- `requirements-runtime.lock`: minimal exact Python build/inference environment for the container.
- `package.json`: dependency-free repository validation and code-index commands.
- `rust-toolchain.toml`: exact Rust compiler profile plus Clippy and rustfmt.
- `Dockerfile`: minimal locked Python install, non-root user, artifacts, and validation entry point.

## GitHub automation

- `.github/dependabot.yml`: weekly Python, Cargo, Docker, and action update proposals.
- `.github/workflows/ci.yml`: Python, locked Rust audit, native agreement, repository, and container gates.
- `.github/workflows/codeql.yml`: Python/Rust no-build analysis and manually built C++ analysis.
- `.github/workflows/dependency-review.yml`: high-severity dependency change rejection on pull requests.

## Python package

- `src/edgevision/__init__.py`: stable public imports and package version.
- `src/edgevision/__main__.py`: `python -m edgevision` entry point.
- `src/edgevision/py.typed`: declares inline type information to static type checkers.
- `src/edgevision/constants.py`: model dimensions, classes, thresholds, versions, and safety bounds.
- `src/edgevision/data.py`: generator, stratified splits, corruptions, digest, and safe NPZ I/O.
- `src/edgevision/model.py`: MLP training, calibration, baseline, inference, and portable format.
- `src/edgevision/export.py`: explicit ONNX graph, preprocessing, quantization, and structural validation.
- `src/edgevision/runtime.py`: single-thread CPU ONNX Runtime wrapper and agreement calculation.
- `src/edgevision/metrics.py`: quality, calibration, latency, throughput, memory, and energy-proxy metrics.
- `src/edgevision/artifacts.py`: checksums, exchange CSVs, and artifact manifest verification.
- `src/edgevision/pipeline.py`: build, validate, benchmark, model-card, threshold, and CLI orchestration.

## C++ native runtime

- `native/cpp/CMakeLists.txt`: C++20 targets, strict warnings, sanitizers, and CTest registration.
- `native/cpp/include/edgevision/model.hpp`: public constants, types, parser, inference, and CSV API.
- `native/cpp/src/model.cpp`: strict portable/CSV parsing, dense inference, softmax, and output.
- `native/cpp/src/main.cpp`: argument validation, orchestration, error reporting, and exit status.
- `native/cpp/tests/model_tests.cpp`: model loading, probability, and invalid-feature tests.

## Rust native runtime

- `native/rust/Cargo.toml`: dependency-free library/binary metadata and optimized release profile.
- `native/rust/Cargo.lock`: exact dependency-free Cargo resolution.
- `native/rust/src/lib.rs`: safe parser, inference engine, vector I/O, prediction output, and tests.
- `native/rust/src/main.rs`: Rust CLI argument handling, orchestration, and failure status.

## Python tests

- `tests/test_data.py`: determinism, balance, stratification, corruption, persistence, and rejection.
- `tests/test_model.py`: convergence, calibration, baseline, serialization, and input guards.
- `tests/test_export_runtime.py`: float export, quantization, ONNX validation, and agreement.
- `tests/test_metrics_artifacts.py`: classification, benchmark, CSV, manifest, and tamper tests.
- `tests/test_pipeline.py`: checked evidence, complete rebuild, validation, benchmark, and CLI integration.
- `tests/test_validation_guards.py`: negative tests for shapes, schemas, values, tokens, files, and limits.

## Scripts

- `scripts/compare-native.py`: row/id validation and maximum probability error gate.
- `scripts/generate-code-index.mjs`: exact source declaration line-index generator/checker.
- `scripts/validate-repository.mjs`: docs, headers, size, hash, marker, and workflow-pin policy.
- `scripts/verify.sh`: full Python, repository, C++, Rust, and cross-runtime verification sequence.
- `scripts/benchmark.sh`: repeatable machine-local benchmark entry point.

## Generated deployment artifacts

- `artifacts/dataset/shapes-v1.npz`: compressed typed train/validation/test arrays.
- `artifacts/dataset/shapes-v1.manifest.json`: dataset digest, shape, format, and split counts.
- `artifacts/model/compact-mlp.evm`: human-inspectable calibrated weights for native inference.
- `artifacts/model/compact-mlp.onnx`: standard float32 ONNX deployment graph.
- `artifacts/model/compact-mlp.int8.onnx`: dynamically signed-int8-quantized dense weights.
- `artifacts/test-vectors.csv`: labeled normalized rows shared by every runtime.
- `artifacts/reference-predictions.csv`: Python portable-model probability reference.
- `artifacts/manifest.json`: SHA-256 and byte size for generated release evidence.

## Design and operating documentation

- `docs/REQUIREMENTS.md`: functional, non-functional, scope, and acceptance requirements.
- `docs/ARCHITECTURE.md`: components, data flow, reliability, scale, observability, and trade-offs.
- `docs/adr/0001-portable-model-and-onnx.md`: alternatives and decision for dual model formats.
- `docs/DATASET_CARD.md`: generation, provenance, splits, transformations, and appropriate use.
- `docs/BENCHMARKING.md`: warmup, percentile, throughput, memory, and energy-proxy methodology.
- `docs/TESTING.md`: risk-based test layers, CI gates, manual tests, and remaining gaps.
- `docs/OPERATIONS.md`: validation, rebuild, diagnosis, rollback, and retention runbook.
- `docs/SECURITY.md`: assets, trust boundaries, controls, and unresolved threats.
- `docs/LIMITATIONS.md`: data, model, runtime, hardware, benchmark, and integrity limitations.
- `docs/RESEARCH.md`: primary ONNX, NIST, GitHub, C++, and Rust sources that informed design.
- `docs/COMPLEXITY.md`: Big-O training/inference analysis and explicit resource limits.
- `docs/DEPLOYMENT_CHECKLIST.md`: pre-deploy, deploy, post-deploy, and rollback verification.
- `docs/FILE_CATALOG.md`: this one-line responsibility catalog for every tracked file.
- `docs/CODE_INDEX.md`: generated exact declaration-to-line map for source review.

## Human and machine-readable reports

- `docs/reports/VALIDATION.md`: release integrity and agreement results.
- `docs/reports/TEST_SUMMARY.md`: Python/native suite and coverage results.
- `docs/reports/PERFORMANCE.md`: interpretation plus local latency/model-size characterization.
- `docs/reports/generated/MODEL_CARD.md`: generated intended use, measured quality, and limits.
- `docs/reports/generated/metrics.json`: clean, baseline, corruption, calibration, and agreement data.
- `docs/reports/generated/benchmark.json`: host-local methodology and runtime measurements.
- `docs/reports/generated/validation.json`: machine-readable integrity/structure/agreement status.
