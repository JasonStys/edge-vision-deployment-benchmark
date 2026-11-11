# Edge Vision Deployment Benchmark

## Native authoring and deployment workbench

**Edge Vision Workbench 1.1.0** adds a real Windows desktop interface to the existing model and
inference engines. Prepare editable held-out CSV data, import your own bounded 8×8 vectors, or draw
a pattern; compare actual NumPy, ONNX float32, ONNX int8, C++ and Rust predictions. Inspect all
model/condition review rows, change advisory gates, export captured evidence, and rebuild the original
reference model into a new retained generation. This remains a synthetic-shape engineering lab;
probabilities and laboratory timings do not establish real-camera performance.

The unsigned Windows candidate bundles patched Python 3.14.10, Tcl/Tk, scientific libraries and both
compiled inference engines. Users launch `EdgeVisionWorkbench.exe` without Python, a compiler, an IDE
or setup commands. The complete extracted folder must stay together. Data lives in a separate
OS-owner profile, with exclusive ownership, no network service and no silent reset of unfamiliar
state. Quit retains actual pending computation and file work before releasing that profile.

See [native usage and build instructions](docs/native-application.md),
[observed validation and packaging fixes](docs/reports/native-application-2026-11-11.md), and
[limitations](docs/LIMITATIONS.md). Public release, installation and clean-main hosted acceptance
remain separate delivery steps until their actual evidence is available.

![Actual Windows app comparing five engines on editable synthetic vectors](docs/images/native-five-engines.png)

![Actual app exposing twelve model and condition rows with strict advisory gate failures](docs/images/native-review-gates.png)

## Candidate review envelope

For a deployment engineer comparing the portable, ONNX float32 and dynamic-int8 variants, `python -m edgevision review --root . --report NEW.json` verifies the artifact manifest and evaluates **each runtime on the same held-out48 synthetic samples under clean, Gaussian noise, center occlusion and dimming conditions**. Every condition reports its confusion matrix, per-class recall, accuracy, calibration error and probability agreement with the portable reference. A weak class cannot hide behind high average accuracy.

Customize laboratory acceptance gates with `--minimum-accuracy`, `--minimum-class-recall`, `--maximum-ece` and `--maximum-runtime-error` (finite0–1). The dossier names failed condition/gate pairs and includes dataset/model fingerprints, model bytes and fresh same-session latency/memory measurements for informed tradeoffs. It never chooses/promotes a model or changes release thresholds; export refuses existing files. Standard output also supports pipeline integration. See [review workflow](docs/DEPLOYMENT_REVIEW.md) and [validation](docs/reports/practical-workflows-20261109.md).

[![CI](https://github.com/JasonStys/edge-vision-deployment-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonStys/edge-vision-deployment-benchmark/actions/workflows/ci.yml)
[![CodeQL](https://github.com/JasonStys/edge-vision-deployment-benchmark/actions/workflows/codeql.yml/badge.svg)](https://github.com/JasonStys/edge-vision-deployment-benchmark/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A reproducible deployment laboratory for a compact image classifier. The repository generates its
own checksummed 8x8 image data, trains and calibrates a neural network, exports and quantizes ONNX,
and then verifies equivalent inference in Python, C++, and Rust. It reports accuracy, per-class
quality, calibration, robustness, latency percentiles, throughput, memory allocation, and a clearly
labeled energy proxy.

This is deliberately an engineering project rather than a notebook: every artifact has a rebuild
path, every parser has explicit limits, native runtimes have no third-party dependencies, and CI
checks the complete deployment contract.

## What it demonstrates

- Deterministic data generation, balanced stratified splits, SHA-256 provenance, and a dataset card.
- A from-scratch 64–24–12–3 ReLU network with full-batch training and held-out temperature scaling.
- A transparent text model format plus a standard ONNX graph and dynamic signed-int8 weight export.
- Cross-runtime test vectors proving Python, C++, and Rust stay within a `1e-5` probability tolerance.
- Clean, noisy, occluded, and dimmed evaluation with confusion matrices and calibration error.
- Warm CPU benchmarks with p50/p95/p99 latency, throughput, Python allocation peaks, and an energy proxy.
- Fail-closed file-size, row-count, dimension, finite-value, schema, and checksum validation.
- Pinned GitHub Actions, CodeQL, dependency review, Dependabot, sanitizers, Clippy, Ruff, mypy, and coverage.

## Architecture at a glance

```mermaid
flowchart LR
    G[Deterministic generator] --> D[Checksummed NPZ dataset]
    D --> T[NumPy training + calibration]
    T --> P[Portable EVM model]
    P --> O[ONNX float32]
    O --> Q[ONNX int8 weights]
    P --> C[C++ inference]
    P --> R[Rust inference]
    P --> Y[Python reference]
    O --> Y
    Q --> Y
    C --> A[Agreement gate]
    R --> A
    Y --> A
    A --> E[Metrics + benchmark evidence]
```

The portable model makes native inference reviewable; ONNX demonstrates ecosystem deployment. The
same weights, preprocessing, class order, calibration temperature, and test vectors bind them
together. See [architecture](docs/ARCHITECTURE.md) and the
[decision record](docs/adr/0001-portable-model-and-onnx.md).

## Quick start

The original CLI supports Python 3.12–3.14, CMake 3.20+, a C++20 compiler, Rust 1.90, and Node.js 24.
The self-contained Windows workbench has its own exact patched build/runtime contract below.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m edgevision validate --root .
python -m pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. To rebuild all small committed
artifacts from source:

```bash
python -m edgevision build --root .
```

Build and compare the native runtimes:

```bash
cmake -S native/cpp -B .runtime/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build .runtime/cpp --config Release
ctest --test-dir .runtime/cpp -C Release --output-on-failure
cargo test --manifest-path native/rust/Cargo.toml --locked

.runtime/cpp/edgevision_cpp
cargo run --release --locked --manifest-path native/rust/Cargo.toml -- artifacts/model/compact-mlp.evm artifacts/test-vectors.csv .runtime/rust.csv
python scripts/compare-native.py --reference artifacts/reference-predictions.csv --candidate .runtime/cpp.csv --candidate .runtime/rust.csv
```

Run the C++ executable from the repository root. It intentionally uses the allow-listed model,
vector, and output paths shown above instead of accepting filesystem paths from command-line input.

`scripts/verify.sh` runs the full local Linux/macOS sequence. Windows users can run the same commands
individually; CI exercises the reference Ubuntu toolchain.

## Major features

| Feature | Implementation | Evidence |
|---|---|---|
| Reproducible dataset | Geometric generator, typed NPZ, deterministic splits | `artifacts/dataset`, dataset card |
| Compact neural model | Two hidden ReLU layers and temperature calibration | portable model, generated model card |
| Portable deployment | Explicit ONNX graph and weight quantization | ONNX checker, Runtime tests |
| Native inference | Dependency-free C++20 and Rust 2024 parsers/runtimes | CTest, sanitizers, Cargo tests |
| Robustness analysis | Noise, occlusion, and brightness corruption suites | generated metrics report |
| Performance analysis | Warmups, percentile latency, throughput, memory, energy proxy | generated benchmark report |
| Supply-chain controls | Exact Python lock, Rust lock, pinned actions, audits | CI and security documentation |

## Repository map

| Path | Purpose |
|---|---|
| `src/edgevision/` | Data, training, serialization, ONNX, evaluation, and CLI code |
| `native/cpp/` | C++20 portable model parser, inference CLI, and unit tests |
| `native/rust/` | Dependency-free Rust parser, inference CLI, and tests |
| `tests/` | Python unit, integration, corruption, export, and artifact tests |
| `artifacts/` | Small generated dataset, models, vectors, predictions, and checksums |
| `scripts/` | Full verification, benchmarking, native comparison, and documentation gates |
| `docs/` | Requirements, design, operations, security, test strategy, reports, and code index |
| `.github/workflows/` | CI, CodeQL, and pull-request dependency review |

Every tracked file is summarized in [the file catalog](docs/FILE_CATALOG.md); declarations and exact
line locations are generated in [the code index](docs/CODE_INDEX.md).

## Release gates

A build is releasable only when clean float32 accuracy is at least 95%, quantization loses no more
than two percentage points, portable-to-ONNX maximum absolute error is at most `1e-5`, both native
runtimes match reference vectors within `1e-5`, and all static analysis, tests, artifact checks, and
security workflows pass. Performance numbers are characterization evidence rather than universal
guarantees because shared-runner hardware varies.

The checked release evidence records 100% accuracy on the 48-example synthetic test split, 97.36%
branch-aware Python coverage, and maximum C++/Rust reference error of `1.1920929e-7`. These results
validate the repository contract; they are not claims about real-camera performance.

## Documentation

- [Requirements](docs/REQUIREMENTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset card](docs/DATASET_CARD.md)
- [Benchmark method](docs/BENCHMARKING.md)
- [Testing strategy](docs/TESTING.md)
- [Operations runbook](docs/OPERATIONS.md)
- [Security model](docs/SECURITY.md)
- [Complexity analysis](docs/COMPLEXITY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Research sources](docs/RESEARCH.md)
- [Deployment checklist](docs/DEPLOYMENT_CHECKLIST.md)
- [Validation report](docs/reports/VALIDATION.md)

## Scope and safety

The generated patterns are intentionally small and non-sensitive. This project is not a production
camera system, medical device, identity system, autonomous-control component, or benchmark across
different machines. Never treat the energy proxy as a physical power measurement. See
[limitations](docs/LIMITATIONS.md) before reusing the design.

## License

MIT. See [LICENSE](LICENSE).

## Documentation editions

Start with the [organized handbook](docs/handbook.md) for current usage, design, security and test guidance. Reviewed downloadable editions: [project-guide.docx](docs/reports/project-guide.docx).

[Publication identities and source scope](docs/reports/publications.json) · [Documentation validation record](docs/reports/documentation-refresh-2026-11-09.md). Historical results retain their own tested revision and date.


## Current Windows application guide

The published **1.1.0** app has a [four-page illustrated guide](docs/reports/windows-application-guide-1.1.0.docx) with actual installed screenshots, a worked user journey, source excerpts and operating limits. Read the [installed validation report](docs/reports/installed-windows-1.1.0-2026-11-11.md) and [review manifest](docs/reports/windows-application-guide-1.1.0.json) for exact scope. Historical editions remain unchanged.
