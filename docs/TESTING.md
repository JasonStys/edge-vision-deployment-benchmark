# Testing Strategy

## Risk model

The most important failures are data leakage, non-deterministic or invalid data, incorrect gradients,
export drift, quantization quality loss, preprocessing/class-order mismatch, unsafe artifact parsing,
misleading benchmarks, and dependency or workflow compromise.

## Test layers

| Layer | Coverage | Primary failures caught |
|---|---|---|
| Unit | rendering, corruptions, metrics, parsers, softmax, checksums | boundary and numerical defects |
| Property/invariant | normalized ranges, split balance, probability sums, finite tensors | malformed state |
| Integration | train → calibrate → serialize → export → infer | contract drift between modules |
| Cross-runtime | shared vectors through Python/C++/Rust | layout, class order, numeric mismatch |
| Robustness | Gaussian, occlusion, dimming | brittle behavior hidden by clean accuracy |
| Performance | warm percentile latency, throughput, allocation peak | regressions and misleading cold timing |
| Security | size/count bounds, audits, CodeQL, dependency review, sanitizers | unsafe inputs and supply-chain risk |
| Reproducibility | content digest, artifact manifest, generated code index | stale or substituted evidence |

## CI gates

Python CI runs Ruff formatting/lint, strict mypy, branch coverage, the complete pytest suite, pip-audit,
artifact validation, and a fresh benchmark. Native CI builds C++ with warnings as errors and sanitizers,
runs CTest, then runs Rust formatting, Clippy with warnings denied, tests, and release inference. The
comparison script enforces a maximum absolute error of `1e-5`. Repository CI validates documentation,
headers, workflow SHA pins, checksums, shell syntax, and the code index.

## Manual testing

Before a release on a target edge device, repeat the benchmark across cold starts and sustained load,
measure process RSS and physical power, test the expected camera preprocessing path, and inspect a
failure gallery using representative field data. Those device- and domain-specific steps are not
faked in hosted CI.

## Current gaps

The included suite has no camera driver, GPU execution provider, ARM cross-compilation, real image
distribution, fuzzing campaign, or physical energy meter. These are deliberate scope boundaries and
must be completed before production reuse.

