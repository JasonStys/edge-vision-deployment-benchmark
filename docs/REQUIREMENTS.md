# Requirements

## Functional requirements

1. Generate a balanced three-class 8x8 image dataset without network access or opaque source data.
2. Produce deterministic train, validation, and test splits with content integrity evidence.
3. Train a compact neural model and calibrate it only on validation data.
4. Export the exact trained graph to ONNX and produce a quantized deployment candidate.
5. Run equivalent inference through portable Python, ONNX Runtime, C++, and Rust paths.
6. Report accuracy, per-class precision/recall/F1, confusion matrix, NLL, ECE, and corruptions.
7. Report warm latency percentiles, throughput, allocation peak, model size, and an energy proxy.
8. Generate small artifacts, reference vectors, a model card, and SHA-256 manifest from source.
9. Validate all artifacts and runtime agreement through local commands and GitHub Actions.

## Non-functional requirements

- Clean float32 accuracy must be at least 95%.
- Quantized accuracy loss must not exceed two percentage points.
- Portable-to-ONNX and native-to-reference maximum probability error must not exceed `1e-5`.
- Input batches, model bytes, dataset samples, CSV rows, tensor dimensions, and temperatures are bounded.
- CI requires warning-free lint/static analysis, tests, coverage, dependency audit, and CodeQL.
- Build inputs are pinned or locked; GitHub Actions use full commit SHAs.
- CPU-only CI completes without a remote dataset, GPU, external service, or large binary download.

## Out of scope

- Production camera ingestion, user interfaces, object detection, identity analysis, or safety control.
- Claims that one shared-runner benchmark generalizes to other processors or energy profiles.
- Accepting arbitrary neural-network architectures in native parsers.
- Treating validation of structure as proof that an untrusted ONNX graph is safe to execute.

## Acceptance criteria

The deployment is accepted only when the release gates in the README pass, the generated artifact
manifest verifies, documentation/code indexes are current, and both CI and CodeQL conclude successfully.

