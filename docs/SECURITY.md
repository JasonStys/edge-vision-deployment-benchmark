# Security Model

## Assets and trust boundaries

Assets are model integrity, dataset provenance, evaluation evidence, and deterministic source. Inputs
cross trust boundaries when the pipeline loads NPZ, portable model, ONNX, vector CSV, prediction CSV,
or manifest files. GitHub Actions and package registries are supply-chain boundaries.

## Controls

- Portable models have a 1 MB limit, fixed version/dimensions/classes, counted tensors, finite values,
  bounded temperature, exact end marker, and no trailing tokens.
- Vector and prediction CSVs have size/row/column limits, exact headers, normalized finite features,
  known labels, and stable identifiers.
- NPZ loading disables pickle and requires exactly the six known arrays.
- ONNX files have a 2 MB limit, checker validation, strict shape inference, CPU-only provider, and
  checksums. Only repository-generated models are executed.
- Artifact manifest paths are resolved beneath the repository root and checked by size and SHA-256.
- The C++ adapter uses fixed repository-relative model, vector, and output paths; untrusted CLI input
  cannot select a filesystem location.
- Native code uses bounds-aware containers, warnings as errors, sanitizers, Clippy, and CodeQL.
- Python dependencies and Rust resolution are locked; actions are pinned to full commit SHAs.
- Workflows default to read-only contents and grant only the minimum security-event permission needed.

## Threats not solved

Structural validation cannot make an arbitrary third-party ONNX graph safe. SHA-256 detects changes
but is not a signature. Hosted runner results do not attest a physical device. Dependency scanning
does not prove absence of unknown vulnerabilities. Production deployment should isolate inference,
sign artifacts, verify signatures before load, apply OS resource limits, and monitor runtime behavior.
