# Deployment Checklist

## Pre-deployment

- [ ] CI and CodeQL pass on the exact commit.
- [ ] Artifact manifest, dataset digest, ONNX checker, and reference agreement pass.
- [ ] Clean accuracy is at least 95%; quantization loss is at most two percentage points.
- [ ] Robustness, calibration, per-class results, and limitations were reviewed for the intended use.
- [ ] Benchmark was repeated on the target device with the production batch and thread settings.
- [ ] Physical memory and energy were measured if requirements depend on them.
- [ ] Artifacts were signed and the signer/trust root was approved for production.
- [ ] Rollback artifact and prior validation evidence are available.

## Deployment

- [ ] Verify signature and SHA-256 before loading.
- [ ] Confirm class order, input layout, normalization, temperature, and model format version.
- [ ] Run the committed reference vectors on the target binary.
- [ ] Apply process memory/CPU limits and isolate untrusted input handling.
- [ ] Record artifact identity, binary version, device, and deployment timestamp.

## Post-deployment

- [ ] Execute a smoke inference and verify finite, normalized output.
- [ ] Compare latency/error telemetry to target-device baselines.
- [ ] Monitor input drift, low confidence, class distribution, and failure samples.
- [ ] Retain deployment and verification evidence with the artifact identity.

## Rollback

- [ ] Stop or drain new inference work.
- [ ] Restore the last signed artifact and matching runtime configuration.
- [ ] Re-run its reference vectors and smoke inference.
- [ ] Confirm telemetry recovery and preserve failed artifacts/logs for analysis.

