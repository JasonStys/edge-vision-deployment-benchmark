# Operations Runbook

## Validate a checkout

Install the locked Python dependencies and editable package, then run `python -m edgevision validate
--root .`. This checks every artifact hash, dataset identity, portable tensor, ONNX graph, and reference
probability. Run `scripts/verify.sh` for the complete Linux/macOS quality suite.

## Rebuild artifacts

Run `python -m edgevision build --root .` from a clean branch. Review the dataset digest, clean and
corruption metrics, calibration error, quantized accuracy delta, ONNX agreement, model sizes, and
benchmark methodology. Regenerate `docs/CODE_INDEX.md`, run all quality gates, then commit source and
evidence together.

## Diagnose failures

- Integrity failure: inspect `git diff`, verify the generator change is intentional, and rebuild.
- Accuracy failure: compare data digest, seed, convergence losses, and per-class confusion matrix.
- ONNX mismatch: inspect tensor orientation, bias broadcasting, temperature, opset, and class order.
- Native mismatch: compare one vector layer-by-layer and verify row-major weight indexing.
- Performance change: confirm host, batch, thread settings, warmups, background load, and model size.
- Audit failure: identify the advisory, update the smallest dependency set, and rerun quality metrics.

## Rollback

Artifacts are immutable within a commit. Roll back by deploying the last commit whose CI, CodeQL,
manifest, and model card passed. Never copy an old model over a new manifest. Re-run validation after
checkout and retain the failed report for root-cause analysis.

## Artifact retention

Git stores only the small release artifacts required to reproduce and compare runtimes. CI retains
ephemeral coverage, benchmark, and native-agreement reports for 14 days. Real deployments should store
signed artifacts and evaluation evidence in an access-controlled registry.

