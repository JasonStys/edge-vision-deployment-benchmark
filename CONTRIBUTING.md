# Contributing

## Workflow

1. Create a focused branch and explain the deployment behavior being changed.
2. Rebuild artifacts only when training, data, export, or report behavior changes.
3. Run `scripts/verify.sh` on a supported Unix-like environment or its documented commands on Windows.
4. Commit source, lockfiles, generated evidence, and code-index changes together.
5. Open a pull request that records accuracy, agreement, robustness, and performance impact.

## Code expectations

- Preserve file headers, focused function documentation, explicit bounds, and generated line indexes.
- Do not silently relax accuracy, agreement, security, coverage, or parser limits.
- Keep generated data and models under two megabytes each; document any format change with an ADR.
- Use deterministic seeds and held-out calibration/evaluation data.
- Treat ONNX and portable model files as untrusted input at runtime boundaries.
- Add tests for success, malformed input, and the failure mode that motivated the change.

## Commit style

Use an imperative summary under 72 characters, then explain why the change is necessary and which
evidence was regenerated. Never commit virtual environments, native build output, caches, or secrets.

