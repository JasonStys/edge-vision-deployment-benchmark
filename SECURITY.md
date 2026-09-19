# Security Policy

## Supported version

The `main` branch is the supported development version. Dependency updates are proposed weekly and
must pass the complete quality and security suite before merge.

## Reporting a vulnerability

Use GitHub private vulnerability reporting when available. Do not open a public issue containing an
exploit, secret, or sensitive environment detail. Include the affected commit, reproduction steps,
impact, and any safe mitigation. Acknowledgment is targeted within seven days.

## Trust boundary

Model, dataset, vector, and prediction files are untrusted. Parsers enforce file-size, count, shape,
finite-number, range, schema, and trailing-data checks before allocation or inference. ONNX validation
does not make arbitrary third-party models safe; only locally generated, checksummed artifacts are in
scope. See [the detailed security model](docs/SECURITY.md).

