# Edge Vision Deployment Benchmark handbook

By Jason Stys.

Compare verified runtimes across shared held-out clean, noise, occlusion and dim conditions with per-class recall, calibration and explicit configurable failure gates.

Supports deployment tradeoffs using the same evidence rather than incompatible aggregate timing or accuracy claims.

## Read the documentation

| Edition | Audience and contents |
| --- | --- |
| [project-guide.docx](reports/project-guide.docx) | User and developer guide: purpose, current usage, design, boundaries and source references. |

The editable Markdown remains the maintenance source. The Word files are readable editions, and the workbook is an evidence index rather than a test runner. See [publication identities and review](reports/publications.json) and the [documentation refresh record](reports/documentation-refresh-2026-11-09.md).

## Current usage and help

[README quick start](../README.md) is the entry point. Use the project-specific guides below before changing inputs, operating modes or deployment assumptions.

### Usage and operations

- [DEPLOYMENT REVIEW](DEPLOYMENT_REVIEW.md)
- [OPERATIONS](OPERATIONS.md)

### Design and contracts

- [ARCHITECTURE](ARCHITECTURE.md)
- [COMPLEXITY](COMPLEXITY.md)
- [RESEARCH](RESEARCH.md)
- [adr/0001 portable model and onnx](adr/0001-portable-model-and-onnx.md)

### Security and validation

- [SECURITY](SECURITY.md)
- [TESTING](TESTING.md)

## Historical results and reproduction

The reviewed source is [`a506fe82222da8a8e7e566404dddd88b4f002fd3`](https://github.com/JasonStys/edge-vision-deployment-benchmark/tree/a506fe82222da8a8e7e566404dddd88b4f002fd3). The observed functional workflow baseline is `a506fe82222da8a8e7e566404dddd88b4f002fd3`; those observations are recorded at `2026-11-10T05:25:40.628667+00:00`. They do not certify an unrun later revision.

- [Detailed recorded validation](reports/practical-workflows-20261109.md)
- [Current Actions](https://github.com/JasonStys/edge-vision-deployment-benchmark/actions) records later integration checks separately.
- Preserve dated measurements and follow the existing testing guide to reproduce them; a test-source inventory alone does not establish execution.

## Practical boundaries

- Synthetic 8x8 shapes only; no camera, people or safety-critical deployment qualification.
- Timing is host-specific, memory tracing is Python-only and energy uses an assumed-wattage proxy rather than measured power.
- Quality gates use controlled held-out conditions; they are not certification of arbitrary model/runtime combinations.

## Maintaining these editions

Update the editable usage/design/test documents first. Rebuild and visually review every Word page and workbook tab before replacing an edition; verify the exact binary digests and evidence source in publications.json. Keep account data, logs with secrets, installed-app state and private test stores out of documentation. Source and release assets retain their existing security and licensing boundaries.
