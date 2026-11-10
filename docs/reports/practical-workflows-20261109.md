# Deployment quality-envelope validation â€” 2026-11-09

Author: Jason Stys. Baseline `1d283dda172960a9a66b898677638d951b9af91d`.

Added a read-only engineer review dossier comparing portable NumPy, float ONNX and dynamic-int8 ONNX on the same 48 held-out synthetic samples under clean/Gaussian/occlusion/dim conditions. Each class/condition retains its confusion evidence and explicit accuracy, worst-recall, ECE and probability-agreement gates. Configurable laboratory limits never alter release thresholds or automatically select/promote a model. Fresh same-run benchmark measurements, bytes, dataset/model/manifest fingerprints and failed gate names support tradeoff review.

Local Windows / Python 3.14.6 validation:

- Ruff format/check and strict mypy passed; **36/36 pytest tests** passed, **97.59% branch-inclusive overall coverage**, new review module 100%. Hand-calculated four-row fixture demonstrates 75% accuracy with zero recall for one class, inclusive thresholds and all four negative gates. Real three-runtime/four-condition integration, invalid limits/corruption, no-overwrite and manifest-tamper cases passed.
- Actual `edgevision validate` passed existing artifact/dataset/ONNX/reference checks; actual create-new `review` exported all 12 condition/runtime evaluations with 48 samples each and all three default envelopes eligible. Trained artifacts/manifest remain unchanged.
- Fresh MSVC 19.52 C++ build and 1/1 CTest passed; Rust format, strict Clippy and 3/3 tests passed. Both freshly executed native runtimes matched 18 Python reference vectors with maximum absolute error `1.1920928955078125e-07`, below unchanged `1e-5` tolerance.
- Full development lock audit initially found 15 advisories in two transitive tools. Exact `filelock 4.2.1` and `msgpack 1.2.3` pins fixed them, following primary [filelock advisory](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8) and [MessagePack advisory](https://github.com/advisories/GHSA-j586-36cw-2gc2). Reinstalled exact fixes, `pip check`, repeated full tests and strict development/runtime lock audits passed with **zero known vulnerabilities**. Existing locks pin versions without hashes; this change preserves that limitation, rather than claiming hashed reproducibility.
- Source index and repository hygiene passed. CI adds actual review generation/evidence upload while retaining existing native, sanitizer, container, release and coverage gates.

The sandbox denied Windows pytest-owned temporary cleanup; a fresh scoped test directory outside that restriction passed. Containers and hosted workflows were not executed in this dated local session; local Docker was left unchanged. CPU timing is host/session-specific, tracemalloc excludes native memory, and assumed-wattage energy remains a proxy. Small synthetic 8Ã—8 shapes do not qualify camera, people or safety deployments. Native validation is separate from the review's ONNX/portable gate; no operational deployment occurs.
