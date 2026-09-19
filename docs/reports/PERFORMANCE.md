# Performance Report

The checked machine-readable benchmark is generated at `docs/reports/generated/benchmark.json`. It
records the batch, warmup count, iteration count, portable NumPy, ONNX float32, and dynamically
weight-quantized ONNX results. Latency percentiles and throughput are meaningful only for the recorded
host/run; the energy value is an assumed-wattage proxy and not a measurement.

No universal performance claim is made. Follow `docs/BENCHMARKING.md` and repeat on the actual target
device before selecting a runtime or quantization configuration.

## Local release characterization

The 2026-09-18 Windows run used a 48-image batch after five warmups and 30 measured iterations.

| Runtime | p50 ms/image | Artifact bytes |
|---|---:|---:|
| Portable NumPy | 0.001375 | 24,482 |
| ONNX float32 | 0.000934 | 8,536 |
| ONNX dynamic int8 weights | 0.001219 | 5,296 |

All three achieved 100% accuracy on the 48-example synthetic test split. Portable-to-ONNX maximum
absolute probability error was `1.1920929e-7`; float-to-quantized error was `1.0728836e-6`. These tiny
measurements are particularly sensitive to host and timer noise, so they document the run rather than
claiming a general speed ordering.
