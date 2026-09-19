# Generated Model Card

## Intended use

This small classifier distinguishes generated vertical, horizontal, and diagonal 8x8 patterns. It is
an engineering benchmark for portable deployment, not a production perception system.

## Measured results

- Float32 test accuracy: 1.0000
- Dynamically weight-quantized ONNX test accuracy: 1.0000
- Float32 10-bin expected calibration error: 0.000001
- Model classes: vertical, horizontal, diagonal
- Benchmark runtimes: methodology, onnx_float32, onnx_int8_dynamic_weights, portable_numpy

## Data and evaluation

Data is generated from documented geometric rules with deterministic splits. Evaluation covers clean
data, Gaussian noise, center occlusion, dimming, calibration, native-runtime agreement, and warm CPU
latency. The generated metrics JSON is the source of truth for detailed results.

## Limitations

The data is synthetic, grayscale, low resolution, balanced, and intentionally simple. Results do not
generalize to camera imagery, people, safety decisions, or uncontrolled environments. The energy
number is an elapsed-time proxy based on an assumed wattage, not a physical power measurement.
