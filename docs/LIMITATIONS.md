# Limitations

- The generated 8x8 line data is intentionally simple and has no real-camera distribution validity.
- The model is a compact dense network, not a convolutional architecture for production vision.
- C++ and Rust execute the same weights through the narrow portable format, not a general ONNX parser.
- Dynamic weight quantization may be slower for a tiny model or on CPUs without suitable instructions.
- Calibration uses a small balanced validation set and does not prove calibrated field behavior.
- Corruptions cover only noise, center occlusion, and dimming.
- Python `tracemalloc` omits some native/process allocation; no native peak-RSS profiler is included.
- The energy figure is a stated wattage proxy, not a measurement.
- CI is CPU-only and does not validate GPU, NPU, ARM, mobile, or camera preprocessing.
- Model/dataset checksums provide integrity evidence but no cryptographic signer identity.

