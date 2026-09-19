# Benchmarking Method

## Questions

The benchmark asks whether export or quantization changes quality and how the portable NumPy, ONNX
float32, and ONNX weight-quantized paths behave on one CPU under the same input batch.

## Procedure

Each predictor receives the same test subset. Five untimed warmups precede 30 measured iterations.
Wall time uses a monotonic nanosecond clock. Reported latency divides batch elapsed time by image count;
p50, p95, and p99 are calculated across iterations. Throughput uses total measured images divided by
total measured seconds. Python allocation peak comes from `tracemalloc` and therefore does not include
all native allocator or process memory.

## Energy proxy

The proxy multiplies elapsed seconds per image by an assumed 15-watt envelope and scales to 1,000
images. It is explicitly not a physical power reading. Use RAPL, an external meter, or a platform power
API on dedicated hardware before making energy claims.

## Interpretation rules

- Compare runtimes only within the same report and host.
- Treat shared-runner changes as noise unless repeated on controlled hardware.
- Examine accuracy and calibration alongside speed; quantization is not automatically beneficial.
- Re-run at least five process-level trials for a release decision on real target hardware.
- Do not compare the generated evidence to unrelated models, batch sizes, or processor families.

## Reproduction

Run `scripts/benchmark.sh` or `python -m edgevision benchmark --root . --report <path>`. Machine-local
output belongs under `.runtime/`; the checked report is a transparent example from the release build.

