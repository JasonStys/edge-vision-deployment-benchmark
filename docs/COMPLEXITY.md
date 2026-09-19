# Complexity and Resource Analysis

Let `N` be batch size, `I=64`, `H1=24`, `H2=12`, and `C=3`.

## Training

Each full-batch epoch performs three dense forward and three dense backward multiplications. Time is
`O(epochs × N × (I×H1 + H1×H2 + H2×C))`. Parameter memory is
`O(I×H1 + H1×H2 + H2×C)` and activation memory is `O(N×(H1+H2+C))`. The included 260 epochs and 144
training examples keep this deliberately small.

## Inference

One image requires `I×H1 + H1×H2 + H2×C = 1,860` multiply-accumulate pairs plus biases, ReLUs, and
softmax. Time is linear in batch size: `O(N × 1,860)`. The native paths use fixed activation arrays of
24, 12, and 3 floats; model parameter storage is constant for this version.

## Data and metrics

Generation, corruption, hashing, and accuracy are `O(N×I)`. Confusion matrices are `O(N+C²)` and the
10-bin ECE pass is `O(10N)`. The centroid baseline is `O(N×C×I)`. Manifests hash files in streaming
64 KiB chunks, so memory is `O(1)` relative to file size.

## Parser safety

Limits are checked before large allocation: 1 MB portable model, 2 MB ONNX artifact, 5 MB vector CSV,
4,096 vector rows, and 100,000 inference rows. Fixed tensor counts prevent a crafted header from
requesting arbitrary memory.

