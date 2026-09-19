# Dataset Card: Shapes v1

## Summary

Shapes v1 is a generated, balanced dataset of 8x8 single-channel images representing vertical,
horizontal, and diagonal line patterns. It exists to exercise deployment mechanics without network
downloads, personal data, licensing uncertainty, or large artifacts.

## Generation and provenance

`edgevision.data.generate_dataset` uses NumPy's seeded generator. Each example varies line offset and
thickness, then receives clipped Gaussian sensor noise. The default seed is 41. No external image,
person, device, or user data is incorporated. `dataset_digest` hashes split names, dtypes, shapes, and
contiguous array bytes; the adjacent manifest records that identity.

## Splits

Generation is balanced by construction. Indexes are shuffled independently per class into 60% train,
20% validation, and 20% test sets, then each split is shuffled. Validation is used only for temperature
selection. The test split is reserved for reported quality and corruption metrics.

## Transformations and corruptions

Inputs are already normalized float32 values in `[0, 1]`. Evaluation corruptions are deterministic
Gaussian noise, a central 2x2 occlusion, and 45% brightness. They are stress tests, not an exhaustive
model of camera or environmental variation.

## Appropriate use

Use the dataset for CI, model-format, numerical-agreement, calibration, and benchmark experiments.
Do not use its results to claim real-world perception quality, fairness, safety, robustness, or camera
performance.

