#!/usr/bin/env bash
# File: Runs the local Python, repository, C++, Rust, and cross-runtime verification sequence.
# Functions: the script is intentionally linear; variables root and runtime isolate generated output.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="${root}/.runtime/verify"
mkdir -p "${runtime}"
cd "${root}"

python -m ruff format --check src tests scripts/compare-native.py
python -m ruff check src tests scripts/compare-native.py
python -m mypy
python -m pytest
python -m edgevision validate --root "${root}"
node scripts/validate-repository.mjs
node scripts/generate-code-index.mjs --check

cmake -S native/cpp -B "${runtime}/cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build "${runtime}/cpp" --config Release
ctest --test-dir "${runtime}/cpp" -C Release --output-on-failure
cargo fmt --manifest-path native/rust/Cargo.toml --all -- --check
cargo clippy --manifest-path native/rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path native/rust/Cargo.toml

"${runtime}/cpp/edgevision_cpp"
cargo run --quiet --release --manifest-path native/rust/Cargo.toml -- artifacts/model/compact-mlp.evm artifacts/test-vectors.csv "${runtime}/rust.csv"
python scripts/compare-native.py --reference artifacts/reference-predictions.csv --candidate "${runtime}/cpp.csv" --candidate "${runtime}/rust.csv"
