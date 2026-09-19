#!/usr/bin/env bash
# File: Repeats the documented CPU benchmark while keeping generated evidence outside tracked files.
# Functions: the script delegates measurement to edgevision.pipeline; variables root and report locate output.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
report="${1:-${root}/.runtime/benchmark.json}"
mkdir -p "$(dirname "${report}")"
python -m edgevision benchmark --root "${root}" --report "${report}"

