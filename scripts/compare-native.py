"""File: Compares C++ and Rust prediction CSVs against the checked Python reference.

Functions: compare, build_parser, and main validate identifiers, shapes, probabilities, and
tolerance.
Variables: candidate paths and errors are invocation-local; exact lines are in docs/CODE_INDEX.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgevision.artifacts import read_predictions
from edgevision.runtime import maximum_absolute_error

MAXIMUM_ALLOWED_TOLERANCE = 0.01


def compare(reference_path: Path, candidate_path: Path, tolerance: float) -> dict[str, object]:
    """Compare one candidate to a reference and raise when IDs or values exceed the contract."""

    reference_ids, reference = read_predictions(reference_path)
    candidate_ids, candidate = read_predictions(candidate_path)
    if reference_ids != candidate_ids:
        raise ValueError(f"candidate identifiers do not match: {candidate_path}")
    error = maximum_absolute_error(reference, candidate)
    if error > tolerance:
        raise RuntimeError(
            f"candidate exceeds tolerance {tolerance:.2e}: {candidate_path} ({error:.2e})"
        )
    return {"candidate": candidate_path.as_posix(), "max_absolute_error": error, "status": "passed"}


def build_parser() -> argparse.ArgumentParser:
    """Create the explicit reference, candidate, tolerance, and optional report arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path, action="append")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every candidate comparison and optionally persist JSON evidence."""

    arguments = build_parser().parse_args(argv)
    if not 0.0 < arguments.tolerance <= MAXIMUM_ALLOWED_TOLERANCE:
        raise ValueError("tolerance must be within (0, 0.01]")
    report = {
        "reference": arguments.reference.as_posix(),
        "tolerance": arguments.tolerance,
        "comparisons": [
            compare(arguments.reference, candidate, arguments.tolerance)
            for candidate in arguments.candidate
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
