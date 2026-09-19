"""File: Tests the checked-in end-to-end artifact validation and CLI parser behavior.

Functions: test cases exercise the repository evidence as an integration surface. Variables: root
and validation reports are test-local; exact declaration lines are in docs/CODE_INDEX.md.
"""

from pathlib import Path

from edgevision.pipeline import build_artifacts, build_parser, main, validate_artifacts


def test_checked_in_artifacts_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate_artifacts(root)
    assert report["status"] == "passed"
    assert report["onnx_reference_max_abs"] <= 1e-5


def test_cli_parser_accepts_documented_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["validate"]).command == "validate"
    assert parser.parse_args(["benchmark", "--report", "report.json"]).report == Path("report.json")


def test_full_build_reconstructs_all_artifacts(tmp_path: Path) -> None:
    result = build_artifacts(tmp_path)
    assert isinstance(result["validation"], dict)
    assert result["validation"]["status"] == "passed"  # type: ignore[index]
    assert (tmp_path / "artifacts/model/compact-mlp.onnx").is_file()
    assert (tmp_path / "docs/reports/generated/MODEL_CARD.md").is_file()
    assert validate_artifacts(tmp_path)["status"] == "passed"


def test_cli_validate_and_benchmark_commands(tmp_path: Path, capsys: object) -> None:
    build_artifacts(tmp_path)
    assert main(["validate", "--root", str(tmp_path)]) == 0
    report = tmp_path / "benchmark-rerun.json"
    assert main(["benchmark", "--root", str(tmp_path), "--report", str(report)]) == 0
    assert report.is_file()
    assert capsys is not None
