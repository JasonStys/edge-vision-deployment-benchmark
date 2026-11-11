"""File: desktop/acceptance.py
Purpose: Exercise actual native controls/five engines with marked synthetic data and event oracles.
Symbols: exercise owns its finite sequence; require remains active in optimized Python.
No normal user profile, desktop input injection, network or alternate demo executable is involved.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from threading import Event

from desktop import application
from desktop.application import Application
from desktop.capture import capture
from desktop.project import parse_vectors
from desktop.runtime_libraries import loaded_libraries
from desktop.storage import VERSION, Home, new_json, read_regular, strict_json

EXPECTED_ENGINES = 5
EXPECTED_SAMPLES = 6
EXPECTED_REVIEW_ROWS = 12
EXPECTED_CLASSES = 3
ORACLE_ERROR = 1e-5


def require(condition: bool, message: str) -> None:
    """Keep independent native oracles active under optimized interpreters."""
    if not condition:
        raise ValueError(message)


def exercise(window: Application, root: Path, identity: dict, *, reopen: bool) -> None:  # noqa: PLR0915 - one explicit ordered native acceptance with shared synthetic context
    """Use real widget bindings and retain shutdown until a real local write completes."""
    output = root / ("reopen" if reopen else "first")
    output.mkdir()
    deadline = time.monotonic() + 90
    receipt = {
        "schema": 1,
        "version": VERSION,
        "identity": identity,
        "reopen": reopen,
        "passed": False,
        "steps": [],
        "normalProfileOpened": False,
    }
    window.acceptance_passed = False
    chosen = output / "captured.json"
    imported = root / "first/vectors.csv"
    answer = False
    application.filedialog.asksaveasfilename = lambda **_kwargs: str(chosen)
    application.filedialog.askopenfilename = lambda **_kwargs: str(imported)
    application.messagebox.askyesno = lambda *_args, **_kwargs: answer

    def record(name: str) -> None:
        """Record actual completed steps, not planned coverage or private profile contents."""
        receipt["steps"].append(name)

    def check_inference() -> None:
        """Independently check counts, probability domain and original agreement bound."""
        result = window.last_evidence["result"]
        require(len(result["ids"]) == EXPECTED_SAMPLES, "Actual custom input count differs")
        require(len(result["runtimes"]) == EXPECTED_ENGINES, "An actual inference engine is absent")
        for runtime in result["runtimes"].values():
            require(
                runtime["reference_max_abs"] <= ORACLE_ERROR, "Unchanged engine agreement failed"
            )
            for row in runtime["probabilities"]:
                require(
                    len(row) == EXPECTED_CLASSES and all(0 <= value <= 1 for value in row),
                    "Actual probabilities exceed the domain",
                )
                require(
                    abs(sum(row) - 1) < ORACLE_ERROR, "Actual probability row does not normalize"
                )
        require(
            len(window.predictions.get_children()) == EXPECTED_SAMPLES * EXPECTED_ENGINES,
            "Native prediction rows differ from captured results",
        )
        receipt["loadedAppLocalLibraries"] = loaded_libraries()
        require(os.environ.get("ORT_DISABLE_TELEMETRY") == "1", "Vendor opt-out is not configured")
        receipt["vendorTelemetryOptOutConfigured"] = True

    def first_journey():  # noqa: PLR0915 - explicit sequential real controls and independent oracles
        """Prepare/edit/import/infer/review/rebuild/export through normal application controls."""
        nonlocal chosen, imported, answer
        window.prepare.invoke()
        text = window.vectors.get("1.0", "end-1c")
        require(len(parse_vectors(text)[0]) == EXPECTED_SAMPLES, "Prepared rows differ")
        chosen = output / "vectors.csv"
        window.save_vectors()
        require(read_regular(chosen, 1024 * 1024).decode() == text, "Edited CSV save differs")
        imported = output / "bad.csv"
        imported.write_text("unrecognized,columns\n", encoding="utf-8")
        window.import_vectors()
        require("Import refused" in window.status.get(), "Invalid CSV import was accepted")
        require(window.vectors.get("1.0", "end-1c") == text, "Invalid import replaced the editor")
        imported = output / "vectors.csv"
        window.import_vectors()
        require(window.vectors.get("1.0", "end-1c") == text, "Valid file import differs")
        window.native.invoke()
        yield
        check_inference()
        capture(window, output / "native-five-engines.png")
        record(
            "editable CSV save/import, invalid input preservation and five real inference engines"
        )
        original = window.workspace.state["active"]
        for gate, value in zip(window.gates, ("1", "1", "0", "0"), strict=True):
            gate.set(value)
        window.review_button.invoke()
        yield
        require(
            len(window.review_rows.get_children()) == EXPECTED_REVIEW_ROWS,
            "Model-by-condition review is incomplete",
        )
        metrics = [
            result["metrics"]["ece_10_bin"]
            for candidate in window.last_evidence["result"]["candidates"].values()
            for result in candidate["conditions"].values()
        ]
        for row, actual in zip(window.review_rows.get_children(), metrics, strict=True):
            shown = float(window.review_rows.item(row)["values"][4])
            require(
                (shown > 0 if actual > 0 else shown == 0)
                and abs(shown - actual) <= max(1e-14, abs(actual) * 0.001),
                "Native calibration display hid or misstated the actual metric",
            )
        require(
            all(
                not item["eligible"]
                for item in window.last_evidence["result"]["candidates"].values()
            ),
            "Strict review hid a genuine quality failure",
        )
        window.tabs.select(1)
        capture(window, output / "native-review-gates.png")
        chosen = output / "review.json"
        window.export.invoke()
        preserved = read_regular(chosen, 8 * 1024 * 1024)
        require(strict_json(preserved) == window.last_evidence, "Captured JSON export differs")
        window.export.invoke()
        require(read_regular(chosen, 8 * 1024 * 1024) == preserved, "Export overwrote evidence")
        for gate in window.gates:
            gate.set("nan")
        window.review_button.invoke()
        require(
            not window.busy and "Review refused" in window.status.get(), "NaN gate started work"
        )
        record(
            "all twelve review rows, strict failures, finite gate refusal and exact new-file export"
        )
        window.rebuild()
        require(
            not window.busy and window.workspace.state["active"] == original,
            "Cancelled model rebuild changed state",
        )
        answer = True
        window.rebuild()
        yield
        require(
            window.workspace.state["active"] != original,
            "Actual rebuild did not publish a generation",
        )
        require((window.workspace.candidates / original).is_dir(), "Rebuild removed old generation")
        window.tabs.select(0)
        window.native.invoke()
        yield
        check_inference()
        record(
            "cancelled rebuild retains data; genuine original training exports a new "
            "verified generation"
        )
        receipt["retained"] = {
            "generation": window.workspace.state["active"],
            "lastReportSha256": hashlib.sha256(
                read_regular(
                    window.workspace.reports / window.workspace.state["last_report"],
                    8 * 1024 * 1024,
                )
            ).hexdigest(),
            "csvSha256": hashlib.sha256(
                read_regular(output / "vectors.csv", 1024 * 1024)
            ).hexdigest(),
        }

    def reopen_journey():
        """Restore exact captured generation and report, then use the imported vectors again."""
        prior = strict_json(read_regular(root / "first/acceptance.json", 8192))
        retained = prior["retained"]
        require(
            window.workspace.state["active"] == retained["generation"],
            "Native restart lost generation",
        )
        require(
            hashlib.sha256(
                read_regular(
                    window.workspace.reports / window.workspace.state["last_report"],
                    8 * 1024 * 1024,
                )
            ).hexdigest()
            == retained["lastReportSha256"],
            "Native restart changed recorded results",
        )
        require(
            hashlib.sha256(read_regular(imported, 1024 * 1024)).hexdigest()
            == retained["csvSha256"],
            "Native restart changed saved authoring data",
        )
        window.import_vectors()
        window.native.invoke()
        yield
        check_inference()
        receipt["retained"] = retained
        record(
            "same generation, captured report and CSV survive restart; all five engines "
            "execute again"
        )

    sequence = reopen_journey() if reopen else first_journey()

    def fail(error: Exception) -> None:
        """Retain the actual failure and request ordinary completion-aware shutdown."""
        receipt["error"] = f"{type(error).__name__}: {error}"[:500]
        new_json(output / "acceptance.json", receipt)
        window.close()

    def pending_close() -> None:
        """Prove a pending actual write retains both native window and exclusive owner handle."""
        entered, release = Event(), Event()

        def write_after_release():
            entered.set()
            if not release.wait(10):
                raise ValueError("Synthetic pending-write handshake expired")
            new_json(output / "completed-write.json", {"writtenAfterRelease": True})
            return {"pendingWriteCompleted": True}

        window.run(write_after_release, lambda _value: None)

        def confirm() -> None:
            try:
                require(time.monotonic() < deadline, "Native acceptance deadline exceeded")
                if not entered.is_set():
                    window.root.after(10, confirm)
                    return
                window.quit.invoke()
                window.close()
                require(
                    window.busy and not window.closed and bool(window.root.winfo_exists()),
                    "Quit destroyed the window while real file work was pending",
                )
                try:
                    other = Home(window.workspace.path)
                except OSError:
                    pass
                else:
                    other.close()
                    raise ValueError(
                        "Quit released the real profile lock before file work completed"
                    )
                record(
                    "repeated Quit retains actual pending write/window/profile until "
                    "explicit release"
                )
                receipt["passed"] = window.acceptance_passed = True
                new_json(output / "acceptance.json", receipt)
            except Exception as error:
                fail(error)
            finally:
                release.set()

        window.root.after(0, confirm)

    def advance() -> None:
        """Advance only after real retained worker completion under one unchanged finite budget."""
        try:
            require(time.monotonic() < deadline, "Native acceptance deadline exceeded")
            if window.busy:
                window.root.after(20, advance)
                return
            next(sequence)
        except StopIteration:
            pending_close()
            return
        except Exception as error:
            fail(error)
            return
        window.root.after(0, advance)

    window.root.after_idle(advance)
