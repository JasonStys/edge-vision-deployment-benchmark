"""File: tests/test_desktop_widgets.py
Purpose: Exercise actual Windows Tk widgets, error handling and retained-work Quit with events.
Symbols: only pytest-owned profiles and withdrawn windows are used; no generic OS input injection.
The native Windows workflow additionally exercises visible compiled windows and relocated packages.
"""

import sys
import time
from pathlib import Path
from threading import Event

import pytest

from desktop.project import Workspace, parse_vectors
from desktop.storage import Home

if sys.platform == "win32":
    import tkinter as tk

    from desktop.application import Application

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Actual Windows native widget scope"
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def window(tmp_path):
    """Open only a disposable profile and retain/close actual worker resources on teardown."""
    workspace = Workspace(tmp_path / "profile", ROOT)
    root = tk.Tk()
    root.withdraw()
    app = Application(root, workspace)
    yield app
    if app.worker is not None:
        app.worker.join(timeout=5)
    if not app.closed:
        workspace.close()
        root.destroy()


def pump(window, condition):
    """Wait on actual Tk/worker completion with a finite test bound, never a blind delay."""
    deadline = time.monotonic() + 5
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail("The actual native widget operation did not complete")
        window.root.update()


def test_drawing_and_prepared_inputs_are_editable_without_implicitly_saving(window):
    """Actual drawing/import controls produce valid vectors but never invent saved user inputs."""
    window.pixels[0][1] = 1.0
    window.use_drawing()
    identifiers, labels, features = parse_vectors(window.vectors.get("1.0", "end-1c"))
    assert identifiers == ["drawn-pattern"] and labels.tolist() == [0]
    assert features[0, 1] == 1 and features.sum() == 1
    window.clear_drawing()
    assert not any(value for row in window.pixels for value in row)
    window.prepare.invoke()
    assert len(parse_vectors(window.vectors.get("1.0", "end-1c"))[0]) == 6
    assert window.workspace.last_report() is None
    assert window.vectors.cget("xscrollcommand")
    assert window.predictions.cget("yscrollcommand")


def test_invalid_review_does_not_start_work_and_prior_capture_remains(window):
    """NaN and malformed thresholds remain visible rejections before scientific work starts."""
    original = {"retained": "independent captured evidence"}
    window.last_evidence = original
    for value in ("nan", "inf", "wrong", "1.1", "-0.1"):
        window.gates[0].set(value)
        window.review_button.invoke()
        assert not window.busy and "Review refused" in window.status.get()
        assert window.last_evidence is original


def test_result_renderer_error_retains_evidence_and_does_not_break_the_close_pump(window):
    """A real completed worker with a failing renderer remains exportable and closable."""
    payload = {"independent": "actual worker result"}

    def fail(_):
        raise ValueError("Synthetic display-only failure")

    window.run(lambda: payload, fail)
    pump(window, lambda: not window.busy)
    assert window.last_evidence == payload and "display failed" in window.status.get()
    window.close()
    pump(window, lambda: window.closed)


def test_repeated_quit_retains_pending_real_worker_and_owner_until_release(window):
    """No forced close bypasses actual retained work or the exclusive profile owner handle."""
    entered, release = Event(), Event()

    def held():
        entered.set()
        if not release.wait(5):
            raise ValueError("Test handshake timed out")
        return {"complete": True}

    window.run(held, lambda _: None)
    assert entered.wait(1)
    try:
        window.quit.invoke()
        window.close()
        assert window.busy and not window.closed and window.root.winfo_exists()
        with pytest.raises(OSError):
            Home(window.workspace.path)
    finally:
        release.set()
    pump(window, lambda: window.closed)
    owner = Home(window.workspace.path)
    owner.close()
