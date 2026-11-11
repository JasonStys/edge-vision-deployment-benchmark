"""File: desktop/launcher.py
Purpose: Open the genuine offline scientific workbench with external durable profile ownership.
Symbols: main retains window/work; arguments are admitted before any normal OS-data lookup.
Only explicit marked acceptance mode drives synthetic data; normal mode uses real editable controls.
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from desktop.application import Application
from desktop.project import Workspace
from desktop.storage import PRODUCT, VERSION, default_profile, new_json, read_regular, strict_json

MARKER = b"edgevision-owned-native-acceptance-v1\n"


def resource_root() -> Path:
    """Resolve authored seed/native resources independently of a caller's working directory."""
    return (
        Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    )


def package_identity() -> dict:
    """Bind frozen product metadata to the actually loaded patched interpreter."""
    if not getattr(sys, "frozen", False):
        return {"product": PRODUCT, "version": VERSION, "mode": "development"}
    identity = strict_json(read_regular(resource_root() / "SOURCE.json", 8192))
    if (
        not isinstance(identity, dict)
        or set(identity)
        != {"schema", "product", "version", "source", "sourceDirty", "python", "tcl", "tk"}
        or type(identity["schema"]) is not int
        or identity["schema"] != 1
        or identity["product"] != PRODUCT
        or identity["version"] != VERSION
        or identity["python"] != platform.python_version()
        or type(identity["sourceDirty"]) is not bool
        or not isinstance(identity["source"], str)
        or re.fullmatch(r"[a-f0-9]{40}", identity["source"]) is None
        or not all(isinstance(identity[key], str) for key in ("tcl", "tk"))
    ):
        raise ValueError("Actual native product/runtime identity differs")
    return identity


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Refuse unmarked or ambiguous test modes before resolving any normal OS profile."""
    parser = argparse.ArgumentParser(description="Edge Vision Workbench by Jason Stys")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--runtime-info-output", type=Path)
    parser.add_argument("--native-acceptance-root", type=Path)
    parser.add_argument("--acceptance-reopen", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.acceptance_reopen and not arguments.native_acceptance_root:
        parser.error("Reopen requires a marked synthetic acceptance root")
    if arguments.native_acceptance_root:
        root = arguments.native_acceptance_root
        if arguments.profile or arguments.runtime_info_output:
            parser.error("Acceptance cannot select a different profile or runtime-only mode")
        if not root.is_absolute() or read_regular(root / "acceptance.marker", 128) != MARKER:
            raise ValueError("Choose the exact marked synthetic acceptance directory")
        if (root / "profile").exists() != arguments.acceptance_reopen:
            raise ValueError("Use a fresh acceptance profile or explicitly reopen it")
    return arguments


def checked_runtime(root: tk.Tk, identity: dict) -> dict[str, str]:
    """Inspect actual Tcl/Tk and refuse a frozen runtime that differs from its build record."""
    runtime = {
        "python": platform.python_version(),
        "tcl": str(root.tk.call("info", "patchlevel")),
        "tk": str(root.tk.call("package", "require", "Tk")),
    }
    if getattr(sys, "frozen", False):
        if any(identity[key] != value for key, value in runtime.items()):
            raise ValueError("Actual loaded Tcl/Tk differs from the built product record")
        root.iconbitmap(str(resource_root() / "application.ico"))
    return runtime


def main(argv: list[str] | None = None) -> int:
    """Open genuine controls and retain actual computation until normal profile/window close."""
    arguments = parse_arguments(argv)
    root = tk.Tk()
    root.withdraw()
    workspace = application = None
    try:
        identity = package_identity()
        runtime = checked_runtime(root, identity)
        if arguments.runtime_info_output:
            new_json(arguments.runtime_info_output, {"package": identity, "runtime": runtime})
            return 0
        profile = (
            arguments.native_acceptance_root / "profile"
            if arguments.native_acceptance_root
            else arguments.profile or default_profile()
        )
        resources = resource_root()
        seed = resources / "seed" if getattr(sys, "frozen", False) else resources
        native = (
            resources / "kernels"
            if getattr(sys, "frozen", False)
            else (resources / ".runtime/native-kernels-001")
        )
        workspace = Workspace(profile, seed, native if native.is_dir() else None)
        application = Application(root, workspace)
        root.deiconify()
        if arguments.native_acceptance_root:
            from desktop.acceptance import exercise  # noqa: PLC0415 - isolated opt-in test driver

            exercise(
                application,
                arguments.native_acceptance_root,
                identity,
                reopen=arguments.acceptance_reopen,
            )
        while not application.closed:
            try:
                root.mainloop()
            except KeyboardInterrupt:
                application.close()
        return 0 if not arguments.native_acceptance_root or application.acceptance_passed else 2
    except (OSError, ValueError, RuntimeError) as error:
        if arguments.native_acceptance_root:
            new_json(
                arguments.native_acceptance_root / "startup-failure.json",
                {"passed": False, "errorType": type(error).__name__, "message": str(error)[:300]},
            )
        else:
            messagebox.showerror("Edge Vision could not open", str(error), parent=root)
        return 2
    finally:
        if application is not None and application.worker is not None:
            application.worker.join()
        if workspace is not None:
            workspace.close()
        if application is None or not application.closed:
            root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
