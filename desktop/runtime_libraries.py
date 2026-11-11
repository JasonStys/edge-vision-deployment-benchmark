"""File: desktop/runtime_libraries.py
Purpose: Verify actual loaded vendor runtime DLLs during explicit frozen Windows acceptance.
Symbols: loaded_libraries inspects only this process and returns names/hashes, not local paths.
System OS libraries remain OS dependencies; this is not a clean-machine certification.
"""

from __future__ import annotations

import ctypes
import hashlib
import re
import sys
from pathlib import Path

from desktop.storage import read_regular, strict_json

NAMES = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
}
REQUIRED = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
MAX_PATH = 4096


def loaded_libraries() -> dict[str, str]:
    """Require used vendor runtimes to originate in this frozen bundle and match recorded bytes."""
    if not getattr(sys, "frozen", False):
        return {}
    if sys.platform != "win32":
        raise ValueError("The current native distribution targets Windows x64")
    root = Path(sys._MEIPASS).resolve()
    inventory = strict_json(read_regular(root / "NATIVE_DLLS.json", 4096))
    if (
        not isinstance(inventory, dict)
        or set(inventory) != NAMES
        or any(
            not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None
            for value in inventory.values()
        )
    ):
        raise ValueError("Fixed native runtime inventory differs")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = (
        [ctypes.c_wchar_p],
        ctypes.c_void_p,
    )
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint
    loaded = {}
    for name in sorted(NAMES):
        handle = kernel.GetModuleHandleW(name)
        if not handle:
            if name in REQUIRED:
                raise ValueError("Expected actual scientific runtime module was not loaded")
            continue
        buffer = ctypes.create_unicode_buffer(MAX_PATH)
        count = kernel.GetModuleFileNameW(handle, buffer, MAX_PATH)
        if not 0 < count < MAX_PATH:
            raise ValueError("Actual loaded runtime path cannot be bounded")
        path = Path(buffer.value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Actual vendor runtime came from outside this application")
        digest = hashlib.sha256(read_regular(path, 4 * 1024 * 1024)).hexdigest()
        if digest != inventory[name]:
            raise ValueError("Actual loaded runtime differs from the shipped vendor bytes")
        loaded[name] = digest
    return loaded
