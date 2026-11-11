"""File: scripts/build-native.py
Purpose: Freeze the actual scientific/Tk application into a fresh source-bound Windows directory.
Symbols: main owns public resources; notices reads installed dependency terms.
No installer, normal profile, global change, Docker action or publication occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import struct
import subprocess
import sys
import tkinter as tk
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop.project import FILES, verify_bundle  # noqa: E402 - fixed source-root CLI bootstrap
from desktop.storage import PRODUCT, VERSION, read_regular, strict_json  # noqa: E402 - fixed source-root CLI bootstrap

MIN_NOTICE_BYTES = 100
MAX_NOTICE_BYTES = 65536


def icon(path: Path) -> None:
    """Author a distinct bounded 8x8 inference-grid glyph, without external artwork or tools."""
    size = 256
    grid_start, grid_end, cell_pitch, stem = 8, 56, 6, 3
    pixels = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            u, v = x // 4, y // 4
            inside = grid_start <= u < grid_end and grid_start <= v < grid_end
            row, column = (v - grid_start) // cell_pitch, (u - grid_start) // cell_pitch
            ink = inside and column in (row, stem)
            grid = inside and (
                (u - grid_start) % cell_pitch == 0 or (v - grid_start) % cell_pitch == 0
            )
            color = (
                bytes((145, 225, 75, 255))
                if ink
                else (bytes((110, 80, 45, 255)) if grid else bytes((62, 32, 14, 255)))
            )
            offset = ((size - 1 - y) * size + x) * 4
            pixels[offset : offset + 4] = color
    bitmap = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, len(pixels), 0, 0, 0, 0)
    image = bitmap + pixels + bytes(size * size // 8)
    directory = struct.pack("<HHHBBBBHHII", 0, 1, 1, 0, 0, 0, 0, 1, 32, len(image), 22)
    with path.open("xb") as output:
        output.write(directory + image)


def command(arguments: list[str], *, timeout: int = 600) -> str:
    """Invoke fixed build tools with vectors, bounded time and no shell expansion."""
    # Only this builder's fixed Git argv is supplied; no application/input content reaches commands.
    return subprocess.check_output(arguments, cwd=ROOT, text=True, timeout=timeout)  # noqa: S603


def notices() -> tuple[bytes, dict[str, str]]:
    """Retain actual Python/Tcl/Tk and scientific-library terms, including DLL zipfs resources."""
    root = tk.Tk()
    root.withdraw()
    versions = {
        "python": platform.python_version(),
        "tcl": str(root.tk.call("info", "patchlevel")),
        "tk": str(root.tk.call("package", "require", "Tk")),
    }
    blocks = [
        ("Authored application - MIT", read_regular(ROOT / "LICENSE", 2 * 1024 * 1024)),
        ("Python runtime", read_regular(Path(sys.base_prefix) / "LICENSE.txt", 2 * 1024 * 1024)),
    ]
    try:
        for title, directory in [
            ("Tcl", root.tk.call("info", "library")),
            ("Tk", root.tk.eval("set tk_library")),
        ]:
            channel = root.tk.call("open", str(directory) + "/license.terms", "rb")
            try:
                data = root.tk.call("read", channel, 65537)
            finally:
                root.tk.call("close", channel)
            if isinstance(data, str):
                data = data.encode("utf-8")
            if not isinstance(data, bytes) or not MIN_NOTICE_BYTES < len(data) <= MAX_NOTICE_BYTES:
                raise ValueError("Loaded Tcl/Tk notice is missing or exceeds its bound")
            blocks.append((title + " " + versions[title.lower()], data))
    finally:
        root.destroy()
    names = [
        "numpy",
        "onnx",
        "onnxruntime",
        "protobuf",
        "flatbuffers",
        "packaging",
        "ml_dtypes",
        "typing-extensions",
        "pyinstaller",
    ]
    for name in names:
        distribution = importlib.metadata.distribution(name)
        paths = [
            entry
            for entry in distribution.files or []
            if entry.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
            and entry.suffix not in (".py", ".pyc")
        ]
        if not paths:
            if name == "flatbuffers" and distribution.version == "25.12.19":
                blocks.append(
                    (
                        "FlatBuffers 25.12.19 upstream LICENSE (wheel omits it)",
                        read_regular(ROOT / "licenses/flatbuffers-25.12.19.LICENSE", 65536),
                    )
                )
                continue
            raise ValueError("Review missing license inventory for " + name)
        blocks.extend(
            (
                name + " " + distribution.version + " / " + entry.name,
                read_regular(Path(distribution.locate_file(entry)), 2 * 1024 * 1024),
            )
            for entry in sorted(paths)
        )
    return b"\n\n".join(title.encode() + b"\n" + data for title, data in blocks), versions


def stage_resources(generated: Path, kernel_source: Path, crt_source: Path) -> tuple[Path, Path]:
    """Copy only fixed public seed/kernel files and verify licensed vendor DLL identities."""
    verify_bundle(ROOT)
    seed = generated / "seed"
    for name in (*FILES, "artifacts/manifest.json"):
        target = seed / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(read_regular(ROOT / name, 5_000_000))
    verify_bundle(seed)
    source_kernels = kernel_source.resolve()
    kernels = generated / "kernels"
    kernels.mkdir()
    inventory = strict_json(read_regular(source_kernels / "KERNELS.json", 4096))
    if not isinstance(inventory, dict) or set(inventory) != {
        "edgevision_cpp.exe",
        "edgevision-rust.exe",
    }:
        raise ValueError("The exact two native inference engines are required")
    for name in ("edgevision_cpp.exe", "edgevision-rust.exe", "KERNELS.json"):
        data = read_regular(source_kernels / name, 32 * 1024 * 1024)
        if name != "KERNELS.json" and hashlib.sha256(data).hexdigest() != inventory[name]:
            raise ValueError("Actual native inference executable digest differs")
        with (kernels / name).open("xb") as stream:
            stream.write(data)
    crt = crt_source.resolve()
    powershell = Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    subprocess.run(  # noqa: S603 - fixed System32 verifier and explicit SDK path
        [
            str(powershell),
            "-NoProfile",
            "-File",
            str(ROOT / "scripts/check-crt.ps1"),
            "-Directory",
            str(crt),
        ],
        check=True,
        timeout=30,
        env={key: value for key, value in os.environ.items() if key.casefold() != "psmodulepath"},
    )
    return seed, kernels


def write_metadata(
    generated: Path, source: str, dirty: bool, runtime: dict, license_bytes: bytes
) -> dict:
    """Create fixed source/version resources and actual third-party notices in new staging."""
    identity = {
        "schema": 1,
        "product": PRODUCT,
        "version": VERSION,
        "source": source,
        "sourceDirty": dirty,
        **runtime,
    }
    (generated / "SOURCE.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    (generated / "THIRD-PARTY-NOTICES.txt").write_bytes(license_bytes)
    icon(generated / "application.ico")
    # PyInstaller parses this fixed generated version resource; it contains no caller text.
    version_resource = """VSVersionInfo(ffi=FixedFileInfo(filevers=(1,1,0,0),prodvers=(1,1,0,0),
    mask=0x3f,flags=0,OS=0x40004,fileType=1,subtype=0,date=(0,0)),kids=[StringFileInfo([
    StringTable('040904B0',[StringStruct('CompanyName','Jason Stys'),
    StringStruct('FileDescription','Edge Vision Deployment Workbench'),
    StringStruct('FileVersion','1.1.0'),StringStruct('ProductVersion','1.1.0'),
    StringStruct('ProductName','Edge Vision Deployment Workbench')])]),
    VarFileInfo([VarStruct('Translation',[1033,1200])])])"""
    (generated / "version.txt").write_text(version_resource, encoding="utf-8")
    return identity


def reviewed_toolchain() -> None:
    """Refuse unsupported runtime/freezer/vendor versions before resource staging."""
    if sys.platform != "win32" or sys.version_info[:3] != (3, 14, 10):
        raise ValueError("Native Windows builds require the reviewed CPython3.14.10 SDK")
    if importlib.metadata.version("pyinstaller") != "6.22.3":
        raise ValueError("Reviewed PyInstaller6.22.3 is required")
    if importlib.metadata.version("onnxruntime") != "1.31.0":
        raise ValueError("Review a different ONNX Runtime before changing native import packaging")


def main(argv: list[str] | None = None) -> int:
    """Require coherent patched build tools and freeze only allowlisted public source/resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--crt-sdk", type=Path, required=True)
    parser.add_argument("--console-diagnostics", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.console_diagnostics and os.environ.get("CI"):
        raise ValueError("Hosted native acceptance must use the normal windowed application")
    reviewed_toolchain()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if metadata["name"] != PRODUCT or metadata["version"] != VERSION:
        raise ValueError("Native/core/project product identity differs")
    output = (ROOT / arguments.output).resolve()
    if not output.is_relative_to(ROOT / "release") or output.exists():
        raise ValueError("Choose a new output below release; earlier candidates are preserved")
    git = ["git", "-c", "safe.directory=" + ROOT.as_posix(), "-C", str(ROOT)]
    source = command([*git, "rev-parse", "HEAD"]).strip()
    changed = command([*git, "status", "--porcelain", "--untracked-files=all"]).strip()
    dirty = bool(changed)
    if os.environ.get("CI") and dirty:
        raise ValueError("Hosted native acceptance requires a clean exact checkout")
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_SHA") != source:
        raise ValueError("Hosted native checkout differs from the declared workflow source")
    generated = ROOT / ".runtime" / ("native-metadata-" + secrets.token_hex(8))
    generated.mkdir(parents=True)
    seed, kernels = stage_resources(generated, arguments.kernels, arguments.crt_sdk)
    crt = arguments.crt_sdk.resolve()
    license_bytes, runtime = notices()
    license_bytes += (
        b"\n\nMicrosoft C++ Runtime (unmodified, Microsoft Corporation).\n"
        b"Licensed separately from this application's MIT source.\n"
        b"Official terms and redistribution list:\n"
        b"https://visualstudio.microsoft.com/license-terms/vs2026-ga-community/\n"
        b"https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution\n"
    )
    identity = write_metadata(generated, source, dirty, runtime, license_bytes)
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--console" if arguments.console_diagnostics else "--windowed",
        "--onedir",
        "--name",
        "EdgeVisionWorkbench",
        "--paths",
        str(ROOT),
        # This exact vendor version uses bare transformer-helper imports during quantization.
        # Resolve them from its locked installed package, never a user input directory.
        "--paths",
        str(importlib.metadata.distribution("onnxruntime").locate_file("onnxruntime/transformers")),
        "--distpath",
        str(output),
        "--workpath",
        str(generated / "build"),
        "--specpath",
        str(generated),
        "--version-file",
        str(generated / "version.txt"),
        "--icon",
        str(generated / "application.ico"),
        "--add-data",
        str(generated / "application.ico") + ":.",
        "--add-data",
        str(seed) + ":seed",
        "--add-data",
        str(kernels) + ":kernels",
        "--add-data",
        str(generated / "SOURCE.json") + ":.",
        "--copy-metadata",
        "onnxruntime",
        "--copy-metadata",
        "onnx",
        "--copy-metadata",
        "numpy",
        str(ROOT / "desktop/launcher.py"),
    ]
    crt_names = (
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
    )
    hashes = {}
    for name in crt_names:
        hashes[name] = hashlib.sha256(read_regular(crt / name, 4 * 1024 * 1024)).hexdigest()
        args[-1:-1] = ["--add-binary", str(crt / name) + ":."]
    (generated / "NATIVE_DLLS.json").write_text(json.dumps(hashes) + "\n", encoding="utf-8")
    args[-1:-1] = ["--add-data", str(generated / "NATIVE_DLLS.json") + ":."]
    environment = {**os.environ, "PYINSTALLER_CONFIG_DIR": str(generated / "cache")}
    # Fixed reviewed PyInstaller argv; user datasets and model files are never executed.
    subprocess.run(args, cwd=ROOT, check=True, timeout=600, env=environment)  # noqa: S603
    if (
        command([*git, "rev-parse", "HEAD"]).strip() != source
        or command([*git, "status", "--porcelain", "--untracked-files=all"]).strip() != changed
    ):
        raise ValueError("Source identity changed during native build")
    bundle = output / "EdgeVisionWorkbench"
    for original, name in [
        (ROOT / "LICENSE", "LICENSE.txt"),
        (generated / "SOURCE.json", "SOURCE.json"),
        (generated / "THIRD-PARTY-NOTICES.txt", "THIRD-PARTY-NOTICES.txt"),
    ]:
        (bundle / name).write_bytes(read_regular(original, 8 * 1024 * 1024))
    executable = bundle / "EdgeVisionWorkbench.exe"
    with executable.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({"built": True, "identity": identity, "executableSha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
