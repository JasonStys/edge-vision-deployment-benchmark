"""File: desktop/storage.py
Purpose: Own one durable local workbench without following links or resetting unknown data.
Functions/variables: see docs/CODE_INDEX.md; Home retains its OS lock until real work completes.
The owner-local profile is plaintext, not authenticated multi-user or tamper-proof storage.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

PRODUCT = "edge-vision-deployment-benchmark"
VERSION = "1.1.0"
MAX_JSON = 8 * 1024 * 1024

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def read_regular(path: Path, limit: int) -> bytes:
    """Bound the actual opened descriptor and reject links, aliases, special files and growth."""
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    with os.fdopen(descriptor, "rb") as source:
        opened, named = os.fstat(source.fileno()), path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or path.is_symlink()
            or path.is_junction()
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            or not 0 <= opened.st_size <= limit
        ):
            raise ValueError("Choose a bounded regular local file without links")
        raw = source.read(min(opened.st_size + 1, limit + 1))
    if len(raw) != opened.st_size or len(raw) > limit:
        raise ValueError("The local file changed or exceeded its byte limit")
    return raw


def strict_json(raw: bytes) -> Any:
    """Refuse duplicate object keys and nonfinite numbers before the exact state schema."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in items:
            if name in result:
                raise ValueError("Duplicate JSON field")
            result[name] = value
        return result

    def number(text: str) -> float:
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        return value

    def invalid(_: str) -> None:
        raise ValueError("Nonfinite JSON literal")

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=number, parse_constant=invalid
    )


def new_json(path: Path, payload: object) -> None:
    """Create a complete new bounded file; callers never overwrite an exported report."""
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8") + b"\n"
    if len(raw) > MAX_JSON:
        raise ValueError("Report exceeds 8 MiB")
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def atomic_json(path: Path, payload: object) -> None:
    """Replace only identified profile state using a fresh flushed sibling, never an export."""
    temporary = path.with_name(".state-" + secrets.token_hex(16) + ".json")
    created = False
    try:
        new_json(temporary, payload)
        created = True
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def default_profile() -> Path:
    """Keep durable user data outside the source, frozen executable and temporary resources."""
    if sys.platform == "win32":
        base = Path(os.environ["LOCALAPPDATA"])
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    if not base.is_absolute():
        raise ValueError("The OS application-data root must be absolute")
    return base / "JasonStysPortfolio" / PRODUCT / "workspace"


def real_ancestors(path: Path) -> None:
    """Refuse existing directory aliases, including Windows junctions, before creating state."""
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Linked profile/resource ancestors are unsupported")


class Home:
    """Own one exact identified workspace and a genuine cross-process OS file lock."""

    def __init__(self, path: Path):
        """Admit empty or identified state only; never reset a corrupt or unfamiliar directory."""
        self.path = path.absolute()
        real_ancestors(self.path)
        if getattr(sys, "frozen", False):
            roots = (Path(sys.executable).resolve().parent, Path(sys._MEIPASS).resolve())
            if any(self.path.resolve().is_relative_to(root) for root in roots):
                raise ValueError("Choose durable data outside the installed application")
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        with os.scandir(self.path) as entries:
            nonempty = next(entries, None) is not None
        marker = self.path / "workspace.json"
        if nonempty and not marker.exists():
            raise ValueError("Unknown profile preserved; choose a new empty directory")
        owner = self.path / "owner.lock"
        if owner.is_symlink() or owner.is_junction():
            raise ValueError("Linked ownership file refused")
        descriptor = os.open(
            owner,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            0o600,
        )
        self.owner = os.fdopen(descriptor, "r+b")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            self.owner.close()
            raise ValueError("Ownership must be a regular unaliased file")
        try:
            if opened.st_size == 0:
                self.owner.write(b"0")
                self.owner.flush()
            self.owner.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = {"schema": 1, "product": PRODUCT}
            if not nonempty:
                new_json(marker, identity)
            observed = strict_json(read_regular(marker, 1024))
            if observed != identity or type(observed.get("schema")) is not int:
                raise ValueError("Unrecognized workspace identity; no reset attempted")
        except BaseException:
            self.owner.close()
            raise

    def close(self) -> None:
        """Release only this owner's OS handle after actual native/background work completes."""
        self.owner.close()
