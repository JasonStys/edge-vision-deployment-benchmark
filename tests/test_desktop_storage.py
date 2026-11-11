"""File: tests/test_desktop_storage.py
Purpose: Verify real native profile ownership, alias refusal and non-destructive state writes.
Symbols: see docs/CODE_INDEX.md. Only new pytest-owned paths are used, never normal AppData.
"""

import json
import os
import sys

if sys.platform == "win32":
    import _winapi

import pytest

from desktop.storage import Home, atomic_json, new_json, read_regular, strict_json


def test_real_profile_lock_and_reopen(tmp_path):
    """A second real owner is refused until the first descriptor actually closes."""
    first = Home(tmp_path / "profile")
    with pytest.raises(OSError):
        Home(first.path)
    first.close()
    reopened = Home(first.path)
    reopened.close()


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_json_is_unambiguous_and_finite(raw):
    """The independent malformed fixtures cannot become ordinary accepted state."""
    with pytest.raises(ValueError):
        strict_json(raw)


def test_unknown_and_bad_marker_preserve_existing_files(tmp_path):
    """An unfamiliar directory or boolean schema never becomes a reset workspace."""
    target = tmp_path / "unknown"
    target.mkdir()
    (target / "keep.txt").write_bytes(b"Retain this unrelated original file")
    with pytest.raises(ValueError, match="Unknown profile"):
        Home(target)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == {
        "keep.txt": b"Retain this unrelated original file"
    }
    (target / "workspace.json").write_text(
        json.dumps({"schema": True, "product": "edge-vision-deployment-benchmark"})
    )
    with pytest.raises(ValueError, match="identity"):
        Home(target)
    assert (target / "keep.txt").read_bytes() == b"Retain this unrelated original file"


@pytest.mark.parametrize("member", ["workspace.json", "owner.lock"])
def test_real_hardlinked_profile_members_are_refused_without_target_changes(tmp_path, member):
    """Hardlink metadata is tested on actual files, without mocked filesystem attributes."""
    home = Home(tmp_path / "profile")
    home.close()
    selected = home.path / member
    outside = tmp_path / "external-sentinel"
    original = selected.read_bytes()
    outside.write_bytes(original)
    selected.unlink()
    os.link(outside, selected)
    with pytest.raises(ValueError):
        Home(home.path)
    assert outside.read_bytes() == original


def test_real_directory_alias_is_refused_before_creating_profile(tmp_path):
    """Actual NTFS junction/POSIX symlink target remains empty."""
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    if sys.platform == "win32":
        _winapi.CreateJunction(str(target), str(alias))
    else:
        alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked profile"):
        Home(alias / "new-profile")
    assert list(target.iterdir()) == []


def test_frozen_program_roots_reject_profile_before_creation(tmp_path, monkeypatch):
    """A writable extracted program is still not the durable profile location."""
    install, internal = tmp_path / "installation", tmp_path / "resources"
    install.mkdir()
    internal.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(install / "workbench.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    for root in (install, internal):
        with pytest.raises(ValueError, match="outside"):
            Home(root / "not-created")
        assert not (root / "not-created").exists()
    outside = Home(tmp_path / "external-data")
    outside.close()


def test_atomic_state_collision_and_export_no_overwrite(tmp_path, monkeypatch):
    """An independently precreated temporary name is never removed by failed exclusive creation."""
    target = tmp_path / "state.json"
    new_json(target, {"generation": "original"})
    before = target.read_bytes()
    collision = tmp_path / (".state-" + "f" * 32 + ".json")
    collision.write_bytes(b"Keep this independently created file")
    monkeypatch.setattr("desktop.storage.secrets.token_hex", lambda _: "f" * 32)
    with pytest.raises(FileExistsError):
        atomic_json(target, {"generation": "replacement"})
    with pytest.raises(FileExistsError):
        new_json(target, {"other": "overwrite"})
    assert target.read_bytes() == before
    assert collision.read_bytes() == b"Keep this independently created file"


def test_regular_reader_rejects_link_aliases_and_size_without_writes(tmp_path):
    """Opened regular-file budget and link count are meaningful independent boundaries."""
    target, alias = tmp_path / "target", tmp_path / "alias"
    target.write_bytes(b"bounded actual file")
    with pytest.raises(ValueError):
        read_regular(target, 2)
    os.link(target, alias)
    with pytest.raises(ValueError):
        read_regular(alias, 100)
    assert target.read_bytes() == b"bounded actual file"
