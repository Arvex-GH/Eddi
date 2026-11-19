"""Cross-platform path helpers.

Eddi runs on Linux, macOS and Windows.  Older builds hard-coded ``/tmp`` which
breaks on Windows (there is no ``/tmp`` there).  Every default path now goes
through this module so the same code works everywhere.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def temp_dir() -> Path:
    """Return a writable per-user temporary directory (``/tmp``, ``%TEMP%``, ...)."""
    override = os.environ.get("EDDI_TMPDIR")
    if override:
        p = Path(override).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(tempfile.gettempdir())


def default_context_path(name: str = "context") -> str:
    """Default mmap file used for the L2 disk-backed context."""
    return str(temp_dir() / f"eddi_{name}.mmap")


def default_brain_path() -> str:
    """Default location of a saved brain (facts + language model)."""
    return str(temp_dir() / "eddi_brain.pkl")


def user_data_dir() -> Path:
    """Per-user data directory used for saved brains."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        p = base / "Eddi"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        p = base / "eddi"
    p.mkdir(parents=True, exist_ok=True)
    return p


def core_binary_name() -> str:
    """Name of the compiled C++ core for this platform."""
    return "eddi_core.exe" if sys.platform == "win32" else "eddi_core"


def find_core_binary(start: Path | None = None) -> Path | None:
    """Locate the compiled C++ core relative to the repository/installation."""
    here = (start or Path(__file__).resolve().parent).resolve()
    candidates = []
    for parent in [here, *here.parents]:
        candidates.append(parent / "bin" / core_binary_name())
        candidates.append(parent / core_binary_name())
        if parent == Path.home() or parent.parent == parent:
            break
    for c in candidates:
        if c.is_file():
            return c
    return None
