"""Keep the AGENTS.md file map honest: every path it names must exist."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_LINE = re.compile(r"^- `([^`]+)`:")


def _mapped_paths() -> list[str]:
    text = (ROOT / "AGENTS.md").read_text()
    section = text.split("# Project File Map", 1)[1].split("\n# ", 1)[0]
    return [m.group(1) for line in section.splitlines() if (m := MAP_LINE.match(line))]


def test_file_map_is_not_empty():
    assert len(_mapped_paths()) > 20


def test_every_mapped_path_exists():
    missing = []
    for entry in _mapped_paths():
        if "<" in entry:
            # A pattern such as tests/test_<module>.py: check its directory.
            exists = (ROOT / entry.split("<", 1)[0].rsplit("/", 1)[0]).is_dir()
        elif "*" in entry:
            exists = any(ROOT.glob(entry))
        else:
            exists = (ROOT / entry).exists()
        if not exists:
            missing.append(entry)
    assert not missing, f"AGENTS.md file map names paths that do not exist: {missing}"
