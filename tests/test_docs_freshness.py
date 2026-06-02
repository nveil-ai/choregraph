# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ensure documentation stays in sync with the actual codebase."""

import re
from pathlib import Path

import pytest

from choregraph.library import TRANSFORM_REGISTRY


DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
LIBRARY_INDEX = DOCS_ROOT / "reference" / "library" / "index.md"


def _documented_functions(path: Path) -> set[str]:
    """Extract function names from markdown table rows like | `func_name` | ..."""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\| `(\w+)` \|", text))


class TestDocsFreshness:
    """Tests that documentation reflects the current code."""

    def test_all_transforms_documented(self):
        """Every function in TRANSFORM_REGISTRY must appear in the library index."""
        documented = _documented_functions(LIBRARY_INDEX)
        registered = set(TRANSFORM_REGISTRY.keys())
        missing = registered - documented
        assert not missing, (
            f"Transforms registered but not in docs/reference/library/index.md:\n"
            f"  {sorted(missing)}"
        )

    def test_no_phantom_transforms(self):
        """No function should be documented if it's not in TRANSFORM_REGISTRY."""
        documented = _documented_functions(LIBRARY_INDEX)
        registered = set(TRANSFORM_REGISTRY.keys())
        phantom = documented - registered
        assert not phantom, (
            f"Transforms in docs/reference/library/index.md but not registered:\n"
            f"  {sorted(phantom)}"
        )

    def test_transform_count_claim(self):
        """The claimed count in the library index must match the registry."""
        text = LIBRARY_INDEX.read_text(encoding="utf-8")
        match = re.search(r"provides (\d+) built-in", text)
        assert match, "Could not find transform count claim in library index"
        claimed = int(match.group(1))
        actual = len(TRANSFORM_REGISTRY)
        assert claimed == actual, (
            f"Docs claim {claimed} transforms but registry has {actual}"
        )

    def test_mkdocstrings_module_paths(self):
        """Every ::: directive in reference pages must point to an importable module."""
        for md_file in DOCS_ROOT.rglob("reference/**/*.md"):
            text = md_file.read_text(encoding="utf-8")
            for module_path in re.findall(r"^::: (.+)$", text, re.MULTILINE):
                module_path = module_path.strip()
                if not module_path.startswith("choregraph"):
                    continue
                parts = module_path.split(".")
                try:
                    __import__(module_path)
                except ImportError:
                    pytest.fail(
                        f"{md_file.relative_to(DOCS_ROOT)}: "
                        f"cannot import '{module_path}'"
                    )
