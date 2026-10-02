"""Every source file carries the copyright and attribution header."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    p
    for p in [*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]
    if p.name != "_version.py"  # written by setuptools_scm at build time
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_spdx_header(path):
    head = path.read_text().splitlines()[:6]
    assert any(
        "SPDX-License-Identifier: Apache-2.0" in line for line in head
    ), f"{path.relative_to(ROOT)} lacks the SPDX header; copy it from any module"
    assert any("https://github.com/opera-adt/geepers" in line for line in head)


def test_notice_and_citation_present():
    notice = (ROOT / "NOTICE").read_text()
    assert "Section 4(d)" in notice
    assert "AI tools" in notice
    assert "repository-code" in (ROOT / "CITATION.cff").read_text()
