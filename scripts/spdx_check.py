#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers.
# Kit copy of 00_tools/standards/scripts/spdx_check.py; re-apply the kit, do not edit.
"""Check (or add) SPDX license headers on Python files.

Every ``.py`` file must declare the repository's license in its first lines::

    # SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
    # SPDX-License-Identifier: Apache-2.0

The check is deliberately narrow: it looks for ``SPDX-License-Identifier: <ID>``
within the leading comment block (after an optional shebang and encoding line)
and nothing else. Empty files are skipped.

Usage (as a pre-commit hook, which passes the staged file names)::

    spdx_check.py --license Apache-2.0 src/pkg/a.py src/pkg/b.py

Add missing headers in place::

    spdx_check.py --license Apache-2.0 --fix \
        --copyright '2025-2026 California Institute of Technology ("Caltech")' \
        --notice 'Part of geepers, https://github.com/opera-adt/geepers.' \
        $(git ls-files '*.py')

Exit code 1 when any file is missing or has a different identifier.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER_LINES_TO_SCAN = 12
IDENTIFIER_RE = re.compile(r"^#\s*SPDX-License-Identifier:\s*(?P<spdx>\S+)\s*$")


def find_identifier(text: str) -> str | None:
    """Return the SPDX identifier declared in the leading comment block, if any."""
    for line in text.splitlines()[:HEADER_LINES_TO_SCAN]:
        match = IDENTIFIER_RE.match(line)
        if match:
            return match.group("spdx")
        if line.strip() and not line.startswith("#"):
            # The leading comment block has ended; a header further down does
            # not count, which is what keeps the convention greppable.
            return None
    return None


def header_insert_index(lines: list[str]) -> int:
    """Index at which a header goes: after a shebang and an encoding line."""
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if len(lines) > index and re.match(r"^#.*coding[:=]", lines[index]):
        index += 1
    return index


def build_header(
    license_id: str, copyright_text: str | None, notices: list[str] | None
) -> list[str]:
    """Build the header lines for `--fix`."""
    header = []
    if copyright_text:
        header.append(f"# SPDX-FileCopyrightText: {copyright_text}")
    header.append(f"# SPDX-License-Identifier: {license_id}")
    for notice in notices or []:
        header.append(f"# {notice}")
    return header


def check_file(
    path: Path,
    license_id: str,
    *,
    fix: bool = False,
    copyright_text: str | None = None,
    notices: list[str] | None = None,
) -> str | None:
    """Check one file. Return a problem description, or None when it passes.

    With ``fix=True`` a missing header is inserted and None is returned; a
    *wrong* identifier is never rewritten automatically.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    found = find_identifier(text)
    if found == license_id:
        return None
    if found is not None:
        return f"{path}: SPDX-License-Identifier is {found}, expected {license_id}"
    if not fix:
        return f"{path}: missing '# SPDX-License-Identifier: {license_id}' header"
    lines = text.splitlines(keepends=True)
    index = header_insert_index([line.rstrip("\n") for line in lines])
    header = [line + "\n" for line in build_header(license_id, copyright_text, notices)]
    path.write_text("".join(lines[:index] + header + lines[index:]), encoding="utf-8")
    return None


def main(argv: list[str] | None = None) -> int:
    """Run the check on the given files; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--license", required=True, help="Expected SPDX identifier, e.g. Apache-2.0"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Insert a header where none exists"
    )
    parser.add_argument(
        "--copyright", help="SPDX-FileCopyrightText value used by --fix"
    )
    parser.add_argument(
        "--notice",
        action="append",
        help="Extra comment line after the identifier, used by --fix; repeatable",
    )
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    problems = []
    for path in args.files:
        if path.suffix != ".py" or not path.is_file():
            continue
        problem = check_file(
            path,
            args.license,
            fix=args.fix,
            copyright_text=args.copyright,
            notices=args.notice,
        )
        if problem:
            problems.append(problem)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
