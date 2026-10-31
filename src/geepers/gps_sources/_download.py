# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Atomic, verified file downloads.

Ported from cal-disp's UNR staging (``cal_disp/download/_stage_unr.py``) so
every UNR consumer gets the same guarantees: the body streams to
``<dest>.part`` and is renamed into place only after the checks pass, so an
existing ``dest`` is always a complete file; an HTML error page (UNR serves
one for some missing files with status 200) or a body shorter than its
``Content-Length`` raises instead of landing on disk.
"""

from __future__ import annotations

from pathlib import Path

import requests

__all__ = ["DownloadError", "download_file"]

PART_SUFFIX = ".part"
CHUNK_SIZE = 1 << 16


class DownloadError(RuntimeError):
    """The server answered, but not with the expected file."""


def _looks_like_html(head: bytes) -> bool:
    start = head.lstrip()[:15].lower()
    return start.startswith((b"<!doctype html", b"<html"))


def _expected_length(response: requests.Response) -> int | None:
    """``Content-Length`` of a response whose body is not transfer-encoded."""
    if response.headers.get("Content-Encoding"):
        # iter_content yields decoded bytes; the header counts encoded ones
        return None
    value = response.headers.get("Content-Length")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def download_file(
    url: str,
    dest: Path,
    session: requests.Session | None = None,
    timeout: float | tuple[float, float] = (10, 120),
) -> Path:
    """Download `url` to `dest` atomically and return `dest`.

    Raises
    ------
    requests.HTTPError
        On a non-2xx status (after the session's retries).
    DownloadError
        When the body is an HTML page or shorter than ``Content-Length``.

    """
    dest = Path(dest)
    part = dest.with_name(dest.name + PART_SUFFIX)
    getter = session.get if session is not None else requests.get
    try:
        _stream_to(part, url, getter, timeout)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return dest


def _stream_to(part: Path, url: str, getter, timeout) -> None:
    """Stream `url` into `part`, raising on HTML bodies or short reads."""
    with getter(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            msg = f"{url} returned an HTML page ({content_type}), not a data file"
            raise DownloadError(msg)
        expected = _expected_length(response)
        received = 0
        with part.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if received == 0 and _looks_like_html(chunk):
                    msg = f"{url} returned an HTML page, not a data file"
                    raise DownloadError(msg)
                f.write(chunk)
                received += len(chunk)
    if expected is not None and received != expected:
        msg = (
            f"{url}: received {received} bytes but Content-Length is"
            f" {expected}; discarding the partial file"
        )
        raise DownloadError(msg)
