# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Atomic, verified downloads (ported from cal-disp's UNR staging)."""

from __future__ import annotations

import pytest
import requests

from geepers.gps_sources._download import DownloadError, download_file
from geepers.gps_sources.unr_grid import UnrGridSource

BODY = b"2020.0 0 0 0 1 1 1 0\n" * 100


class FakeResponse:
    def __init__(self, chunks, headers=None, status=200):
        self.chunks = chunks
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            msg = f"{self.status} Client Error"
            raise requests.HTTPError(msg)

    def iter_content(self, chunk_size):  # noqa: ARG002 - matches requests
        yield from self.chunks


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, stream=False, timeout=None):
        self.calls.append((url, stream, timeout))
        return self.response


def test_writes_the_whole_body_atomically(tmp_path):
    dest = tmp_path / "000001_IGS20.tenv8"
    session = FakeSession(
        FakeResponse([BODY[:1000], BODY[1000:]], {"Content-Length": str(len(BODY))})
    )
    assert download_file("https://x/f", dest, session=session) == dest
    assert dest.read_bytes() == BODY
    assert not (tmp_path / "000001_IGS20.tenv8.part").exists()
    assert session.calls[0][1] is True  # streamed


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (
            FakeResponse([b"<html>nope</html>"], {"Content-Type": "text/html"}),
            "HTML page",
        ),
        (FakeResponse([b"  <!DOCTYPE html><p>moved</p>"]), "HTML page"),
        (
            FakeResponse([BODY[:50]], {"Content-Length": str(len(BODY))}),
            "Content-Length",
        ),
    ],
    ids=["html-content-type", "html-body", "short-body"],
)
def test_bad_bodies_raise_and_leave_nothing(tmp_path, response, match):
    dest = tmp_path / "f.tenv8"
    with pytest.raises(DownloadError, match=match):
        download_file("https://x/f", dest, session=FakeSession(response))
    assert list(tmp_path.iterdir()) == []


def test_http_errors_propagate_and_leave_nothing(tmp_path):
    with pytest.raises(requests.HTTPError, match="404"):
        download_file(
            "https://x/f",
            tmp_path / "f",
            session=FakeSession(FakeResponse([], status=404)),
        )
    assert list(tmp_path.iterdir()) == []


def test_encoded_bodies_skip_the_length_check(tmp_path):
    response = FakeResponse(
        [BODY], {"Content-Length": "10", "Content-Encoding": "gzip"}
    )
    download_file("https://x/f", tmp_path / "f", session=FakeSession(response))
    assert (tmp_path / "f").read_bytes() == BODY


def test_grid_source_uses_the_verified_download(tmp_path):
    """Regression: an HTML error page used to be saved as the .tenv8 file."""
    src = UnrGridSource(version="0.3", gridded_type="constant", cache_dir=tmp_path)
    bad = FakeSession(
        FakeResponse([b"<html>not found</html>"], {"Content-Type": "text/html"})
    )
    with pytest.raises(DownloadError):
        src._download_file(
            "000001",
            plate="IGS20",
            output_dir=tmp_path,
            session=bad,
            version="0.3",
            gridded_type="constant",
        )
    assert not (tmp_path / "000001_IGS20.tenv8").exists()
    good = FakeSession(FakeResponse([BODY], {"Content-Length": str(len(BODY))}))
    out = src._download_file(
        "1",
        plate="IGS20",
        output_dir=tmp_path,
        session=good,
        version="0.3",
        gridded_type="constant",
    )
    assert out.read_bytes() == BODY
    assert good.calls[0][0].endswith("time_contsant_gridded/IGS20/000001_IGS20.tenv8")
    # an existing complete file is not fetched again
    again = FakeSession(FakeResponse([b"should not be read"]))
    src._download_file(
        "1",
        plate="IGS20",
        output_dir=tmp_path,
        session=again,
        version="0.3",
        gridded_type="constant",
    )
    assert again.calls == []
