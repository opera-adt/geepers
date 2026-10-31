"""Shared test fixtures."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regen",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/* from the current code (then read git diff).",
    )


def _compare(actual, expected, rtol: float, atol: float, where: str) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert sorted(actual) == sorted(expected), f"{where}: keys differ"
        for key in expected:
            _compare(actual[key], expected[key], rtol, atol, f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), where
        assert len(actual) == len(expected), f"{where}: length differs"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, rtol, atol, f"{where}[{i}]")
    elif isinstance(expected, float) and not isinstance(actual, str):
        if math.isnan(expected):
            assert math.isnan(actual), f"{where}: {actual} != nan"
        else:
            assert math.isclose(
                actual, expected, rel_tol=rtol, abs_tol=atol
            ), f"{where}: {actual} != {expected}"
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


@pytest.fixture
def golden(pytestconfig: pytest.Config):
    """Compare a JSON-able result with tests/golden/<name>, or rewrite it.

    Floats are compared with a tolerance (results move in the last digits
    across BLAS builds); everything else must match exactly. A failure is
    a change in behavior: if it is intended, run ``pytest --regen``, read
    the diff and explain it in the commit message.
    """

    def check(name: str, actual, rtol: float = 1e-6, atol: float = 1e-9) -> None:
        path = GOLDEN_DIR / name
        # A round trip makes `actual` the same types the golden file holds
        actual = json.loads(json.dumps(actual))
        if pytestconfig.getoption("--regen"):
            GOLDEN_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
            pytest.skip(f"regenerated {name}")
        assert path.exists(), f"{name} is missing; create it with pytest --regen"
        _compare(actual, json.loads(path.read_text()), rtol, atol, name)

    return check
