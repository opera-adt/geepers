"""The web viewer's earthquake and volcano helpers.

`scripts/browse_unr_grid.html` builds the USGS earthquake query and groups the
Smithsonian volcano types in plain JavaScript. This runs the page's own code in
Node; it is skipped where Node is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

VIEWER = Path(__file__).parents[1] / "scripts" / "browse_unr_grid.html"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")

# From the earthquake / volcano constants to the first function that needs the map
_START = "        const EQ_URL = "
_END = "        function eqColorExpr()"


def _run(body: str) -> Any:
    html = VIEWER.read_text()
    i = html.index(_START)
    source = (
        html[i : html.index(_END, i)]
        + f"\nconsole.log(JSON.stringify((() => {{ {body} }})()));\n"
    )
    assert NODE is not None
    out = subprocess.run(
        [NODE, "-e", source], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_earthquake_query_for_a_period_and_a_box() -> None:
    query = _run(
        "const q = eqQuery('4.5', '30', '', '', [-125, 32, -114, 42],"
        " new Date('2026-10-04T00:00:00Z'));"
        " return Object.fromEntries(q.entries());"
    )
    assert query["minmagnitude"] == "4.5"
    assert query["starttime"] == "2026-09-04"
    assert "endtime" not in query
    assert (query["minlongitude"], query["maxlatitude"]) == ("-125.000", "42.000")
    assert query["limit"] == "20000"


def test_earthquake_query_for_dates_and_the_world() -> None:
    query = _run(
        "return Object.fromEntries("
        "eqQuery('6', 'custom', '2025-01-01', '2025-12-31', null).entries());"
    )
    assert query["starttime"] == "2025-01-01"
    assert query["endtime"] == "2025-12-31T23:59:59"
    assert "minlatitude" not in query


def test_volcano_types_are_grouped() -> None:
    groups = _run(
        "return ['Stratovolcano(es)', 'Shield(s)', 'Compound', 'Submarine volcano',"
        " 'Maar(s)', ''].map(voTypeOf);"
    )
    assert groups == [
        "Stratovolcano",
        "Shield",
        "Complex",
        "Submarine",
        "Other",
        "Other",
    ]


def test_volcano_years_read_as_ce_or_bce() -> None:
    assert _run("return [voYear(1980), voYear(-8300), voYear(null)];") == [
        "1980 CE",
        "8300 BCE",
        "unknown",
    ]
