"""The web viewer's linearity test must agree with `geepers.linearity`.

`scripts/browse_unr_grid.html` carries a JavaScript port of the linearity test
(and of the step detector it feeds). This runs the page's own code in Node on
the same series and compares the results; it is skipped where Node is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geepers.linearity import linearity_test
from geepers.synthetic import power_law_noise

VIEWER = Path(__file__).parents[1] / "scripts" / "browse_unr_grid.html"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node.js")

# Consecutive blocks of the page script, from each start marker to the next one
_BLOCKS = [
    ("        const median = arr =>", "        function haversineKm"),
    ("        function invert(", "        function fitSteps("),
    (
        "        // White-noise level from the first differences",
        "        function stepParams(",
    ),
    (
        "        // ================= linearity",
        "        // --- spatial structure function",
    ),
]

_RUNNER = r"""
const cases = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));
const out = cases.map(c => {
  const r = linearityTest(c.sec, c.y, { sampling: c.sampling, simulations: 0, steps: c.steps });
  return { model: r.model, rate: r.rate, sigma: r.rateSigma, departure: r.departure,
           steps: r.steps, stepSizes: r.stepSizes };
});
process.stdout.write(JSON.stringify(out));
"""


def _viewer_js() -> str:
    html = VIEWER.read_text()
    parts = ["const DAY_SEC = 86400, YEAR_SEC = 31557600;"]
    for start, end in _BLOCKS:
        i = html.index(start)
        parts.append(html[i : html.index(end, i)])
    # The page reads the sampling from its loaded dataset; the test passes it
    return "\n".join(parts).replace("sampling = meanSamplingDays()", "sampling")


def _cases() -> list[dict]:
    """Monthly series: linear, with an offset (declared), and curved."""
    dates = pd.date_range("2014-01-01", periods=132, freq="30D")
    t = np.arange(132) * 30 / 365.25
    sec = (dates - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
    rng = np.random.default_rng(5)
    noise = 1.5 * power_law_noise(132, -1, seed=5) + rng.normal(0, 1.5, 132)
    base = -3.0 * t + 3.0 * np.sin(2 * np.pi * t) + noise
    step_at = dates[40]
    series = {
        "linear": (base, []),
        "offset": (base + 25.0 * (dates >= step_at), [step_at]),
        "curved": (base - 0.4 * (t - t.mean()) ** 2, []),
    }
    return [
        {
            "label": label,
            "dates": dates,
            "sec": sec.tolist(),
            "y": values.tolist(),
            "sampling": 30,
            "steps": [
                (pd.Timestamp(d) - pd.Timestamp("1970-01-01")).total_seconds()
                for d in steps
            ],
            "step_dates": steps,
        }
        for label, (values, steps) in series.items()
    ]


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    cases = _cases()
    folder = tmp_path_factory.mktemp("viewer")
    script = folder / "run.js"
    script.write_text(_viewer_js() + _RUNNER)
    data = folder / "cases.json"
    data.write_text(
        json.dumps(
            [{k: c[k] for k in ("sec", "y", "sampling", "steps")} for c in cases]
        )
    )
    run = subprocess.run(
        [NODE, str(script), str(data)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return cases, json.loads(run.stdout)


def test_viewer_matches_python(results):
    cases, viewer = results
    for case, js in zip(cases, viewer, strict=True):
        py = linearity_test(
            case["dates"],
            np.asarray(case["y"]),
            sampling_days=case["sampling"],
            step_dates=case["step_dates"] or None,
            n_simulations=50,
        )
        assert js["model"] == py.model, case["label"]
        assert js["rate"] == pytest.approx(py.trend.velocity, abs=1e-3)
        assert js["sigma"] == pytest.approx(py.trend.velocity_uncertainty, rel=1e-3)
        assert js["departure"] == pytest.approx(py.departure, abs=1e-3)
        for j, size in enumerate(js["stepSizes"]):
            assert size == pytest.approx(py.trend.parameters[f"step_{j}"][0], abs=1e-2)


def test_declared_offset_is_estimated_not_called_non_linear(results):
    cases, viewer = results
    by_label = {c["label"]: js for c, js in zip(cases, viewer, strict=True)}
    assert by_label["linear"]["model"] == "linear"
    assert by_label["curved"]["model"] != "linear"
    offset = by_label["offset"]
    assert offset["model"] == "linear"
    assert len(offset["steps"]) == 1
    assert offset["stepSizes"][0] == pytest.approx(25.0, abs=5.0)
