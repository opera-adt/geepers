"""Tests for the UNR monument table (geepers.gps_sources.unr)."""

from __future__ import annotations

import pandas as pd
import pytest

import geepers.gps_sources.unr as unr
from geepers.gps_sources import UnrSource

TABLE = """\
Site    : Monument Description    : Height of Monument (m) : Monument Foundation : Foundation Depth (m) : Geologic Characteristic
==========================================================
AAAA    : DEEP-DRILLED BRACED     :                        :  UNKNOWN            :                      :
BBBB    : SHALLOW-DRILLED BRACED  :  1.5                   :                     :                      : BEDROCK
CCCC    : STEEL MAST              :  3                     :  ROOF               :                      :
DDDD    : STEEL MAST              :  2.0                   :  CONCRETE BLOCK     :  1.2                 : SAND
EEEE    : CONCRETE PILLAR         :  ~0.18                 :                     :                      :
FFFF    : STEEL PIN IN ROCK       :                        :                     :                      :
GGGG    :                         :                        :                     :                      :
HHHH    : PILLAR ON THE EDGE OF A ROOF :                   :                     :                      :
IIII    : DEEP WYATT/AGNEW DRILLED-BRACED :                :                     :                      :
JJJJ    : Aluminum Pedestal       : 1.5  : 2019-04-01 : 1. "Foundation" is : ~0.18  : BEDROCK
KKKK    : PERMANENT STATION UNSPECIFIED :                  :  UNKNOWN            :                      :
LLLL    : DEEP DRILL-BRACE        :                        :                     :                      :
short line
"""


@pytest.fixture
def table():
    return unr._parse_monuments(TABLE).set_index("id")


class TestParse:
    def test_every_station_is_read_once(self, table):
        assert len(table) == 12
        assert table.index.is_unique
        assert "Site" not in table.index

    @pytest.mark.parametrize(
        ("station", "expected"),
        [
            ("AAAA", "deep_braced"),
            ("BBBB", "braced"),
            ("CCCC", "building"),
            ("DDDD", "mast"),
            ("EEEE", "pillar"),
            ("FFFF", "rock_pin"),
            ("GGGG", "unknown"),
            ("HHHH", "building"),
            ("IIII", "deep_braced"),
            ("KKKK", "unknown"),
            ("LLLL", "deep_braced"),
        ],
    )
    def test_monument_class(self, table, station, expected):
        assert table.loc[station, "monument_class"] == expected

    def test_numbers_are_parsed_and_others_are_nan(self, table):
        assert table.loc["DDDD", "height_m"] == 2.0
        assert table.loc["DDDD", "foundation_depth_m"] == 1.2
        assert pd.isna(table.loc["EEEE", "height_m"])

    def test_a_colon_in_a_foundation_keeps_the_columns_aligned(self, table):
        assert table.loc["JJJJ", "geology"] == "BEDROCK"
        assert "2019-04-01" in table.loc["JJJJ", "foundation"]

    def test_text_fields_are_kept(self, table):
        assert table.loc["DDDD", "geology"] == "SAND"
        assert table.loc["DDDD", "foundation"] == "CONCRETE BLOCK"


class _Response:
    text = TABLE

    def raise_for_status(self):
        pass


class TestMonuments:
    @pytest.fixture(autouse=True)
    def fake_download(self, monkeypatch):
        unr.UnrSource._read_monuments_table.cache_clear()
        calls = []

        def get(url, timeout):
            calls.append((url, timeout))
            return _Response()

        monkeypatch.setattr(unr.requests, "get", get)
        yield calls
        unr.UnrSource._read_monuments_table.cache_clear()

    def test_downloads_the_table_once(self, fake_download):
        source = UnrSource()
        source.monuments()
        source.monuments()
        assert [url for url, _ in fake_download] == [unr.MONUMENTS_URL]

    def test_selects_stations_in_any_case(self):
        out = UnrSource().monuments(["cccc", "AAAA", "NOPE"])
        assert sorted(out["id"]) == ["AAAA", "CCCC"]

    def test_changing_the_result_does_not_change_the_cache(self):
        source = UnrSource()
        first = source.monuments()
        first.loc[0, "monument_class"] = "changed"
        assert "changed" not in set(source.monuments()["monument_class"])


def test_summary_counts_by_class():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "unr-monuments.py"
    spec = importlib.util.spec_from_file_location("unr_monuments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    joined = pd.DataFrame(
        {
            "monument_class": ["mast", "mast", "pillar"],
            "signal": ["local", "none", "shared"],
            "local_random_walk_mm": [4.0, 2.0, 1.0],
            "noisy": [True, False, False],
        }
    )
    out = module.summarize(joined)
    assert out.loc["mast", "n"] == 2
    assert out.loc["mast", "noisy_share"] == 0.5
    assert out.loc["mast", "n_local"] == 1
    assert out.loc["pillar", "n_local"] == 0
    assert out.loc["pillar", "n_smooth"] == 1
