# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""Pull the UNR monument table, and relate it to unstable stations.

    python scripts/unr-monuments.py -o monuments.csv
    python scripts/unr-monuments.py -o joined.csv --neighbors trends/neighbors.csv

`--neighbors` takes the table `geepers-trends --neighbors` writes. It adds the
monument of each station and prints, per monument class, how many stations
exceed the random-walk limit and how many smooth signals are local to them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geepers.gps_sources import UnrSource


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    """Return the stability of the stations of each monument class."""
    groups = joined.groupby("monument_class", sort=True)
    smooth = joined[joined["signal"].isin(["shared", "local"])]
    return pd.DataFrame(
        {
            "n": groups.size(),
            "median_local_random_walk_mm": groups["local_random_walk_mm"].median(),
            "noisy_share": groups["noisy"].mean(),
            "n_smooth": smooth.groupby("monument_class").size(),
            "n_local": (
                smooth[smooth["signal"] == "local"].groupby("monument_class").size()
            ),
        }
    ).fillna({"n_smooth": 0, "n_local": 0})


def main() -> None:
    """Write the table, joined with a neighbor check if one is given."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-o", "--output", type=Path, default=Path("monuments.csv"))
    parser.add_argument("--neighbors", type=Path, help="neighbors.csv to join")
    args = parser.parse_args()

    table = UnrSource().monuments()
    print(table["monument_class"].value_counts().to_string())
    if args.neighbors is not None:
        checks = pd.read_csv(args.neighbors)
        table = checks.merge(table, on="id", how="left")
        table["monument_class"] = table["monument_class"].fillna("unknown")
        print(summarize(table).round(2).to_string())
    table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
