# GPS series drops the first acquisition's day (off by one)

Status: both fixes are on `feature/gps-start-date`, with regression tests.
Found 2026-10-09 during the disp-xr T34 GNSS validation
(`disp-xr/studies/gnss_validation`). disp-xr has the matching fix on its
branch `feature/gnss-daily-day`.

## Symptom

On DISP-S1 frame F08882 (first acquisition `2016-09-27T00:26:23Z`), the
`combined_data.csv` from `geepers.workflows.main` starts the GPS series on
2016-09-28 for 93 stations. The first InSAR epoch is paired with the
**next day's** daily GPS solution. Every later epoch lands on its own day.

## Cause

`workflows.py` passes the first acquisition *timestamp* as the GPS start
bound:

```python
# src/geepers/workflows.py:225
start_date = insar_reader.da.time[0].to_pandas()   # 2016-09-27 00:26:23
```

UNR daily solutions are dated at midnight (`YYMMMDD`, parsed in
`gps_sources/unr.py:251`), and `_filter_by_date` compares timestamps:

```python
# src/geepers/gps_sources/base.py:292 (before the fix)
if start_date:
    df = df[df["date"] >= _naive(start_date)]   # 2016-09-27 00:00 < 00:26 -> dropped
```

So the acquisition day's solution is filtered out. `merge_gps_insar`
(`workflows.py:42`) then matches the acquisition to the nearest remaining GPS
row within its 1-day tolerance, which is 2016-09-28 00:00, 23.6 h away. The
match isn't flagged.

The end bound is unaffected, because the last day's midnight row is still
`<=` the last acquisition time.

## Reproduction

```python
import pandas as pd
from geepers.gps_sources.unr import UnrSource
from geepers.workflows import merge_gps_insar

gps = pd.DataFrame({"date": pd.date_range("2016-09-25", "2016-09-30"), "east": range(6)})
start = pd.Timestamp("2016-09-27 00:26:23")
kept = UnrSource._filter_by_date(UnrSource.__new__(UnrSource), gps, start, None)
kept.date.dt.date.tolist()        # starts 2016-09-28: the 27th is gone
insar = pd.DataFrame({"los_insar": [0.0]}, index=pd.DatetimeIndex([start]))
merge_gps_insar(kept.set_index("date"), insar).los_insar.dropna().index[0]
# 2016-09-28, should be 2016-09-27
```

## Impact

- It affects the first epoch only, but that epoch is the DISP temporal
  reference (displacement 0). Any comparison that zeroes GPS at the first
  common epoch therefore carries that day's GPS noise as a constant offset
  over the whole series. On F08882 this was 3.9 mm at the 95th percentile
  across stations.
- Velocities are barely affected: at most 0.16 mm/yr against disp-xr on
  F08882, F11116 and F39362 after double differencing.
- Any acquisition time after midnight UTC hits this, which is every S1 and
  NISAR frame whose first acquisition isn't exactly at 00:00.

## Proposed fix

1. Compare calendar days in `_filter_by_date`: floor the start bound to the
   day (`_naive(start_date).normalize()`). Do it there rather than in
   `workflows.py`, so that every source and every caller is covered.
2. Done: when every GPS stamp is at midnight, `merge_gps_insar` matches each
   acquisition's floored day (exact for the day, nearest within `tolerance`
   if that day is missing). See the next section.

## Related: afternoon acquisitions get the next day's solution

`merge_gps_insar` picks the GPS row nearest in time. Daily solutions are
stamped at midnight, so for any acquisition after 12:00 UTC the nearest
stamp is the *next* day's. A UNR daily solution covers the whole UTC day,
so a 14:07 acquisition belongs to that day's solution. This affects every
epoch of such a frame, not just the first. Measured on DISP-S1, against
matching on the acquisition's own day:

| Frame | Acquisition (UTC) | Epoch GPS difference, median / p95 | Velocity difference, p95 / max |
| --- | --- | --- | --- |
| F08882 | 00:26 | 0 / 0 mm | 0 / 0 mm/yr |
| F11116 | 14:07 | 2.6 / 8.3 mm | 0.41 / 4.5 mm/yr |
| F39362 | 19:16 | 2.2 / 11.5 mm | 0.37 / 0.40 mm/yr |

disp-xr's `gnss.align` has the same behaviour, so the two tools agree with
each other while both being wrong. Fixed: for daily solutions, `merge_gps_insar` matches on the acquisition's
floored day (`insar_times.normalize()`). Floored days have no ties, unlike
centring the stamps at noon, where a 00:00 acquisition is 12 h from both
neighbours. disp-xr fixed `gnss.align` the same way (stamps centred at noon).

## Regression test

In `tests/gps_sources/`: a midnight-dated daily series and a start bound of
`2016-09-27 00:26:23` must keep the 2016-09-27 row. In `tests/` for
workflows: `merge_gps_insar` with that series must match the acquisition to
2016-09-27.
