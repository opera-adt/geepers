# SPDX-FileCopyrightText: 2025-2026 California Institute of Technology ("Caltech")
# SPDX-License-Identifier: Apache-2.0
# Part of geepers, https://github.com/opera-adt/geepers. If you copy or adapt
# any of this code, keep this notice and cite the repository (see NOTICE).
"""UnrGridSource.download_data_files forwards its arguments to each download."""

from __future__ import annotations

from pathlib import Path

from geepers.gps_sources.unr_grid import UnrGridSource


def test_download_data_files_binds_arguments(tmp_path, monkeypatch):
    """Regression: the keyword arguments used to go to tqdm's thread_map,
    which rejected them, so every file failed with 'Unknown argument(s)'."""
    calls = []

    def fake_one(self, grid_id, plate, output_dir, session, version, gridded_type):
        calls.append((grid_id, plate, Path(output_dir), version, gridded_type))
        dest = Path(output_dir) / f"{int(grid_id):06d}_{plate}.tenv8"
        dest.write_text("2020.0 0 0 0 1 1 1 0\n")
        return dest

    monkeypatch.setattr(UnrGridSource, "_download_file", fake_one)
    src = UnrGridSource(version="0.3", gridded_type="constant", cache_dir=tmp_path)
    files = src.download_data_files(
        ["000001", "000002"],
        plate="IGS20",
        max_workers=2,
        output_dir=tmp_path,
        version="0.3",
        gridded_type="constant",
    )
    assert sorted(p.name for p in files) == ["000001_IGS20.tenv8", "000002_IGS20.tenv8"]
    assert sorted(c[0] for c in calls) == ["000001", "000002"]
    assert {c[1:] for c in calls} == {("IGS20", tmp_path, "0.3", "constant")}
