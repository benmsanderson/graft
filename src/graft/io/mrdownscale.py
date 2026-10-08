"""Read mrdownscale's gridded land use onto the LUH grid.

mrdownscale's downscaled land use (``calcLandHighRes``) is a magpie object:
area in Mha per 0.25 degree cell, crops split by irrigation and biofuel
type. ``scripts/export_mrdownscale_states.R`` writes it as a CSV of x, y,
year and the LUH states, crop subtypes summed onto their LUH state. This
turns one year of that into LUH's own form - state fractions of cell area on
the full grid, NaN off the cells mrdownscale covers - so it can be scored by
exactly the code that reads LUH3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from graft import luh_schema as S


def read_states_csv(path: str | Path, static: xr.Dataset, year: int) -> xr.Dataset:
    """One year of an exported mrdownscale grid as LUH state fractions."""
    import pandas as pd

    cells = pd.read_csv(path)
    cells = cells[cells["year"] == year]
    if cells.empty:
        raise KeyError(f"year {year} not in {path}")

    lat = static["lat"].values
    lon = static["lon"].values
    # LUH's latitude runs north to south
    i = np.searchsorted(-lat, -cells["y"].values)
    j = np.searchsorted(lon, cells["x"].values)
    if not (np.allclose(lat[i], cells["y"].values) and np.allclose(lon[j], cells["x"].values)):
        raise ValueError(f"{path} is not on the grid of the static file")

    km2_per_mha = 1e4
    area = np.asarray(static[S.STATIC_CELL_AREA].values, dtype="float64") / km2_per_mha
    data = {}
    for state in S.AREA_STATES:
        if state not in cells:
            continue
        grid = np.full(area.shape, np.nan)
        grid[i, j] = cells[state].values / area[i, j]
        data[state] = (("lat", "lon"), grid)
    return xr.Dataset(data, coords={"lat": lat, "lon": lon})
