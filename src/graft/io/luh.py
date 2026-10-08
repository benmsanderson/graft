"""Read LUH3 (CMIP7) gridded forcing.

Thin, explicit wrappers over :func:`xarray.open_dataset`. The point is not to hide
xarray but to (a) give the four LUH3 components names, (b) keep reads lazy so a
0.25 deg scenario never lands in memory whole, and (c) expose a per-year iterator
that the validator and (later) the bookkeeping model both stream over.

No dask required: xarray + the netCDF4 backend read a single ``isel(time=i)`` slice
straight from disk, so streaming years keeps memory bounded on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

# open_dataset kwargs shared by every LUH3 read. decode_times stays on so the
# time coordinate is usable; mask_and_scale on so _FillValue becomes NaN.
_OPEN_KW = dict(chunks=None, decode_times=True, mask_and_scale=True)


def _open(path: str | Path) -> xr.Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LUH3 file not found: {path}")
    try:
        return xr.open_dataset(path, **_OPEN_KW)
    except ValueError as error:
        if "unable to decode time units" not in str(error):
            raise
        # mrdownscale writes "years since 1970-01-01" on a 365_day calendar,
        # which cftime refuses - "years" is not a unit it accepts for a
        # non-standard calendar. select_year reads such an axis by hand.
        return xr.open_dataset(path, **{**_OPEN_KW, "decode_times": False})


def open_states(path: str | Path) -> xr.Dataset:
    """Open a LUH3 ``multiple-states`` file (land-state fractions + secma/secmb)."""
    return _open(path)


def open_transitions(path: str | Path) -> xr.Dataset:
    """Open a LUH3 ``multiple-transitions`` file (``X_to_Y`` area fluxes)."""
    return _open(path)


def open_management(path: str | Path) -> xr.Dataset:
    """Open a LUH3 ``multiple-management`` file (wood harvest, fertiliser, ...)."""
    return _open(path)


def open_static(path: str | Path) -> xr.Dataset:
    """Open ``staticData_quarterdeg.nc`` (carea, icwtr, ptbio, ccode, ...)."""
    return _open(path)


@dataclass
class LUHScenario:
    """One LUH3 scenario as its component datasets plus the static grid.

    Construct with :meth:`from_paths`. Datasets are opened lazily and should be
    closed with :meth:`close` (or use as a context manager).
    """

    states: xr.Dataset
    transitions: xr.Dataset
    management: xr.Dataset | None = None
    static: xr.Dataset | None = None

    @classmethod
    def from_paths(
        cls,
        states: str | Path,
        transitions: str | Path,
        management: str | Path | None = None,
        static: str | Path | None = None,
    ) -> "LUHScenario":
        return cls(
            states=open_states(states),
            transitions=open_transitions(transitions),
            management=open_management(management) if management is not None else None,
            static=open_static(static) if static is not None else None,
        )

    # -- streaming -----------------------------------------------------------
    @property
    def years(self) -> list[int]:
        """Integer years present in the *states* file, in order."""
        t = self.states["time"]
        try:
            return [int(v) for v in t.dt.year.values]
        except (TypeError, AttributeError):
            return [int(v) for v in t.values]

    def transition_years(self) -> list[int]:
        """Years for which a transitions slice exists (states has one more)."""
        t = self.transitions["time"]
        try:
            return [int(v) for v in t.dt.year.values]
        except (TypeError, AttributeError):
            return [int(v) for v in t.values]

    def iter_state_steps(
        self, *, stride: int = 1, max_steps: int | None = None
    ) -> Iterator[tuple[int, xr.Dataset, xr.Dataset, xr.Dataset]]:
        """Yield ``(year, states[t], states[t+1], transitions[t])``, loaded.

        Aligns the ``t`` transitions slice with the state pair it drives. Each
        yielded triple is fully loaded (``.load()``) so downstream code operates
        on numpy-backed arrays and the previous year's slices can be released.

        ``stride``/``max_steps`` subsample the time axis and, importantly, only
        the yielded indices are read from disk -- a strided pass over a 0.25 deg
        scenario does not pay to load the years it skips.
        """
        n_trans = self.transitions.sizes["time"]
        n_states = self.states.sizes["time"]
        n = min(n_trans, n_states - 1)
        tr_years = self.transition_years()
        yielded = 0
        for i in range(0, n, stride):
            if max_steps is not None and yielded >= max_steps:
                break
            s0 = self.states.isel(time=i).load()
            s1 = self.states.isel(time=i + 1).load()
            tr = self.transitions.isel(time=i).load()
            yielded += 1
            yield tr_years[i], s0, s1, tr

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        for ds in (self.states, self.transitions, self.management, self.static):
            if ds is not None:
                ds.close()

    def __enter__(self) -> "LUHScenario":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def select_year(ds: xr.Dataset, year: int) -> xr.Dataset:
    """One year of a LUH file, time dimension dropped.

    LUH3 times decode to ``noleap`` cftime dates (850-2024 historic,
    2022-2100 for the scenarios), so match on the year. Plain integer years,
    as in the test fixtures, match directly.
    """
    time = ds["time"]
    if hasattr(time, "dt") and time.dtype.kind in "OM":
        years = time.dt.year.values
    else:
        # an undecoded axis: "years since 1970-01-01" counts years from 1970,
        # anything else is already a year
        units = str(time.attrs.get("units", ""))
        offset = 0
        if units.startswith("years since"):
            offset = int(units.split("since")[1].strip()[:4])
        years = time.values + offset
    matches = [i for i, y in enumerate(years) if int(y) == year]
    if not matches:
        raise KeyError(f"year {year} not in file ({int(years[0])}-{int(years[-1])})")
    return ds.isel(time=matches[0])


def region_of_cell(mask: str | Path, mapping: str | Path, static: xr.Dataset):
    """Row and column indices of every masked land cell, and its region.

    ``mask`` is the ``x,y,country`` table :mod:`scripts.luh_country_mask`
    builds from LUH's own ``ccode``, and ``mapping`` a ``region,country``
    table for one model. Returns ``(i, j, region)``, so that ``field[i, j]``
    takes any grid field to the masked cells in the order the regions name
    them.
    """
    import pandas as pd

    cells = pd.read_csv(mask).merge(pd.read_csv(mapping), on="country")
    lat, lon = static["lat"].values, static["lon"].values
    i = np.searchsorted(-lat, -cells["y"].values)
    j = np.searchsorted(lon, cells["x"].values)
    if not (np.allclose(lat[i], cells["y"]) and np.allclose(lon[j], cells["x"])):
        raise ValueError("the mask is not on the grid of the static file")
    return i, j, cells["region"].values


def state_years(ds: xr.Dataset) -> list[int]:
    """The calendar years of a states dataset, decoded or not."""
    values = ds["time"].values
    if np.issubdtype(np.asarray(values).dtype, np.number):
        units = str(ds["time"].attrs.get("units", ""))
        epoch = int(units.split("since")[-1].strip().split("-")[0]) if "since" in units else 0
        if units.strip().startswith("days"):
            return [int(round(epoch + v / 365.0)) for v in values]
        return [int(round(epoch + v)) for v in values]
    return [int(getattr(v, "year", np.datetime64(v, "Y").astype(int) + 1970)) for v in values]
