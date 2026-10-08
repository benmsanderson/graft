"""Summarise an extended product where there is no LUH3 extension to compare with.

Global land groups at a few years, the harvest across the 2150 join, and the
plausibility checks that do not need a benchmark: no state below zero or above
the cell, primary land never gaining area, managed land static after 2150.

    python scripts/extend_summary.py ~/madrat/output/lx2_iamc
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import xarray as xr

from graft import paths

STATIC = str(paths.STATIC)
CROPS = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per")
GROUPS = {"primf": ("primf",), "secdf": ("secdf",), "primn": ("primn",), "secdn": ("secdn",),
          "crop": CROPS, "pastr": ("pastr",), "range": ("range",), "urban": ("urban",)}
MANAGED = CROPS + ("pastr", "range", "urban")


def years_of(ds: xr.Dataset) -> list[int]:
    units = ds.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    return list(np.round(epoch + ds.time.values / 365).astype(int))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    args = ap.parse_args()
    cell = xr.open_dataset(STATIC)["carea"].values.astype("float64") / 1e4      # Mha
    one = lambda sub, kind: xr.open_dataset(glob.glob(f"{args.folder}/{sub}/multiple-{kind}_*.nc")[0],
                                            decode_times=False)
    ramp, ext = one("annual", "states"), one("extension", "states")
    rampT, extT = one("annual", "transitions"), one("extension", "transitions")
    at = lambda y: (ramp, years_of(ramp).index(y)) if y < 2150 else (ext, years_of(ext).index(y))
    area = lambda y, names: sum(float(np.nansum(at(y)[0][v].isel(time=at(y)[1]).values * cell)) for v in names)

    print("Mha   " + "".join(f"{g:>8s}" for g in GROUPS))
    for y in (2100, 2125, 2150, 2200, 2300, 2500):
        print(f"{y}  " + "".join(f"{area(y, n):8.0f}" for n in GROUPS.values()))

    def harvest(y):
        ds, i = (rampT, y - 2020) if y < 2150 else (extT, y - 2150)
        total = sum(float(np.nansum(ds[v].isel(time=i).values)) for v in ds.data_vars if v.endswith("_bioh"))
        return total / 1e12, float(np.nansum(ds["primf_bioh"].isel(time=i).values)) / 1e12
    print("harvest PgC/yr, total (primary forest): " + "  ".join(
        f"{y} {harvest(y)[0]:.3f} ({harvest(y)[1]:.3f})" for y in (2100, 2149, 2150, 2151, 2200, 2300, 2499)))

    problems = []
    for name, ds in (("2020-2150", ramp), ("2150-2500", ext)):
        years = years_of(ds)
        for k in range(0, len(years), 10):
            s = ds.isel(time=k)
            total = sum(np.nan_to_num(s[v].values) for v in s.data_vars)
            low = min(float(np.nanmin(s[v].values)) for v in s.data_vars)
            if low < -1e-6 or float(total.max()) > 1 + 1e-4:
                problems.append(f"{years[k]}: state range {low:.2g}, cell total up to {float(total.max()):.5f}")
        for v in ("primf", "primn"):
            gain = max(float(np.nanmax(ds[v].isel(time=k + 1).values - ds[v].isel(time=k).values))
                       for k in range(0, len(years) - 1, 10))
            if gain > 1e-6:
                problems.append(f"{name}: {v} gains up to {gain:.2g} of a cell")
    moved = max(abs(area(2500, (v,)) - area(2150, (v,))) for v in MANAGED)
    if moved > 0.01:
        problems.append(f"managed land moves after 2150 by up to {moved:.2f} Mha")
    print("plausibility: " + ("ok" if not problems else "; ".join(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
