"""Compare an extended product with a LUH3 extension, year by selected year.

Global areas of the land groups and the wood harvest, from the ramp
(the annual product, to its last year) and the static period (the files
scripts/extend_product.py writes), against ``UofMD-landState-<key>-ext-3-1``.
LUH3 is a benchmark here, not a target.

    python scripts/extend_compare.py ~/madrat/output/vlx1_iamc vl
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import xarray as xr

from graft import paths

EXT = str(paths.LUH3_EXT)
STATIC = str(paths.STATIC)
CROPS = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per")
GROUPS = {"primf": ("primf",), "secdf": ("secdf", "pltns"), "primn": ("primn",), "secdn": ("secdn",),
          "crop": CROPS, "pastr": ("pastr",), "range": ("range",), "urban": ("urban",)}
YEARS = (2100, 2125, 2150, 2200, 2300, 2400, 2500)


def years_of(ds: xr.Dataset) -> list[int]:
    units = ds.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    t = ds.time.values
    return list(np.round(epoch + (t / 365 if units.startswith("days") else t)).astype(int))


class Product:
    """States and transitions from one or more folders, looked up by year."""

    def __init__(self, folders: list[str]):
        opened = lambda kind: [xr.open_dataset(f, decode_times=False)
                               for d in folders for f in sorted(glob.glob(f"{d}/multiple-{kind}_*.nc"))]
        # later folders win where years overlap: the static period restates its first year
        self.states = {y: (ds, i) for ds in opened("states") for i, y in enumerate(years_of(ds))}
        self.transitions = {y: (ds, i) for ds in opened("transitions") for i, y in enumerate(years_of(ds))}

    def area(self, year: int, names, cell: np.ndarray) -> float:
        ds, i = self.states[year]
        return sum(float(np.nansum(ds[v].isel(time=i).values * cell)) for v in names if v in ds) / 1e4

    def harvest(self, year: int) -> tuple[float, float]:
        """Harvested carbon, PgC, in total and from primary forest."""
        if year not in self.transitions:
            return float("nan"), float("nan")
        ds, i = self.transitions[year]
        total = sum(float(np.nansum(ds[v].isel(time=i).values)) for v in ds.data_vars if v.endswith("_bioh"))
        return total / 1e12, float(np.nansum(ds["primf_bioh"].isel(time=i).values)) / 1e12


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="the run's output folder, holding annual/ and extension/")
    ap.add_argument("key", choices=["vl", "h"])
    args = ap.parse_args()
    cell = xr.open_dataset(STATIC)["carea"].values.astype("float64") * 100     # km2 -> ha
    ours = Product([f"{args.folder}/annual", f"{args.folder}/extension"])
    luh = Product([f"{paths.LUH3_SCENARIOS}/UofMD-landState-{args.key}-3-1",
                   f"{EXT}/UofMD-landState-{args.key}-ext-3-1"])
    print("Mha, ours / LUH3;  harvest PgC/yr, total (primary forest)")
    print("year  " + "  ".join(f"{g:>15s}" for g in GROUPS) + "   harvest ours        harvest LUH3")
    for year in YEARS:
        cells = "  ".join(f"{ours.area(year, names, cell) / 1e2:7.0f}/{luh.area(year, names, cell) / 1e2:7.0f}"
                          for names in GROUPS.values())
        y = min(year, 2499)
        (a, ap_), (b, bp) = ours.harvest(y), luh.harvest(y)
        print(f"{year}  {cells}   {a:.3f} ({ap_:.3f})   {b:.3f} ({bp:.3f})", flush=True)


if __name__ == "__main__":
    main()
