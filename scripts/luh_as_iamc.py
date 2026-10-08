"""Write a LUH3 product back out as an IAMC release, for the round trip.

The round trip: take a published LUH3 scenario, aggregate it to a
model's regions, and run it back through the pipeline as if an IAM had
reported it. The regional totals are then the target's own, so whatever
remains when the result is scored against that same product is what the
downscaling step costs - the number every other gap should be read against.

The land tree is written from the states file, in the variables readIAMC
expects. Fertiliser comes from the target's own management history, held
constant, and wood harvest is left out entirely, which makes the reader hold
the target's harvest (both say "no scenario information", which is true
here). Needs pandas, xarray, netCDF4:

    python scripts/luh_as_iamc.py STATES.nc --static STATIC.nc \\
        --mask country_cell.csv.gz --mapping REMIND-MAgPIE_..._R10.csv \\
        --management MANAGEMENT.nc --out land_roundtrip.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from graft.io import luh

# the LUH states behind each IAMC variable readIAMC reads
TREE = {
    "Land Cover": ["primf", "secdf", "primn", "secdn", "urban", "pastr", "range",
                   "c3ann", "c4ann", "c3per", "c4per", "c3nfx"],
    "Land Cover|Cropland": ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"],
    "Land Cover|Pasture": ["pastr", "range"],
    "Land Cover|Forest": ["primf", "secdf"],
    "Land Cover|Forest|Primary": ["primf"],
    "Land Cover|Forest|Secondary": ["secdf"],
    "Land Cover|Other Natural": ["primn", "secdn"],
    "Land Cover|Built-Up Area": ["urban"],
}
CROPS = ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"]


def region_of_cell(mask: Path, mapping: Path, static: xr.Dataset):
    """Row and column indices of every masked cell, and its region."""
    cells = pd.read_csv(mask).merge(pd.read_csv(mapping), on="country")
    lat, lon = static["lat"].values, static["lon"].values
    i = np.searchsorted(-lat, -cells["y"].values)
    j = np.searchsorted(lon, cells["x"].values)
    if not (np.allclose(lat[i], cells["y"]) and np.allclose(lon[j], cells["x"])):
        raise ValueError("the mask is not on the grid of the static file")
    return i, j, cells["region"].values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("states", help="LUH3 scenario states file")
    ap.add_argument("--static", required=True)
    ap.add_argument("--mask", required=True, help="x,y,country from luh_country_mask.py")
    ap.add_argument("--mapping", required=True, help="region,country for the model")
    ap.add_argument("--management", help="LUH3 historic management, for the fertiliser level")
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2025, 2030, 2035, 2040, 2045, 2050, 2060, 2070, 2080, 2090, 2100])
    ap.add_argument("--model", default="LUH3-VL")
    ap.add_argument("--scenario", default="Round trip (LUH3)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    static = luh.open_static(args.static)
    carea = np.asarray(static["carea"].values, dtype="float64")
    i, j, region = region_of_cell(Path(args.mask), Path(args.mapping), static)
    states = luh.open_states(args.states)

    rows = []
    for year in args.years:
        one = luh.select_year(states, year)
        area = {s: np.nan_to_num(one[s].values) * carea / 1e4 for s in TREE["Land Cover"]}
        for variable, members in TREE.items():
            cell = sum(area[m] for m in members)[i, j]
            for name, value in pd.Series(cell).groupby(region).sum().items():
                rows.append({"Model": args.model, "Scenario": args.scenario, "Region": name,
                             "Variable": variable, "Unit": "million ha", "Year": year,
                             "Value": value})
        # LUH3 reports no plantations, and its states do not separate energy
        # crops; both are zero rather than absent, so the reader does not
        # mistake them for something it should hold at the target's value
        for variable in ("Land Cover|Forest|Planted", "Land Cover|Cropland|Energy Crops"):
            for name in pd.unique(region):
                rows.append({"Model": args.model, "Scenario": args.scenario, "Region": name,
                             "Variable": variable, "Unit": "million ha", "Year": year, "Value": 0.0})

    if args.management:
        # the target's own fertiliser, held constant: a round trip should say
        # nothing the target does not say
        management = luh.open_states(args.management)
        last = luh.select_year(management, int(management["time"].dt.year.values.max()))
        latest = luh.select_year(states, max(args.years))
        applied = sum(np.nan_to_num(last[f"fertl_{c}"].values) * np.nan_to_num(latest[c].values)
                      for c in CROPS) * carea * 100 / 1e9  # kg/ha * ha -> Tg
        for name, value in pd.Series(applied[i, j]).groupby(region).sum().items():
            for year in args.years:
                rows.append({"Model": args.model, "Scenario": args.scenario, "Region": name,
                             "Variable": "Fertilizer Use|Nitrogen|Synthetic", "Unit": "Tg N/yr",
                             "Year": year, "Value": value})

    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["Model", "Scenario", "Region", "Variable", "Unit"],
                            columns="Year", values="Value").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {wide['Region'].nunique()} regions, "
          f"{wide['Variable'].nunique()} variables, {len(args.years)} years")
    print(f"global land {long[long.Variable == 'Land Cover'].groupby('Year').Value.sum().iloc[0]:.1f} Mha, "
          f"forest {long[long.Variable == 'Land Cover|Forest'].groupby('Year').Value.sum().iloc[0]:.1f} Mha "
          f"in {args.years[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
