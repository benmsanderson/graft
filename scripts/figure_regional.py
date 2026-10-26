"""Regional change in each land group over 2025-2100: the IAM, LUH3 and the products.

A companion to ``figure_data.py`` for the three-way comparison of regional
amounts. For each marker and each R10 region of the marker's model, writes the
change over 2025-2100 (Mha) of cropland, pasture (managed pasture and
rangeland), forest and other natural land, as the IAM reports it (from the
regional release), as the product has it, and as LUH3 has it where LUH3 has
published the marker. Forest includes plantations in LUH3 (``secdf + pltns``),
as the products fold them. Writes ``regional.csv`` to ``--out``.

    python scripts/figure_regional.py --release REL --version 0-2 --iamc land_r10.csv --out ../graft-paper/figure-data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from graft import paths
from figure_data import BENCHMARKS, MARKERS, MODELS, only, region_map, years_of
from package_release import PREFIX

YEARS = (2025, 2100)
GROUPS = {"cropland": ("c3ann", "c3nfx", "c3per", "c4ann", "c4per"), "pasture": ("pastr", "range"),
          "forest": ("primf", "secdf", "pltns"), "other natural": ("primn", "secdn")}
IAMC = {"cropland": "Land Cover|Cropland", "pasture": "Land Cover|Pasture",
        "forest": "Land Cover|Forest", "other natural": "Land Cover|Other Natural"}


def gridded(path: str, reg: np.ndarray, names: list[str], cell: np.ndarray) -> list[dict]:
    ds = xr.open_dataset(path, decode_times=False)
    index = {int(y): i for i, y in enumerate(years_of(ds))}
    rows = []
    for g, states in GROUPS.items():
        total = {}
        for y in YEARS:
            x = sum(np.nan_to_num(np.asarray(ds[v].isel(time=index[y]).values, dtype="float64"))
                    for v in states if v in ds)
            total[y] = np.bincount(reg[reg >= 0], weights=(x * cell)[reg >= 0], minlength=len(names))
        rows += [{"group": g, "region": n, "Mha": float(total[YEARS[1]][i] - total[YEARS[0]][i])}
                 for i, n in enumerate(names)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", required=True, help="folder holding the packaged release datasets")
    ap.add_argument("--version", default="0-2")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--iamc", required=True, help="the regional release's land subset (iamc_coverage.py --export)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    static = xr.open_dataset(paths.STATIC)
    cell = np.nan_to_num(np.asarray(static["carea"].values, dtype="float64")) / 1e4   # Mha per unit fraction
    iamc = pd.read_csv(args.iamc)
    iamc = iamc[iamc.Scenario.str.contains("(Marker)", regex=False)]

    rows = []
    for mk in MARKERS:
        print(mk, flush=True)
        reg, names = region_map(mk, static["lat"].values, static["lon"].values)
        model = iamc[iamc.Model == MODELS[mk].replace("_", " ")]
        for g, v in IAMC.items():
            d = model[model.Variable == v].set_index("Region")
            rows += [{"source": "IAM", "marker": mk, "group": g, "region": n,
                      "Mha": float(d.loc[n, str(YEARS[1])] - d.loc[n, str(YEARS[0])])}
                     for n in names if n in d.index]
        rel = only(f"{args.release}/{args.prefix}-{mk}-{args.version}/multiple-states_*.nc")
        rows += [{"source": "mrdownscale", "marker": mk, **r} for r in gridded(rel, reg, names, cell)]
        if mk in BENCHMARKS:
            luh = only(f"{paths.LUH3_SCENARIOS}/{BENCHMARKS[mk][0]}/multiple-states_*.nc")
            rows += [{"source": "LUH3", "marker": mk, **r} for r in gridded(luh, reg, names, cell)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "regional.csv", index=False, float_format="%.3f")
    print("wrote regional.csv")


if __name__ == "__main__":
    main()
