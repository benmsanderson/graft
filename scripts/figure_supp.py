"""Reduce the released products and LUH3 to the tables the supplementary figures need.

A companion to ``figure_data.py``, for the figures that answer questions about
the join at 2025, gross turnover and management. Writes to ``--out``:

- ``join.csv``: for each one-year step, the share of land cells whose largest
  change in any land state (fraction of the cell) exceeds each threshold, per
  source (``mrdownscale``, ``LUH3``; LUH3 history as marker ``history``),
  marker and step;
- ``turnover.csv``: global gross transitions (Mha/yr) per source, marker and
  year, split into crop-to-crop flows, other flows that return within a cell
  (twice the smaller of each pair a->b, b->a), and the rest;
- ``management.csv``: global irrigated area, first- and second-generation
  bioenergy crop area, flooded area (Mha) and fertilizer (Tg N/yr) per source,
  marker and year;
- ``supp_maps.nc``, for each marker LUH3 has published, both sources: the
  largest one-year change of any land state over 2024-2026 (``join_*``), and
  the within-cell return flows other than crop-to-crop in 2030
  (``turnover_*``), as fractions of the cell (per year).

Secondary forest includes plantations in LUH3 (``secdf + pltns``), as the
products fold them.

    python scripts/figure_supp.py --release REL --version 0-2 --out ../graft-paper/figure-data
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from graft import paths
from figure_data import BENCHMARKS, MARKERS, only, years_of
from package_release import PREFIX

LAND = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per", "pastr", "primf", "primn", "range", "secdf", "secdn", "urban")
CROPS = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per")
THRESHOLDS = np.round(np.logspace(-4, 0, 81), 8)
JOIN_STEPS = {"mrdownscale": [(2023, 2024), (2024, 2025), (2025, 2026), (2030, 2031)],
              "LUH3": [(2024, 2025), (2025, 2026), (2030, 2031)],
              "history": [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]}
TURNOVER_YEARS = {"mrdownscale": [2020, 2023, 2024, 2025, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2099],
                  "LUH3": [2024, 2025, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2099],
                  "history": [2000, 2005, 2010, 2015, 2020, 2023]}
MAP_TURNOVER_YEAR = 2030
MANAGEMENT_YEARS = list(range(2015, 2101, 5)) + [2024]


class Years:
    """One file's fields by year, with LUH3's plantations folded into secondary forest."""

    def __init__(self, path: str):
        self.ds = xr.open_dataset(path, decode_times=False)
        self.index = {int(y): i for i, y in enumerate(years_of(self.ds))}

    def __contains__(self, year: int) -> bool:
        return year in self.index

    def get(self, v: str, year: int) -> np.ndarray:
        if v not in self.ds:
            return np.zeros((self.ds.sizes["lat"], self.ds.sizes["lon"]))
        x = np.nan_to_num(np.asarray(self.ds[v].isel(time=self.index[year]).values, dtype="float64"))
        if v == "secdf" and "pltns" in self.ds:
            x = x + self.get("pltns", year)
        return x


def largest_change(s: Years, a: int, b: int) -> np.ndarray:
    return np.max([np.abs(s.get(v, b) - s.get(v, a)) for v in LAND], axis=0)


def exceedance(change: np.ndarray, land: np.ndarray) -> np.ndarray:
    x = np.sort(change[land])
    return 1 - np.searchsorted(x, THRESHOLDS, side="right") / x.size


def turnover(t: Years, year: int, cell: np.ndarray) -> tuple[dict, np.ndarray]:
    """Global gross transitions by kind (Mha/yr), and the cell field of non-crop return flows."""
    names = [v for v in t.ds.data_vars if "_to_" in v]
    f = {v: t.get(v, year) for v in names}
    total = sum(float((x * cell).sum()) for x in f.values())
    crop = sum(float((f[v] * cell).sum()) for v in names
               if v.split("_to_")[0] in CROPS and v.split("_to_")[1] in CROPS)
    back = np.zeros_like(cell)
    for v in names:
        a, b = v.split("_to_")
        w = f"{b}_to_{a}"
        if w in f and a < b and not (a in CROPS and b in CROPS):
            back += 2 * np.minimum(f[v], f[w])
    ret = float((back * cell).sum())
    return {"total": total, "crop_to_crop": crop, "returning": ret, "other": total - crop - ret}, back


def management(s: Years, m: Years, year: int, cell: np.ndarray) -> dict:
    crop = {c: s.get(c, year) for c in CROPS}
    out = {}
    for k in ("irrig", "cpbf1", "cpbf2"):
        out[k] = sum(float((m.get(f"{k}_{c}", year) * crop[c] * cell).sum()) for c in CROPS)
    out["flood"] = float((m.get("flood", year) * crop["c3ann"] * cell).sum()) if "flood" in m.ds else np.nan
    # fertilizer, kg N/ha/yr on each crop's area: Mha x kg/ha = 1e6 x kg = 1e-3 Tg
    out["fertl"] = sum(float((m.get(f"fertl_{c}", year) * crop[c] * cell).sum()) for c in CROPS) * 1e-3
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", required=True, help="folder holding the packaged release datasets")
    ap.add_argument("--version", default="0-1")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    static = xr.open_dataset(paths.STATIC)
    cell = np.nan_to_num(np.asarray(static["carea"].values, dtype="float64")) / 1e4   # Mha per unit fraction
    rel = Path(args.release)
    history = lambda kind: only(f"{paths.LUH3}/multiple-{kind}_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_*.nc")
    files = {("history", "history"): {k: history(k) for k in ("states", "transitions", "management")}}
    for mk in MARKERS:
        files[("mrdownscale", mk)] = {k: only(f"{rel}/{args.prefix}-{mk}-{args.version}/multiple-{k}_*.nc")
                                      for k in ("states", "transitions", "management")}
    for mk, (scen, _) in BENCHMARKS.items():
        files[("LUH3", mk)] = {k: only(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-{k}_*.nc")
                               for k in ("states", "transitions")}
        found = glob.glob(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-management_*.nc")
        files[("LUH3", mk)]["management"] = found[0] if len(found) == 1 else None
    land = np.isfinite(np.asarray(xr.open_dataset(history("states"), decode_times=False)["primf"].isel(time=-1).values))

    join, turn, mgmt, maps = [], [], [], {}
    for (source, mk), f in files.items():
        print(source, mk, flush=True)
        s, t = Years(f["states"]), Years(f["transitions"])
        m = Years(f["management"]) if f["management"] else None
        if m is None:
            print(f"  no management file for {source} {mk}: management skipped", flush=True)
        key = "history" if source == "history" else source
        source_label = "LUH3" if source == "history" else source
        for a, b in JOIN_STEPS[key]:
            if a in s and b in s:
                share = exceedance(largest_change(s, a, b), land)
                join += [{"source": source_label, "marker": mk, "step": f"{a}-{b}", "threshold": th, "share": sh}
                         for th, sh in zip(THRESHOLDS, share)]
        for y in TURNOVER_YEARS[key]:
            if y in t:
                kinds, back = turnover(t, y, cell)
                turn.append({"source": source_label, "marker": mk, "year": y, **kinds})
                if y == MAP_TURNOVER_YEAR and mk in BENCHMARKS:
                    maps[f"turnover_{mk}_{source}"] = (("lat", "lon"), np.where(land, back, np.nan).astype("float32"))
        for y in MANAGEMENT_YEARS:
            if m is not None and y in s and y in m and (source != "history" or y <= 2024):
                mgmt.append({"source": source_label, "marker": mk, "year": y, **management(s, m, y, cell)})
        if mk in BENCHMARKS:
            big = np.max([largest_change(s, a, a + 1) for a in (2024, 2025)], axis=0)
            maps[f"join_{mk}_{source}"] = (("lat", "lon"), np.where(land, big, np.nan).astype("float32"))

    pd.DataFrame(join).to_csv(out / "join.csv", index=False, float_format="%.6g")
    pd.DataFrame(turn).sort_values(["source", "marker", "year"]).to_csv(out / "turnover.csv", index=False, float_format="%.3f")
    pd.DataFrame(mgmt).sort_values(["source", "marker", "year"]).to_csv(out / "management.csv", index=False, float_format="%.3f")
    enc = {k: {"zlib": True, "complevel": 5, "least_significant_digit": 4} for k in maps}
    xr.Dataset(maps, coords={"lat": static["lat"], "lon": static["lon"]},
               attrs={"turnover_year": MAP_TURNOVER_YEAR, "join_steps": "2024-2025, 2025-2026"}).to_netcdf(
        out / "supp_maps.nc", encoding=enc)
    print("wrote join.csv, turnover.csv, management.csv, supp_maps.nc")


if __name__ == "__main__":
    main()
