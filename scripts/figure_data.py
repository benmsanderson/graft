"""Reduce the released products and LUH3 to the small tables the figures need.

The slow half of making figures: everything that reads the 0.25 degree files.
Run it once per data version; the plotting notebook (paper repository,
``figures.ipynb``) reads only what this writes, and redraws in seconds.

Writes to ``--out``:

- ``states.csv``: global and latitude-band area (Mha) of every land state, per
  source (``mrdownscale`` or ``LUH3``), marker and year, 2015-2500;
- ``harvest.csv``: wood harvest carbon (Pg C/yr) and area (Mha/yr) per source
  pool, source, marker and year;
- ``maps.nc``: forest, cropland, pasture and natural-land fractions in 2050 and
  2100 for VL and H, both sources;
- ``skill.csv``: per-group gaps and cell correlations against LUH3 (VL, H;
  2050, 2100), as ``graft.compare.score_cells`` computes them;
- ``manifest.json``: the datasets read, their checksums file, graft's commit.

    python scripts/figure_data.py --release REL --out ../graft-paper/figure-data
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from graft import compare, paths

MARKERS = ["vl", "l", "ln", "m", "ml", "h", "hl"]
BENCHMARKED = ["vl", "h"]
BLOCK = 25
MAP_YEARS = (2050, 2100)
MAP_FIELDS = {"forest": ("primf", "secdf", "pltns"), "cropland": ("c3ann", "c3nfx", "c3per", "c4ann", "c4per"),
              "pasture": ("pastr",), "rangeland": ("range",), "primn": ("primn",), "secdn": ("secdn",)}


def years_of(ds: xr.Dataset) -> np.ndarray:
    units = ds.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    t = np.asarray(ds.time.values, dtype="float64")
    return np.round(epoch + (t / 365 if units.startswith("days") else t)).astype(int)


def only(pattern: str) -> str:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {len(found)}")
    return found[0]


def reduce_states(path: str, keep, weights: dict[str, np.ndarray]) -> list[dict]:
    """Area of every land-fraction variable, per year and region weight."""
    ds = xr.open_dataset(path, decode_times=False)
    years = years_of(ds)
    idx = [i for i, y in enumerate(years) if keep(y)]
    rows = []
    for v in ds.data_vars:
        if "time" not in ds[v].dims or v in ("secma", "secmb") or "bnds" in v:
            continue
        for start in range(0, len(idx), BLOCK):
            part = idx[start:start + BLOCK]
            x = np.nan_to_num(np.asarray(ds[v].isel(time=part).values, dtype="float64"))
            for name, w in weights.items():
                totals = np.tensordot(x, w, axes=([1, 2], [0, 1]))
                rows += [{"year": int(years[i]), "variable": v, "region": name, "Mha": float(t)}
                         for i, t in zip(part, totals)]
    return rows


def reduce_harvest(path: str, keep, cell: np.ndarray) -> list[dict]:
    """Harvest carbon (Pg C/yr) and area (Mha/yr) per source pool."""
    ds = xr.open_dataset(path, decode_times=False)
    years = years_of(ds)
    idx = [i for i, y in enumerate(years) if keep(y)]
    rows = []
    pools = sorted({v[:-5] for v in ds.data_vars if v.endswith("_bioh") or v.endswith("_harv")})
    for p in pools:
        for start in range(0, len(idx), BLOCK):
            part = idx[start:start + BLOCK]
            out = {}
            for kind, weight, scale in (("bioh", None, 1e-12), ("harv", cell, 1.0)):
                name = f"{p}_{kind}"
                if name not in ds:
                    continue
                x = np.nan_to_num(np.asarray(ds[name].isel(time=part).values, dtype="float64"))
                out[kind] = (x.sum(axis=(1, 2)) if weight is None else np.tensordot(x, weight, axes=([1, 2], [0, 1]))) * scale
            for k, i in enumerate(part):
                rows.append({"year": int(years[i]), "pool": p,
                             "PgC": float(out["bioh"][k]) if "bioh" in out else np.nan,
                             "Mha": float(out["harv"][k]) if "harv" in out else np.nan})
    return rows


def state_at(path: str, year: int) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=False)
    return ds.isel(time=int(np.where(years_of(ds) == year)[0][0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", required=True, help="folder holding the packaged release datasets")
    ap.add_argument("--version", default="0-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps-only", action="store_true",
                    help="redo maps.nc and skill.csv only, keeping the series tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    static = xr.open_dataset(paths.STATIC)
    cell = np.asarray(static["carea"].values, dtype="float64") / 1e4          # Mha per unit fraction
    lat = static["lat"].values
    a = np.abs(lat)[:, None] * np.ones((1, static.sizes["lon"]))
    weights = {"global": cell, "boreal": cell * (lat[:, None] > 50),
               "temperate": cell * ((a >= 23) & (a <= 50)), "tropical": cell * (a < 23)}

    rel = Path(args.release)
    sources = []   # (source, marker, kind, path, keep)
    for m in MARKERS:
        main_id = f"CICERO-graft-landState-{m}-{args.version}"
        ext_id = f"CICERO-graft-landState-{m}-ext-{args.version}"
        sources += [("mrdownscale", m, k, only(f"{rel}/{main_id}/multiple-{k}_*.nc"), lambda y: True)
                    for k in ("states", "transitions")]
        sources += [("mrdownscale", m, k, only(f"{rel}/{ext_id}/multiple-{k}_*.nc"), lambda y: y > 2100)
                    for k in ("states", "transitions")]
    history = lambda kind: only(f"{paths.LUH3}/multiple-{kind}_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_*.nc")
    for m in BENCHMARKED:
        sources += [("LUH3", m, k, history(k), lambda y: 2015 <= y <= 2021) for k in ("states", "transitions")]
        sources += [("LUH3", m, k, only(f"{paths.LUH3_SCENARIOS}/UofMD-landState-{m}-3-1/multiple-{k}_*.nc"),
                     lambda y: True) for k in ("states", "transitions")]
        sources += [("LUH3", m, k, only(f"{paths.LUH3_EXT}/UofMD-landState-{m}-ext-3-1/multiple-{k}_*.nc"),
                     lambda y: y > 2100) for k in ("states", "transitions")]

    states, harvest = [], []
    for source, m, kind, path, keep in ([] if args.maps_only else sources):
        print(f"{source} {m} {kind}: {Path(path).name}", flush=True)
        rows = reduce_states(path, keep, weights) if kind == "states" else reduce_harvest(path, keep, cell)
        for r in rows:
            r.update(source=source, marker=m)
        (states if kind == "states" else harvest).extend(rows)
    cols = ["source", "marker", "year"]
    if not args.maps_only:
        pd.DataFrame(states)[cols + ["variable", "region", "Mha"]].to_csv(out / "states.csv", index=False, float_format="%.4f")
        pd.DataFrame(harvest)[cols + ["pool", "PgC", "Mha"]].to_csv(out / "harvest.csv", index=False, float_format="%.5f")

    # maps and skill, VL and H
    fields, skill = {}, []
    for m in BENCHMARKED:
        ours = only(f"{rel}/CICERO-graft-landState-{m}-{args.version}/multiple-states_*.nc")
        luh = only(f"{paths.LUH3_SCENARIOS}/UofMD-landState-{m}-3-1/multiple-states_*.nc")
        for year in MAP_YEARS:
            o, r = state_at(ours, year), state_at(luh, year)
            for source, ds in (("mrdownscale", o), ("LUH3", r)):
                # LUH3 stores its plantation variables as no-data: count missing as zero on
                # land, and keep the ocean missing
                land = np.isfinite(np.asarray(ds["primf"].values))
                for name, vs in MAP_FIELDS.items():
                    x = sum(np.nan_to_num(np.asarray(ds[v].values, dtype="float32")) for v in vs if v in ds)
                    fields[f"{name}_{m}_{source}_{year}"] = (("lat", "lon"), np.where(land, x, np.nan))
            t = compare.score_cells(o, r, cell * 1e4).reset_index()
            t.insert(0, "year", year)
            t.insert(0, "marker", m)
            skill.append(t)
    xr.Dataset(fields, coords={"lat": static["lat"], "lon": static["lon"]}).to_netcdf(
        out / "maps.nc", encoding={k: {"zlib": True, "complevel": 5} for k in fields})
    pd.concat(skill).to_csv(out / "skill.csv", index=False, float_format="%.4f")

    commit = subprocess.run(["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    sums = {p.parent.name: p.read_text() for p in sorted(rel.glob("*/SHA256SUMS"))}
    json.dump({"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "graft_commit": commit, "release": str(rel), "release_checksums": sums,
               "luh3": sorted({Path(p).name for s, _, _, p, _ in sources if s == "LUH3"})},
              open(out / "manifest.json", "w"), indent=1)
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
