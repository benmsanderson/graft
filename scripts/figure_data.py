"""Reduce the released products and LUH3 to the small tables the figures need.

The slow half of making figures: everything that reads the 0.25 degree files.
Run it once per data version; the plotting notebook (paper repository,
``figures.ipynb``) reads only what this writes, and redraws in seconds.

Writes to ``--out``:

- ``states.csv``: global and latitude-band area (Mha) of every land state, per
  source (``mrdownscale`` or ``LUH3``), marker and year, 2015-2500;
- ``harvest.csv``: wood harvest carbon (Pg C/yr) and area (Mha/yr) per source
  pool, source, marker and year;
- ``maps.nc``, for each marker LUH3 has published (VL, H, M, HL), both
  sources: forest, primary forest, cropland, pasture and rangeland fractions
  and secondary mean age in 2024 and 2100; cumulative wood harvest carbon
  2025-2099 (all and primary, kg C/m2); and the model's R10 region of each
  cell (``region_<marker>``, names in its ``regions`` attribute);
- ``skill.csv``: per-group gaps and cell correlations against LUH3 (the same
  markers; 2050, 2100), as ``graft.compare.score_cells`` computes them;
- ``manifest.json``: the datasets read, their checksums file, graft's commit,
  and for each LUH3 scenario how far its first years (2022-2024) are from the
  history the products are harmonized to: zero, or the script stops, so every
  comparison is against the same starting point.

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
from package_release import PREFIX

MARKERS = ["vl", "l", "ln", "m", "ml", "h", "hl"]
# the LUH3 scenarios published on input4MIPs: marker -> (2020-2100 dataset, extension or None)
BENCHMARKS = {"vl": ("UofMD-landState-vl-3-1-1", "UofMD-landState-vl-ext-3-1"),
              "h": ("UofMD-landState-h-3-1-1", "UofMD-landState-h-ext-3-1"),
              "m": ("UofMD-landState-m-3-1", "UofMD-landState-m-ext-3-1"),
              "hl": ("UofMD-landState-hl-3-1", None)}
MODELS = {"vl": "REMIND-MAgPIE_3.5-4.11", "l": "MESSAGEix-GLOBIOM-GAINS_2.1-M-R12", "ln": "AIM_3.0",
          "m": "IMAGE_3.4", "ml": "COFFEE_1.6", "h": "GCAM_8s", "hl": "WITCH_6.0"}
BLOCK = 25
MAP_YEARS = (2024, 2100)
HARVEST_YEARS = (2025, 2099)
MAP_FIELDS = {"forest": ("primf", "secdf", "pltns"), "primf": ("primf",),
              "cropland": ("c3ann", "c3nfx", "c3per", "c4ann", "c4per"),
              "pasture": ("pastr",), "rangeland": ("range",)}


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


def cumulative_harvest(path: str, area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wood harvest carbon over HARVEST_YEARS per unit cell area (kg C/m2): all pools, and primary land."""
    ds = xr.open_dataset(path, decode_times=False)
    years = years_of(ds)
    idx = [i for i, y in enumerate(years) if HARVEST_YEARS[0] <= y <= HARVEST_YEARS[1]]
    total, primary = np.zeros(area.shape), np.zeros(area.shape)
    for v in (v for v in ds.data_vars if v.endswith("_bioh")):
        for start in range(0, len(idx), BLOCK):
            x = np.nan_to_num(np.asarray(ds[v].isel(time=idx[start:start + BLOCK]).values, dtype="float64")).sum(axis=0)
            total += x
            if v.startswith("prim"):
                primary += x
    return total / area, primary / area


def region_map(marker: str, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Index of the marker's model's R10 region for every cell (-1 where none), and the region names."""
    regions = pd.read_csv(Path(__file__).parents[1] / "data" / "region_mappings" / f"{MODELS[marker]}_R10.csv")
    cells = pd.read_csv(Path(__file__).parents[1] / "data" / "country_cell.csv.gz").merge(regions, on="country")
    names = sorted(regions.region.unique())
    out = np.full((lat.size, lon.size), -1, dtype="int8")
    row = {round(float(v), 3): i for i, v in enumerate(lat)}
    col = {round(float(v), 3): i for i, v in enumerate(lon)}
    out[[row[round(y, 3)] for y in cells.y], [col[round(x, 3)] for x in cells.x]] = \
        [names.index(r) for r in cells.region]
    return out, names


OVERLAP = (2022, 2023, 2024)
HISTORY_TOLERANCE = 0.01        # Mha, summed over land states and cells
LAND_STATES = ("primf", "secdf", "primn", "secdn", "c3ann", "c4ann", "c3per", "c4per", "c3nfx", "pastr", "range", "urban")


def history_offset(path: str, history: str, area: np.ndarray) -> float:
    """Summed absolute difference (Mha) of a LUH3 scenario's land states from the history over OVERLAP."""
    s, h = xr.open_dataset(path, decode_times=False), xr.open_dataset(history, decode_times=False)
    sy, hy = years_of(s), years_of(h)
    total = 0.0
    for y in OVERLAP:
        i, j = int(np.where(sy == y)[0][0]), int(np.where(hy == y)[0][0])
        for v in LAND_STATES:
                total += float(np.sum(np.abs(np.nan_to_num(np.asarray(s[v].isel(time=i).values, dtype="float64"))
                                             - np.nan_to_num(np.asarray(h[v].isel(time=j).values, dtype="float64"))) * area))
    return total


def state_at(path: str, year: int) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_times=False)
    return ds.isel(time=int(np.where(years_of(ds) == year)[0][0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", required=True, help="folder holding the packaged release datasets")
    ap.add_argument("--version", default="0-1")
    ap.add_argument("--prefix", default=PREFIX, help=f"the datasets' name before the marker (default {PREFIX})")
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
        main_id = f"{args.prefix}-{m}-{args.version}"
        ext_id = f"{args.prefix}-{m}-ext-{args.version}"
        sources += [("mrdownscale", m, k, only(f"{rel}/{main_id}/multiple-{k}_*.nc"), lambda y: True)
                    for k in ("states", "transitions")]
        sources += [("mrdownscale", m, k, only(f"{rel}/{ext_id}/multiple-{k}_*.nc"), lambda y: y > 2100)
                    for k in ("states", "transitions")]
    history = lambda kind: only(f"{paths.LUH3}/multiple-{kind}_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_*.nc")
    for m, (scen, ext) in BENCHMARKS.items():
        sources += [("LUH3", m, k, history(k), lambda y: 2015 <= y <= 2021) for k in ("states", "transitions")]
        sources += [("LUH3", m, k, only(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-{k}_*.nc"),
                     lambda y: True) for k in ("states", "transitions")]
        if ext:
            sources += [("LUH3", m, k, only(f"{paths.LUH3_EXT}/{ext}/multiple-{k}_*.nc"),
                         lambda y: y > 2100) for k in ("states", "transitions")]

    # every benchmark must start from the history the products are harmonized to
    offsets = {}
    for m, (scen, _) in BENCHMARKS.items():
        offsets[scen] = history_offset(only(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-states_*.nc"), history("states"), cell)
        print(f"{scen}: {offsets[scen]:.4f} Mha from {Path(history('states')).name} over {OVERLAP}", flush=True)
    if max(offsets.values()) > HISTORY_TOLERANCE:
        raise SystemExit(f"a LUH3 benchmark does not start from the harmonization target: {offsets}")

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

    # maps and skill, for every marker LUH3 has published
    fields, skill, encoding = {}, [], {}
    fraction = {"dtype": "int16", "scale_factor": 1e-4, "_FillValue": -32768, "zlib": True, "complevel": 5}
    area = np.asarray(static["carea"].values, dtype="float64") * 1e6        # km2 to m2
    for m, (scen, _) in BENCHMARKS.items():
        print(f"maps {m}", flush=True)
        ours = only(f"{rel}/{args.prefix}-{m}-{args.version}/multiple-states_*.nc")
        luh = only(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-states_*.nc")
        for year in MAP_YEARS:
            o, r = state_at(ours, year), state_at(luh, year)
            for source, ds in (("mrdownscale", o), ("LUH3", r)):
                # LUH3 stores its plantation variables as no-data: count missing as zero on
                # land, and keep the ocean missing
                land = np.isfinite(np.asarray(ds["primf"].values))
                for name, vs in MAP_FIELDS.items():
                    x = sum(np.nan_to_num(np.asarray(ds[v].values, dtype="float32")) for v in vs if v in ds)
                    fields[f"{name}_{m}_{source}_{year}"] = (("lat", "lon"), np.where(land, x, np.nan))
                    encoding[f"{name}_{m}_{source}_{year}"] = fraction
                fields[f"secma_{m}_{source}_{year}"] = (("lat", "lon"), np.asarray(ds["secma"].values, dtype="float32"))
                encoding[f"secma_{m}_{source}_{year}"] = {"zlib": True, "complevel": 5, "least_significant_digit": 1}
        for year in (2050, 2100):
            t = compare.score_cells(state_at(ours, year), state_at(luh, year), cell * 1e4).reset_index()
            t.insert(0, "year", year)
            t.insert(0, "marker", m)
            skill.append(t)
        for source, path in (("mrdownscale", only(f"{rel}/{args.prefix}-{m}-{args.version}/multiple-transitions_*.nc")),
                             ("LUH3", only(f"{paths.LUH3_SCENARIOS}/{scen}/multiple-transitions_*.nc"))):
            for name, x in zip(("harvest", "harvest_primary"), cumulative_harvest(path, area)):
                fields[f"{name}_{m}_{source}"] = (("lat", "lon"), x.astype("float32"))
                encoding[f"{name}_{m}_{source}"] = {"zlib": True, "complevel": 5, "least_significant_digit": 3}
        reg, names = region_map(m, static["lat"].values, static["lon"].values)
        fields[f"region_{m}"] = xr.Variable(("lat", "lon"), reg, attrs={"regions": "; ".join(names)})
        encoding[f"region_{m}"] = {"zlib": True, "complevel": 5}
    xr.Dataset(fields, coords={"lat": static["lat"], "lon": static["lon"]},
               attrs={"harvest_years": f"{HARVEST_YEARS[0]}-{HARVEST_YEARS[1]}", "harvest_units": "kg C m-2"}).to_netcdf(
        out / "maps.nc", encoding=encoding)
    pd.concat(skill).to_csv(out / "skill.csv", index=False, float_format="%.4f")

    commit = subprocess.run(["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    sums = {p.parent.name: p.read_text() for p in sorted(rel.glob("*/SHA256SUMS"))}
    json.dump({"created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "graft_commit": commit, "release": str(rel), "release_checksums": sums,
               "luh3": sorted({Path(p).name for s, _, _, p, _ in sources if s == "LUH3"}),
               "luh3_offset_from_history_Mha": offsets},
              open(out / "manifest.json", "w"), indent=1)
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
