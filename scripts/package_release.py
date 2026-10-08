"""Package one marker's build as release datasets, in LUH3's layout.

A build (release/build.sh on the mrdownscale fork) leaves, per marker, annual
files for 2020-2150 and the static period 2150-2500. This writes two datasets,
split at 2100 as LUH publishes them:

- ``CICERO-graft-landState-<marker>-<version>``: states 2020-2100 (with
  secma and secmb), transitions 2020-2099, management 2020-2100;
- ``CICERO-graft-landState-<marker>-ext-<version>``: states 2100-2500,
  transitions 2100-2499, management 2100-2500.

The build's last ramp transition year, which only repeats 2149's rates, is
replaced by the static period's own 2150. ``secma``/``secmb`` for 2020-2023,
before the age tracker starts, are LUH3 history's own, as are the states those
years carry. Files follow LUH3's conventions: a 365-day calendar with
``time_bnds``, latitude and longitude bounds, float32 fields chunked one year
at a time, and global attributes that say what the product is - and that it
is independent of LUH3. A README and SHA-256 checksums go with each dataset.

    python scripts/package_release.py ~/madrat/output/vlr01_iamc vl OUT --manifest WORK/manifest.txt
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from graft import paths

MARKERS = {"vl": ("Very Low", "REMIND-MAgPIE 3.5-4.11"), "l": ("Low", "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12"),
           "ln": ("Low-to-Negative", "AIM 3.0"), "m": ("Medium", "IMAGE 3.4"),
           "ml": ("Medium-to-Low", "COFFEE 1.6"), "h": ("High", "GCAM 8s"), "hl": ("High-to-Low", "WITCH 6.0")}
FILL = np.float32(1e20)
DAYS = 365
NOT_LUH3 = ("Independent of LUH3 and not part of the official CMIP7 forcing datasets: land-use inputs produced "
            "rapidly from regionally aggregated public IAM data (the IIASA ScenarioMIP release) with a fork of "
            "mrdownscale and graft. See references.")


def years_of(ds: xr.Dataset) -> np.ndarray:
    units = ds.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    t = ds.time.values
    return np.round(epoch + (t / DAYS if units.startswith("days") else t)).astype(int)


def only(pattern: str) -> str:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {len(found)}")
    return found[0]


class Source:
    """A variable's yearly fields drawn from several files, later files winning."""

    def __init__(self, files: list[str]):
        self.where = {}
        for f in files:
            ds = xr.open_dataset(f, decode_times=False)
            for i, y in enumerate(years_of(ds)):
                for v in ds.data_vars:
                    if "time" in ds[v].dims:
                        self.where[(v, int(y))] = (ds, i)

    def block(self, v: str, years: list[int]) -> np.ndarray:
        """All of a variable's years at once: runs of consecutive steps from one
        file are read in a single call, since the files are compressed in
        chunks of many years and reading year by year decompresses each chunk
        over and over."""
        out, run = [], []
        for y in years + [None]:
            here = self.where.get((v, y)) if y is not None else None
            if run and (here is None or here[0] is not run[0][0] or here[1] != run[-1][1] + 1):
                ds = run[0][0]
                out.append(np.asarray(ds[v].isel(time=slice(run[0][1], run[-1][1] + 1)).values, dtype="float32"))
                run = []
            if here is not None:
                run.append(here)
        return np.concatenate(out)

    def attrs(self, v: str, year: int) -> dict:
        return dict(self.where[(v, year)][0][v].attrs)


def write(path: Path, src: Source, names: list[str], years: list[int], grid: xr.Dataset, attrs: dict):
    nc = netCDF4.Dataset(path, "w", format="NETCDF4")
    nc.createDimension("time", None)
    nc.createDimension("lat", grid.sizes["lat"])
    nc.createDimension("lon", grid.sizes["lon"])
    nc.createDimension("bnds", 2)
    for dim in ("lat", "lon"):
        v = nc.createVariable(dim, "f8", (dim,))
        v[:] = grid[dim].values
        v.setncatts({"units": "degrees_north" if dim == "lat" else "degrees_east",
                     "standard_name": "latitude" if dim == "lat" else "longitude",
                     "long_name": "latitude" if dim == "lat" else "longitude",
                     "axis": "Y" if dim == "lat" else "X", "bounds": f"{dim}_bnds"})
        b = nc.createVariable(f"{dim}_bnds", "f8", (dim, "bnds"))
        half = abs(float(grid[dim].values[1] - grid[dim].values[0])) / 2
        b[:] = np.stack([grid[dim].values - half, grid[dim].values + half], axis=1)
    t = nc.createVariable("time", "f8", ("time",))
    t.setncatts({"standard_name": "time", "long_name": "time", "bounds": "time_bnds",
                 "units": "days since 1970-01-01 00:00:00", "calendar": "365_day", "axis": "T"})
    tb = nc.createVariable("time_bnds", "f8", ("time", "bnds"))
    start = np.array([(y - 1970) * DAYS for y in years], dtype="f8")
    t[:] = start
    tb[:] = np.stack([start, start + DAYS], axis=1)
    for name in names:
        v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4,
                              fill_value=FILL, chunksizes=(1, grid.sizes["lat"], grid.sizes["lon"]))
        a = {k: x for k, x in src.attrs(name, years[0]).items() if k.lower() not in ("_fillvalue", "missing_value")}
        a.setdefault("cell_methods", "time: mean")
        v.setncatts(a)
        x = src.block(name, years)
        assert x.shape[0] == len(years), f"{name}: {x.shape[0]} of {len(years)} years found"
        v[:] = np.where(np.isnan(x), FILL, x)
    nc.setncatts(attrs)
    nc.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build", help="one marker's build folder, holding annual/ and extension/")
    ap.add_argument("marker", choices=list(MARKERS))
    ap.add_argument("out")
    ap.add_argument("--version", default="0-1")
    ap.add_argument("--manifest", help="the build's manifest.txt, copied into the provenance")
    args = ap.parse_args()

    b = Path(args.build)
    annual = {k: only(f"{b}/annual/multiple-{k}_*.nc") for k in ("states", "transitions", "management")}
    ext = {k: only(f"{b}/extension/multiple-{k}_*.nc") for k in ("states", "transitions", "management", "secondary")}
    history = only(f"{paths.LUH3}/multiple-states_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_*.nc")
    grid = xr.open_dataset(annual["states"], decode_times=False)

    # secma/secmb: LUH3 history before the tracker starts, then tracked, then the static period
    tracked = xr.open_dataset(f"{b}/annual/secondary_2024-2150.nc", decode_times=False)
    tracked = tracked.assign_coords(time=("time", (tracked.time.values.astype(int) - 1970) * DAYS))
    tracked.time.attrs["units"] = "days since 1970-01-01"
    tmp = Path(args.out) / f".secondary_{args.marker}.nc"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tracked[["secma", "secmb"]].to_netcdf(tmp, encoding={v: {"zlib": True, "complevel": 1} for v in ("secma", "secmb")})
    hist = xr.open_dataset(history, decode_times=False)
    early = hist[["secma", "secmb"]].isel(time=[i for i, y in enumerate(years_of(hist)) if 2020 <= y <= 2023])
    early = early.assign_coords(time=("time", np.array([(y - 1970) * DAYS for y in range(2020, 2024)], dtype="f8")))
    early.time.attrs["units"] = "days since 1970-01-01"
    early_tmp = Path(args.out) / f".secondary_hist_{args.marker}.nc"
    early.to_netcdf(early_tmp)

    states = Source([str(early_tmp), str(tmp), annual["states"], ext["states"], ext["secondary"]])
    transitions = Source([annual["transitions"], ext["transitions"]])
    management = Source([annual["management"], ext["management"]])
    stateNames = list(xr.open_dataset(annual["states"], decode_times=False).data_vars) + ["secma", "secmb"]
    transitionNames = list(xr.open_dataset(annual["transitions"], decode_times=False).data_vars)
    managementNames = list(xr.open_dataset(annual["management"], decode_times=False).data_vars)

    scenario, model = MARKERS[args.marker]
    provenance = Path(args.manifest).read_text() if args.manifest else ""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    built = []
    for ext_part, label, span, tspan in ((False, "", range(2020, 2101), range(2020, 2100)),
                                         (True, "-ext", range(2100, 2501), range(2100, 2500))):
        source_id = f"CICERO-graft-landState-{args.marker}{label}-{args.version}"
        folder = Path(args.out) / source_id
        folder.mkdir(parents=True, exist_ok=True)
        for kind, src, names, yrs in (("states", states, stateNames, list(span)),
                                      ("transitions", transitions, transitionNames, list(tspan)),
                                      ("management", management, managementNames, list(span))):
            attrs = {
                "title": f"Land-use states, transitions and management for the ScenarioMIP {scenario} marker "
                         f"({model}){', extension' if ext_part else ''}, LUH format, from public IAM data",
                "source": f"graft/mrdownscale land-use forcing from regional IAM data, version {args.version.replace('-', '.')}",
                "source_id": source_id, "source_version": args.version.replace("-", "."),
                "comment": NOT_LUH3, "institution_id": "CICERO",
                "institution": "CICERO Center for International Climate Research, Oslo, Norway",
                "contact": "benjamin.sanderson@cicero.oslo.no", "license_id": "CC BY 4.0",
                "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
                "references": "https://github.com/benmsanderson/graft; https://github.com/benmsanderson/mrdownscale "
                              "(fork of https://github.com/pik-piam/mrdownscale); harmonized to LUH3 history, "
                              "UofMD-landState-3-1-1 (Hurtt et al. 2020, https://doi.org/10.5194/gmd-13-5425-2020)",
                "input_data": f"ScenarioMIP {scenario} marker by the {model} modelling team, from the IIASA "
                              "'ScenarioMIP/CMIP7 Ensemble - Data at the R10 region level', release v0.1 "
                              "(September 2026), https://scenariomip.apps.ece.iiasa.ac.at; LUH3 history "
                              "UofMD-landState-3-1-1 (CC BY 4.0)",
                "input_data_license": "The IIASA release is copyright IIASA and the contributing modelling teams "
                                      "and used under its licence, https://scenariomip.apps.ece.iiasa.ac.at/license; "
                                      "this dataset is derived from it and does not reproduce it",
                "acknowledgement": f"Scenario land-use results by the {model} team, as published in the ScenarioMIP "
                                   "release; harmonized to LUH3 history (University of Maryland; Hurtt et al. 2020, "
                                   "Chini et al. 2021); built with mrdownscale (Sauer and Dietrich, PIK, "
                                   "https://doi.org/10.5281/zenodo.11244475)",
                "activity_id": "input4MIPs", "target_mip": "ScenarioMIP", "mip_era": "CMIP7",
                "dataset_category": "landState", "variable_id": f"multiple-{kind}", "realm": "land",
                "frequency": "yr", "grid_label": "gn", "nominal_resolution": "25 km",
                "data_structure": "grid", "Conventions": "CF-1.6", "creation_date": now,
                "provenance": provenance}
            name = f"multiple-{kind}_input4MIPs_landState_ScenarioMIP_{source_id}_gn_{yrs[0]}-{yrs[-1]}.nc"
            write(folder / name, src, names, yrs, grid, attrs)
            built.append(folder / name)
            print("wrote", name, flush=True)
        with open(folder / "SHA256SUMS", "w") as f:
            for p in sorted(folder.glob("*.nc")):
                f.write(f"{sha256(p)}  {p.name}\n")
        (folder / "README.txt").write_text(
            f"{source_id}\n\n{attrs['title']}.\n\n{NOT_LUH3}\n\n"
            f"Annual, 0.25 degree, {span[0]}-{span[-1]} (transitions {tspan[0]}-{tspan[-1]}). Variables as in LUH3; "
            "plantations are folded into secondary forest; wood harvest moves primary land to secondary as in LUH3.\n\n"
            f"Input: {attrs['input_data']}.\n{attrs['input_data_license']}.\n\n"
            f"Acknowledgement: {attrs['acknowledgement']}.\n\n"
            f"Licence: CC BY 4.0. Provenance (build manifest):\n\n{provenance}\n")
    tmp.unlink()
    early_tmp.unlink()


if __name__ == "__main__":
    main()
