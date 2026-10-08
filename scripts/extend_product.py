"""Carry an annual product from the end of the ramp to 2500.

After the ramp (scripts/extend_iamc.py) land use is static, so the years to
2500 need no downscaling run. This starts from the product's last year and
writes states, transitions, management and secondary age and biomass for every
year after it:

- managed land, and natural land that is not harvested, stay as they are;
- wood harvest continues (graft.extend): over 49 years it is handed from the
  last ramp year's harvest, carried cell by cell, to the maintenance term,
  and harvested primary land becomes secondary, as in LUH3;
- every other transition is zero, and management is held at its last year.

The product's own secma and secmb at the start come from scripts/age_track.py,
run with the same regrowth curve (``--asymptote 0.75``).

    python scripts/extend_product.py ANNUAL_DIR OUT_DIR --secondary SECONDARY.nc
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

from graft import age, extend, paths

STATIC = str(paths.STATIC)
FILL = np.float32(1e20)
BLOCK = 25   # years written at a time
#: what the harvest moves or fells, as the product's transitions name it
HARVEST = {"primf": ("primary_area", "primary_carbon", None),
           "primn": ("nonforest_primary_area", "nonforest_primary_carbon", None),
           "secdf": ("secondary_area", "secondary_carbon", "secondary_reported"),
           "secnf": ("nonforest_secondary_area", "nonforest_secondary_carbon", "nonforest_secondary_reported")}


def only(pattern: str) -> str:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {len(found)}")
    return found[0]


def years_of(ds: xr.Dataset) -> np.ndarray:
    units = ds.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    t = ds.time.values
    return np.round(epoch + (t / 365 if units.startswith("days") else t)).astype(int)


class Writer:
    """One output file, shaped like ``template``, filled a block of years at a time."""

    def __init__(self, path: Path, template: xr.Dataset, names: list[str], years: list[int], comment: str):
        self.nc = netCDF4.Dataset(path, "w")
        self.years, self.buffer, self.done = years, {}, 0
        nc = self.nc
        for dim in ("lat", "lon"):
            nc.createDimension(dim, template.sizes[dim])
            v = nc.createVariable(dim, "f8", (dim,))
            v[:] = template[dim].values
            v.setncatts({k: a for k, a in template[dim].attrs.items() if k != "bounds"})
        nc.createDimension("time", len(years))
        t = nc.createVariable("time", "f8", ("time",))
        t[:] = [(y - 1970) * 365 for y in years]
        t.setncatts({"units": "days since 1970-01-01 0:0:0", "calendar": "365_day", "axis": "T",
                     "standard_name": "time", "long_name": "time"})
        for name in names:
            v = nc.createVariable(name, "f4", ("time", "lat", "lon"), zlib=True, complevel=4,
                                  fill_value=FILL, chunksizes=(min(BLOCK, len(years)), 240, 480))
            if name in template:
                v.setncatts({k: a for k, a in template[name].attrs.items() if k.lower() != "_fillvalue"})
        nc.setncatts({**template.attrs, "comment": comment, "frequency": "yr"})

    def add(self, fields: dict[str, np.ndarray]) -> None:
        """One year of every variable; NaN marks cells outside the variable's mask."""
        for name, field in fields.items():
            self.buffer.setdefault(name, []).append(np.where(np.isnan(field), FILL, field).astype("f4"))
        if len(next(iter(self.buffer.values()))) == BLOCK:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        n = len(next(iter(self.buffer.values())))
        for name, stack in self.buffer.items():
            self.nc[name][self.done:self.done + n] = np.stack(stack)
        self.done += n
        self.buffer = {}

    def close(self) -> None:
        self.flush()
        assert self.done == len(self.years), f"{self.done} of {len(self.years)} years written"
        self.nc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="folder holding the annual states, transitions and management")
    ap.add_argument("out")
    ap.add_argument("--secondary", required=True, help="secma and secmb from scripts/age_track.py")
    ap.add_argument("--end", type=int, default=2500)
    ap.add_argument("--static", default=STATIC)
    args = ap.parse_args()

    static = xr.open_dataset(args.static)
    area = np.asarray(static["carea"].values, dtype="float64") * 1e6          # m2
    ptbio = np.nan_to_num(np.asarray(static["ptbio"].values, dtype="float64"))
    zone = np.nan_to_num(static["ccode"].values).astype(int)
    table = extend.load_probability()
    regrowth = age.Regrowth(asymptote=extend.ABOVEGROUND)

    paths = {k: only(f"{args.folder}/multiple-{k}_*.nc") for k in ("states", "transitions", "management")}
    states, transitions, management = (xr.open_dataset(paths[k], decode_times=False)
                                       for k in ("states", "transitions", "management"))
    start = int(years_of(states)[-1])
    years = list(range(start, args.end + 1))
    last = {v: np.asarray(states[v].isel(time=-1).values, dtype="float64") for v in states.data_vars}
    lastManagement = {v: management[v].isel(time=-1).values for v in management.data_vars}
    # a transition's cells, from the last year it was written: zero there, missing elsewhere
    blank = {v: np.where(np.isnan(transitions[v].isel(time=-1).values), np.nan, 0.0)
             for v in transitions.data_vars}

    secondary = xr.open_dataset(args.secondary, decode_times=False)
    at = secondary.isel(time=int(np.where(secondary.time.values == start)[0][0]))
    seedFrom = xr.Dataset({v: (("lat", "lon"), np.nan_to_num(last[v])) for v in ("secdf", "secdn")}
                          | {v: (("lat", "lon"), at[v].values) for v in ("secma", "secmb")})
    forest, other = age.seed(seedFrom, static, regrowth)
    primf, primn = np.nan_to_num(last["primf"]), np.nan_to_num(last["primn"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = lambda kind, a, b: Path(paths[kind]).name.rsplit("_", 1)[0] + f"_{a}-{b}.nc"
    comment = (f"static land use after {start}, wood harvest by graft.extend; "
               f"continues {Path(paths['states']).name}")
    writers = {
        "states": Writer(out / name("states", start, args.end), states, list(states.data_vars), years, comment),
        "transitions": Writer(out / name("transitions", start, args.end - 1), transitions,
                              list(transitions.data_vars), years[:-1], comment),
        "management": Writer(out / name("management", start, args.end), management,
                             list(management.data_vars), years, comment),
        "secondary": Writer(out / name("states", start, args.end).replace("multiple-states", "multiple-secondary"),
                            states.drop_vars(list(states.data_vars)), ["secma", "secmb"], years, comment),
    }
    writers["secondary"].nc["secma"].setncatts({"units": "years", "long_name": "secondary mean age"})
    writers["secondary"].nc["secmb"].setncatts({"units": "kg m-2",
                                                "long_name": "secondary mean biomass carbon density"})
    land = ~np.isnan(last["primf"])

    def write_year(primf, primn, forest, other):
        moved = {"primf": primf, "primn": primn, "secdf": forest.area, "secdn": other.area}
        writers["states"].add({v: np.where(np.isnan(last[v]), np.nan, moved[v]) if v in moved else last[v]
                               for v in last})
        writers["management"].add(lastManagement)
        secma, secmb = age.combine(forest, other)
        writers["secondary"].add({"secma": np.where(land, secma, np.nan), "secmb": np.where(land, secmb, np.nan)})

    # the last ramp year's harvest, which the static period carries on from
    # (the file's final step only repeats it)
    before = transitions.isel(time=int(np.where(years_of(transitions) == start - 1)[0][0]))
    field = lambda v: np.nan_to_num(np.asarray(before[v].values, dtype="float64")) if v in before else 0.0
    lastHarvest = extend.Harvest(
        primary_area=field("primf_harv"), primary_carbon=field("primf_bioh"),
        secondary_area=field("secdf_harv"), secondary_carbon=field("secdf_bioh"),
        unmet=np.zeros(int(zone.max()) + 1),
        nonforest_primary_area=field("primn_harv"), nonforest_primary_carbon=field("primn_bioh"),
        nonforest_secondary_area=field("secnf_harv"), nonforest_secondary_carbon=field("secnf_bioh"))

    mha = lambda frac: float(np.sum(frac * area)) / 1e10
    write_year(primf, primn, forest, other)
    for year, h, primf, primn, forest, other in extend.run(primf, primn, forest, other, ptbio, area,
                                                           zone, table, regrowth, years[:-1],
                                                           last=lastHarvest):
        flows = dict(blank)
        for pool, (areaField, carbonField, reportedField) in HARVEST.items():
            harvested = getattr(h, reportedField) if reportedField else None
            if harvested is None:
                harvested = getattr(h, areaField)
            for suffix, value in (("harv", harvested), ("bioh", getattr(h, carbonField))):
                if f"{pool}_{suffix}" in flows:
                    flows[f"{pool}_{suffix}"] = np.where(np.isnan(blank[f"{pool}_{suffix}"]), np.nan,
                                                         np.zeros_like(primf) + value)
        writers["transitions"].add(flows)
        write_year(primf, primn, forest, other)
        if year % 25 == 0 or year == years[0]:
            print(f"{year}: harvest {h.carbon.sum() / 1e12:.3f} PgC (primary forest {h.primary_carbon.sum() / 1e12:.3f}, "
                  f"unmet {h.unmet.sum() / 1e12:.3f}) | primf {mha(primf):7.1f} secdf {mha(forest.area):7.1f} Mha",
                  flush=True)
    for w in writers.values():
        w.close()
    print("wrote", ", ".join(sorted(p.name for p in out.glob("multiple-*.nc"))))


if __name__ == "__main__":
    main()
