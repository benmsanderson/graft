"""Turn a five- and ten-yearly product into the annual one an ESM needs.

LUH3 is annual; mrdownscale's output carries the IAM's own reporting years,
which jump a decade after 2050. No Earth system model can be driven by that.

The conversion is cheap for a reason worth stating, because it is also the
reason it has to be checked. mrdownscale derives transitions from the endpoint
states as **per-year rates**, so a linearly interpolated state series has
exactly the constant annual net change the transitions already assert. States
and management interpolate; each interval's transitions hold across its years;
and the two then agree by construction rather than by luck.

``--check`` verifies that instead of assuming it, comparing every annual state
change against the transitions that claim to produce it.

    python scripts/annualise.py OUT_DIR --states S.nc --transitions T.nc \\
        --management M.nc --check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from graft import luh_schema as S
from graft.io import luh

#: written so the axis decodes: days on a 365-day calendar, as LUH3 does
TIME_UNITS = "days since 1970-01-01 0:0:0"
CALENDAR = "365_day"
DAYS = 365


def time_coord(years: list[int]) -> xr.DataArray:
    values = np.array([(y - 1970) * DAYS for y in years], dtype="float64")
    return xr.DataArray(values, dims="time", name="time",
                        attrs={"units": TIME_UNITS, "calendar": CALENDAR,
                               "axis": "T", "standard_name": "time", "long_name": "time"})


def gridded(ds: xr.Dataset) -> list[str]:
    return [v for v in ds.data_vars
            if "time" in ds[v].dims and ds[v].ndim >= 3 and "bound" not in v]


def interpolate(ds: xr.Dataset, years: list[int], reported: list[int]) -> xr.Dataset:
    """Linear in time between reported years, variable by variable."""
    out = {}
    for name in gridded(ds):
        source = ds[name].values
        stack = np.empty((len(years), *source.shape[1:]), dtype="float32")
        for k, year in enumerate(years):
            after = int(np.searchsorted(reported, year, side="left"))
            if year in reported:
                stack[k] = source[reported.index(year)]
                continue
            lo, hi = reported[after - 1], reported[after]
            w = (year - lo) / (hi - lo)
            stack[k] = (1 - w) * source[after - 1] + w * source[after]
        out[name] = (("time", *ds[name].dims[1:]), stack, dict(ds[name].attrs))
    return out


def hold(ds: xr.Dataset, years: list[int], reported: list[int]) -> dict:
    """Each interval's per-year rate, held across the years it covers."""
    out = {}
    index = [max(0, int(np.searchsorted(reported, y, side="right")) - 1) for y in years]
    for name in gridded(ds):
        source = ds[name].values
        stack = np.stack([source[i] for i in index]).astype("float32")
        out[name] = (("time", *ds[name].dims[1:]), stack, dict(ds[name].attrs))
    return out


def write(path: Path, built: dict, template: xr.Dataset, years: list[int], source: str):
    coords = {d: template[d] for d in template[gridded(template)[0]].dims[1:]}
    ds = xr.Dataset({k: v for k, v in built.items()},
                    coords={"time": time_coord(years), **coords})
    ds.attrs = {**template.attrs,
                "comment": f"annual, interpolated from {source} by graft's annualise.py",
                "frequency": "yr"}
    encoding = {k: {"zlib": True, "complevel": 4, "dtype": "float32"} for k in built}
    # units and calendar ride on the coordinate's attrs, since the values are
    # raw numbers rather than datetimes; xarray refuses them in encoding too
    encoding["time"] = {"dtype": "float64"}
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding)
    print(f"  wrote {path.name}: {len(built)} variables, {len(years)} years")


def check(states: xr.Dataset, transitions: xr.Dataset, years: list[int],
          static: xr.Dataset, tolerance: float = 1e-6) -> int:
    """Every annual state change against the flows that claim to make it.

    Read with LUH3's rule, as an ESM reads it: primary land leaves through its
    transitions and through ``primf_harv``/``primn_harv``, which it gives to
    secondary land. Checking transitions alone, and only some states, once let
    a product that moved the same hectares twice pass.
    """
    area = np.asarray(static["carea"].values, dtype="float64")
    names = list(transitions.data_vars)
    pools = [s for s in S.AREA_STATES if s in states]
    moving = {src: (hv, dst) for src, (hv, dst) in S.AREA_MOVING_HARVEST.items()
              if hv in transitions and src in states and dst in states}
    worst, worstAt = 0.0, None
    flows = [v for v in names if "_to_" in v]
    for k in range(len(years) - 1):
        # read each year's fields once; the per-pool sums below reuse them
        step = {v: np.nan_to_num(transitions[v].isel(time=k).values) for v in flows}
        harvest = {src: np.nan_to_num(transitions[hv].isel(time=k).values)
                   for src, (hv, _) in moving.items()}
        for pool in pools:
            change = (states[pool].isel(time=k + 1).values
                      - states[pool].isel(time=k).values)
            gain = sum(step[v] for v in flows if v.endswith(f"_to_{pool}"))
            loss = sum(step[v] for v in flows if v.startswith(f"{pool}_to_"))
            for src, (_, dst) in moving.items():
                if pool == src:
                    loss = loss + harvest[src]
                if pool == dst:
                    gain = gain + harvest[src]
            residual = np.abs(np.nan_to_num(change) - (gain - loss))
            mha = float(np.nansum(residual * area) / 1e4)
            if mha > worst:
                worst, worstAt = mha, (years[k], pool)
    verdict = "ok" if worst < 1.0 else "FAILED"
    print(f"  consistency (LUH3 rule, harvest included): worst annual residual {worst:.4f} Mha"
          f"{f' at {worstAt[0]} {worstAt[1]}' if worstAt else ''} - {verdict}")
    return 0 if worst < 1.0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="directory to write the annual files into")
    ap.add_argument("--states", required=True)
    ap.add_argument("--transitions")
    ap.add_argument("--management")
    ap.add_argument("--static", help="needed by --check, for cell area")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    states = xr.open_dataset(args.states, decode_times=False)
    reported = luh.state_years(states)
    years = list(range(min(reported), max(reported) + 1))
    print(f"{len(reported)} reported years -> {len(years)} annual, "
          f"{years[0]} to {years[-1]}")

    write(out / Path(args.states).name, interpolate(states, years, reported),
          states, years, Path(args.states).name)

    if args.management:
        management = xr.open_dataset(args.management, decode_times=False)
        write(out / Path(args.management).name,
              interpolate(management, years, luh.state_years(management)),
              management, years, Path(args.management).name)

    annualTransitions = None
    if args.transitions:
        transitions = xr.open_dataset(args.transitions, decode_times=False)
        built = hold(transitions, years, luh.state_years(transitions))
        write(out / Path(args.transitions).name, built, transitions, years,
              Path(args.transitions).name)
        annualTransitions = xr.open_dataset(out / Path(args.transitions).name,
                                            decode_times=False)

    if args.check:
        if not (annualTransitions and args.static):
            raise SystemExit("--check needs --transitions and --static")
        return check(xr.open_dataset(out / Path(args.states).name, decode_times=False),
                     annualTransitions, years, luh.open_static(args.static))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
