"""Can the age tracker reproduce a published secma and secmb from its own land?

The test that decides whether :mod:`graft.age` is worth anything. Seed both
secondary pools from a published product's own ``secma`` and ``secmb`` at the
start year, then propagate to the end using nothing but that product's gross
transitions and wood harvest, and compare with what it actually published.
Nothing about the answer is used along the way.

    python scripts/age_validate.py --static STATIC.nc --states LUH3-VL.nc \\
        --transitions LUH3-VL-transitions.nc --from 2025 --to 2100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from graft import age as A
from graft.io import luh


def transition_years(transitions: xr.Dataset, override: int | None = None) -> list[int]:
    """The calendar year of each time step, for either file convention.

    LUH3 writes ``days since <year>-1-1`` and mrdownscale ``years since
    1970-01-01``, so the axis has to be read rather than assumed.
    """
    values = np.asarray(transitions["time"].values, dtype="float64")
    units = str(transitions["time"].attrs.get("units", ""))
    epoch = override if override is not None else int(units.split("since")[-1].strip().split("-")[0])
    if units.strip().startswith("days"):
        return [int(round(epoch + v / 365.0)) for v in values]
    return [int(round(epoch + v)) for v in values]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--static", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--from", dest="start", type=int, default=2025)
    ap.add_argument("--to", dest="end", type=int, default=2100)
    ap.add_argument("--base", type=int, help="first year of the transitions file "
                                             "(default: the start of its time units)")
    ap.add_argument("--asymptote", type=float)
    ap.add_argument("--residual", type=float, default=0.0)
    ap.add_argument("--seed-from", help="states file holding the secma and secmb to start "
                                        "from, when the product being propagated has none "
                                        "of its own - LUH3's history, which it is "
                                        "harmonized to, is the natural choice")
    ap.add_argument("--seed-year", type=int, help="year to seed at (default: --from)")
    ap.add_argument("--out", help="write secma and secmb at every state year to this netCDF")
    ap.add_argument("--fit", action="store_true",
                    help="also report the cross-sectional fit, which is not the "
                         "same curve and should not be used to propagate")
    args = ap.parse_args(argv)

    static = luh.open_static(args.static)
    states = luh.open_states(args.states)
    transitions = xr.open_dataset(args.transitions, decode_times=False)
    area = np.asarray(static["carea"].values, dtype="float64")

    regrowth = A.Regrowth() if args.asymptote is None else A.Regrowth(asymptote=args.asymptote)
    if args.fit:
        fitted, detail = A.fit_regrowth(luh.select_year(states, args.start), static)
        for name, d in detail.items():
            print(f"cross-sectional {name:10s} asymptote {d['asymptote']:.3f} "
                  f"timescale {d['timescale']:5.1f} yr  R2 {d['r2']:.3f}  ({d['cells']} cells)")
        print("  that curve is mean-against-mean and sits below the cohort curve "
              "(Jensen); propagation uses the calibrated asymptote instead\n")

    years = transition_years(transitions, args.base)

    target = A.target_biomass(static, regrowth)
    seedStates = luh.open_states(args.seed_from) if args.seed_from else states
    seedYear = args.seed_year if args.seed_year is not None else args.start
    forest, other = A.seed(luh.select_year(seedStates, seedYear), static, regrowth)
    if args.seed_from:
        # the seed carries the reference's secondary areas; the product's own
        # are what the transitions then move, so start from those
        start = luh.select_year(states, args.start)
        for pool, key in ((forest, "secdf"), (other, "secdn")):
            pool.area = np.nan_to_num(np.asarray(start[key].values, dtype="float64"))
    # every step whose year lies in the window, the last one running to the end
    # transitions are per-year rates, so a ten-year step is ten one-year steps
    # rather than one step of ten times the rate: over a long step the rate can
    # exceed the land a cell has left, and clamping that would manufacture area
    wanted = sorted(y for y in luh.state_years(states) if args.start <= y <= args.end)
    saved = {}
    usable = [(i, y) for i, y in enumerate(years) if args.start <= y < args.end]
    for k, (index, year) in enumerate(usable):
        if args.out and year in wanted:
            saved[year] = A.combine(forest, other)
        span = (usable[k + 1][1] if k + 1 < len(usable) else args.end) - year
        forestFlows, otherFlows = A.flows_from_transitions(transitions, index, dt=1.0, area=area * 1e6)
        for offset in range(int(span)):
            forest, other = A.advance(forest, other, forestFlows, otherFlows,
                                      target, regrowth, dt=1.0, residual=args.residual)
            if args.out and (year + offset + 1) in wanted:
                saved[year + offset + 1] = A.combine(forest, other)

    ourAge, ourBiomass = A.combine(forest, other)

    if args.out:
        order = sorted(saved)
        template = states["secdf"]
        dims = [d for d in template.dims if d != "time"]
        coords = {d: states[d] for d in dims}
        written = xr.Dataset(
            {name: (("time", *dims), np.stack([saved[y][i] for y in order]))
             for i, name in enumerate(("secma", "secmb"))},
            coords={"time": ("time", np.asarray(order, dtype="int32")), **coords})
        written["secma"].attrs = {"units": "years", "long_name": "secondary mean age"}
        written["secmb"].attrs = {"units": "kg m-2",
                                  "long_name": "secondary mean biomass carbon density"}
        written["time"].attrs = {"units": "years since 0-1-1", "long_name": "year"}
        written.attrs = {
            "title": "secondary land age and biomass, propagated by graft.age",
            "source": f"states {Path(args.states).name}, "
                      f"transitions {Path(args.transitions).name}",
            "seeded_from": Path(args.seed_from).name if args.seed_from else "own fields",
            "regrowth": f"asymptote {regrowth.asymptote} of ptbio, timescales "
                        f"{regrowth.forest_timescale} and {regrowth.nonforest_timescale} yr",
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        written.to_netcdf(args.out)
        print(f"wrote {args.out}: {len(order)} years, {order[0]} to {order[-1]}")
    final = luh.select_year(states, args.end)
    get = lambda v: np.nan_to_num(np.asarray(final[v].values, dtype="float64"))
    secondary = get("secdf") + get("secdn")
    if "secma" not in final.data_vars:
        weight = secondary * area
        live = secondary > 0.01
        print(f"{args.start} -> {args.end}, {int(live.sum())} cells, no published fields "
              f"to compare with; this is the product's own secma and secmb")
        for name, ours, unit in (("secma", ourAge, "yr"), ("secmb", ourBiomass, "kg C/m2")):
            print(f"  {name}  {np.average(ours[live], weights=weight[live]):7.2f} {unit}")
        print(f"  secondary biomass  {float(np.nansum(ourBiomass * secondary * area * 1e6) * 1e-12):6.1f} Pg C")
        print(f"  area               {np.nansum(secondary * area) / 1e4:8.1f} Mha  "
              f"(tracker {np.nansum((forest.area + other.area) * area) / 1e4:8.1f})")
        return 0
    weight = secondary * area
    live = secondary > 0.01

    print(f"{args.start} -> {args.end}, {int(live.sum())} cells, "
          f"asymptote {regrowth.asymptote}, residual {args.residual}")
    for name, ours, published, unit in (("secma", ourAge, get("secma"), "yr"),
                                        ("secmb", ourBiomass, get("secmb"), "kg C/m2")):
        mean = np.average(ours[live], weights=weight[live])
        reference = np.average(published[live], weights=weight[live])
        mae = np.average(np.abs(ours - published)[live], weights=weight[live])
        r = np.corrcoef(ours[live], published[live])[0, 1]
        print(f"  {name}  ours {mean:7.2f}  published {reference:7.2f} {unit:8s}"
              f"  MAE {mae:6.2f}  corr {r:.3f}")
    pg = lambda b: float(np.nansum(b * secondary * area * 1e6) * 1e-12)
    print(f"  secondary biomass  ours {pg(ourBiomass):6.1f}  published {pg(get('secmb')):6.1f} Pg C")
    print(f"  area               ours {np.nansum((forest.area + other.area) * area) / 1e4:8.1f}"
          f"  published {np.nansum(secondary * area) / 1e4:8.1f} Mha")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
