"""Does a gridded land-use forcing carry the carbon its own scenario reported?

A downscaler conserves area, not carbon; this checks the carbon, and the
round trip of a LUH3 scenario (scripts/luh_as_iamc.py) gives it a denominator. Two questions, in order of how much the answer can be
trusted:

1. **Placement.** Two griddings of the same regional areas hold different
   carbon if they put the land in different cells. :func:`graft.carbon.decompose`
   splits the difference exactly into the part from differing area and the part
   from differing placement. The bookkeeping priors largely cancel in that
   difference, so this is a statement about the downscaler.
2. **Level.** The same bookkeeping against the IAM's own reported
   ``Emissions|CO2|AFOLU``. A one-parameter-per-class rule cannot reproduce an
   IAM's land-carbon model, so this is a consistency band and a sign test, not
   a validation - but a forcing whose implied flux has the wrong *sign* is
   telling the Earth system model something its own scenario did not say.

    python scripts/carbon_check.py OURS.nc --static STATIC.nc \\
        --mask country_cell.csv.gz --mapping region_mapping.csv \\
        --reference LUH3-VL.nc --iamc ScenarioMIP_v0.1_R10.xlsx \\
        --model 'REMIND-MAgPIE 3.5-4.11' --scenario 'Very Low - SSP1 (Marker)'
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import xarray as xr

from graft import carbon
from graft.compare import fold_for_scoring
from graft.io import luh


def trajectory(yearly, static, years, label):
    """Stock and flux over ``years``, from one single-year dataset per year."""
    stocks = np.array([float(carbon.total_stock(y, static).sum()) for y in yearly])
    return pd.DataFrame({"year": years, f"{label} stock": stocks.round(1),
                         f"{label} flux": np.r_[np.nan, carbon.flux(stocks, np.asarray(years))].round(2)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("states", help="the gridded product to check")
    ap.add_argument("--static", required=True)
    ap.add_argument("--mask", help="only needed with --reference, for the decomposition")
    ap.add_argument("--mapping", help="only needed with --reference, for the decomposition")
    ap.add_argument("--reference", help="a second product to decompose against")
    ap.add_argument("--iamc", help="the IAMC release holding the reported AFOLU")
    ap.add_argument("--model")
    ap.add_argument("--scenario")
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2025, 2030, 2040, 2050, 2070, 2100])
    ap.add_argument("--decompose-year", type=int, default=2050)
    ap.add_argument("--label", default="ours", help="what to call the product being checked")
    ap.add_argument("--secondary", help="a secma/secmb file from scripts/age_track.py, "
                                        "merged into the product so its secondary land is "
                                        "valued with its own tracked biomass rather than a "
                                        "prior or a borrowed field")
    ap.add_argument("--borrow-secmb", action="store_true",
                    help="give the checked product the reference's own secmb, cell by "
                         "cell and year by year, instead of the constant prior. It "
                         "assumes the two agree on the age structure of secondary land "
                         "in a cell, which is wrong in detail and much less wrong than a "
                         "global constant - and it is the only way to score a product "
                         "that does not carry secmb of its own")
    ap.add_argument("--history", metavar="STATES",
                    help="LUH3's historical states, to check the bookkeeping against "
                         "the published historical land-use flux before trusting it "
                         "on a scenario")
    args = ap.parse_args(argv)

    static = luh.open_static(args.static)

    if args.history:
        # the one place the answer is already known: the Global Carbon Budget
        # puts land-use change at about 1.2 +/- 0.7 Gt C/yr over the last
        # decade, rising through the twentieth century from well under 1
        history = luh.open_states(args.history)
        print("the bookkeeping on LUH3's own history, with its own secmb:")
        for a, b in [(1850, 1900), (1900, 1950), (1950, 2000), (2000, 2024), (2014, 2024)]:
            s = np.array([float(carbon.total_stock(luh.select_year(history, y), static).sum())
                          for y in (a, b)])
            f = carbon.flux(s, np.array([a, b]))[0]
            print(f"  {a}-{b}  {s[0]:6.1f} -> {s[1]:6.1f} Pg C   "
                  f"{f:+5.2f} Gt CO2/yr = {f / carbon.C_TO_CO2:+5.2f} Gt C/yr")
        print()

    ours = fold_for_scoring(luh.open_states(args.states))

    print(f"potential biomass carbon: {carbon.density(static).max():.1f} kg C/m2 at most, "
          f"priors {carbon.POTENTIAL_FRACTION}\n")

    oursYearly = [luh.select_year(ours, y) for y in args.years]
    if args.secondary:
        tracked = xr.open_dataset(args.secondary, decode_times=False)
        years = luh.state_years(tracked)
        for k, year in enumerate(args.years):
            at = tracked.isel(time=years.index(year))
            oursYearly[k] = oursYearly[k].assign(
                secmb=(("lat", "lon"), np.asarray(at["secmb"].values, dtype="float64")))
        table = trajectory(oursYearly, static, args.years, args.label)
        print(f"secondary land in {args.label} is valued with its own tracked secmb\n")
    table = trajectory(oursYearly, static, args.years, args.label)

    if args.reference:
        reference = fold_for_scoring(luh.open_states(args.reference))
        referenceYearly = [luh.select_year(reference, y) for y in args.years]
        if args.borrow_secmb:
            if "secmb" not in reference.data_vars:
                raise SystemExit("the reference carries no secmb to borrow")
            oursYearly = [o.assign(secmb=r["secmb"])
                          for o, r in zip(oursYearly, referenceYearly)]
            table = trajectory(oursYearly, static, args.years, args.label)
            print(f"secondary land in {args.label} is valued with the reference's "
                  f"own secmb, not the prior\n")
        table = table.merge(trajectory(referenceYearly, static, args.years, "reference"), on="year")

    if args.iamc:
        release = pd.read_excel(args.iamc, sheet_name="data")
        sub = release[(release.Variable == "Emissions|CO2|AFOLU")
                      & (release.Model == args.model) & (release.Scenario == args.scenario)]
        if sub.empty:
            raise SystemExit(f"no reported AFOLU for {args.model} / {args.scenario}")
        reported = sub[[str(y) for y in args.years]].sum()
        table["reported AFOLU"] = (reported.to_numpy() / 1e3).round(2)

    print(table.to_string(index=False))
    print("\nstock in Pg C, flux in Gt CO2/yr, positive a source to the atmosphere")

    if args.iamc:
        years = np.asarray(args.years)
        claimed = carbon.cumulative(reported, years)
        print(f"\nover {years[0]}-{years[-1]} the scenario reports the land "
              f"{'gaining' if claimed > 0 else 'losing'} {abs(claimed):.0f} Pg C")
        for label, yearly in ([(args.label, oursYearly)]
                              + ([("reference", referenceYearly)] if args.reference else [])):
            stocks = np.array([float(carbon.total_stock(y, static).sum()) for y in yearly])
            need = carbon.implied_regrowth_fraction(yearly, static, years, claimed)
            says = carbon.effective_fraction(yearly[-1], static)
            source = "its own secmb" if "secmb" in yearly[-1].data_vars else "the prior"
            print(f"  {label:9s} says {stocks[-1] - stocks[0]:+7.1f} Pg C, with secondary land at "
                  f"{says:.2f} of potential from {source}")
            print(f"  {'':9s} reconciling would need {need:.2f} "
                  f"({'admissible' if 0 <= need <= 1 else 'NOT ADMISSIBLE, above potential vegetation'})")

    if args.reference and args.mask and args.mapping:
        index_i, index_j, region = luh.region_of_cell(args.mask, args.mapping, static)
        index = (index_i, index_j)
        year = args.decompose_year
        split = carbon.decompose(oursYearly[args.years.index(year)],
                                 referenceYearly[args.years.index(year)],
                                 static, index, region, year)
        print(f"\n{args.label} minus reference at {year}: {split}"
              f"\n(the priors are used on both sides here, so a secmb the reference "
              f"carries does not make the two incomparable)")
        gap = (split.area["a"] - split.area["b"]).sum()
        print(f"\nby state, {args.label} minus reference:")
        out = pd.DataFrame({
            "area gap Mha": gap.round(1),
            "kg C/m2": (split.mean_density["a"] * split.area["a"]).sum() /
                            split.area["a"].sum(),
            "reference kg C/m2": (split.mean_density["b"] * split.area["b"]).sum() /
                                 split.area["b"].sum(),
        })
        out["density gap %"] = (100 * (out["kg C/m2"] / out["reference kg C/m2"] - 1))
        print(out.round(3).to_string())

    sens = carbon.sensitivity(oursYearly, static, np.asarray(args.years))
    print(f"\nsensitivity of the period-mean flux ({sens.attrs['base_flux']:.2f} Gt CO2/yr) "
          f"to each prior, moved +/-20%:")
    print(sens.head(6).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
