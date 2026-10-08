"""Audit an IAMC-format scenario release for land-use downscaling readiness.

Answers three questions about a ScenarioMIP (or any IAMC) release before any
downscaling work starts:

1. Does it carry sub-global regions?  Downscaling needs sub-global regions;
   a World-only release cannot drive it at all.  For a regional release the
   question sharpens: is the land tree reported in *every* region, or only in
   some?  Several IAMs report land cover globally but thin out regionally, and
   that is the risk this script exists to measure.
2. Which land-cover, wood-harvest and AFOLU variables does each scenario
   actually report?  The LUH category set is not fully determined by IAMC
   reporting, and which priors we need depends on what is missing per model.
3. Which IAM produced each ScenarioMIP marker?  Read off the "(Marker)" tag
   rather than assumed.

Needs pandas and openpyxl, which are deliberately not graft dependencies:

    pip install pandas openpyxl
    python scripts/iamc_coverage.py ScenarioMIP_v0.1_global.xlsx
    python scripts/iamc_coverage.py ScenarioMIP_v0.1_R10.xlsx --export land_r10.csv

The --export subset is the land tree plus the wood harvest and fertiliser
variables, which is small enough to move around freely and is the input a
readIAMC path would consume.
"""

from __future__ import annotations

import argparse
import sys

# Variables that matter for the IAMC -> LUH translate stage. The comment on
# each says what it feeds; "prior" means IAMC cannot determine it and the
# mapping has to supply an assumption.
LAND_VARIABLES = [
    "Land Cover",                                    # closure check
    "Land Cover|Cropland",                           # c3ann+c4ann+c3per+c4per+c3nfx (C3/C4 split is a prior)
    "Land Cover|Cropland|Cereals",                   # annual crops
    "Land Cover|Cropland|Energy Crops",              # biofuel management layer
    "Land Cover|Cropland|Trees",                     # perennial crops
    "Land Cover|Pasture",                            # pastr+range (split is a prior)
    "Land Cover|Forest",
    "Land Cover|Forest|Primary",                     # primf
    "Land Cover|Forest|Secondary",                   # secdf
    "Land Cover|Forest|Planted",                     # secdf, managed fraction
    "Land Cover|Built-Up Area",                      # urban
    "Land Cover|Other Natural",                      # primn/secdn (split is a prior)
    "Land Cover|Other Land",
    "Forestry Production|Roundwood",                 # wood harvest demand
    "Forestry Production|Roundwood|Industrial Roundwood",
    "Emissions|CO2|AFOLU",                           # carbon-consistency target
    "Gross Emissions|CO2|AFOLU",                     # transition channel target
    "Gross Removals|CO2|AFOLU",                      # regrowth channel target
    "Carbon Removal|Land Use|Re/Afforestation",
    "Primary Energy|Biomass|Energy Crops",
]

# The top-level partition of Land Cover. Children should sum to the total; a
# model that nests one inside another (AIM's Other Land) shows as > 1.
PARTITION = [
    "Land Cover|Cropland",
    "Land Cover|Pasture",
    "Land Cover|Forest",
    "Land Cover|Built-Up Area",
    "Land Cover|Other Natural",
    "Land Cover|Other Land",
]

# Variables the nonland stage needs: wood harvest demand and fertiliser.
# IAMC reports one harvest volume, where LUH wants harvested carbon and area
# per source forest, so the split is a prior taken from LUH's own history
# (mrdownscale's toolIAMCWoodHarvest).
NONLAND_VARIABLES = [
    "Forestry Production|Roundwood",                       # wood harvest demand
    "Forestry Production|Roundwood|Industrial Roundwood",
    "Forestry Production|Roundwood|Wood Fuel",             # rndwd/fulwd split
    "Fertilizer Use|Nitrogen",                             # LUH fertiliser layer
    "Fertilizer Use|Nitrogen|Synthetic",
]

CHECK_YEARS = ("2020", "2050", "2100")


def load(path: str):
    import pandas as pd

    if path.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path, sheet_name="data")
    return pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="IAMC wide-format .xlsx (sheet 'data') or .csv")
    ap.add_argument(
        "--scenarios",
        nargs="*",
        help="scenario names to audit; default is every scenario tagged (Marker)",
    )
    ap.add_argument("--region", help="restrict the audit to a single region")
    ap.add_argument(
        "--export",
        metavar="PATH",
        help="write a tidy CSV of just the land variables for the audited scenarios",
    )
    args = ap.parse_args(argv)

    import pandas as pd

    df = load(args.path)
    years = [c for c in df.columns if str(c).isdigit()]

    if args.region:
        df = df[df["Region"] == args.region]
        if df.empty:
            print(f"no rows for region {args.region!r}", file=sys.stderr)
            return 1

    regions = sorted(df["Region"].unique())
    print(f"regions ({len(regions)}): {', '.join(regions[:12])}"
          + (" ..." if len(regions) > 12 else ""))
    if regions == ["World"]:
        print("  !! World only - this release cannot drive spatial downscaling.")
        print("     A native-model-region release is required.")
    print(f"year columns: {years[0]}..{years[-1]} (reported steps vary by model, below)")
    print(f"variables: {df['Variable'].nunique()}")

    if args.scenarios:
        pairs = (
            df[df["Scenario"].isin(args.scenarios)][["Model", "Scenario"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
    else:
        pairs = (
            df[df["Scenario"].str.contains(r"\(Marker\)", na=False)][["Model", "Scenario"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
    pairs = sorted(pairs, key=lambda r: r.Scenario)
    if not pairs:
        print("no matching scenarios", file=sys.stderr)
        return 1

    print("\nscenario -> model, and the years it actually reports Land Cover")
    for r in pairs:
        lc = df[(df["Model"] == r.Model) & (df["Scenario"] == r.Scenario)
                & (df["Variable"] == "Land Cover")]
        steps = [y for y in years if lc[y].notna().all()] if len(lc) else []
        print(f"  {r.Scenario:35s} {r.Model:35s} {len(steps):3d} steps")

    labels = [r.Scenario.split(" - ")[0] for r in pairs]
    width = max(len(s) for s in LAND_VARIABLES) + 2
    print("\ncoverage  (x every region, N/M that many of M regions, - none)")
    print("a variable reported globally but in only some regions shows as N/M,")
    print("and every such case needs a gap-filling decision before downscaling.\n")
    print(" " * width + "".join(f"{s[:11]:>13s}" for s in labels))
    for var in LAND_VARIABLES:
        cells = []
        for r in pairs:
            sub = df[(df["Model"] == r.Model)
                     & (df["Scenario"] == r.Scenario)
                     & (df["Variable"] == var)]
            total = df[(df["Model"] == r.Model)
                       & (df["Scenario"] == r.Scenario)]["Region"].nunique()
            ok = sum(
                1
                for _, row in sub.iterrows()
                if all(pd.notna(row.get(y)) for y in CHECK_YEARS)
            )
            if ok == 0:
                cells.append("-")
            elif ok == total:
                cells.append("x")
            else:
                cells.append(f"{ok}/{total}")
        print(f"{var:{width}s}" + "".join(f"{c:>13s}" for c in cells))

    if len(regions) > 1:
        # Common region names (R5, R10, ...) are unions of whole native regions,
        # so the same name covers different countries in different models. Total
        # land area per region makes that visible; closure checks the partition.
        print("\nland area 2020 (Mha) per region - same name, different countries")
        print("where columns disagree; see IAMconsortium/common-definitions mappings\n")
        lc = df[df["Variable"] == "Land Cover"]
        rw = max(len(s) for s in regions) + 2
        print(" " * rw + "".join(f"{s[:11]:>13s}" for s in labels))
        for reg in regions:
            cells = []
            for r in pairs:
                v = lc[(lc["Model"] == r.Model) & (lc["Scenario"] == r.Scenario)
                       & (lc["Region"] == reg)]["2020"]
                cells.append(f"{v.iloc[0]:.0f}" if len(v) and pd.notna(v.iloc[0]) else "-")
            print(f"{reg:{rw}s}" + "".join(f"{c:>13s}" for c in cells))

    print("\npartition closure: sum of top-level Land Cover children / Land Cover,")
    print("worst case over regions and check years (1.000 = closes)\n")
    for r, label in zip(pairs, labels):
        sub = df[(df["Model"] == r.Model) & (df["Scenario"] == r.Scenario)]
        tot = sub[sub["Variable"] == "Land Cover"].set_index("Region")[list(CHECK_YEARS)]
        kids = sub[sub["Variable"].isin(PARTITION)].groupby("Region")[list(CHECK_YEARS)].sum()
        q = (kids / tot).stack()
        flag = "" if q.sub(1).abs().max() < 0.01 else "  !! does not close"
        print(f"  {label:20s} {q.min():.3f} .. {q.max():.3f}{flag}")

    if args.export:
        keys = {(r.Model, r.Scenario) for r in pairs}
        out = df[
            df["Variable"].isin(LAND_VARIABLES + NONLAND_VARIABLES)
            & df.apply(lambda r: (r["Model"], r["Scenario"]) in keys, axis=1)
        ]
        out.to_csv(args.export, index=False)
        print(f"\nwrote {len(out)} rows to {args.export}: the land tree plus the "
              f"{len(NONLAND_VARIABLES)} wood harvest and fertiliser variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
