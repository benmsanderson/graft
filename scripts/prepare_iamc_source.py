"""Write the source folder mrdownscale's readIAMC expects.

readIAMC reads three files from a madrat source folder named IAMC:

    data.csv            IAMC wide format, one marker
    region_mapping.csv  region,country (ISO3) for that marker's model
    country_cell.csv    x,y,country per grid cell, from luh_country_mask.py

This script assembles them from an iamc_coverage.py --export, a mapping
written by region_masks.py and a mask written by luh_country_mask.py, so
they always belong to the same model:

    python scripts/prepare_iamc_source.py land_r10.csv \\
        data/region_mappings/REMIND-MAgPIE_3.5-4.11_R10.csv \\
        ~/madrat/sources/IAMC --country-cell country_cell.csv.gz

Needs pandas.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def copy(src, dst) -> None:
    """Copy, unless the source folder already holds that very file.

    The mask is the same for every marker, so it is natural to leave it in the
    source folder and point at it there when restaging the next one.
    """
    if Path(src).resolve() == Path(dst).resolve():
        return
    shutil.copyfile(src, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="iamc_coverage.py --export CSV")
    ap.add_argument("mapping", help="region,country CSV from region_masks.py")
    ap.add_argument("out", help="source folder to write, named IAMC")
    ap.add_argument("--country-cell",
                    help="x,y,country mask from luh_country_mask.py; without it the "
                         "source folder is incomplete and readIAMC will say so")
    ap.add_argument("--scenario", default="Very Low - SSP1 (Marker)",
                    help="scenario to keep (default: the VL marker)")
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.data)
    keep = df[df["Scenario"] == args.scenario]
    if keep.empty:
        print(f"scenario {args.scenario!r} not in {args.data}; available: "
              + ", ".join(sorted(df["Scenario"].unique())), file=sys.stderr)
        return 1
    models = sorted(keep["Model"].unique())
    if len(models) > 1:
        print(f"several models report {args.scenario!r}: {models}", file=sys.stderr)
        return 1

    mapping = pd.read_csv(args.mapping)
    missing = sorted(set(keep["Region"]) - set(mapping["region"]) - {"World"})
    if missing:
        print(f"regions with no countries in {args.mapping}: {missing}", file=sys.stderr)
        return 1

    out = Path(args.out)
    if out.name != "IAMC":
        print(f"madrat looks for a folder named IAMC, not {out.name!r}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out / "data.csv", index=False)
    copy(args.mapping, out / "region_mapping.csv")
    if args.country_cell:
        suffix = ".csv.gz" if args.country_cell.endswith(".gz") else ".csv"
        cells = pd.read_csv(args.country_cell)
        uncovered = sorted(set(cells.country) - set(mapping.country))
        if uncovered:
            print(f"note: {len(uncovered)} countries in the mask are not in the region "
                  f"mapping, their cells will be dropped: {', '.join(uncovered)}")
        copy(args.country_cell, out / f"country_cell{suffix}")

    # the land tree drives the downscaling and is reported 5- or 10-yearly,
    # while some other variables are annual; report the grid that matters
    land = keep[keep["Variable"] == "Land Cover"]
    years = [c for c in keep.columns if c.isdigit() and land[c].notna().all()]
    print(f"{out}/data.csv: {models[0]}, {args.scenario}, "
          f"{keep['Region'].nunique()} regions, {keep['Variable'].nunique()} variables, "
          f"land reported in {len(years)} years, {years[0]} to {years[-1]}")
    print(f"{out}/region_mapping.csv: {len(mapping)} countries, "
          f"{mapping['region'].nunique()} regions")
    if args.country_cell:
        print(f"{out}/country_cell{suffix}: {len(cells)} grid cells, "
              f"{cells.country.nunique()} countries")
    else:
        print("no --country-cell given; run scripts/luh_country_mask.py and pass it")
    print("\nrun mrdownscale with input = \"iamc:" + args.scenario + "\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
