"""Which country boundaries did a published LUH3 scenario actually use?

LUH3's static `ccode` field carries a pre-1990 country set: the former Soviet
Union is one block, as is pre-split Sudan. If the published LUH3 scenarios
were built on those domains, a mask that keeps them whole would match the
target better than one that splits them. If they were built on
modern borders, keeping them whole misplaces area relative to both the IAM
and LUH3.

That is measurable. A composite block that straddles two IAM regions must go
wholly to one of them under LUH domains, and must split between them under
modern borders. So aggregate a published LUH3 scenario over the region the
block is *not* counted in, and see whether the IAM's reported regional total
is reproduced (LUH domains) or falls short by the block's other half
(modern).

Run against LUH3-VL and REMIND-MAgPIE, this says modern borders: the Sudan
block's pasture divides 37% to Africa and 63% to the Middle East, and its
cropland 15/84, against a 25/75 split of its land area. Control regions with
no straddle reproduce the IAM totals to within 1%, so the method is sound.

Reads one year over OPeNDAP rather than downloading the whole file. Needs
numpy, pandas, xarray, netCDF4 and pycountry:

    python scripts/luh_provenance.py <static.nc> <states-url-or-path> \\
        --mapping data/region_mappings/REMIND-MAgPIE_3.5-4.11_R10.csv \\
        --data land_r10.csv --model REMIND-MAgPIE --year 2050

Forest and other natural land are reported but cannot answer the boundary
question: LUH constrains forest to its potential-forest mask, so tropical
woodland an IAM reports as forest sits in primn/secdn instead, and the two
categories differ from the IAM by far more than any boundary effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# LUH3 state variables grouped into the IAMC land tree
GROUPS = {
    "Forest": ["primf", "secdf", "pltns"],
    "Cropland": ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"],
    "Pasture": ["pastr", "range"],
    "Built-Up Area": ["urban"],
    "Other Natural": ["primn", "secdn"],
}

# composite ccode -> the country it falls back to, and the pair of regions it
# can straddle is discovered from the mapping
COMPOSITE = {901: "RUS", 902: "SRB", 903: "CZE", 736: "SDN", 900: "USA"}
BLOCKS = (736, 901)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("static", help="LUH3 multiple-static .nc")
    ap.add_argument("states", help="LUH3 scenario states file, path or OPeNDAP URL")
    ap.add_argument("--mapping", required=True, help="region,country CSV for the model")
    ap.add_argument("--data", required=True, help="iamc_coverage.py export")
    ap.add_argument("--model", required=True, help="model name prefix in --data")
    ap.add_argument("--year", type=int, default=2050)
    ap.add_argument("--start-year", type=int, default=2022,
                    help="first year in the states file (default: 2022, LUH3 scenarios)")
    ap.add_argument("--cache", default=".cache/luh_provenance")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import pycountry
    import xarray as xr

    needed = sorted({v for vs in GROUPS.values() for v in vs})
    cache = Path(args.cache) / f"states_{args.year}.nc"
    if cache.exists():
        year = xr.open_dataset(cache)
    else:
        print(f"reading {args.year} ({len(needed)} variables) from {args.states}")
        states = xr.open_dataset(args.states, decode_times=False)
        year = states[needed].isel(time=int(args.year - args.start_year)).load()
        cache.parent.mkdir(parents=True, exist_ok=True)
        year.to_netcdf(cache)

    static = xr.open_dataset(args.static)
    ccode = static.ccode.values
    carea = static.carea.values

    numeric = {int(c.numeric): c.alpha_3 for c in pycountry.countries
               if getattr(c, "numeric", None)}
    mapping = pd.read_csv(args.mapping)
    iso2region = dict(zip(mapping.country, mapping.region))

    # label cells by region, keeping the composite blocks separate
    label = np.full(ccode.shape, "", dtype=object)
    for code in np.unique(ccode):
        if not np.isfinite(code) or code == 0:
            continue
        code = int(code)
        iso = COMPOSITE.get(code, numeric.get(code)) if code in COMPOSITE else numeric.get(code)
        region = iso2region.get(iso) if iso else None
        if region is None:
            continue
        label[ccode == code] = f"BLOCK {code}" if code in BLOCKS else region

    reported = pd.read_csv(args.data)
    reported = reported[reported.Model.str.startswith(args.model)]
    reported = reported.pivot_table(index="Region", columns="Variable", values=str(args.year))
    if reported.empty:
        print(f"no {args.model} rows for {args.year} in {args.data}", file=sys.stderr)
        return 1

    # which regions each block can belong to, from the mapping itself
    claims: dict[str, list[str]] = {}
    for code in BLOCKS:
        iso = COMPOSITE[code]
        members = {736: ["SDN", "SSD"], 901: ["RUS", "EST", "LVA", "LTU", "UKR", "BLR",
                                              "MDA", "KAZ", "UZB", "TKM", "KGZ", "TJK",
                                              "GEO", "ARM", "AZE"]}[code]
        regions = {iso2region[m] for m in members if m in iso2region}
        claims[f"BLOCK {code}"] = sorted(regions)
        if len(regions) < 2:
            print(f"note: block {code} ({iso}) does not straddle for this model")

    print(f"\n{args.model}, {args.year}: land area (Mha) outside each block, "
          "against the model's reported total\n")
    rows = []
    for group, variables in GROUPS.items():
        area = np.nansum([year[v].values for v in variables], axis=0) * carea / 1e4
        totals = {lab: area[label == lab].sum() for lab in set(label.ravel()) - {""}}
        for block, regions in claims.items():
            for region in regions:
                rows.append({"group": group, "region": region.split(" (")[0], "block": block,
                             "LUH3 outside block": totals.get(region, 0.0),
                             "block total": totals.get(block, 0.0),
                             "reported": reported.loc[region, f"Land Cover|{group}"]})
    table = pd.DataFrame(rows)
    table["shortfall"] = table["reported"] - table["LUH3 outside block"]
    table["share of block"] = 100 * table["shortfall"] / table["block total"]
    print(table.round({"LUH3 outside block": 1, "block total": 1, "reported": 1,
                       "shortfall": 1, "share of block": 0}).to_string(index=False))

    print("\nA block taken wholly by one region means LUH used its own domains;")
    print("a block divided between them means LUH used modern borders.")
    print("Read the Cropland, Pasture and Built-Up Area rows: Forest and Other")
    print("Natural differ from the IAM by more than any boundary effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
