"""One row per marker: what its forcing says about carbon, and what it reported.

`carbon_check.py` examines one product in detail. This runs the same
bookkeeping across a whole set and puts the answers side by side, which is
where the cross-marker results live - both the reconciling fraction, and
whether the primary-forest trajectory responds to the scenario at all.

Only VL and H are published as LUH3 products, so the others cannot borrow a
`secmb` and are valued with the prior throughout. The prior cannot represent
secondary land ageing, so it understates the regrowth sink and biases the
carbon column towards loss. The reconciling fraction does not care: it solves
for the constant fraction that would close the gap whatever the stock used, so
it is the column to compare across markers.

    python scripts/carbon_summary.py ~/madrat/output --static STATIC.nc \\
        --iamc ScenarioMIP_v0.1_R10.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from graft import carbon
from graft.compare import fold_for_scoring, global_area
from graft.io import luh

#: folder tag -> the model and scenario it was built from
MARKERS = {
    "vl": ("REMIND-MAgPIE 3.5-4.11", "Very Low - SSP1 (Marker)"),
    "l": ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "Low - SSP2 (Marker)"),
    "ln": ("AIM 3.0", "Low-to-Negative - SSP2 (Marker)"),
    "m": ("IMAGE 3.4", "Medium - SSP2 (Marker)"),
    "ml": ("COFFEE 1.6", "Medium-to-Low - SSP2 (Marker)"),
    "h": ("GCAM 8s", "High - SSP3 (Marker)"),
    "hl": ("WITCH 6.0", "High-to-Low - SSP5 (Marker)"),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outputs", help="madrat output folder holding <tag>_iamc runs")
    ap.add_argument("--static", required=True)
    ap.add_argument("--iamc", required=True)
    ap.add_argument("--years", nargs="+", type=int,
                    default=[2025, 2030, 2040, 2050, 2070, 2100])
    ap.add_argument("--csv", help="also write the table here")
    args = ap.parse_args(argv)

    static = luh.open_static(args.static)
    carea = np.asarray(static["carea"].values, dtype="float64")
    release = pd.read_excel(args.iamc, sheet_name="data")
    years = np.asarray(args.years)

    rows = []
    for tag, (model, scenario) in MARKERS.items():
        found = sorted(Path(args.outputs).glob(f"{tag}_iamc/multiple-states_*.nc"))
        if not found:
            print(f"{tag}: no product, skipped")
            continue
        states = fold_for_scoring(luh.open_states(found[-1]))
        yearly = [luh.select_year(states, y) for y in args.years]
        stocks = np.array([float(carbon.total_stock(y, static).sum()) for y in yearly])

        sub = release[(release.Variable == "Emissions|CO2|AFOLU")
                      & (release.Model == model) & (release.Scenario == scenario)]
        reported = carbon.cumulative(sub[[str(y) for y in args.years]].sum(), years)

        first, last = (global_area(yearly[0], carea), global_area(yearly[-1], carea))
        span = args.years[-1] - args.years[0]
        rows.append({
            "marker": tag,
            "model": model.split()[0],
            "forcing Pg C": stocks[-1] - stocks[0],
            "reported Pg C": reported,
            "reconciling frac": carbon.implied_regrowth_fraction(yearly, static, years, reported),
            "primf Mha/yr": (last["primf"] - first["primf"]) / span,
            "forest Mha": (last["primf"] + last["secdf"]) - (first["primf"] + first["secdf"]),
        })

    table = pd.DataFrame(rows)
    print(f"\n{args.years[0]}-{args.years[-1]}, secondary land at the prior throughout\n")
    print(table.round(2).to_string(index=False))
    if len(table):
        print(f"\nprimary forest declines at {table['primf Mha/yr'].min():.2f} to "
              f"{table['primf Mha/yr'].max():.2f} Mha/yr across the set, a spread of "
              f"{table['primf Mha/yr'].max() - table['primf Mha/yr'].min():.2f}; "
              f"fadeForest extrapolates 10.49")
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
