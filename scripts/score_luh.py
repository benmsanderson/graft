"""Score gridded land use against a LUH3 reference, cell by cell.

Every experiment on the pipeline is judged here, so every score is taken the
same way (graft.compare):

- plantations folded into secdf on both sides, since LUH3 carries none;
- one set of cells throughout, the reference's land, with cells ours leaves
  empty counted as zero rather than dropped;
- per group, global totals, the gap, mean absolute error per cell and cell
  correlation;
- forest skill as the grid is aggregated, 0.25 up to 5 degrees;
- with --base-year, how much forest sits and changes on cells LUH's
  potential-forest mask excludes, where LUH itself never changes forest.

Ours is either a CSV from scripts/export_mrdownscale_states.R or a LUH-format
states file; the reference is a LUH states file or an OPeNDAP URL, of which
one year is read and cached. Needs pandas beyond graft's own dependencies:

    python scripts/score_luh.py ours.csv.gz REFERENCE --static static.nc --year 2050
    python scripts/score_luh.py ours.csv.gz REFERENCE --static static.nc --year 2050 \\
        --base-year 2025 --json scores.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from graft import compare as C
from graft import luh_schema as S
from graft.io import luh
from graft.io.mrdownscale import read_states_csv


def load_ours(path: str, static: xr.Dataset, year: int) -> xr.Dataset:
    if path.endswith((".csv", ".csv.gz")):
        return read_states_csv(path, static, year)
    return luh.select_year(luh.open_states(path), year)


def load_reference(source: str, year: int, cache: Path) -> xr.Dataset:
    """One year of a reference file or OPeNDAP URL, cached as a small file."""
    key = hashlib.sha1(source.encode()).hexdigest()[:12]
    cached = cache / f"reference_{key}_{year}.nc"
    if cached.exists():
        return xr.open_dataset(cached)
    ds = xr.open_dataset(source) if "://" in source else luh.open_states(source)
    states = [s for s in S.AREA_STATES if s in ds]
    one = luh.select_year(ds[states], year).load()
    cache.mkdir(parents=True, exist_ok=True)
    one = one.drop_vars("time", errors="ignore")
    one.encoding.pop("unlimited_dims", None)  # the time dimension it named is gone
    one.to_netcdf(cached)
    return one


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ours", help="mrdownscale export CSV, or a LUH-format states file")
    ap.add_argument("reference", help="LUH states file or OPeNDAP URL")
    ap.add_argument("--static", required=True, help="LUH3 static file (carea, and fstnf for --base-year)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--base-year", type=int,
                    help="also report forest on cells LUH's potential-forest mask excludes, "
                         "from this year to --year")
    ap.add_argument("--json", help="write the scores here as well")
    ap.add_argument("--cache", default=".cache/score_luh")
    args = ap.parse_args(argv)

    static = luh.open_static(args.static)
    carea = np.asarray(static[S.STATIC_CELL_AREA].values, dtype="float64")
    ours = load_ours(args.ours, static, args.year)
    reference = load_reference(args.reference, args.year, Path(args.cache))

    report: dict = {"year": args.year, "ours": args.ours, "reference": args.reference}

    cells = C.score_cells(ours, reference, carea)
    report["cells"] = cells.reset_index().to_dict(orient="records")
    print(f"{args.year}: ours against the reference, cell by cell over the reference's land "
          "(plantations folded into secdf)\n")
    print(cells.round({"ours (Mha)": 1, "reference (Mha)": 1, "gap (Mha)": 1, "gap (%)": 1,
                       "cell MAE (kha)": 2, "cell corr": 3}).to_string())

    land = C.land_cells(reference)
    a, b = C.group_areas(ours, carea), C.group_areas(reference, carea)
    resolution = C.score_by_resolution(a["forest"], b["forest"], land)
    report["forest_by_resolution"] = resolution.reset_index().to_dict(orient="records")
    print("\nforest skill as the grid is aggregated (factor 4 = 1 degree, 20 = 5 degrees)\n")
    print(resolution.round(3).to_string())

    if args.base_year is not None:
        if "fstnf" not in static:
            print(f"\n{args.static} has no fstnf; skipping the mask check", file=sys.stderr)
        else:
            base = load_ours(args.ours, static, args.base_year)
            forest = {args.base_year: C.group_areas(base, carea)["forest"], args.year: a["forest"]}
            mask = C.mask_diagnostics(forest, static["fstnf"].values)
            report["mask"] = {"by_year": mask.reset_index().to_dict(orient="records"), **mask.attrs}
            print("\nforest on cells LUH's potential-forest mask excludes (LUH itself never "
                  "changes forest there)\n")
            print(mask.round(1).to_string())
            for name, value in mask.attrs.items():
                print(f"  {args.base_year} -> {args.year} {name}: {value:.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
