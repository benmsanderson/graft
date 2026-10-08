"""Compare two builds of the same product, file by file and variable by variable.

For checking that a release build reproduces an earlier one: every netCDF file
in the first folder is matched by kind (states, transitions, management,
secondary) and period to one in the second, and every variable is compared on
every ``--stride``-th year. Reports the largest absolute difference per file
and fails if any exceeds ``--tolerance``.

    python scripts/compare_products.py ~/madrat/output/vlr01_iamc ~/madrat/output/vlx7_iamc
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def key(path: Path) -> tuple[str, str, str]:
    """(subfolder, kind, period) of a product file, whatever its date stamp."""
    kind = re.match(r"multiple-([a-z]+)_", path.name)
    period = re.search(r"_(\d{4}-\d{4})\.nc$", path.name)
    sub = path.parent.name if path.parent.name in ("annual", "extension") else ""
    return (sub, kind.group(1) if kind else path.stem, period.group(1) if period else "")


def files(folder: Path) -> dict:
    found = {}
    for sub in ("", "annual", "extension"):
        for f in sorted((folder / sub).glob("*.nc")) if (folder / sub).is_dir() else []:
            found[key(f)] = f
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("new")
    ap.add_argument("reference")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=1e-5,
                    help="largest absolute difference allowed, in the variable's own units")
    args = ap.parse_args()
    new, ref = files(Path(args.new)), files(Path(args.reference))
    failed = False
    for k in sorted(set(new) | set(ref)):
        if k not in new or k not in ref:
            print(f"{'/'.join(k)}: only in {'new' if k in new else 'reference'}")
            failed = True
            continue
        a, b = xr.open_dataset(new[k], decode_times=False), xr.open_dataset(ref[k], decode_times=False)
        if a.sizes.get("time") != b.sizes.get("time") or set(a.data_vars) != set(b.data_vars):
            print(f"{'/'.join(k)}: different shape or variables")
            failed = True
            continue
        worst, where = 0.0, ""
        for v in a.data_vars:
            if "time" not in a[v].dims:
                continue
            for t in range(0, a.sizes["time"], args.stride):
                x = np.asarray(a[v].isel(time=t).values, dtype="float64")
                y = np.asarray(b[v].isel(time=t).values, dtype="float64")
                if not np.array_equal(np.isnan(x), np.isnan(y)):
                    worst, where = np.inf, f"{v} (missing-value mask) at step {t}"
                    break
                d = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
                if d > worst:
                    worst, where = d, f"{v} at step {t}"
        ok = worst <= args.tolerance
        failed = failed or not ok
        print(f"{'/'.join(k)}: max |diff| {worst:.3g}{' (' + where + ')' if worst else ''} - {'ok' if ok else 'DIFFERS'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
