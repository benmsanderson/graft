"""Check an annual product's states against its transitions under LUH3's rule.

Runs ``annualise.check`` - every area state, primary land leaving through its
transitions and through primf_harv/primn_harv - on every Nth year pair, since
a full pass reads every transition field for every year (about 50 s a year
at 0.25 degrees). One line per checked year; exit status 1 if any fails.

    python scripts/check_closure.py ~/madrat/output/vlp1_iamc/annual
    python scripts/check_closure.py DIR --stride 1
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from graft import paths

sys.path.insert(0, str(Path(__file__).parent))
import annualise  # noqa: E402

STATIC = str(paths.STATIC)


def only(pattern: str) -> str:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise SystemExit(f"expected one file for {pattern}, found {len(found)}")
    return found[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="folder holding one annual states and one transitions file")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--static", default=STATIC)
    args = ap.parse_args()

    static = xr.open_dataset(args.static)
    states = xr.open_dataset(only(f"{args.folder}/multiple-states_*.nc"), decode_times=False)
    transitions = xr.open_dataset(only(f"{args.folder}/multiple-transitions_*.nc"), decode_times=False)
    units = states.time.attrs["units"]
    epoch = int(units.split("since")[1].strip().split("-")[0])
    years = list(np.round(epoch + states.time.values / 365).astype(int))

    failed = False
    for k in range(0, len(years) - 1, args.stride):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = annualise.check(states.isel(time=[k, k + 1]), transitions.isel(time=[k, k + 1]),
                                     years[k:k + 2], static)
        failed = failed or status != 0
        print(out.getvalue().strip(), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
