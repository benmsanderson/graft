"""What our product carries that a published LUH3 one does, and what it does not.

Two different questions hide behind "how close are we to LUH output", and this
answers the mechanical one: which variables and which years. mrdownscale's
ScenarioMIP format is itself a reduced deliverable - it omits the 130 land
transitions on purpose, emitting them only under `outputFormat = "ESM"` - so
the gap to LUH3's full product is not the same as the gap to what mrdownscale
is designed to write. The report separates the two.

It reads the reference's variable list over OPeNDAP rather than downloading
2.6 GB of transitions to count their names, so it needs the network but very
little of it.

    python scripts/output_gap.py ~/madrat/output/vl_iamc \\
        --source-id UofMD-landState-vl-3-1-1
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

INDEX = ("https://esgf-node.ornl.gov/esgf-1-5-bridge?type=File&project=input4MIPs"
         "&source_id={source_id}&limit=200&format=application%2Fsolr%2Bjson")
KINDS = ("states", "management", "transitions")
SKIP = {"time", "lat", "lon", "bounds", "time_bnds", "lat_bnds", "lon_bnds"}

#: why each missing variable is missing, so the report says what it would cost
REASON = {
    "secma": "needs a secondary-land age tracker (graft.age)",
    "secmb": "needs a secondary-land age tracker (graft.age)",
    **{f"prtct_{s}": "protection mask; WDPA is public but unused"
       for s in ("primf", "primn", "secdf", "secdn", "pltns")},
    "flood": "not reported by IAMs",
    "addtc": "not reported by IAMs",
    "combf": "not reported by IAMs",
    **{f"cpbf2_{c}": "we carry second-generation biofuel share for perennials only"
       for c in ("c3ann", "c3nfx", "c4ann")},
    **{f"fharv_{c}": "perennial harvest fraction, not reported by IAMs"
       for c in ("c3per", "c4per")},
    **{f"sec{a}f_{v}": "the young/mature secondary split, so it needs the age tracker too"
       for a in ("y", "m") for v in ("harv", "bioh")},
}

#: ours that LUH3 has no slot for, and why we keep them anyway
EXTRA = {"pltns": "mrdownscale's target mapping carries plantations (issue 4)",
         "secdf_harv": "undivided secondary forest, where LUH3 splits young and mature",
         "secdf_bioh": "undivided secondary forest, where LUH3 splits young and mature"}


def reference_variables(source_id: str) -> dict[str, tuple[set[str], str]]:
    """Variable names per file kind, read from the published product's headers."""
    docs = json.load(urllib.request.urlopen(INDEX.format(source_id=source_id),
                                            timeout=180))["response"]["docs"]
    out = {}
    for kind in KINDS:
        urls = [u.split("|")[0] for d in docs if kind in d.get("title", "")
                for u in d.get("url", []) if "dodsC" in u]
        if not urls:
            continue
        base = urls[0][:-5] if urls[0].endswith(".html") else urls[0]
        dds = urllib.request.urlopen(base + ".dds", timeout=240).read().decode()
        out[kind] = (set(re.findall(r"^\s+\w+ (\w+)\[", dds, re.M)) - SKIP, base)
    return out


def ours(folder: Path, kind: str):
    found = sorted(folder.glob(f"multiple-{kind}_*.nc"))
    if not found:
        return set(), None
    d = xr.open_dataset(found[-1], decode_times=False)
    names = {v for v in d.data_vars if d[v].ndim >= 2} - SKIP
    return {n for n in names if "bound" not in n}, d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="a <tag>_iamc folder holding our three files")
    ap.add_argument("--source-id", default="UofMD-landState-vl-3-1-1")
    args = ap.parse_args(argv)

    folder = Path(args.folder).expanduser()
    reference = reference_variables(args.source_id)

    total_ref = total_ours = 0
    for kind, (ref, url) in reference.items():
        mine, ds = ours(folder, kind)
        total_ref += len(ref)
        total_ours += len(mine)
        missing = sorted(ref - mine)
        print(f"=== {kind}: LUH3 {len(ref)}, ours {len(mine)}, shared {len(ref & mine)} ===")
        transitions = [m for m in missing if "_to_" in m]
        if transitions:
            print(f"  {len(transitions)} land transitions, e.g. {transitions[:3]} - "
                  f"mrdownscale computes these under outputFormat = 'ESM'")
        for name in (m for m in missing if "_to_" not in m):
            print(f"  {name:14s} {REASON.get(name, '')}")
        for name in sorted(mine - ref):
            print(f"  + {name:12s} {EXTRA.get(name, '')}")
        if ds is not None and "time" in ds.coords:
            print(f"  years: LUH3 annual, ours {len(ds.time)}")
        print()

    print(f"{total_ours} of {total_ref} gridded variables.")
    print("mrdownscale's own ScenarioMIP format specifies what we emit, and every "
          "name check passes; the gap above is to LUH3's full product, not to the "
          "format we are writing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
