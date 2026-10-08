"""Check what each marker model means by a common region (R5, R9, R10).

A common region is a union of each model's native regions, so the same name
covers different countries in different models.  This script
answers two questions before any per-model region mask is built:

1. Where do the markers disagree?  Countries each model assigns to a
   different region of the set, weighted by land area.
2. Does the mapping describe the data?  Each model's reported 2020
   `Land Cover` against the area of the countries its mapping assigns.  A
   cell far from 1 means the land model aggregated differently from the
   energy model's mapping file (MESSAGE and WITCH, via GLOBIOM), and the
   gap is usually one country's area.

Country lists come from IAMconsortium/common-definitions at a pinned commit,
land areas from the World Bank (AG.LND.TOTL.K2, 2020).  Both are cached.
Needs pandas, pyyaml and pycountry, which are deliberately not graft
dependencies:

    pip install pandas pyyaml pycountry
    python scripts/region_masks.py R10
    python scripts/region_masks.py R10 --data land_r10.csv
    python scripts/region_masks.py R10 --write data/region_mappings \
        --models "REMIND-MAgPIE 3.5-4.11"

The --data file is an --export from scripts/iamc_coverage.py.  --write emits
one region,country (ISO3) CSV per model, the shape mrdownscale's COFFEE path
reads as a regionMapping; only write a model whose cells pass the --data
check, since a failing cell means the mapping does not describe its data.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

COMMON_DEFINITIONS = "IAMconsortium/common-definitions"
PINNED_REF = "e31d4672022047e48b3794cfa83ff3c7aa034474"  # main, 2026-09-18

# Marker model -> (directory, mapping file, native-region file).  The two
# files are named differently for IMAGE and MESSAGE, so they are listed
# rather than derived.
MARKERS = {
    "AIM 3.0": ("AIM", "AIM_3.0", "AIM_3.0"),
    "COFFEE 1.6": ("COFFEE", "COFFEE_1.6", "COFFEE_1.6"),
    "GCAM 8s": ("GCAM", "GCAM_8s", "GCAM_8s"),
    "IMAGE 3.4": ("IMAGE", "IMAGE_v3.4", "IMAGE_v3.4"),
    "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12": (
        "MESSAGEix-GLOBIOM",
        "MESSAGEix-GLOBIOM-GAINS_2.1-M-R12",
        "MESSAGEix-GLOBIOM-GAINS_2.1-R12",
    ),
    "REMIND-MAgPIE 3.5-4.11": (
        "REMIND-MAgPIE", "REMIND-MAgPIE_3.5-4.11", "REMIND-MAgPIE_3.5-4.11",
    ),
    "WITCH 6.0": ("WITCH", "WITCH_6.0", "WITCH_6.0"),
}

WORLD_BANK = (
    "https://api.worldbank.org/v2/country/all/indicator/AG.LND.TOTL.K2"
    "?format=json&per_page=20000&date=2020"
)

# common-definitions names pycountry cannot resolve, needed both for land
# areas and for the ISO3 codes --write emits.  XKX (Kosovo) is the World Bank
# code, not ISO 3166.  AREA_FALLBACK covers areas the World Bank does not
# report (Mha); smaller unreported territories count as zero.
ISO3_ALIAS = {
    "Democratic Republic of the Congo": "COD",
    "Kosovo": "XKX",
    "Micronesia": "FSM",
    "North Korea": "PRK",
    "Palestine": "PSE",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Taiwan": "TWN",
    "Turkey": "TUR",
    "United States Virgin Islands": "VIR",
    "Vatican": "VAT",
    "Western Sahara": "ESH",
}
AREA_FALLBACK = {"TWN": 3.6, "GRL": 41.0, "ESH": 26.6}


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path


def _pair(x):
    return next(iter(x.items())) if isinstance(x, dict) else (x, x)


def load_model(model: str, cache: Path, ref: str) -> dict[str, set[str]]:
    """Common region name -> set of countries, for one model."""
    import yaml

    folder, mapping, native = MARKERS[model]
    base = f"https://raw.githubusercontent.com/{COMMON_DEFINITIONS}/{ref}"
    mp = yaml.safe_load(fetch(
        f"{base}/mappings/{folder}/{mapping}.yaml",
        cache / ref[:12] / f"map_{mapping}.yaml",
    ).read_text())
    nat = yaml.safe_load(fetch(
        f"{base}/definitions/region/native_regions/{folder}/{native}.yaml",
        cache / ref[:12] / f"native_{native}.yaml",
    ).read_text())

    # native_regions lists "code: full name" pairs and, for some models,
    # bare names too; the explicit pair must win over a bare repeat.
    code_to_name = {}
    for x in mp["native_regions"]:
        code, name = _pair(x)
        if isinstance(x, dict) or code not in code_to_name:
            code_to_name[code] = name

    name_to_countries = {}
    for block in nat:
        for regions in block.values():
            for r in regions:
                name, spec = _pair(r)
                c = spec.get("countries", []) if isinstance(spec, dict) else []
                name_to_countries[name] = [c] if isinstance(c, str) else c

    out = {}
    for cr in mp.get("common_regions", []):
        region, codes = _pair(cr)
        out[region] = {
            c for code in codes
            for c in name_to_countries.get(code_to_name.get(code, code), [])
        }
    return out


def land_area(cache: Path):
    """Country name -> land area in Mha."""
    import pycountry

    rows = json.loads(fetch(WORLD_BANK, cache / "wb_land_2020.json").read_text())[1]
    by_iso = {r["countryiso3code"]: r["value"] / 1e4 for r in rows if r["value"]}

    def area(country: str) -> float:
        iso = ISO3_ALIAS.get(country)
        if iso is None:
            try:
                iso = pycountry.countries.lookup(country).alpha_3
            except LookupError:
                return 0.0
        return by_iso.get(iso, AREA_FALLBACK.get(iso, 0.0))

    return area


def short(model: str) -> str:
    return model.split()[0].split("-")[0][:7]


def disagreement(maps, tag: str, area) -> None:
    suffix = f" ({tag})"
    assign = collections.defaultdict(dict)  # country -> model -> region
    for model, regions in maps.items():
        for region, countries in regions.items():
            if region.endswith(suffix):
                for c in countries:
                    assign[c][model] = region[: -len(suffix)]
    present = [m for m in maps if any(r.endswith(suffix) for r in maps[m])]
    if not present:
        print(f"no model maps to {tag}", file=sys.stderr)
        return
    split = {
        c: a for c, a in assign.items()
        if len(set(a.values())) > 1 or len(a) < len(present)
    }
    total = sum(area(c) for c in assign)
    split_area = sum(area(c) for c in split)
    print(f"{tag}: {len(present)} of {len(maps)} markers map to it; "
          f"{len(split)} of {len(assign)} countries are assigned differently,")
    print(f"  {split_area:.0f} Mha = {100 * split_area / total:.1f}% of land "
          "(countries under 1 Mha not listed)\n")
    for c, a in sorted(split.items(), key=lambda kv: -area(kv[0])):
        if area(c) < 1:
            continue
        groups = collections.defaultdict(list)
        for m in present:
            groups[a.get(m, "<none>")].append(short(m))
        print(f"  {c:26s}{area(c):5.0f}  "
              + " | ".join(f"{k}: {','.join(v)}" for k, v in groups.items()))


def area_check(maps, tag: str, area, path: str) -> None:
    import pandas as pd

    df = pd.read_csv(path)
    df = df[df["Variable"] == "Land Cover"]
    rows = []
    for model, regions in maps.items():
        for region, countries in regions.items():
            if not region.endswith(f" ({tag})"):
                continue
            rep = df[(df["Model"] == model) & (df["Region"] == region)]["2020"]
            if len(rep):
                mask = sum(area(c) for c in countries)
                rows.append((short(model), region, mask, rep.iloc[0]))
    if not rows:
        print(f"\nno {tag} Land Cover rows in {path}", file=sys.stderr)
        return
    t = pd.DataFrame(rows, columns=["model", "region", "mask", "reported"])
    t["ratio"] = t["reported"] / t["mask"]
    t["gap"] = t["reported"] - t["mask"]
    print(f"\nreported 2020 Land Cover / area of the countries the mapping assigns")
    print(f"median {t.ratio.median():.3f}; cells off by more than 10%, with the gap "
          "in Mha\n(a gap near one country's area names the likely culprit):\n")
    bad = t[t.ratio.sub(1).abs() > 0.10].sort_values(["model", "region"])
    print(bad.round({"mask": 0, "reported": 0, "ratio": 3, "gap": 0})
          .to_string(index=False) if len(bad) else "  none")
    ok = sorted(set(t.model) - set(bad.model))
    print(f"\nevery cell within 10%: {', '.join(ok) or 'none'}")


def iso3(country: str) -> str | None:
    import pycountry

    if country in ISO3_ALIAS:
        return ISO3_ALIAS[country]
    try:
        return pycountry.countries.lookup(country).alpha_3
    except LookupError:
        return None


def write_mappings(maps, tag: str, out: Path, models) -> None:
    """One region,country CSV per model, the shape mrdownscale's readCOFFEE
    regionMapping takes; region names match the IAMC Region column."""
    import pycountry

    suffix = f" ({tag})"
    every_iso = {c.alpha_3 for c in pycountry.countries}
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {tag} region,country mappings to {out}/")
    for model in models:
        rows, unresolved, seen = [], [], collections.Counter()
        for region, countries in sorted(maps[model].items()):
            if not region.endswith(suffix):
                continue
            for c in sorted(countries):
                code = iso3(c)
                if code is None:
                    unresolved.append(c)
                    continue
                rows.append((region, code))
                seen[code] += 1
        if not rows:
            print(f"  {model}: no {tag} regions, skipped")
            continue
        twice = sorted(k for k, n in seen.items() if n > 1)
        missing = sorted(every_iso - set(seen))
        path = out / f"{model.replace(' ', '_')}_{tag}.csv"
        with path.open("w") as f:
            f.write("region,country\n")
            f.writelines(f'"{r}",{c}\n' for r, c in rows)
        print(f"  {path.name}: {len(rows)} countries"
              + (f"; in two regions: {', '.join(twice)}" if twice else "")
              + (f"; unresolved names: {', '.join(unresolved)}" if unresolved else "")
              + (f"; ISO 3166 codes not mapped: {', '.join(missing)}" if missing else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag", help="common region set, e.g. R5, R9, R10")
    ap.add_argument("--data", help="iamc_coverage.py --export CSV to check against")
    ap.add_argument("--write", metavar="DIR",
                    help="write one region,country CSV per model into DIR")
    ap.add_argument("--models", nargs="*", default=list(MARKERS),
                    help="models to write (default: all seven markers)")
    ap.add_argument("--ref", default=PINNED_REF,
                    help="common-definitions commit or branch (default: pinned)")
    ap.add_argument("--cache", default=".cache/region_masks",
                    help="where downloaded definitions and areas are kept")
    args = ap.parse_args(argv)

    cache = Path(args.cache)
    area = land_area(cache)
    maps = {m: load_model(m, cache, args.ref) for m in MARKERS}
    disagreement(maps, args.tag, area)
    if args.data:
        area_check(maps, args.tag, area, args.data)
    if args.write:
        write_mappings(maps, args.tag, Path(args.write), args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
