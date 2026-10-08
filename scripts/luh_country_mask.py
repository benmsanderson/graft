"""Build a country-per-grid-cell mask on the LUH3 grid, from public data only.

mrdownscale gets country-per-cell from a MAgPIE gdx clustermap, which makes
its regional input paths depend on a model run nobody outside PIK has.
LUH3's own static file carries `ccode`, a country code per 0.25
degree cell, on exactly the target grid, so the mask can be built from the
dataset we are already downscaling to.

Two things need care:

1. `ccode` is ISO 3166-1 numeric, but its country set predates 1990: the
   former Soviet Union is one code (901), as are Yugoslavia (902),
   Czechoslovakia (903) and pre-split Sudan (736); Alaska (900) is separate
   from the USA. IAM regions follow modern borders, so those blocks can
   straddle two regions - for REMIND-MAgPIE at R10, South Sudan (63 Mha) and
   the Baltics (17 Mha).

2. Whether to split them is a real choice, so it is a flag, not an
   assumption. `--boundaries luh` keeps LUH's own accounting domains, which
   would be right if the published LUH3 scenarios were built on them.
   Measured against LUH3-VL, they are not: the Sudan block's area is split
   between Africa and the Middle East in the published product, so
   `--boundaries modern` (the default) is what matches the target. Rerun
   scripts/luh_provenance.py to check that for another release.

Country shapes for the split come from Natural Earth, which is public
domain. Needs numpy, pandas, xarray, netCDF4, pycountry and shapely:

    python scripts/luh_country_mask.py multiple-static_...gn.nc \\
        --out country_cell.csv.gz
    python scripts/luh_country_mask.py multiple-static_...gn.nc \\
        --mapping data/region_mappings/REMIND-MAgPIE_3.5-4.11_R10.csv \\
        --data land_r10.csv --model REMIND-MAgPIE
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

NATURAL_EARTH = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                 "master/geojson/ne_50m_admin_0_countries.geojson")

# ccode values that are not ISO 3166-1 numeric, with the modern country whose
# code we fall back to when a cell cannot be resolved against Natural Earth.
# The blocks are split per cell under --boundaries modern.
COMPOSITE = {
    901: ("RUS", "former Soviet Union"),
    902: ("SRB", "Yugoslavia"),
    903: ("CZE", "Czechoslovakia"),
    736: ("SDN", "Sudan before the 2011 split"),
    900: ("USA", "Alaska"),
    530: (None, "Netherlands Antilles"),
    125: (None, "single unattributed cell"),
}


def iso_by_number() -> dict[int, str]:
    import pycountry

    return {int(c.numeric): c.alpha_3 for c in pycountry.countries
            if getattr(c, "numeric", None)}


def natural_earth(cache: Path):
    """ISO3 -> shapely geometry, from Natural Earth (public domain)."""
    from shapely.geometry import shape

    path = cache / "ne_50m_admin_0_countries.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(NATURAL_EARTH) as r:
            path.write_bytes(r.read())
    out = {}
    for feature in json.loads(path.read_text())["features"]:
        p = feature["properties"]
        iso = p.get("ISO_A3_EH") or p.get("ISO_A3") or p.get("ADM0_A3")
        if not iso or iso == "-99":
            continue
        geometry = shape(feature["geometry"])
        out[iso] = geometry.union(out[iso]) if iso in out else geometry
    return out


# Natural Earth maps the Caribbean Netherlands inside the Netherlands polygon
# at every resolution, so a Bonaire cell resolves to NLD and would follow
# Europe's trajectory rather than Latin America's. Territories that sit far
# from their sovereign need saying explicitly; the bounds are the cell's.
OVERSEAS = [("NLD", (-70, -62, 10, 19), "BES")]


def overseas_override(iso, lon2d, lat2d) -> list[str]:
    """Reassign cells of a sovereign that lie inside a named territory's box."""
    fixed = []
    for sovereign, (west, east, south, north), territory in OVERSEAS:
        hit = ((iso == sovereign) & (lon2d >= west) & (lon2d <= east)
               & (lat2d >= south) & (lat2d <= north))
        if hit.any():
            iso[hit] = territory
            fixed.append(f"{hit.sum()} {sovereign} cells -> {territory}")
    return fixed


def nearest_country(iso, mask, lon2d, lat2d, cache) -> list[str]:
    """Assign the cells selected by `mask` to the nearest country, in place."""
    import numpy as np
    from shapely import STRtree
    from shapely.geometry import Point

    shapes = natural_earth(cache)
    names = list(shapes)
    tree = STRtree([shapes[n] for n in names])

    cells = np.argwhere(mask)
    points = [Point(lon2d[i, j], lat2d[i, j]) for i, j in cells]
    nearest = tree.nearest(points)
    for k, (i, j) in enumerate(cells):
        iso[i, j] = names[int(nearest[k])]
    return sorted({names[int(n)] for n in nearest})


def split_blocks(iso, ccode, lon2d, lat2d, codes, cache) -> int:
    """Resolve cells of composite codes to modern countries, in place."""
    import numpy as np
    from shapely import STRtree
    from shapely.geometry import Point

    shapes = natural_earth(cache)
    names = list(shapes)
    tree = STRtree([shapes[n] for n in names])

    resolved = 0
    for code in codes:
        cells = np.argwhere(ccode == code)
        points = [Point(lon2d[i, j], lat2d[i, j]) for i, j in cells]
        # containment first; nearest catches coastal cells whose centre falls
        # outside every polygon
        hits = tree.query(points, predicate="intersects")
        found = {}
        for point_index, shape_index in zip(hits[0], hits[1]):
            found.setdefault(int(point_index), names[int(shape_index)])
        nearest = None
        missing = [k for k in range(len(points)) if k not in found]
        if missing:
            nearest = tree.nearest([points[k] for k in missing])
        for n, k in enumerate(missing):
            found[k] = names[int(nearest[n])]
        for k, (i, j) in enumerate(cells):
            iso[i, j] = found[k]
            resolved += 1
    return resolved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("static", help="LUH3 multiple-static .nc (carries ccode, carea, icwtr)")
    ap.add_argument("--boundaries", choices=("modern", "luh"), default="modern",
                    help="split LUH's pre-1990 composite codes into modern countries "
                         "(default) or keep LUH's own accounting domains")
    ap.add_argument("--out", help="write the mask as a csv(.gz) of x, y, country")
    ap.add_argument("--mapping", help="region,country CSV; report land area per region")
    ap.add_argument("--data", help="iamc_coverage.py export to check the mapping against")
    ap.add_argument("--model", help="model name prefix to check in --data")
    ap.add_argument("--cache", default=".cache/luh_country_mask")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import xarray as xr

    static = xr.open_dataset(args.static)
    for variable in ("ccode", "carea", "icwtr"):
        if variable not in static:
            print(f"{args.static} has no {variable}", file=sys.stderr)
            return 1
    ccode = static.ccode.values
    land = static.carea.values * (1 - static.icwtr.values) / 1e4  # Mha
    lon2d, lat2d = np.meshgrid(static.lon.values, static.lat.values)

    numeric = iso_by_number()
    iso = np.full(ccode.shape, "", dtype=object)
    for code in np.unique(ccode):
        if not np.isfinite(code) or code == 0:
            continue
        code = int(code)
        fallback = COMPOSITE[code][0] if code in COMPOSITE else numeric.get(code)
        if fallback:
            iso[ccode == code] = fallback

    straddling = [c for c in COMPOSITE if c in ccode and COMPOSITE[c][0]]
    if args.boundaries == "modern":
        n = split_blocks(iso, ccode, lon2d, lat2d, straddling, Path(args.cache))
        print(f"resolved {n} cells of {len(straddling)} composite codes "
              f"({', '.join(COMPOSITE[c][1] for c in straddling)}) to modern countries")
    else:
        print("keeping LUH accounting domains; composite codes stay whole: "
              + ", ".join(f"{COMPOSITE[c][1]} -> {COMPOSITE[c][0]}" for c in straddling))

    # LUH leaves some land cells without a usable country: ccode 0 over the
    # Galapagos, deprecated codes such as the Netherlands Antilles. They are
    # land in the target grid, so leaving them out means the downscaler fills
    # them by nearest neighbour over any distance - up to 1200 km, which can
    # cross a region boundary. Assign them here instead, where the nearest
    # country is at least chosen deliberately.
    orphan = (iso == "") & (land > 0)
    if orphan.any():
        if args.boundaries == "modern":
            countries = nearest_country(iso, orphan, lon2d, lat2d, Path(args.cache))
            print(f"assigned {orphan.sum()} land cells with no usable ccode to the "
                  f"nearest country: {', '.join(countries)}")
        else:
            print(f"{orphan.sum()} land cells have no usable ccode ({land[orphan].sum():.1f} Mha); "
                  "--boundaries modern assigns them to the nearest country")

    for note in overseas_override(iso, lon2d, lat2d):
        print(f"overseas territory: {note}")

    assigned = iso != ""
    print(f"{assigned.sum()} cells carry a country, {land[~assigned].sum():.1f} Mha unassigned")

    if args.out:
        out = pd.DataFrame({"x": lon2d[assigned], "y": lat2d[assigned],
                            "country": iso[assigned]})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out)} cells to {args.out}")

    if args.mapping:
        mapping = pd.read_csv(args.mapping)
        iso2region = dict(zip(mapping.country, mapping.region))
        totals: dict[str, float] = {}
        for country in np.unique(iso[assigned]):
            region = iso2region.get(country)
            if region is None:
                continue
            totals[region] = totals.get(region, 0.0) + land[iso == country].sum()
        regions = pd.Series(totals, name="mask").sort_index()

        if args.data and args.model:
            reported = pd.read_csv(args.data)
            reported = reported[reported.Model.str.startswith(args.model)
                                & (reported.Variable == "Land Cover")]
            reported = reported.set_index("Region")["2020"].sort_index()
            table = pd.concat([regions, reported.rename("reported")], axis=1)
            table["gap"] = table["mask"] - table["reported"]
            table["gap %"] = 100 * table["gap"] / table["reported"]
            print()
            print(table.round({"mask": 0, "reported": 0, "gap": 1, "gap %": 2}).to_string())
            print(f"\ntotal mask {regions.sum():.0f} Mha, reported {reported.sum():.0f} Mha")
        else:
            print()
            print(regions.round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
