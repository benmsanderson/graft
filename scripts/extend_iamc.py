"""Carry an IAMC export past 2100 on the extension protocol's ramp.

The ScenarioMIP extensions take AFOLU CO2 linearly from its 2100 value to zero
in 2149 and hold it there. Land use follows the same ramp here: each region's
land categories keep changing at their end-of-century rate, scaled by a
multiplier that falls linearly from one in 2100 to zero in 2149, so land use is
static from 2150. This is the rule the LUH3 extensions were built on, applied
to the regional input instead of the grid.

- *Rate.* The mean annual change over 2080-2100, which every marker reports at
  both ends; a single last step can be a jump.
- *Land.* Declining categories stop at zero. Expanding ones share whatever room
  the declines leave, so a region's land total stays at its 2100 value.
- *Inside cropland and forest.* Energy crops and the forest parts follow their
  own rates, kept between zero and their parent.
- *Wood demand.* Linearly from its 2100 value to half of it in 2149, then
  held: part-way to the zero the CMIP7 file reaches, because the maintenance
  term that floors it in that design cannot act inside the downscaling run
  (see the paper).
  A model that reports no roundwood (COFFEE) gets an index in its place, so
  the ramp has something to act on: each region's LUH3 2011-2020 harvest
  carbon, constant to 2100 and split by LUH3's own 2020 roundwood and fuel
  shares. mrdownscale calibrates harvest to the target's history by ratio,
  so a constant index reproduces exactly the history it holds when nothing
  is reported; the index is not a volume, only its shape is used.
- *Fertilizer and everything else not named below.* Held at 2100.
- *AFOLU fluxes.* Linearly to zero in 2149, as the protocol prescribes.

Writes the input export with five-yearly columns added to the end year.

    python scripts/extend_iamc.py land_r10.csv land_r10_ext.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL = "Land Cover"
#: the categories that make up a region's land
LAND = ["Land Cover|Cropland", "Land Cover|Pasture", "Land Cover|Forest",
        "Land Cover|Built-Up Area", "Land Cover|Other Natural"]
#: fluxes the protocol takes linearly to zero
FLUXES = ["Emissions|CO2|AFOLU", "Gross Emissions|CO2|AFOLU", "Gross Removals|CO2|AFOLU",
          "Carbon Removal|Land Use|Re/Afforestation"]
#: wood demand, which ramps part-way (see WOOD_END)
WOOD = ["Forestry Production|Roundwood", "Forestry Production|Roundwood|Industrial Roundwood",
        "Forestry Production|Roundwood|Wood Fuel"]
#: the share of its 2100 value wood demand ramps to by 2149
WOOD_END = 0.5
LAST, ZERO = 2100, 2149
#: the end-of-century rate is the mean change over 2060-2100: forty years, so
#: that a model's last reporting step (COFFEE's cropland jumps 2090-2100 in
#: several regions) does not set fifty years of change on its own
RATE_FROM = 2060
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from graft import paths  # noqa: E402

LUH3 = str(paths.LUH3)
STATIC = str(paths.STATIC)
SOURCES = ("primf", "primn", "secmf", "secyf", "secnf", "pltns")


def ramp(year: int) -> float:
    """The multiplier on end-of-century rates: one in 2100, zero from 2149."""
    return max(0.0, 1.0 - (year - LAST) / (ZERO - LAST))


def extend_land(start: pd.Series, rate: pd.Series, years: list[int]) -> pd.DataFrame:
    """One region's land categories, year by year (rows: category, columns: year).

    ``start`` and ``rate`` are indexed by category, the rates summing to zero.
    Declines are applied and stopped at zero; expansions are scaled to the room
    the declines leave.
    """
    total = start.sum()
    now, out = start.astype(float).copy(), {}
    for year in range(LAST + 1, max(years) + 1):
        step = rate * ramp(year)
        kept = np.maximum(now + np.minimum(step, 0.0), 0.0)
        growth = np.maximum(step, 0.0)
        room = max(total - kept.sum(), 0.0)
        now = kept + growth * (min(1.0, room / growth.sum()) if growth.sum() > 0 else 0.0)
        if year in years:
            out[year] = now.copy()
    return pd.DataFrame(out)


def extend(df: pd.DataFrame, end: int = 2150, step: int = 5) -> pd.DataFrame:
    years = list(range(LAST + step, end + 1, step))
    cumulative = {y: sum(ramp(t) for t in range(LAST + 1, y + 1)) for y in years}
    new = pd.DataFrame(np.nan, index=df.index, columns=[str(y) for y in years])
    rate = (df[str(LAST)] - df[str(RATE_FROM)]) / (LAST - RATE_FROM)

    for _, group in df.groupby(["Model", "Scenario", "Region"]):
        index = group.set_index("Variable").index
        rows = pd.Series(group.index, index=index)
        land = [v for v in LAND if v in index]
        if land:
            rates = pd.Series(rate[rows[land]].values, index=land).fillna(0.0)
            # reported categories need not balance to the last digit; the
            # imbalance goes to other natural land, as the reader's residual does
            rates[land[-1]] -= rates.sum()
            path = extend_land(group.set_index("Variable")[str(LAST)][land].fillna(0.0), rates, years)
            for v in land:
                new.loc[rows[v], :] = path.loc[v].values
        for v in index:
            row, last = rows[v], df.at[rows[v], str(LAST)]
            if v in land:
                continue
            if v == TOTAL or not v.startswith(TOTAL):
                scale = {y: ramp(y) if v in FLUXES else
                         WOOD_END + (1.0 - WOOD_END) * ramp(y) if v in WOOD else 1.0 for y in years}
                new.loc[row, :] = [last * scale[y] for y in years]
                continue
            # a part of cropland or forest: its own ramped rate, within its parent
            parent = v.rsplit("|", 1)[0]
            if np.isnan(last):
                continue
            own = 0.0 if np.isnan(rate[row]) else rate[row]
            values = np.array([last + own * cumulative[y] for y in years])
            ceiling = new.loc[rows[parent]].values.astype(float) if parent in land else np.inf
            new.loc[row, :] = np.clip(values, 0.0, ceiling)
    return pd.concat([df, new], axis=1)


def harvest_history(mapping: str, mask: str, window=range(2011, 2021), share_year: int = 2020) -> pd.DataFrame:
    """Each region's LUH3 harvest carbon (mean over ``window``, Tg C/yr) and
    the share of it the management layer calls roundwood in ``share_year``,
    as mrdownscale computes its priors."""
    import glob

    import xarray as xr

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from graft.io import luh

    static = xr.open_dataset(STATIC)
    i, j, region = luh.region_of_cell(mask, mapping, static)
    transitions = xr.open_dataset(glob.glob(f"{LUH3}/multiple-transitions_*.nc")[0], decode_times=False)
    management = xr.open_dataset(glob.glob(f"{LUH3}/multiple-management_*.nc")[0], decode_times=False)
    years = luh.state_years(transitions)
    bioh = lambda y: sum(np.nan_to_num(transitions[f"{s}_bioh"].isel(time=years.index(y)).values)
                         for s in SOURCES)
    per = lambda field: pd.Series(field[i, j], index=region).groupby(level=0).sum()
    weight = sum(per(bioh(y)) for y in window) / len(window) / 1e9
    last = bioh(share_year)
    rndwd = np.nan_to_num(management["rndwd"].isel(time=luh.state_years(management).index(share_year)).values)
    share = per(rndwd * last) / per(last).clip(lower=np.finfo(float).eps)
    return pd.DataFrame({"weight": weight, "share": share})


def add_roundwood_index(df: pd.DataFrame, history: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Rows of the roundwood index for one scenario, on its fertilizer years."""
    rows = df[(df.Scenario == scenario) & (df.Variable == "Fertilizer Use|Nitrogen|Synthetic")
              & (df.Region != "World")]
    years = [c for c in df.columns if c.isdigit() and int(c) <= LAST]
    added = []
    for _, row in rows.iterrows():
        h = history.loc[row.Region]
        for variable, value in ((WOOD[0], h.weight), (WOOD[1], h.weight * h.share),
                                (WOOD[2], h.weight * (1 - h.share))):
            new = row.copy()
            new["Variable"], new["Unit"] = variable, "million m3/yr"
            for y in years:
                new[y] = value if not np.isnan(row[y]) else np.nan
            added.append(new)
    return pd.concat([df, pd.DataFrame(added)], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", help="iamc_coverage.py --export CSV, to 2100")
    ap.add_argument("out")
    ap.add_argument("--end", type=int, default=2150)
    ap.add_argument("--mappings", default="data/region_mappings",
                    help="folder of <model>_R10.csv region mappings, for the roundwood index")
    ap.add_argument("--mask", default=str(Path(__file__).parents[1] / "data" / "country_cell.csv.gz"))
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    for (model, scenario), group in df.groupby(["Model", "Scenario"]):
        if WOOD[0] in set(group.Variable):
            continue
        mapping = f"{args.mappings}/{model.replace(' ', '_')}_R10.csv"
        print(f"{model} reports no roundwood: index from LUH3's history, mapping {mapping}")
        df = add_roundwood_index(df, harvest_history(mapping, args.mask), scenario)
    out = extend(df, args.end)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} rows, to {args.end}")


if __name__ == "__main__":
    main()
