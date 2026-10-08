"""Put two LUH-format state datasets on a common footing before scoring them.

The published LUH3 products and mrdownscale disagree on one convention, and a
comparison that ignores it measures the convention rather than the land:

- LUH3 carries no plantation state. ``pltns`` is 0 in every cell of the
  historic file and NaN throughout the published scenarios; plantations sit
  inside ``secdf``.
- mrdownscale reports ``pltns`` separately, because its LUH3 target mapping
  sends the forestry reference categories there. For VL that is 322 Mha in
  2020 rising to 859 Mha by 2100.

The pipeline keeps ``pltns``, since the IAM reports planted forest and
mrdownscale's own format carries it. Scoring folds it back into
``secdf`` on both sides, so each dataset is compared in LUH's convention and a
plantation that is really there is not counted as missing secondary forest.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# state -> the states it absorbs when scoring. Folding is NaN-aware: a state
# that is all NaN (LUH3's pltns) contributes nothing rather than blanking the
# sum.
SCORING_FOLD: dict[str, tuple[str, ...]] = {"secdf": ("secdf", "pltns")}


def fold_for_scoring(states: xr.Dataset) -> xr.Dataset:
    """Return ``states`` in LUH's scoring convention.

    Each key of :data:`SCORING_FOLD` becomes the NaN-aware sum of its members,
    and the absorbed members are dropped. Variables not involved are passed
    through, so area closure is unchanged by the fold. A dataset that lacks an
    absorbed member (one produced without ``pltns`` at all) is left as it is.
    """
    out = states.copy()
    for target, members in SCORING_FOLD.items():
        present = [m for m in members if m in out]
        if target not in present:
            continue
        absorbed = [m for m in present if m != target]
        if not absorbed:
            continue
        stacked = xr.concat([out[m] for m in present], dim="_fold")
        # keep NaN only where every member is NaN, e.g. over ocean
        folded = stacked.sum("_fold", skipna=True).where(stacked.notnull().any("_fold"))
        out[target] = folded.assign_attrs(out[target].attrs)
        out = out.drop_vars(absorbed)
    return out


def global_area(states: xr.Dataset, carea: np.ndarray) -> dict[str, float]:
    """Global area of each state in Mha, from fractions and cell area in km^2."""
    return {
        name: float(np.nansum(np.asarray(states[name].values, dtype="float64") * carea) / 1e4)
        for name in states.data_vars
        if states[name].dims[-2:] == ("lat", "lon")
    }


# --- Scoring ------------------------------------------------------------------
# LUH states grouped the way scores are reported. Scores fold plantations into
# secdf first, so forest here is primf + secdf.
SCORE_GROUPS: dict[str, tuple[str, ...]] = {
    "forest": ("primf", "secdf"),
    "primf": ("primf",),
    "secdf": ("secdf",),
    "primn": ("primn",),
    "secdn": ("secdn",),
    "crop": ("c3ann", "c4ann", "c3per", "c4per", "c3nfx"),
    "pastr": ("pastr",),
    "range": ("range",),
    "urban": ("urban",),
}


def land_cells(reference: xr.Dataset) -> np.ndarray:
    """Cells the reference counts as land: finite in any of its states.

    Every score is taken over this one set of cells. Taking correlations over
    different sets - cells holding forest on either side in one place, land
    cells in another - made the same forest score read 0.966 and 0.984.
    """
    states = [v for v in reference.data_vars if reference[v].dims[-2:] == ("lat", "lon")]
    return np.isfinite(np.stack([reference[v].values for v in states])).any(axis=0)


def group_areas(states: xr.Dataset, carea: np.ndarray,
                groups: dict[str, tuple[str, ...]] = SCORE_GROUPS) -> dict[str, np.ndarray]:
    """Area per cell in Mha of each group, plantations folded into secdf."""
    folded = fold_for_scoring(states)
    out = {}
    for name, members in groups.items():
        present = [m for m in members if m in folded]
        if present:
            out[name] = sum(np.nan_to_num(folded[m].values) for m in present) * carea / 1e4
    return out


def score_cells(ours: xr.Dataset, reference: xr.Dataset, carea: np.ndarray,
                groups: dict[str, tuple[str, ...]] = SCORE_GROUPS):
    """Per group: global totals, gap, mean absolute error per cell, correlation.

    Cells ``ours`` leaves empty inside the reference's land count as zero, so
    missing coverage shows up as error rather than dropping out of the score.
    """
    import pandas as pd

    land = land_cells(reference)
    a, b = group_areas(ours, carea, groups), group_areas(reference, carea, groups)
    rows = []
    for name in groups:
        if name not in a or name not in b:
            continue
        x, y = a[name][land], b[name][land]
        rows.append({
            "group": name,
            "ours (Mha)": x.sum(),
            "reference (Mha)": y.sum(),
            "gap (Mha)": x.sum() - y.sum(),
            "gap (%)": 100 * (x.sum() - y.sum()) / y.sum() if y.sum() else np.nan,
            "cell MAE (kha)": 1e3 * np.abs(x - y).mean(),
            "cell corr": np.corrcoef(x, y)[0, 1] if x.std() and y.std() else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")


def score_by_resolution(ours_grid: np.ndarray, reference_grid: np.ndarray, land: np.ndarray,
                        factors: tuple[int, ...] = (1, 4, 8, 20)):
    """Skill of one group's area when aggregated to coarser blocks.

    A map can be right at 2 degrees and wrong at 0.25. A factor of k
    sums k by k cells; on the 0.25 degree grid, 4 is 1 degree and 20 is 5.
    Factors that do not divide the grid are skipped.
    """
    import pandas as pd

    rows = []
    nlat, nlon = land.shape
    a = np.where(land, ours_grid, 0.0)
    b = np.where(land, reference_grid, 0.0)
    for k in factors:
        if nlat % k or nlon % k:
            continue
        agg = lambda g: g.reshape(nlat // k, k, nlon // k, k).sum(axis=(1, 3))
        x, y = agg(a), agg(b)
        m = agg(land.astype(float)) > 0
        rows.append({
            "factor": k,
            "corr": np.corrcoef(x[m], y[m])[0, 1],
            "relative absolute error": np.abs(x - y)[m].sum() / y[m].sum(),
        })
    return pd.DataFrame(rows).set_index("factor")


def mask_diagnostics(forest_by_year: dict[int, np.ndarray], potential_forest: np.ndarray):
    """How much forest sits, and changes, where LUH's mask says it cannot.

    LUH lets forest change only on cells whose potential vegetation is forest
    (fstnf = 1); every hectare of LUH3-VL's forest change falls there.
    """
    import pandas as pd

    off = potential_forest == 0
    years = sorted(forest_by_year)
    rows = []
    for year in years:
        f = forest_by_year[year]
        rows.append({"year": year, "forest (Mha)": f.sum(),
                     "on fstnf = 0 (Mha)": f[off].sum(),
                     "on fstnf = 0 (%)": 100 * f[off].sum() / f.sum() if f.sum() else np.nan})
    out = pd.DataFrame(rows).set_index("year")
    if len(years) >= 2:
        change = forest_by_year[years[-1]] - forest_by_year[years[0]]
        out.attrs["gain on fstnf = 0 (Mha)"] = float(change[off & (change > 0)].sum())
        out.attrs["gain on fstnf = 1 (Mha)"] = float(change[~off & (change > 0)].sum())
    return out
