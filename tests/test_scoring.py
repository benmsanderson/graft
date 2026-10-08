import numpy as np
import pandas as pd
import pytest
import xarray as xr

from graft import compare as C
from graft.io.luh import select_year
from graft.io.mrdownscale import read_states_csv
from tests.fixtures import make_consistent_scenario, make_static


def _states(nlat=4, nlon=8, seed=0, ocean=True):
    """LUH-shaped state fractions with some ocean, summing to 1 on land."""
    rng = np.random.default_rng(seed)
    names = ["primf", "secdf", "primn", "secdn", "c3ann", "pastr", "range", "urban"]
    frac = rng.dirichlet(np.ones(len(names)), size=(nlat, nlon))
    data = {n: (("lat", "lon"), frac[..., k].copy()) for k, n in enumerate(names)}
    ds = xr.Dataset(data, coords={"lat": np.linspace(80, -80, nlat), "lon": np.linspace(-170, 170, nlon)})
    if ocean:
        for n in names:
            ds[n].values[0, :2] = np.nan
    return ds


def test_identical_grids_score_perfectly():
    ref = _states()
    scores = C.score_cells(ref.copy(deep=True), ref, np.full((4, 8), 1e4))

    assert np.allclose(scores["gap (Mha)"], 0)
    assert np.allclose(scores["cell MAE (kha)"], 0)
    assert np.allclose(scores["cell corr"], 1)


def test_scores_fold_plantations_into_secondary_forest():
    ref = _states()
    ref["pltns"] = xr.full_like(ref.secdf, np.nan)  # LUH3: no plantation state
    ours = ref.copy(deep=True)
    ours["pltns"] = ours.secdf * 0.25  # mrdownscale: plantations reported apart
    ours["secdf"] = ours.secdf * 0.75

    scores = C.score_cells(ours, ref, np.full((4, 8), 1e4))

    assert np.isclose(scores.loc["secdf", "gap (Mha)"], 0)
    assert np.isclose(scores.loc["forest", "cell MAE (kha)"], 0)


def test_cells_ours_leaves_empty_count_as_error():
    ref = _states()
    ours = ref.copy(deep=True)
    ours["primf"].values[3, 5] = np.nan  # a land cell missing from ours

    scores = C.score_cells(ours, ref, np.full((4, 8), 1e4))

    assert scores.loc["primf", "cell MAE (kha)"] > 0
    assert scores.loc["primf", "gap (Mha)"] < 0


def test_resolution_skill_aggregates_and_skips_factors_that_do_not_divide():
    ref = _states()
    land = C.land_cells(ref)
    grid = C.group_areas(ref, np.full((4, 8), 1e4))["forest"]

    skill = C.score_by_resolution(grid, grid, land, factors=(1, 2, 3, 4))

    assert list(skill.index) == [1, 2, 4]  # 3 divides neither 4 nor 8
    assert np.allclose(skill["relative absolute error"], 0)


def test_mask_diagnostics_find_forest_where_the_mask_excludes_it():
    potential = np.array([[1, 0], [1, 0]])
    forest = {2025: np.array([[5.0, 0.0], [5.0, 1.0]]),
              2050: np.array([[6.0, 3.0], [5.0, 1.0]])}

    mask = C.mask_diagnostics(forest, potential)

    assert mask.loc[2050, "on fstnf = 0 (Mha)"] == 4.0
    assert mask.attrs["gain on fstnf = 0 (Mha)"] == 3.0
    assert mask.attrs["gain on fstnf = 1 (Mha)"] == 1.0


def test_select_year_picks_one_year():
    states, _, _ = make_consistent_scenario(nyear=3)

    one = select_year(states, 2021)

    assert "time" not in one.dims
    np.testing.assert_allclose(one.primf.values, states.primf.isel(time=1).values)
    with pytest.raises(KeyError):
        select_year(states, 1999)


def test_exported_mrdownscale_grid_reads_back_as_luh_fractions(tmp_path):
    static = make_static()  # 3 x 4 cells of 1000 km2, i.e. 0.1 Mha
    lat, lon = static.lat.values, static.lon.values
    rows = [{"x": lon[j], "y": lat[i], "year": 2050, "primf": 0.05, "secdf": 0.02, "urban": 0.03}
            for i in range(3) for j in range(4) if (i, j) != (0, 0)]
    path = tmp_path / "states.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    ds = read_states_csv(path, static, 2050)

    np.testing.assert_allclose(ds.primf.values[1, 1], 0.5)  # 0.05 Mha of a 0.1 Mha cell
    assert np.isnan(ds.primf.values[0, 0])  # not a cell mrdownscale covered
    with pytest.raises(KeyError):
        read_states_csv(path, static, 2100)
