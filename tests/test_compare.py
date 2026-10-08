import numpy as np
import xarray as xr

from graft.compare import fold_for_scoring, global_area


def _states(secdf, pltns=None, other=0.5):
    data = {
        "secdf": (("lat", "lon"), np.array(secdf, dtype="float64")),
        "primn": (("lat", "lon"), np.full((1, 2), other)),
    }
    if pltns is not None:
        data["pltns"] = (("lat", "lon"), np.array(pltns, dtype="float64"))
    return xr.Dataset(data, coords={"lat": [0.0], "lon": [0.0, 1.0]})


def test_plantations_are_folded_into_secondary_forest():
    # mrdownscale's convention: plantations reported separately
    folded = fold_for_scoring(_states(secdf=[[0.2, 0.1]], pltns=[[0.1, 0.3]]))

    assert "pltns" not in folded
    np.testing.assert_allclose(folded.secdf.values, [[0.3, 0.4]])


def test_luh3_nan_plantations_leave_secondary_forest_intact():
    # LUH3's published scenarios: pltns present but all NaN
    folded = fold_for_scoring(_states(secdf=[[0.2, 0.1]], pltns=[[np.nan, np.nan]]))

    np.testing.assert_allclose(folded.secdf.values, [[0.2, 0.1]])


def test_ocean_stays_nan_rather_than_becoming_zero():
    folded = fold_for_scoring(_states(secdf=[[np.nan, 0.1]], pltns=[[np.nan, 0.2]]))

    assert np.isnan(folded.secdf.values[0, 0])
    np.testing.assert_allclose(folded.secdf.values[0, 1], 0.3)


def test_fold_conserves_area_and_leaves_other_states_alone():
    states = _states(secdf=[[0.2, 0.1]], pltns=[[0.1, 0.3]])
    carea = np.full((1, 2), 1e4)  # 1 Mha per cell

    before, after = global_area(states, carea), global_area(fold_for_scoring(states), carea)

    assert np.isclose(sum(before.values()), sum(after.values()))
    assert after["primn"] == before["primn"]


def test_a_dataset_without_plantations_is_unchanged():
    states = _states(secdf=[[0.2, 0.1]])

    xr.testing.assert_identical(fold_for_scoring(states), states)
