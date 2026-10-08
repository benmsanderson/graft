import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import extend_iamc as E  # noqa: E402


def frame(rows):
    years = [str(y) for y in (2060, 2080, 2100)]
    return pd.DataFrame([{"Model": "m", "Scenario": "s", "Region": "r", "Variable": v, "Unit": "u",
                          **dict(zip(years, values))} for v, values in rows.items()])


def test_ramp_is_one_at_2100_and_zero_from_2149():
    assert E.ramp(2100) == 1.0
    assert E.ramp(2149) == 0.0 and E.ramp(2200) == 0.0
    assert 0.0 < E.ramp(2125) < 1.0


def test_land_is_static_after_the_ramp_and_keeps_its_total():
    out = E.extend(frame({"Land Cover": [100, 100, 100],
                          "Land Cover|Cropland": [20, 25, 30], "Land Cover|Pasture": [20, 20, 20],
                          "Land Cover|Forest": [40, 37, 34], "Land Cover|Built-Up Area": [1, 1, 1],
                          "Land Cover|Other Natural": [19, 17, 15]}), end=2160)
    land = out[out.Variable.isin(E.LAND)].set_index("Variable")
    for year in ("2105", "2125", "2150", "2160"):
        assert land[year].sum() == pytest.approx(100.0)
    assert (land["2150"] == land["2160"]).all()
    # cropland gained 0.25 a year over 2060-2100: the ramp adds 0.25 * sum of multipliers
    assert land.at["Land Cover|Cropland", "2150"] == pytest.approx(30 + 0.25 * sum(E.ramp(t) for t in range(2101, 2150)))


def test_a_category_that_runs_out_stops_the_expansion_it_fed():
    out = E.extend(frame({"Land Cover": [100, 100, 100],
                          "Land Cover|Cropland": [50, 70, 90], "Land Cover|Pasture": [0, 0, 0],
                          "Land Cover|Forest": [10, 10, 10], "Land Cover|Built-Up Area": [0, 0, 0],
                          "Land Cover|Other Natural": [40, 20, 0]}))
    land = out[out.Variable.isin(E.LAND)].set_index("Variable")
    assert land.at["Land Cover|Other Natural", "2150"] == 0.0
    assert land.at["Land Cover|Cropland", "2150"] == pytest.approx(90.0)
    assert land["2150"].sum() == pytest.approx(100.0)


def test_parts_stay_within_their_parent_and_demand_is_held_and_fluxes_ramp():
    out = E.extend(frame({"Land Cover": [100, 100, 100],
                          "Land Cover|Cropland": [30, 30, 30], "Land Cover|Cropland|Energy Crops": [10, 5, 0],
                          "Land Cover|Pasture": [20, 20, 20], "Land Cover|Forest": [40, 40, 40],
                          "Land Cover|Forest|Primary": [np.nan, 20, 20],
                          "Land Cover|Built-Up Area": [0, 0, 0], "Land Cover|Other Natural": [10, 10, 10],
                          "Forestry Production|Roundwood": [5, 6, 7],
                          "Emissions|CO2|AFOLU": [100, 100, 98]})).set_index("Variable")
    assert out.at["Land Cover|Cropland|Energy Crops", "2150"] == 0.0
    assert out.at["Land Cover|Forest|Primary", "2150"] == 20.0     # no 2080 value: held
    assert out.at["Forestry Production|Roundwood", "2150"] == pytest.approx(7.0 * E.WOOD_END)
    assert out.at["Forestry Production|Roundwood", "2105"] == pytest.approx(
        7.0 * (E.WOOD_END + (1 - E.WOOD_END) * E.ramp(2105)))
    assert out.at["Emissions|CO2|AFOLU", "2150"] == 0.0
    assert out.at["Emissions|CO2|AFOLU", "2105"] == pytest.approx(98 * (1 - 5 / 49))
    # every variable reported in 2100 is reported in every added year
    assert not out[[str(y) for y in range(2105, 2151, 5)]].isna().any().any()


def test_the_roundwood_index_is_constant_and_split_by_the_history_share():
    df = frame({"Fertilizer Use|Nitrogen|Synthetic": [5, np.nan, 6]})
    history = pd.DataFrame({"weight": [100.0], "share": [0.25]}, index=["r"])
    out = E.add_roundwood_index(df, history, "s").set_index("Variable")
    assert out.at[E.WOOD[0], "2060"] == 100.0 and out.at[E.WOOD[0], "2100"] == 100.0
    assert np.isnan(out.at[E.WOOD[0], "2080"])                  # only where the model reports
    assert out.at[E.WOOD[1], "2100"] == 25.0 and out.at[E.WOOD[2], "2100"] == 75.0
