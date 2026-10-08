"""The carbon check's arithmetic, on a grid small enough to verify by hand."""

import numpy as np
import pytest
import xarray as xr

from graft import carbon


@pytest.fixture
def static():
    """Two cells of 1e10 m^2 (1 Mha), one carbon-rich and one poor."""
    return xr.Dataset(
        {"ptbio": (("lat", "lon"), np.array([[10.0, 1.0]])),
         "carea": (("lat", "lon"), np.array([[1e4, 1e4]]))},
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )


def states(primf, secdf):
    return xr.Dataset({"primf": (("lat", "lon"), np.array([primf])),
                       "secdf": (("lat", "lon"), np.array([secdf]))},
                      coords={"lat": [0.0], "lon": [0.0, 1.0]})


def test_stock_is_density_times_area_times_the_prior(static):
    # all of the rich cell as primary forest: 1e10 m2 * 10 kg C/m2 = 100 Pg C
    s = carbon.stock(states([1.0, 0.0], [0.0, 0.0]), static)
    assert s["primf"] == pytest.approx(0.1)
    # the same cell as secondary forest holds the declared fraction of it
    s = carbon.stock(states([0.0, 0.0], [1.0, 0.0]), static)
    assert s["secdf"] == pytest.approx(0.1 * carbon.POTENTIAL_FRACTION["secdf"])


def test_flux_is_per_year_and_signed_towards_the_atmosphere(static):
    # losing 1 Pg C over 10 years is a source of 0.1 Pg C/yr
    got = carbon.flux(np.array([10.0, 9.0]), np.array([2020, 2030]))
    assert got[0] == pytest.approx(0.1 * carbon.C_TO_CO2)


def test_decomposition_is_exact_and_separates_placement(static):
    index = (np.array([0, 0]), np.array([0, 1]))
    region = np.array(["R", "R"])
    # the same area of primary forest, moved from the rich cell to the poor one
    rich = states([1.0, 0.0], [0.0, 0.0])
    poor = states([0.0, 1.0], [0.0, 0.0])

    split = carbon.decompose(rich, poor, static, index, region, 2050)

    assert split.from_area == pytest.approx(0.0)  # same hectares either way
    assert split.from_placement == pytest.approx(0.1 - 0.01)
    assert split.total == pytest.approx(split.from_area + split.from_placement)


def test_decomposition_splits_a_mixed_difference(static):
    index = (np.array([0, 0]), np.array([0, 1]))
    region = np.array(["R", "R"])
    a = states([1.0, 1.0], [0.0, 0.0])
    b = states([0.0, 1.0], [0.0, 0.0])

    split = carbon.decompose(a, b, static, index, region, 2050)

    # a holds one extra Mha of primary forest, and its forest sits on richer
    # land on average, so both terms are positive and they sum to the total
    assert split.from_area > 0 and split.from_placement > 0
    assert split.total == pytest.approx(0.1)


def test_implied_fraction_recovers_the_prior_it_was_given(static):
    # a run that converts primary forest to secondary, and the carbon change
    # the declared prior itself implies; the inversion must return that prior
    yearly = [states([1.0, 0.0], [0.0, 0.0]), states([0.0, 0.0], [1.0, 0.0])]
    years = np.array([2025, 2100])
    change = (carbon.total_stock(yearly[1], static).sum()
              - carbon.total_stock(yearly[0], static).sum())

    got = carbon.implied_regrowth_fraction(yearly, static, years, float(change))

    assert got == pytest.approx(carbon.POTENTIAL_FRACTION["secdf"])


def test_published_secmb_replaces_the_prior(static):
    # the same secondary forest, once under the prior and once under a
    # published density of 2 kg C/m^2 on a cell whose potential is 10
    plain = states([0.0, 0.0], [1.0, 0.0])
    withField = plain.assign(secmb=(("lat", "lon"), np.array([[2.0, 0.0]])))

    assert carbon.stock(plain, static)["secdf"] == pytest.approx(
        0.1 * carbon.POTENTIAL_FRACTION["secdf"])
    assert carbon.stock(withField, static)["secondary"] == pytest.approx(0.02)
    # and the effective fraction reads back the ratio to potential
    assert carbon.effective_fraction(withField, static) == pytest.approx(0.2)


def test_implied_fraction_is_blind_to_which_model_the_stock_used(static):
    # replacing the prior with a published field changes the stock but not the
    # question the inversion asks, which is what secondary land would have to
    # hold; both must answer the same
    def run(withField):
        yearly = [states([1.0, 0.0], [0.0, 0.0]), states([0.0, 0.0], [1.0, 0.0])]
        if withField:
            yearly = [y.assign(secmb=(("lat", "lon"), np.array([[2.0, 0.0]]))) for y in yearly]
        return carbon.implied_regrowth_fraction(yearly, static, np.array([2025, 2100]), -0.05)

    assert run(False) == pytest.approx(run(True))


def test_implied_fraction_goes_above_one_when_nothing_reconciles(static):
    yearly = [states([1.0, 0.0], [0.0, 0.0]), states([0.0, 0.0], [1.0, 0.0])]
    years = np.array([2025, 2100])

    # the scenario claims the land gained carbon while the forcing clears it
    got = carbon.implied_regrowth_fraction(yearly, static, years, +5.0)

    assert got > 1.0


def test_cumulative_flips_the_sign_of_reported_emissions():
    import pandas as pd
    # a steady 1000 Mt CO2/yr source for 10 years is 10 Gt CO2 out of the land
    got = carbon.cumulative(pd.Series([1000.0, 1000.0]), np.array([2020, 2030]))
    assert got == pytest.approx(-10 / carbon.C_TO_CO2)
