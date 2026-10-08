"""The age tracker's arithmetic, on cells small enough to verify by hand."""

import numpy as np
import pytest
import xarray as xr

from graft import age as A


@pytest.fixture
def static():
    return xr.Dataset(
        {"ptbio": (("lat", "lon"), np.array([[10.0, 10.0]])),
         "carea": (("lat", "lon"), np.array([[1e4, 1e4]])),
         "fstnf": (("lat", "lon"), np.array([[1.0, 1.0]]))},
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )


def pool(area, age, biomass):
    f = lambda v: np.array([[v, 0.0]], dtype="float64")
    return A.Pool(area=f(area), age=f(age), biomass=f(biomass))


def test_undisturbed_land_just_ages(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.5, 20.0, 3.0)

    out = A.step(p, target, 62.0, A.Inflow(np.zeros((1, 2))), np.zeros((1, 2)), 1.0)

    assert out.age[0, 0] == pytest.approx(21.0)
    assert out.area[0, 0] == pytest.approx(0.5)
    # one year of exponential approach towards 0.745 * 10
    ceiling = target[0, 0]
    assert out.biomass[0, 0] == pytest.approx(ceiling - (ceiling - 3.0) * np.exp(-1 / 62))


def test_proportional_loss_moves_neither_mean(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.5, 20.0, 3.0)

    kept = A.step(p, target, 62.0, A.Inflow(np.zeros((1, 2))), np.array([[0.2, 0.0]]), 0.0)

    assert kept.area[0, 0] == pytest.approx(0.3)
    assert kept.age[0, 0] == pytest.approx(20.0)
    assert kept.biomass[0, 0] == pytest.approx(3.0)


def test_new_land_enters_young_and_bare(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.5, 40.0, 5.0)

    out = A.step(p, target, 62.0, A.Inflow(np.array([[0.5, 0.0]])), np.zeros((1, 2)), 0.0)

    # half the land is now age 0 and the other half aged to 40, so the mean halves
    assert out.area[0, 0] == pytest.approx(1.0)
    assert out.age[0, 0] == pytest.approx(20.0)
    assert out.biomass[0, 0] == pytest.approx(5.0 / 2, rel=1e-2)


def test_carrying_means_is_exact_against_explicit_cohorts(static):
    """The point of the design: means propagate without a Jensen error.

    Two cohorts stepped separately and then averaged must equal one pool
    started at their average, because ageing and exponential approach are both
    affine in what is tracked.
    """
    target = A.target_biomass(static, A.Regrowth())
    zero = np.zeros((1, 2))
    young, old = pool(0.5, 5.0, 1.0), pool(0.5, 95.0, 6.0)
    mean = pool(1.0, 50.0, 3.5)

    for _ in range(30):
        young = A.step(young, target, 62.0, A.Inflow(zero), zero, 1.0)
        old = A.step(old, target, 62.0, A.Inflow(zero), zero, 1.0)
        mean = A.step(mean, target, 62.0, A.Inflow(zero), zero, 1.0)

    cohortAge, cohortBiomass = A.combine(young, old)
    assert cohortAge[0, 0] == pytest.approx(mean.age[0, 0])
    assert cohortBiomass[0, 0] == pytest.approx(mean.biomass[0, 0])


def test_a_multi_year_step_matches_repeated_single_years(static):
    """Only the age of land arriving during the step should differ."""
    target = A.target_biomass(static, A.Regrowth())
    zero = np.zeros((1, 2))
    one, five = pool(0.5, 20.0, 3.0), pool(0.5, 20.0, 3.0)

    for _ in range(5):
        one = A.step(one, target, 62.0, A.Inflow(zero), zero, 1.0)
    five = A.step(five, target, 62.0, A.Inflow(zero), zero, 5.0)

    assert one.age[0, 0] == pytest.approx(five.age[0, 0])
    assert one.biomass[0, 0] == pytest.approx(five.biomass[0, 0])


def test_land_that_all_disappears_leaves_nothing_behind(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.4, 30.0, 4.0)

    out = A.step(p, target, 62.0, A.Inflow(np.zeros((1, 2))), np.array([[0.4, 0.0]]), 1.0)

    assert out.area[0, 0] == pytest.approx(0.0)
    assert out.age[0, 0] == pytest.approx(0.0)
    assert out.biomass[0, 0] == pytest.approx(0.0)


def test_transferred_land_keeps_the_age_it_brought(static):
    """secdf_to_secdn moves land that already has a history; it is not new."""
    target = A.target_biomass(static, A.Regrowth())
    empty = A.Pool(area=np.zeros((1, 2)), age=np.zeros((1, 2)), biomass=np.zeros((1, 2)))
    moved = A.Inflow(area=np.array([[0.3, 0.0]]),
                     age=np.array([[50.0, 0.0]]), biomass=np.array([[4.0, 0.0]]))

    out = A.step(empty, target, 7.0, moved, np.zeros((1, 2)), 0.0)

    assert out.age[0, 0] == pytest.approx(50.0)
    assert out.biomass[0, 0] == pytest.approx(4.0)


def test_several_inflows_are_area_weighted(static):
    target = A.target_biomass(static, A.Regrowth())
    empty = A.Pool(area=np.zeros((1, 2)), age=np.zeros((1, 2)), biomass=np.zeros((1, 2)))
    bare = A.Inflow(area=np.array([[0.1, 0.0]]))
    old = A.Inflow(area=np.array([[0.3, 0.0]]), age=np.array([[40.0, 0.0]]))

    out = A.step(empty, target, 62.0, [bare, old], np.zeros((1, 2)), 0.0)

    assert out.area[0, 0] == pytest.approx(0.4)
    assert out.age[0, 0] == pytest.approx((0.1 * 0 + 0.3 * 40) / 0.4)


def test_biomass_never_passes_the_asymptote(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(1.0, 0.0, 0.0)

    for _ in range(500):
        p = A.step(p, target, 62.0, A.Inflow(np.zeros((1, 2))), np.zeros((1, 2)), 1.0)

    ceiling = A.Regrowth().asymptote * 10.0
    assert p.biomass[0, 0] < ceiling + 1e-9
    assert p.biomass[0, 0] == pytest.approx(ceiling, rel=1e-3)


def test_seeding_reproduces_the_published_biomass_exactly(static):
    """A seeded year must carry the product's own secmb forward unchanged."""
    states = xr.Dataset(
        {"secdf": (("lat", "lon"), np.array([[0.4, 0.1]])),
         "secdn": (("lat", "lon"), np.array([[0.2, 0.3]])),
         "secma": (("lat", "lon"), np.array([[40.0, 15.0]])),
         "secmb": (("lat", "lon"), np.array([[2.5, 1.1]]))},
        coords={"lat": [0.0], "lon": [0.0, 1.0]},
    )

    forest, other = A.seed(states, static, A.Regrowth())
    _, biomass = A.combine(forest, other)

    assert biomass == pytest.approx(np.array([[2.5, 1.1]]))
    # and the forest pool, regrowing slowly, holds less than the other
    assert forest.biomass[0, 0] < other.biomass[0, 0]


def test_harvest_keeps_the_area_but_returns_the_stand_to_young(static):
    """The flow that mattered most: LUH fells secondary without unmaking it."""
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.6, 80.0, 6.0)
    zero = np.zeros((1, 2))
    flows = A.Flows(arriving=zero, leaving=zero, harvested=np.array([[0.3, 0.0]]),
                    fromPrimary=zero, transferredIn=zero, transferredOut=zero)
    empty = A.Pool(area=zero.copy(), age=zero.copy(), biomass=zero.copy())
    none = A.Flows(*(zero for _ in range(6)))

    after, _ = A.advance(p, empty, flows, none, target, A.Regrowth(), dt=0.0)

    assert after.area[0, 0] == pytest.approx(0.6)       # still secondary
    assert after.age[0, 0] == pytest.approx(40.0)       # half of it back to zero
    assert after.biomass[0, 0] == pytest.approx(3.0)    # and bare, at residual 0


def test_harvest_cannot_fell_more_than_is_there(static):
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.2, 50.0, 4.0)
    zero = np.zeros((1, 2))
    flows = A.Flows(arriving=zero, leaving=zero, harvested=np.array([[0.9, 0.0]]),
                    fromPrimary=zero, transferredIn=zero, transferredOut=zero)
    empty = A.Pool(area=zero.copy(), age=zero.copy(), biomass=zero.copy())
    none = A.Flows(*(zero for _ in range(6)))

    after, _ = A.advance(p, empty, flows, none, target, A.Regrowth(), dt=1.0)

    assert after.area[0, 0] == pytest.approx(0.2)
    assert after.age[0, 0] == pytest.approx(0.5)  # all of it felled, then half a step


def test_harvest_with_its_carbon_fells_only_the_stands_that_carbon_amounts_to(static):
    """mrdownscale's undivided harvest area is not whole stands."""
    target = A.target_biomass(static, A.Regrowth())
    p = pool(0.6, 80.0, 6.0)
    zero = np.zeros((1, 2))
    # 0.3 of the cell reported as harvested, but only 0.6 kg C per m2 of cell
    # taken: a tenth of the cell's worth of stands at 6 kg C/m2
    flows = A.Flows(arriving=zero, leaving=zero, harvested=np.array([[0.3, 0.0]]),
                    fromPrimary=zero, transferredIn=zero, transferredOut=zero,
                    harvestedCarbon=np.array([[0.6, 0.0]]))
    empty = A.Pool(area=zero.copy(), age=zero.copy(), biomass=zero.copy())
    none = A.Flows(*(zero for _ in range(6)))

    after, _ = A.advance(p, empty, flows, none, target, A.Regrowth(), dt=0.0)

    assert after.area[0, 0] == pytest.approx(0.6)
    assert after.biomass[0, 0] == pytest.approx(6.0 * 0.5 / 0.6)     # 0.6 kg C/m2 of cell gone
    assert after.age[0, 0] == pytest.approx(80.0 * 0.5 / 0.6)
