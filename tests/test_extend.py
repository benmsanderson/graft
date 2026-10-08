import numpy as np
import pytest

from graft import age, extend

TABLE = np.linspace(0.02, 1.0, 50)   # a monotone stand-in for GLM's table


def pool(area, biomass):
    area = np.asarray(area, dtype=float)
    return age.Pool(area=area, age=np.full_like(area, 50.0), biomass=np.asarray(biomass, dtype=float))


def test_probability_bins_follow_glm():
    b = np.array([0.0, 0.3, 0.5, 0.51, 24.9, 100.0])
    p = extend.harvest_probability(b, TABLE)
    assert p[0] == 0.0
    assert p[1] == TABLE[0] and p[2] == TABLE[0] and p[3] == TABLE[1]
    assert p[-1] == TABLE[-1]


def test_demand_split_by_available_biomass_and_met_exactly():
    zone = np.array([0, 0])
    primf = np.array([0.5, 0.0])
    ptbio = np.array([10.0, 10.0])
    forest = pool([0.0, 0.5], [0.0, 8.0])
    area = np.array([1.0, 1.0])
    p = extend.harvest_probability(np.array([8.0]), TABLE)[0]
    vb, smb = 0.5 * 10.0 * extend.ABOVEGROUND, 0.5 * 8.0 * p
    demand = np.array([0.5 * (vb + smb)])

    h = extend.allocate(demand, zone, primf, ptbio, forest, area, TABLE, primary_mature=False)

    assert h.primary_carbon.sum() == pytest.approx(demand[0] * vb / (vb + smb))
    assert h.primary_carbon.sum() + h.secondary_carbon.sum() == pytest.approx(demand[0])
    assert h.unmet[0] == 0.0
    # whole stands: area taken follows the carbon at the stand's aboveground density
    assert h.primary_area[0] == pytest.approx(h.primary_carbon[0] / (10.0 * extend.ABOVEGROUND))


def test_primary_as_mature_competes_on_the_secondary_scale_and_is_capped():
    zone = np.array([0])
    primf, ptbio, area = np.array([0.5]), np.array([10.0]), np.array([1.0])
    forest = pool([0.5], [8.0])
    p, top = extend.harvest_probability(np.array([8.0]), TABLE)[0], TABLE.max()
    vb, smb = 0.5 * 10.0 * extend.ABOVEGROUND * top, 0.5 * 8.0 * p

    h = extend.allocate(np.array([1.0]), zone, primf, ptbio, forest, area, TABLE)
    assert h.primary_carbon[0] == pytest.approx(vb / (vb + smb))
    assert h.primary_carbon[0] + h.secondary_carbon[0] == pytest.approx(1.0)

    # demand beyond both: primary gives its mature share and no more, young secondary the rest
    h = extend.allocate(np.array([vb + smb + 0.1]), zone, primf, ptbio, forest, area, TABLE)
    assert h.primary_area[0] == pytest.approx(0.5 * top)
    assert h.primary_carbon[0] + h.secondary_carbon[0] == pytest.approx(vb + smb + 0.1)


def test_excess_demand_goes_to_young_secondary_then_unmet():
    zone = np.array([0])
    primf, ptbio, area = np.array([0.1]), np.array([10.0]), np.array([1.0])
    forest = pool([0.5], [4.0])
    everything = 0.1 * 10.0 * extend.ABOVEGROUND + 0.5 * 4.0   # primary plus all secondary
    h = extend.allocate(np.array([everything + 1.0]), zone, primf, ptbio, forest, area, TABLE,
                        primary_mature=False)
    assert h.primary_area[0] == pytest.approx(0.1)
    assert h.secondary_area[0] == pytest.approx(0.5)
    assert h.unmet[0] == pytest.approx(1.0)


def test_step_moves_primary_into_secondary_and_keeps_forest_area():
    primf = np.array([0.4])
    forest, other = pool([0.3], [5.0]), pool([0.1], [1.0])
    h = extend.Harvest(primary_area=np.array([0.1]), secondary_area=np.array([0.05]),
                       primary_carbon=np.zeros(1), secondary_carbon=np.zeros(1), unmet=np.zeros(1))
    regrowth = age.Regrowth()
    primf2, forest2, other2 = extend.step(primf, forest, other, h, np.array([10.0]), regrowth)
    assert primf2[0] == pytest.approx(0.3)
    assert forest2.area[0] == pytest.approx(0.4)
    assert primf2[0] + forest2.area[0] == pytest.approx(primf[0] + forest.area[0])
    assert other2.area[0] == pytest.approx(0.1)
    # new and felled stands are young, so mean age falls below 51
    assert forest2.age[0] < 51.0


def test_recovery_is_net_change_plus_harvest_and_ignores_losses():
    zone = np.array([0, 1])
    before, after = np.array([10.0, 10.0]), np.array([11.0, 5.0])
    harvested = np.array([2.0, 1.0])
    r = extend.recovery(before, after, harvested, zone, 2)
    assert r[0] == pytest.approx(3.0)
    assert r[1] == 0.0


def test_maintenance_is_half_the_gross_recovery_and_where_net_recovery_settles():
    zone = np.array([0])
    r = extend.maintenance(np.array([10.0]), np.array([11.0]), np.array([2.0]), zone, 1)
    assert r[0] == pytest.approx(1.5)
    # constant regrowth g: the net rule h(t) = g - h(t-1) alternates about g / 2
    g, h = 3.0, 2.5
    series = []
    for _ in range(6):
        h = max(g - h, 0.0)
        series.append(h)
    assert np.mean(series) == pytest.approx(g / 2)
    assert series[0] != pytest.approx(series[1])


def test_run_conserves_land_and_its_harvest_fades_without_alternating():
    zone = np.array([0, 0])
    ptbio, area = np.array([12.0, 12.0]), np.array([1.0, 1.0])
    primf, primn = np.array([0.4, 0.1]), np.array([0.05, 0.0])
    forest, other = pool([0.3, 0.6], [3.0, 5.0]), pool([0.1, 0.1], [0.5, 0.5])
    land = primf + primn + forest.area + other.area
    taken = []
    for _, h, pf, pn, f, o in extend.run(primf, primn, forest, other, ptbio, area, zone, TABLE,
                                         age.Regrowth(asymptote=extend.ABOVEGROUND), range(60)):
        taken.append(h.carbon.sum())
        assert pf + pn + f.area + o.area == pytest.approx(land)
    assert (pf <= primf).all() and pf.sum() < primf.sum()
    assert taken[0] > 0 and taken[-1] < taken[5]
    steps = np.diff(taken[2:])
    assert (steps <= 1e-12).all()     # monotone: no year-to-year alternation


def test_carried_harvest_joins_the_last_ramp_year_and_fades_out():
    zone = np.array([0, 0])
    ptbio, area = np.array([12.0, 12.0]), np.array([1.0, 1.0])
    primf, primn = np.array([0.4, 0.1]), np.array([0.05, 0.0])
    forest, other = pool([0.3, 0.6], [3.0, 5.0]), pool([0.1, 0.1], [0.5, 0.5])
    # a last ramp year that fells far more than the maintenance term would
    last = extend.Harvest(primary_area=np.array([0.004, 0.001]), secondary_area=np.array([0.1, 0.2]),
                          primary_carbon=np.array([0.03, 0.008]), secondary_carbon=np.array([0.05, 0.1]),
                          unmet=np.zeros(1))
    regrowth = age.Regrowth(asymptote=extend.ABOVEGROUND)
    alone = next(extend.run(primf, primn, forest, other, ptbio, area, zone, TABLE, regrowth, [2150]))[1]
    taken, primary = [], []
    for _, h, pf, pn, f, o in extend.run(primf, primn, forest, other, ptbio, area, zone, TABLE,
                                         regrowth, range(2150, 2210), last=last):
        taken.append(h.carbon.sum())
        primary.append(h.primary_carbon.sum())
        assert pf + pn + f.area + o.area == pytest.approx(primf + primn + forest.area + other.area)
    full = last.primary_carbon.sum() + last.secondary_carbon.sum()
    weight = 1 - 1 / extend.FADE
    # the first year is almost all the last ramp year's harvest, the rest maintenance
    assert taken[0] == pytest.approx(weight * full + (1 - weight) * alone.carbon.sum(), rel=1e-3)
    assert abs(taken[0] - full) < 0.03 * full                       # no step from the ramp
    assert abs(primary[0] - last.primary_carbon.sum()) < 0.1 * last.primary_carbon.sum()
    assert extend.fade(2149 + extend.FADE, 2149) == 0.0 and taken[-1] < taken[0]


def test_carried_harvest_stays_within_the_land():
    primf, primn = np.array([0.001]), np.array([0.0])
    forest, other = pool([0.2], [2.0]), pool([0.0], [0.0])
    last = extend.Harvest(primary_area=np.array([0.01]), secondary_area=np.array([0.5]),
                          primary_carbon=np.array([0.08]), secondary_carbon=np.array([1.0]),
                          unmet=np.zeros(1))
    c = extend.carry(last, 1.0, primf, primn, forest, other, np.array([1.0]))
    assert c.primary_area[0] == pytest.approx(0.001) and c.primary_carbon[0] == pytest.approx(0.008)
    assert c.secondary_carbon[0] == pytest.approx(0.4) and c.secondary_area[0] == pytest.approx(0.2)
    assert c.secondary_reported[0] == pytest.approx(0.2)      # no more than the pool


def test_carried_harvest_a_cell_cannot_supply_is_taken_elsewhere_in_the_country():
    zone = np.array([0, 0])
    ptbio, area = np.array([12.0, 12.0]), np.array([1.0, 1.0])
    primf, primn = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    # the first cell's secondary forest holds 0.1 kg C/m2 of cell, the second 3
    forest, other = pool([0.1, 0.6], [1.0, 5.0]), pool([0.0, 0.0], [0.0, 0.0])
    last = extend.Harvest(primary_area=np.zeros(2), secondary_area=np.array([0.1, 0.0]),
                          primary_carbon=np.zeros(2), secondary_carbon=np.array([0.5, 0.0]),
                          unmet=np.zeros(1))
    regrowth = age.Regrowth(asymptote=extend.ABOVEGROUND)
    alone = next(extend.run(primf, primn, forest, other, ptbio, area, zone, TABLE, regrowth, [2150]))[1]
    h = next(extend.run(primf, primn, forest, other, ptbio, area, zone, TABLE, regrowth, [2150], last=last))[1]
    weight = 1 - 1 / extend.FADE
    assert h.carbon.sum() == pytest.approx(weight * 0.5 + (1 - weight) * alone.carbon.sum(), rel=1e-6)
    assert h.secondary_carbon[0] < 0.11 and h.secondary_carbon[1] > 0.3   # the rest from the second cell


def test_legacy_step_moves_felled_and_harvested_land_to_the_new_pool():
    regrowth = age.Regrowth()
    legacy, new = pool([0.4], [6.0]), pool([0.0], [0.0])
    h = extend.Harvest(primary_area=np.array([0.05]), secondary_area=np.array([0.1]),
                       primary_carbon=np.zeros(1), secondary_carbon=np.zeros(1), unmet=np.zeros(1))
    primf, legacy2, new2, grown = extend.step_legacy(np.array([0.3]), legacy, new, h, TABLE,
                                                     np.array([10.0]), regrowth, np.array([1.0]))
    assert legacy2.area[0] == pytest.approx(0.3)
    assert new2.area[0] == pytest.approx(0.15)
    assert primf[0] == pytest.approx(0.25)
    # total land conserved: primary + both pools
    assert primf[0] + legacy2.area[0] + new2.area[0] == pytest.approx(0.3 + 0.4)
    assert grown[0] > 0   # the remaining legacy stands keep growing
