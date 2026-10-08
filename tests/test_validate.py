import numpy as np

from graft.io.luh import LUHScenario
from graft.validate import check_scenario
from tests.fixtures import make_consistent_scenario, make_static


def _scenario(**kw):
    states, trans, mgmt = make_consistent_scenario(**kw)
    return LUHScenario(states=states, transitions=trans, management=mgmt, static=make_static())


def test_consistent_scenario_passes():
    report = check_scenario(_scenario(), name="synthetic")
    assert report.passed, report
    assert report.worst_closure < 1e-9
    assert report.worst_consistency < 1e-9
    assert report.most_negative > -1e-9


def test_land_area_matches_static():
    # 3x4 cells, 1000 km^2 each, icwtr=0.1 -> land = 0.9 of total area
    report = check_scenario(_scenario(), name="synthetic")
    rec = report.records[0]
    expected = 3 * 4 * 1000.0 * 0.9
    assert np.isclose(rec.land_area_km2, expected, rtol=1e-6)


def test_broken_closure_is_caught():
    states, trans, mgmt = make_consistent_scenario()
    # inject an area leak: delete part of primf without a matching transition
    states["primf"][2, :, :] = states["primf"][2, :, :] - 0.05
    scen = LUHScenario(states=states, transitions=trans, management=mgmt, static=make_static())
    report = check_scenario(scen, name="broken")
    assert not report.passed
    # both the closure and the consistency identity should notice the year-2 leak
    assert report.worst_closure > 1e-3
    assert report.worst_consistency > 1e-3


def test_negative_fraction_is_flagged():
    states, trans, mgmt = make_consistent_scenario()
    states["urban"][1, 0, 0] = -0.01
    scen = LUHScenario(states=states, transitions=trans, management=mgmt, static=make_static())
    report = check_scenario(scen, name="negative")
    assert report.most_negative <= -0.01 + 1e-9
    assert any(r.neg_state_count > 0 for r in report.records)


def test_stride_and_max_steps():
    report = check_scenario(_scenario(nyear=6), year_stride=2, max_steps=2)
    assert len(report.records) == 2


def test_wood_harvest_area_is_folded_in():
    # primf->secdf moves via primf_harv (not a transition); the identity must
    # still close because the validator folds harvest area in.
    states, trans, mgmt = make_consistent_scenario(with_harvest=True)
    assert "primf_harv" in trans
    scen = LUHScenario(states=states, transitions=trans, management=mgmt, static=make_static())
    report = check_scenario(scen, name="harvest")
    assert report.passed, report
    assert report.worst_consistency < 1e-9
