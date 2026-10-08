import numpy as np

from graft import luh_schema as S
from graft.io.luh import LUHScenario
from tests.fixtures import make_consistent_scenario, make_static


def _scenario():
    states, trans, mgmt = make_consistent_scenario()
    return LUHScenario(states=states, transitions=trans, management=mgmt, static=make_static())


def test_schema_partition():
    # secma/secmb are diagnostics, never area states
    assert "secma" not in S.AREA_STATES
    assert "secmb" not in S.AREA_STATES
    assert len(S.AREA_STATES) == 13
    assert set(S.NATURAL_STATES) <= set(S.AREA_STATES)


def test_parse_transition():
    assert S.parse_transition("primf_to_secdf") == ("primf", "secdf")
    assert S.parse_transition("secma") is None  # not a transition
    assert S.parse_transition("fertl_c3ann") is None  # management-shaped noise


def test_inflow_outflow_symmetry():
    trans = {"primf_to_secdf", "c3ann_to_pastr", "secdf_to_c3ann"}
    assert S.outflow_vars("primf", trans) == ["primf_to_secdf"]
    assert S.inflow_vars("secdf", trans) == ["primf_to_secdf"]
    assert set(S.outflow_vars("secdf", trans)) == {"secdf_to_c3ann"}


def test_years_and_streaming():
    scen = _scenario()
    assert scen.years == [2020, 2021, 2022, 2023]
    assert scen.transition_years() == [2020, 2021, 2022]
    steps = list(scen.iter_state_steps())
    assert len(steps) == 3  # one per transition slice
    year, s0, s1, tr = steps[0]
    assert year == 2020
    # streamed slices are loaded (numpy-backed) and 2-D
    assert s0["primf"].ndim == 2
    assert np.isfinite(np.asarray(s0["primf"].values)).all()
