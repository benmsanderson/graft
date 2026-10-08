"""Synthetic LUH3-shaped datasets.

Tiny (a few cells, a few years) but structurally identical to the real files:
same variable names, same dims, same closure and transition identities. CI runs
on these in milliseconds with no network and no 40 GB download.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from graft import luh_schema as S


def _grid(nlat: int = 3, nlon: int = 4):
    lat = np.linspace(80, -80, nlat)
    lon = np.linspace(-170, 170, nlon)
    return lat, lon


def make_static(nlat: int = 3, nlon: int = 4, icwtr: float = 0.1) -> xr.Dataset:
    lat, lon = _grid(nlat, nlon)
    shape = (nlat, nlon)
    return xr.Dataset(
        {
            S.STATIC_CELL_AREA: (("lat", "lon"), np.full(shape, 1000.0)),
            S.STATIC_ICE_WATER: (("lat", "lon"), np.full(shape, icwtr)),
            S.STATIC_POT_BIOMASS: (("lat", "lon"), np.full(shape, 10.0)),
        },
        coords={"lat": lat, "lon": lon},
    )


def make_consistent_scenario(
    nyear: int = 4,
    nlat: int = 3,
    nlon: int = 4,
    icwtr: float = 0.1,
    seed: int = 0,
    with_harvest: bool = False,
):
    """Return ``(states, transitions, management)`` that close exactly.

    Construction guarantees both identities: states sum to ``1 - icwtr`` every
    year, and ``states[t+1] - states[t]`` equals inflow-minus-outflow, because
    the states are *built forward from* the transitions.
    """
    rng = np.random.default_rng(seed)
    lat, lon = _grid(nlat, nlon)
    shape = (nlat, nlon)
    land = 1.0 - icwtr

    states = S.AREA_STATES
    ns = len(states)
    idx = {s: k for k, s in enumerate(states)}

    # initial fractions: random simplex scaled to `land`
    f0 = rng.random((ns, nlat, nlon))
    f0 = f0 / f0.sum(axis=0, keepdims=True) * land
    frac = np.empty((nyear, ns, nlat, nlon))
    frac[0] = f0

    # transitions: for each step pick a small flux from primf->secdf and
    # c3ann->pastr, both guaranteed to leave enough source area.
    trans_pairs = [("primf", "secdf"), ("c3ann", "pastr"), ("secdf", "c3ann")]
    trans = {}
    for a, b in trans_pairs:
        trans[S.transition_name(a, b)] = np.zeros((nyear - 1, nlat, nlon))

    # optional primf->secdf wood harvest, carried as primf_harv (NOT a transition)
    harv = np.zeros((nyear - 1, nlat, nlon)) if with_harvest else None

    for t in range(nyear - 1):
        frac[t + 1] = frac[t].copy()
        for a, b in trans_pairs:
            src = frac[t, idx[a]]
            flux = np.minimum(src * 0.1, 0.02) * rng.random(shape)
            trans[S.transition_name(a, b)][t] = flux
            frac[t + 1, idx[a]] -= flux
            frac[t + 1, idx[b]] += flux
        if with_harvest:
            h = np.minimum(frac[t + 1, idx["primf"]] * 0.1, 0.01) * rng.random(shape)
            harv[t] = h
            frac[t + 1, idx["primf"]] -= h
            frac[t + 1, idx["secdf"]] += h

    time = np.arange(2020, 2020 + nyear)
    tvar = time
    ttime = time[:-1]

    state_ds = xr.Dataset(
        {s: (("time", "lat", "lon"), frac[:, idx[s]]) for s in states},
        coords={"time": tvar, "lat": lat, "lon": lon},
    )
    # add the secma/secmb diagnostics that the real states file carries
    state_ds["secma"] = (("time", "lat", "lon"), np.full((nyear, nlat, nlon), 20.0))
    state_ds["secmb"] = (("time", "lat", "lon"), np.full((nyear, nlat, nlon), 5.0))

    if with_harvest:
        trans["primf_harv"] = harv
    trans_ds = xr.Dataset(
        {name: (("time", "lat", "lon"), arr) for name, arr in trans.items()},
        coords={"time": ttime, "lat": lat, "lon": lon},
    )
    mgmt_ds = xr.Dataset(
        {"fulwd": (("time", "lat", "lon"), np.zeros((nyear, nlat, nlon)))},
        coords={"time": tvar, "lat": lat, "lon": lon},
    )
    return state_ds, trans_ds, mgmt_ds
