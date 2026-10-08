"""Secondary land's age and biomass, which LUH publishes and mrdownscale does not.

LUH3's states files carry ``secma``, the mean age of secondary land in a cell,
and ``secmb``, its mean biomass carbon density. mrdownscale's ScenarioMIP
output carries neither, and they are not decoration: a hectare of
secondary forest is not a carbon stock until you know how old it is, a
product's land carbon can be computed only where ``secmb`` is there, and
LUH's young/mature wood-harvest split needs the age.

This propagates both from an initial condition, given how much secondary land
each cell gains and loses. Two pools are tracked, forest and non-forest, since
they regrow at very different rates, and combined on output the way LUH
reports them.

The step is deliberately the simplest thing that is *exact* given its own
assumptions:

- land entering is new, so it enters at age zero and at ``entry_biomass``;
- land leaving is drawn proportionally from the cohorts already there, so it
  changes neither the mean age nor the mean biomass;
- what remains ages by the step, and its biomass approaches a target
  exponentially.

Under those, tracking the *mean* age and the *mean* biomass is exact rather
than approximate, because both updates are affine in the quantity tracked:
ageing adds a constant, and one step of exponential approach maps ``b`` to
``target - (target - b) * exp(-dt / tau)``, which is affine in ``b``. So no
Jensen error accumulates from carrying means instead of cohorts. What is
approximate is the proportional-loss assumption, and, over a multi-year step,
that land entering during it is treated as entering at the midpoint.

The regrowth parameters are not invented. :func:`fit_regrowth` recovers the
timescales from any product that publishes ``secma`` and ``secmb``, and LUH3's
history gives a forest timescale near 62 years and a non-forest one of exactly
7.0 - stable to three digits from 1500 to 2024, which is LUH's own fixed curve
read back out of its output.

The asymptote is 0.75, GLM's aboveground share of potential biomass: GLM
counts primary land at 0.75 x ``ptbio`` and regrows secondary land towards the
same value (``grid_info.cc``, ``grid_cell.cc``), and a cross-sectional fit to
LUH3's history gives 0.745. GLM evaluates biomass at a cell's *mean* age,
where this tracker carries mean biomass exactly, and by Jensen's inequality
the mean of a convex curve sits below the curve at the mean; so on LUH3-VL's
own transitions this tracker runs about 11% below the published ``secmb`` over
2025-2100. A ceiling of 0.847 was once calibrated to close that gap, but it
lets secondary land outgrow the primary land it replaced and doubles old-stand
regrowth after 2150; one physical value is used throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Regrowth:
    """How secondary land approaches the potential biomass of its cell."""

    asymptote: float = 0.75  #: fraction of ``ptbio`` secondary land tends to (aboveground)
    forest_timescale: float = 62.0  #: years, e-folding for secondary forest
    nonforest_timescale: float = 7.0  #: years, for secondary non-forest

    def timescale(self, forest: bool) -> float:
        return self.forest_timescale if forest else self.nonforest_timescale


@dataclass
class Pool:
    """One secondary pool in every cell: how much, how old, how much carbon."""

    area: np.ndarray  #: fraction of the cell
    age: np.ndarray  #: years
    biomass: np.ndarray  #: kg C per m^2


def fit_regrowth(states: xr.Dataset, static: xr.Dataset,
                 timescales: np.ndarray | None = None) -> tuple[Regrowth, dict]:
    """Recover the regrowth curve from a product that publishes secma and secmb.

    For a fixed timescale the curve is linear in the asymptote, so the
    asymptote is solved exactly and only the timescale is scanned. Cells are
    weighted by their secondary area and split into the ones that are almost
    all forest and the ones that are almost all not, so each pool's parameters
    come from cells where the other pool contributes nothing.

    Returns the fitted :class:`Regrowth` and a dict of diagnostics.
    """
    if timescales is None:
        timescales = np.arange(1.0, 200.0, 0.5)
    get = lambda d, v: np.nan_to_num(np.asarray(d[v].values, dtype="float64"))
    pot = get(static, "ptbio")
    forest_potential = get(static, "fstnf")
    area = np.asarray(static["carea"].values, dtype="float64")
    secdf, secdn = get(states, "secdf"), get(states, "secdn")
    secondary = secdf + secdn
    age, biomass = get(states, "secma"), get(states, "secmb")
    ratio = biomass / np.where(pot > 0, pot, 1.0)

    def one(select):
        a, y, w = age[select], ratio[select], (secondary * area)[select]
        best = None
        for tau in timescales:
            x = 1.0 - np.exp(-a / tau)
            alpha = min(float(np.sum(w * x * y) / np.sum(w * x * x)), 1.0)
            residual = float(np.sum(w * (y - alpha * x) ** 2))
            if best is None or residual < best[0]:
                best = (residual, alpha, float(tau))
        residual, alpha, tau = best
        spread = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        return alpha, tau, 1.0 - residual / spread, int(select.sum())

    isForest = ((secondary > 0.02) & (secdf > 0.9 * secondary)
                & (forest_potential > 0.5) & (pot > 1.0) & (age > 0))
    isOther = (secondary > 0.02) & (secdn > 0.9 * secondary) & (pot > 0.2) & (age > 0)
    fAlpha, fTau, fR2, fN = one(isForest)
    nAlpha, nTau, nR2, nN = one(isOther)
    weight = np.array([fN, nN], dtype="float64")
    fitted = Regrowth(asymptote=float(np.average([fAlpha, nAlpha], weights=weight)),
                      forest_timescale=fTau, nonforest_timescale=nTau)
    return fitted, {"forest": {"asymptote": fAlpha, "timescale": fTau, "r2": fR2, "cells": fN},
                    "nonforest": {"asymptote": nAlpha, "timescale": nTau, "r2": nR2, "cells": nN}}


def target_biomass(static: xr.Dataset, regrowth: Regrowth) -> np.ndarray:
    """The biomass density secondary land tends towards, kg C per m^2."""
    return regrowth.asymptote * np.nan_to_num(np.asarray(static["ptbio"].values,
                                                         dtype="float64"))


@dataclass(frozen=True)
class Inflow:
    """Area arriving in a pool during a step, and what it brings with it.

    Abandonment and clearing bring bare, new land, so the defaults are age and
    biomass zero. Two inflows are not like that and the tracker would be wrong
    to treat them so: land moving between the secondary pools
    (``secdf_to_secdn`` and back) keeps the age and biomass it already had,
    and primary forest that is harvested rather than converted arrives at age
    zero but not bare, since what is taken is the merchantable stem.
    """

    area: np.ndarray
    age: np.ndarray | float = 0.0
    biomass: np.ndarray | float = 0.0


def step(pool: Pool, target: np.ndarray, timescale: float,
         inflows: Inflow | list[Inflow], loss: np.ndarray, dt: float) -> Pool:
    """Advance one pool by ``dt`` years, given what it gains and loses.

    ``loss`` is taken proportionally and so moves neither mean. What stays
    ages by ``dt`` and grows towards ``target``. What arrives does so on
    average at the midpoint of the step, so it lands aged ``dt / 2`` beyond
    whatever age it brought, and grows for that half step too.
    """
    if isinstance(inflows, Inflow):
        inflows = [inflows]
    half = dt / 2.0
    stays = np.maximum(pool.area - loss, 0.0)
    arriving = [Inflow(np.maximum(i.area, 0.0), i.age, i.biomass) for i in inflows]
    after = stays + sum(i.area for i in arriving)
    safe = np.where(after > 0, after, 1.0)

    grown = target - (target - pool.biomass) * np.exp(-dt / timescale)
    age = stays * (pool.age + dt)
    biomass = stays * grown
    for i in arriving:
        age = age + i.area * (i.age + half)
        biomass = biomass + i.area * (target - (target - i.biomass) * np.exp(-half / timescale))

    empty = after <= 0
    return Pool(area=after,
                age=np.where(empty, 0.0, age / safe),
                biomass=np.where(empty, 0.0, biomass / safe))


def combine(forest: Pool, other: Pool) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted mean age and biomass of both pools, as LUH reports them."""
    total = forest.area + other.area
    safe = np.where(total > 0, total, 1.0)
    age = (forest.area * forest.age + other.area * other.age) / safe
    biomass = (forest.area * forest.biomass + other.area * other.biomass) / safe
    return np.where(total > 0, age, 0.0), np.where(total > 0, biomass, 0.0)


def seed(states: xr.Dataset, static: xr.Dataset,
         regrowth: Regrowth) -> tuple[Pool, Pool]:
    """Start both pools from a year that publishes secma and secmb.

    Each pool takes the published mean age, and the biomass its own curve
    gives at that age; the two are then scaled by one factor per cell so that
    their area-weighted mean reproduces the published ``secmb`` exactly. The
    published fields are therefore carried forward unchanged at the start, and
    the split between pools is the curve's business rather than a guess.
    """
    get = lambda v: np.nan_to_num(np.asarray(states[v].values, dtype="float64"))
    target = target_biomass(static, regrowth)
    age = get("secma")
    pools = {}
    for name, key in (("forest", "secdf"), ("other", "secdn")):
        tau = regrowth.timescale(name == "forest")
        pools[name] = Pool(area=get(key), age=age.copy(),
                           biomass=target * (1.0 - np.exp(-age / tau)))
    _, implied = combine(pools["forest"], pools["other"])
    published = get("secmb")
    scale = np.where(implied > 0, published / np.where(implied > 0, implied, 1.0), 1.0)
    for p in pools.values():
        p.biomass = p.biomass * scale
    return pools["forest"], pools["other"]


@dataclass
class Flows:
    """What moves in and out of the two secondary pools over one step.

    All areas are cell fractions per step. ``arriving`` is land coming from
    outside secondary - abandoned cropland, cleared primary non-forest - which
    is new and bare. ``leaving`` is land converted away. ``harvested`` stays
    secondary but its stand is felled, so it returns to age zero in the same
    pool. ``fromPrimary`` is primary forest or non-forest taken by wood
    harvest, which LUH moves into secondary through ``primf_harv`` and
    ``primn_harv`` rather than through a land transition. ``transferred`` moves
    between the two pools and keeps the age and biomass it had.
    """

    arriving: np.ndarray
    leaving: np.ndarray
    harvested: np.ndarray
    fromPrimary: np.ndarray
    transferredIn: np.ndarray
    transferredOut: np.ndarray
    #: kg C per m2 of cell taken by the felling, where the file's harvest area
    #: is not the area of whole stands (see :func:`flows_from_transitions`)
    harvestedCarbon: np.ndarray | None = None


def flows_from_transitions(transitions: xr.Dataset, index: int, dt: float = 1.0,
                           area: np.ndarray | None = None) -> tuple[Flows, Flows]:
    """Read one step's flows for both pools out of a LUH-format transitions file.

    ``index`` selects the time step and ``dt`` its length in years. Transitions
    and harvest are both per-year rates, so they are multiplied by ``dt`` to
    give the area moved over the step; with that, a pool's change closes
    against the published states to rounding.

    Two conventions are handled, because the published product and
    mrdownscale's do not agree on how primary land becomes secondary. LUH3
    carries no ``primf_to_secdf``: harvested primary forest reaches secondary
    through ``primf_harv``, and the budget needs it added. mrdownscale writes
    the transition explicitly, and then adding the harvest would count the
    same hectares twice. Which file is in hand decides, rather than a flag.

    The same goes for felling inside a pool: LUH3 splits secondary forest
    harvest into ``secyf_harv`` and ``secmf_harv``, mrdownscale writes one
    undivided ``secdf_harv``, and either is read.

    The undivided area is not an area of whole stands, though. LUH3's young and
    mature harvest each carry their cell's mean biomass; mrdownscale's single
    area carries about a quarter of it (1.3 against 4.4 kg C/m2 in VL, 2095),
    and felling all of it as whole stands would remove three times the carbon
    the file reports. So for that convention, and given the cell ``area`` (m2),
    the flows also carry the carbon taken, and :func:`advance` fells the area
    that carbon amounts to at the pool's own biomass - which is how GLM itself
    sets the area it returns to age zero.
    """
    names = list(transitions.data_vars)

    def field(name):
        if name not in names:
            return 0.0
        return np.nan_to_num(np.asarray(transitions[name].isel(time=index).values,
                                        dtype="float64")) * dt

    def side(pool, other, primary):
        explicit = f"{primary}_to_{pool}"
        arriving = sum(field(v) for v in names
                       if v.endswith(f"_to_{pool}") and not v.startswith("sec")
                       and v != explicit)
        leaving = sum(field(v) for v in names
                      if v.startswith(f"{pool}_to_") and "_to_sec" not in v)
        # whichever way this file says primary land becomes secondary, once
        fromPrimary = field(explicit) if explicit in names else field(f"{primary}_harv")
        felled = sum(field(v) for v in names
                     if v.endswith("_harv") and v[:-5] in HARVEST_OF[pool])
        carbon = None
        if area is not None and UNDIVIDED[pool] + "_bioh" in names and UNDIVIDED[pool] + "_harv" in names:
            carbon = field(UNDIVIDED[pool] + "_bioh") / area
        return Flows(arriving=arriving, leaving=leaving, harvested=felled,
                     fromPrimary=fromPrimary,
                     transferredIn=field(f"{other}_to_{pool}"),
                     transferredOut=field(f"{pool}_to_{other}"),
                     harvestedCarbon=carbon)

    return side("secdf", "secdn", "primf"), side("secdn", "secdf", "primn")


#: the harvest variables that fell a stand inside each pool, keeping it there
HARVEST_OF = {"secdf": {"secyf", "secmf", "secdf"}, "secdn": {"secnf", "secdn"}}
#: mrdownscale's one harvest variable per pool, whose area is not whole stands
UNDIVIDED = {"secdf": "secdf", "secdn": "secnf"}


def advance(forest: Pool, other: Pool, forestFlows: Flows, otherFlows: Flows,
            target: np.ndarray, regrowth: Regrowth, dt: float = 1.0,
            residual: float = 0.0) -> tuple[Pool, Pool]:
    """Advance both pools one step, resolving their cross-transfers together.

    ``residual`` is the fraction of standing biomass a harvest leaves behind.
    Zero reproduces LUH's published fields best, which says its secondary
    biomass is the merchantable stand rather than everything on the plot.
    """
    def one(pool, flows, partner, timescale):
        felled = np.minimum(np.maximum(flows.harvested, 0.0), pool.area)
        if flows.harvestedCarbon is not None:
            stands = flows.harvestedCarbon / np.where(pool.biomass > 0, pool.biomass, np.inf)
            felled = np.minimum(felled, stands)
        inflows = [Inflow(flows.arriving),
                   Inflow(flows.fromPrimary, biomass=residual * target),
                   Inflow(flows.transferredIn, age=partner.age, biomass=partner.biomass),
                   Inflow(felled, biomass=residual * pool.biomass)]
        leaving = flows.leaving + flows.transferredOut + felled
        return step(pool, target, timescale, inflows, leaving, dt)

    return (one(forest, forestFlows, other, regrowth.forest_timescale),
            one(other, otherFlows, forest, regrowth.nonforest_timescale))
