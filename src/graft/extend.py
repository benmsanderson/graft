"""Wood harvest after the scenarios end, the way LUH's extensions do it.

After 2100 the IAMs say nothing, so LUH holds managed land and keeps only
wood harvest and shifting cultivation going (Hurtt et al. 2020, GMD 13, 5425,
section 2.13). In LUH3's extensions shifting cultivation clears secondary land,
never primary, so all primary forest lost after the ramp is harvest. Two things decide that loss, and this module does both:

- *How much is harvested.* The CMIP7 design is
  ``h_k(t) = max(R_k(t-1), file_k(t))``: the IAM demand ramped to zero by 2149,
  floored by a maintenance term ``R``, the zone's biomass recovery over the
  previous year. :func:`maintenance` is the term used here: half of the gross
  recovery of all live biomass, the level the net-change formula settles at.
- *Where it comes from.* GLM's own rule (``zone.cc``,
  ``precomputeHarvestRatios``): demand is split between primary forest and
  *mature* secondary forest in proportion to the biomass each has available,
  maturity being the secondary biomass weighted by GLM's harvest probability,
  which rises with stand biomass. What both cannot meet is taken from young
  secondary forest. :func:`allocate` implements it, with one change: primary
  forest is weighed as a mature stand, not at all of its biomass. Secondary
  biomass comes from :mod:`graft.age`.

GLM harvests the cells nearest agriculture first within each pool; this
spreads a zone's harvest over its cells in proportion to what each holds, so
it decides how much is taken from each pool as GLM does, and where less
closely.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graft import age, paths

#: GLM's harvest probability table (CMIP6 GLM, ``phbio.average.7states.txt``)
HARVEST_PROBABILITY = paths.DATA / "phbio.average.7states.txt"
#: aboveground share of potential biomass (GLM, ``grid_info.cc``): what a
#: harvest of primary land yields, and what secondary land regrows towards
ABOVEGROUND = 0.75
#: years over which the last ramp year's harvest fades out, as the protocol
#: takes AFOLU from its 2100 value to zero in 2149
FADE = 49


def harvest_probability(biomass: np.ndarray, table: np.ndarray) -> np.ndarray:
    """GLM's ``harvestProbability``: the table's value for the 0.5 kg C/m2 bin
    the biomass falls in; zero for no biomass, the last value beyond the table."""
    index = np.clip(np.ceil(biomass / 0.5).astype(int) - 1, 0, len(table) - 1)
    return np.where(biomass <= 0, 0.0, table[index])


def load_probability(path: Path = HARVEST_PROBABILITY) -> np.ndarray:
    return np.loadtxt(path)[:, 1]


@dataclass
class Harvest:
    """One year's harvest, per cell."""

    primary_area: np.ndarray      #: fraction of the cell moved from primary to secondary forest
    secondary_area: np.ndarray    #: fraction of the cell whose secondary stand is felled
    primary_carbon: np.ndarray    #: kg C taken from primary forest
    secondary_carbon: np.ndarray  #: kg C taken from secondary forest
    unmet: np.ndarray             #: kg C per zone that no land could supply
    nonforest_primary_area: np.ndarray | float = 0.0    #: primary non-forest moved to secondary
    nonforest_secondary_area: np.ndarray | float = 0.0  #: secondary non-forest felled
    nonforest_primary_carbon: np.ndarray | float = 0.0    #: kg C taken from primary non-forest
    nonforest_secondary_carbon: np.ndarray | float = 0.0  #: kg C taken from secondary non-forest
    #: harvest area as a file reports it, where that is not the area of whole
    #: stands (mrdownscale's undivided secondary harvest); None: the stands
    secondary_reported: np.ndarray | None = None
    nonforest_secondary_reported: np.ndarray | None = None

    @property
    def nonforest_carbon(self):
        return self.nonforest_primary_carbon + self.nonforest_secondary_carbon

    @property
    def carbon(self):
        """Everything taken, kg C per cell."""
        return self.primary_carbon + self.secondary_carbon + self.nonforest_carbon


def allocate(demand: np.ndarray, zone: np.ndarray, primf: np.ndarray, ptbio: np.ndarray,
             forest: age.Pool, area: np.ndarray, probability: np.ndarray,
             primn: np.ndarray | None = None, other: age.Pool | None = None,
             primary_mature: bool = True) -> Harvest:
    """Split each zone's demand (kg C) between primary and secondary forest.

    ``zone`` gives each cell's zone index into ``demand``; ``area`` is cell area
    in m2; fractions are of the cell. Primary forest holds the aboveground
    share of its potential biomass, mature secondary forest its mean biomass
    times GLM's harvest probability, young secondary the rest.

    Primary forest is weighed as a mature stand, by the table's highest
    probability, so primary and secondary forest compete on one scale and no
    more than that share of primary forest is felled in a year. With
    ``primary_mature`` off all primary biomass counts against the few percent
    of secondary biomass the table lets through, as in CMIP6's GLM, which fells
    primary forest far faster than LUH3 does.
    """
    n = len(demand)
    total = lambda field: np.bincount(zone.ravel(), weights=field.ravel(), minlength=n)
    p = harvest_probability(forest.biomass, probability)
    ptbio = ABOVEGROUND * ptbio
    pv = probability.max() if primary_mature else 1.0
    primary = primf * ptbio * pv * area                 # kg C available, per cell
    mature = forest.area * forest.biomass * p * area
    young = forest.area * forest.biomass * (1.0 - p) * area
    vb, smb, yb = total(primary), total(mature), total(young)

    both = vb + smb
    share = np.where(both > 0, vb / np.where(both > 0, both, 1.0), 0.0)
    fromPrimary = np.where(demand <= both, demand * share, vb)
    fromMature = np.where(demand <= both, demand - fromPrimary, smb)
    rest = np.maximum(demand - fromPrimary - fromMature, 0.0)
    fromYoung = np.minimum(rest, yb)
    rest = rest - fromYoung
    # GLM's last resort: primary and secondary non-forest, in proportion to biomass
    if primn is None:
        primn = np.zeros_like(primf)
    if other is None:
        other = age.Pool(area=np.zeros_like(primf), age=np.zeros_like(primf), biomass=np.zeros_like(primf))
    nonPrimary = primn * ptbio * area
    nonSecondary = other.area * other.biomass * area
    npb, nsb = total(nonPrimary), total(nonSecondary)
    nonforest = npb + nsb
    fromNonforest = np.minimum(rest, nonforest)
    nshare = np.where(nonforest > 0, npb / np.where(nonforest > 0, nonforest, 1.0), 0.0)
    fromNP, fromNS = fromNonforest * nshare, fromNonforest * (1.0 - nshare)
    unmet = rest - fromNonforest

    ratio = lambda taken, available: np.where(available > 0, taken / np.where(available > 0, available, 1.0), 0.0)
    rp, rm, ry = ratio(fromPrimary, vb)[zone], ratio(fromMature, smb)[zone], ratio(fromYoung, yb)[zone]
    rnp, rns = ratio(fromNP, npb)[zone], ratio(fromNS, nsb)[zone]
    # harvest takes whole stands: the area follows the carbon at the stand's density
    return Harvest(primary_area=primf * pv * rp,
                   secondary_area=forest.area * (p * rm + (1.0 - p) * ry),
                   primary_carbon=primary * rp,
                   secondary_carbon=mature * rm + young * ry,
                   unmet=unmet,
                   nonforest_primary_area=primn * rnp,
                   nonforest_secondary_area=other.area * rns,
                   nonforest_primary_carbon=nonPrimary * rnp,
                   nonforest_secondary_carbon=nonSecondary * rns)


def step(primf: np.ndarray, forest: age.Pool, other: age.Pool, harvest: Harvest,
         target: np.ndarray, regrowth: age.Regrowth) -> tuple[np.ndarray, age.Pool, age.Pool]:
    """Apply one year's harvest and age secondary land by a year.

    Harvested primary forest enters secondary forest at age zero, harvested
    secondary stands are reset in place, and nothing else moves: managed land
    is static in the extension.
    """
    zero = np.zeros_like(primf)
    forestFlows = age.Flows(arriving=zero, leaving=zero, harvested=harvest.secondary_area,
                            fromPrimary=harvest.primary_area, transferredIn=zero, transferredOut=zero)
    otherFlows = age.Flows(arriving=zero, leaving=zero, harvested=zero + harvest.nonforest_secondary_area,
                           fromPrimary=zero + harvest.nonforest_primary_area,
                           transferredIn=zero, transferredOut=zero)
    forest, other = age.advance(forest, other, forestFlows, otherFlows, target, regrowth)
    return np.maximum(primf - harvest.primary_area, 0.0), forest, other


def live_biomass(primf: np.ndarray, primn: np.ndarray, ptbio: np.ndarray,
                 forest: age.Pool, other: age.Pool, area: np.ndarray) -> np.ndarray:
    """Standing live biomass per cell, kg C: primary land at the aboveground
    share of potential biomass, secondary land at its tracked mean."""
    return ((primf + primn) * ABOVEGROUND * ptbio + forest.area * forest.biomass
            + other.area * other.biomass) * area


def harvestable_biomass(primf: np.ndarray, primn: np.ndarray, ptbio: np.ndarray,
                        forest: age.Pool, other: age.Pool, area: np.ndarray,
                        probability: np.ndarray) -> np.ndarray:
    """Biomass GLM would count as available to harvest, kg C per cell: primary
    land at potential biomass, secondary land at its mean biomass weighted by
    the harvest probability, so young regrowth barely counts."""
    p = lambda pool: harvest_probability(pool.biomass, probability)
    return ((primf + primn) * ABOVEGROUND * ptbio + forest.area * forest.biomass * p(forest)
            + other.area * other.biomass * p(other)) * area


def recovery(before: np.ndarray, after: np.ndarray, harvested: np.ndarray,
             zone: np.ndarray, n: int) -> np.ndarray:
    """The maintenance term: a zone's gross biomass recovery over a year, kg C.

    Net change of live biomass plus what was harvested in that year, so that
    harvesting it the next year holds total biomass roughly constant. Net
    losses are ignored.
    """
    total = lambda field: np.bincount(zone.ravel(), weights=field.ravel(), minlength=n)
    return np.maximum(total(after) - total(before) + total(harvested), 0.0)


def maintenance(before: np.ndarray, after: np.ndarray, harvested: np.ndarray,
                zone: np.ndarray, n: int) -> np.ndarray:
    """The maintenance demand: half of a zone's gross recovery over the year.

    The CMIP7 formula takes the net biomass change, ``max(0, B(t) - B(t-1))``.
    Net change is regrowth less harvest, so that rule makes harvest and its
    predecessor sum to the regrowth: it settles at half the regrowth but
    alternates from year to year on the way. Half of the gross recovery is the
    same level without the alternation. Biomass keeps accumulating and the
    demand fades as secondary land fills up.
    """
    return 0.5 * recovery(before, after, harvested, zone, n)


def fade(year: int, last: int, length: int = FADE) -> float:
    """The share of the last ramp year's harvest still carried in ``year``."""
    return max(0.0, 1.0 - (year - last) / length)


def carry(last: Harvest, factor: float, primf: np.ndarray, primn: np.ndarray,
          forest: age.Pool, other: age.Pool, area: np.ndarray) -> Harvest:
    """The last ramp year's harvest, cell by cell, scaled by ``factor`` and kept
    within what the land still holds.

    Primary land is taken by area. Secondary land is taken by carbon: the
    stands felled are the carbon over the pool's own biomass, as the age
    tracker reads such a file, while the area the file reported is kept, scaled,
    for writing.
    """
    def within(wanted, available):
        taken = np.minimum(wanted, available)
        return taken, np.where(wanted > 0, taken / np.where(wanted > 0, wanted, 1.0), 0.0)

    def stands(carbon, reported, pool):
        held = pool.area * pool.biomass * area
        taken = np.minimum(carbon * factor, held)
        felled = np.where(held > 0, taken / np.where(held > 0, held, 1.0), 0.0) * pool.area
        return taken, felled, np.minimum(reported * factor, pool.area)

    zero = np.zeros_like(primf)
    pa, pshare = within(last.primary_area * factor, primf)
    na, nshare = within(zero + last.nonforest_primary_area * factor, primn)
    sc, sa, sr = stands(last.secondary_carbon, last.secondary_area, forest)
    oc, oa, orep = stands(zero + last.nonforest_secondary_carbon, zero + last.nonforest_secondary_area, other)
    return Harvest(primary_area=pa, secondary_area=sa, primary_carbon=last.primary_carbon * factor * pshare,
                   secondary_carbon=sc, unmet=np.zeros_like(last.unmet),
                   nonforest_primary_area=na, nonforest_secondary_area=oa,
                   nonforest_primary_carbon=last.nonforest_primary_carbon * factor * nshare,
                   nonforest_secondary_carbon=oc, secondary_reported=sr, nonforest_secondary_reported=orep)


def add(a: Harvest, b: Harvest) -> Harvest:
    """Two harvests of the same year as one."""
    reported = lambda x, stands, y, other: (None if x is None and y is None else
                                            (stands if x is None else x) + (other if y is None else y))
    return Harvest(
        primary_area=a.primary_area + b.primary_area, secondary_area=a.secondary_area + b.secondary_area,
        primary_carbon=a.primary_carbon + b.primary_carbon,
        secondary_carbon=a.secondary_carbon + b.secondary_carbon, unmet=a.unmet + b.unmet,
        nonforest_primary_area=a.nonforest_primary_area + b.nonforest_primary_area,
        nonforest_secondary_area=a.nonforest_secondary_area + b.nonforest_secondary_area,
        nonforest_primary_carbon=a.nonforest_primary_carbon + b.nonforest_primary_carbon,
        nonforest_secondary_carbon=a.nonforest_secondary_carbon + b.nonforest_secondary_carbon,
        secondary_reported=reported(a.secondary_reported, a.secondary_area,
                                    b.secondary_reported, b.secondary_area),
        nonforest_secondary_reported=reported(a.nonforest_secondary_reported, a.nonforest_secondary_area,
                                              b.nonforest_secondary_reported, b.nonforest_secondary_area))


def removed(harvest: Harvest, ptbio: np.ndarray, forest: age.Pool, other: age.Pool,
            area: np.ndarray) -> np.ndarray:
    """Live biomass a harvest takes off the land, kg C per cell, valued as
    :func:`live_biomass` values it: primary land at the aboveground share of
    potential biomass, secondary stands at their pool's mean."""
    return ((harvest.primary_area + harvest.nonforest_primary_area) * ABOVEGROUND * ptbio
            + harvest.secondary_area * forest.biomass
            + harvest.nonforest_secondary_area * other.biomass) * area


def run(primf: np.ndarray, primn: np.ndarray, forest: age.Pool, other: age.Pool,
        ptbio: np.ndarray, area: np.ndarray, zone: np.ndarray, probability: np.ndarray,
        regrowth: age.Regrowth, years, last: Harvest | None = None):
    """The static period, year by year: only wood harvest moves land.

    Harvest is handed over from the scenario to the maintenance term across
    :data:`FADE` years. ``last`` is the harvest of the final ramp year; it is
    carried cell by cell with a weight falling linearly to zero, and the
    maintenance term from the year before, allocated by :func:`allocate`,
    takes the rest of the weight. So harvest and its split between primary and
    secondary land join the ramp without a step. (Taking the larger of the
    two per country, as the CMIP7 design is written, steps up wherever a
    country's maintenance term is above its last harvest - 13% in total and
    30% on primary forest in VL.) The first year has no year before it, so
    its maintenance term is measured over a
    year without harvest. Yields ``(year, harvest, primf, primn, forest,
    other)``: the harvest applied in that year and the land at the end of it.
    """
    n = int(zone.max()) + 1
    total = lambda field: np.bincount(zone.ravel(), weights=field.ravel(), minlength=n)
    target = regrowth.asymptote * ptbio
    zero = np.zeros_like(primf)
    idle = Harvest(primary_area=zero, secondary_area=zero, primary_carbon=zero,
                   secondary_carbon=zero, unmet=np.zeros(n))
    _, grownForest, grownOther = step(primf, forest, other, idle, target, regrowth)
    demand = maintenance(live_biomass(primf, primn, ptbio, forest, other, area),
                         live_biomass(primf, primn, ptbio, grownForest, grownOther, area),
                         zero, zone, n)
    years = list(years)
    for year in years:
        factor = fade(year, years[0] - 1) if last is not None else 0.0
        carried = carry(last, factor, primf, primn, forest, other, area) if factor > 0 else idle
        # what a cell can no longer supply of its carried harvest is sought
        # elsewhere in the country, by the rule
        shortfall = np.maximum(total(zero + last.carbon) * factor - total(zero + carried.carbon), 0.0) \
            if factor > 0 else 0.0
        harvest = add(carried, allocate((1.0 - factor) * demand + shortfall, zone, primf - carried.primary_area, ptbio, forest, area,
                                        probability, primn=primn - carried.nonforest_primary_area,
                                        other=other))
        taken = removed(harvest, ptbio, forest, other, area)
        before = live_biomass(primf, primn, ptbio, forest, other, area)
        primf, forest, other = step(primf, forest, other, harvest, target, regrowth)
        primn = np.maximum(primn - harvest.nonforest_primary_area, 0.0)
        after = live_biomass(primf, primn, ptbio, forest, other, area)
        yield year, harvest, primf, primn, forest, other
        demand = maintenance(before, after, taken, zone, n)


def merge(a: age.Pool, b: age.Pool) -> age.Pool:
    """Two pools of the same land in each cell, as one: areas added, age and
    biomass area-weighted."""
    area = a.area + b.area
    safe = np.where(area > 0, area, 1.0)
    return age.Pool(area=area,
                    age=np.where(area > 0, (a.area * a.age + b.area * b.age) / safe, 0.0),
                    biomass=np.where(area > 0, (a.area * a.biomass + b.area * b.biomass) / safe, 0.0))


def split_felled(felled: np.ndarray, legacy: age.Pool, new: age.Pool,
                 probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share a cell's felled secondary area between the legacy and new pools in
    proportion to the mature biomass each holds, as the allocation drew it."""
    ml = legacy.area * legacy.biomass * harvest_probability(legacy.biomass, probability)
    mn = new.area * new.biomass * harvest_probability(new.biomass, probability)
    total = ml + mn
    share = np.where(total > 0, ml / np.where(total > 0, total, 1.0), 0.5)
    fromLegacy = np.minimum(felled * share, legacy.area)
    return fromLegacy, np.minimum(felled - fromLegacy, new.area)


def step_legacy(primf: np.ndarray, legacy: age.Pool, new: age.Pool, harvest: Harvest,
                probability: np.ndarray, target: np.ndarray, regrowth: age.Regrowth,
                area: np.ndarray) -> tuple[np.ndarray, age.Pool, age.Pool, np.ndarray]:
    """One year with secondary forest in two pools: the legacy of the scenario
    period and the stands the extension itself has harvested or created.

    Felled legacy stands move to the new pool at age zero, as does harvested
    primary forest; felled new stands are reset in place. Returns the updated
    primary forest, both pools and the legacy pool's gross regrowth per cell
    (kg C): its biomass change plus what was felled from it.
    """
    fromLegacy, fromNew = split_felled(harvest.secondary_area, legacy, new, probability)
    tau = regrowth.forest_timescale
    before = legacy.area * legacy.biomass * area
    felledCarbon = fromLegacy * legacy.biomass * area
    legacy = age.step(legacy, target, tau, [], fromLegacy, 1.0)
    new = age.step(new, target, tau,
                   [age.Inflow(harvest.primary_area), age.Inflow(fromLegacy), age.Inflow(fromNew)],
                   fromNew, 1.0)
    regrowthCarbon = legacy.area * legacy.biomass * area - before + felledCarbon
    return np.maximum(primf - harvest.primary_area, 0.0), legacy, new, regrowthCarbon
