"""Does a gridded land-use forcing carry the carbon its own scenario reported?

A downscaler conserves
*area*: every region ends each year with the hectares the IAM reported. It
does not conserve carbon, because a hectare of deforestation placed in the
Congo basin and the same hectare placed on a dry margin of the same region
are the same hectare and very different emissions. Area closure passes either
way, so nothing in the existing checks can see the difference.

The bookkeeping here is deliberately the simplest defensible one, and it uses
only data that ships with the target. LUH's static file carries ``ptbio``, the
potential biomass carbon content of each cell in kg C per m^2 - the same
potential-vegetation field LUH's own bookkeeping rests on. A cell's carbon is
then its potential density times the fraction of that potential each land
state holds:

    C(cell, year) = ptbio(cell) * area(cell) * sum_s frac(s) * state(s, cell, year)

The fractions are priors, declared in :data:`POTENTIAL_FRACTION` and reported
with every result. They are not a carbon-cycle model and this module does not
pretend otherwise: absolute stocks from a one-parameter-per-class rule carry
an error of tens of Pg C.

What survives that is the comparison, which is why it is the point. Two
griddings of the *same* regional areas differ only in placement, so the
fractions largely cancel and what is left is the downscaler's own carbon
error. :func:`decompose` splits any difference exactly into the part from
differing areas and the part from differing placement:

    C = sum_(region, state) A * rho,  rho = area-weighted mean density

    dC = sum dA * (rho_a + rho_b)/2  +  sum (A_a + A_b)/2 * drho
         \\_______ how much ______/      \\_______ where _______/

The second term is what no area check can reach.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

#: Fraction of a cell's potential biomass carbon standing on each LUH state.
#:
#: A prior, not a measurement. Primary vegetation holds its potential by
#: definition; secondary forest is taken at rather over half, which is where
#: bookkeeping models put a regrowth-age-weighted mean; cropland keeps the
#: standing residue only. Perennial cropland holds more than annual because
#: the trees stand. The sensitivity of every reported number to these is
#: measured in :func:`sensitivity` rather than argued about.
POTENTIAL_FRACTION: dict[str, float] = {
    "primf": 1.00,
    "secdf": 0.55,
    "pltns": 0.45,
    "primn": 1.00,
    "secdn": 0.80,
    "range": 0.75,
    "pastr": 0.30,
    "c3ann": 0.07,
    "c4ann": 0.07,
    "c3nfx": 0.07,
    "c3per": 0.20,
    "c4per": 0.20,
    "urban": 0.05,
}

#: kg C -> Pg C, and Pg C -> Gt CO2
KG_TO_PG = 1e-12
C_TO_CO2 = 44.0 / 12.0


def density(static: xr.Dataset) -> np.ndarray:
    """Potential biomass carbon density, kg C per m^2, with ocean as zero."""
    return np.nan_to_num(np.asarray(static["ptbio"].values, dtype="float64"))


def cell_area_m2(static: xr.Dataset) -> np.ndarray:
    """Grid cell area in m^2, from the static file's km^2."""
    return np.asarray(static["carea"].values, dtype="float64") * 1e6


#: The secondary states LUH's published ``secmb`` covers, when it is there.
SECONDARY = ("secdf", "secdn")


def stock(states: xr.Dataset, static: xr.Dataset,
          fractions: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """Carbon stock per state, Pg C, one value per year.

    ``states`` holds fractions of the cell on the LUH grid; a state the
    fractions do not name is skipped and reported by the caller.

    Where the dataset carries LUH's own ``secmb``, the secondary mean biomass
    carbon density in kg C per m^2, the secondary states use it instead of a
    prior and are returned together under ``"secondary"``. That is the whole
    regrowth question answered by the target rather than guessed at, and it
    matters: LUH's field puts secondary biomass a quarter below the prior in
    2050 and growing faster, because it tracks stand age and the prior cannot.
    A product that omits ``secmb`` - mrdownscale's ScenarioMIP output does -
    leaves the caller no choice but the prior.
    """
    fractions = POTENTIAL_FRACTION if fractions is None else fractions
    area = cell_area_m2(static)
    weight = density(static) * area * KG_TO_PG
    published = "secmb" in states.data_vars
    out = {}
    for name in states.data_vars:
        if name not in fractions or states[name].dims[-2:] != ("lat", "lon"):
            continue
        if published and name in SECONDARY:
            continue
        field = np.nan_to_num(np.asarray(states[name].values, dtype="float64"))
        out[name] = fractions[name] * np.nansum(field * weight, axis=(-2, -1))
    if published:
        secondary = sum(np.nan_to_num(np.asarray(states[s].values, dtype="float64"))
                        for s in SECONDARY if s in states.data_vars)
        secmb = np.nan_to_num(np.asarray(states["secmb"].values, dtype="float64"))
        out["secondary"] = np.nansum(secmb * secondary * area * KG_TO_PG, axis=(-2, -1))
    return out


def total_stock(states: xr.Dataset, static: xr.Dataset,
                fractions: dict[str, float] | None = None) -> np.ndarray:
    """Total land carbon stock, Pg C, one value per year."""
    return sum(stock(states, static, fractions).values())


def flux(stocks: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Net land-use CO2 flux to the atmosphere, Gt CO2 per year, between years.

    Positive is a source. The value for a step is the mean over it, since a
    stock known at 5- or 10-year spacing supports nothing finer.
    """
    return -np.diff(stocks) / np.diff(years) * C_TO_CO2


@dataclass
class Decomposition:
    """An exact split of a carbon difference into amount and placement."""

    area: pd.DataFrame  #: region x state, Mha, both sides and the difference
    mean_density: pd.DataFrame  #: region x state, kg C/m^2, both sides
    from_area: float  #: Pg C explained by differing areas
    from_placement: float  #: Pg C explained by differing placement
    total: float  #: Pg C, a - b

    def __str__(self) -> str:
        return (f"{self.total:+.2f} Pg C = {self.from_area:+.2f} from area "
                f"{self.from_placement:+.2f} from placement")


def _by_region(field: np.ndarray, weight: np.ndarray, index, region: np.ndarray):
    """Sum ``field * weight`` over the masked cells, per region."""
    return pd.Series(field[index] * weight[index]).groupby(region).sum()


def decompose(a: xr.Dataset, b: xr.Dataset, static: xr.Dataset,
              index, region: np.ndarray, year: int,
              fractions: dict[str, float] | None = None) -> Decomposition:
    """Split the carbon difference between two griddings, ``a - b``, at ``year``.

    Both datasets are single years on the LUH grid, as
    :func:`graft.io.luh.select_year` returns them; ``year`` only labels the
    result. ``index`` is a pair of row and column indices selecting the masked
    land cells, and ``region`` the region of each, as
    :func:`graft.io.luh.region_of_cell` returns them. Where the two agree on regional areas - the round trip
    of a LUH3 scenario, or any two products built from the same IAM - ``from_area`` is
    zero by construction and the whole difference is placement.
    """
    fractions = POTENTIAL_FRACTION if fractions is None else fractions
    rho = density(static)
    area_ha = cell_area_m2(static) / 1e4 / 1e6  # -> Mha per unit fraction

    states = [s for s in fractions if s in a.data_vars and s in b.data_vars]
    areas, densities = {}, {}
    for side, ds in (("a", a), ("b", b)):
        for name in states:
            field = np.nan_to_num(np.asarray(ds[name].values, dtype="float64"))
            area = _by_region(field, area_ha, index, region)
            carbon = _by_region(field * rho, area_ha, index, region)
            areas[(side, name)] = area
            # area-weighted mean potential density of the land in this state
            densities[(side, name)] = carbon / area.replace(0.0, np.nan)

    area = pd.DataFrame(areas)
    dens = pd.DataFrame(densities).fillna(0.0)

    # C = A * rho * frac, in Pg: Mha -> 1e10 m2, times kg C/m2, times 1e-12
    scale = np.array([fractions[s] for s in states]) * 1e10 * KG_TO_PG
    aA = area[[("a", s) for s in states]].to_numpy()
    bA = area[[("b", s) for s in states]].to_numpy()
    aR = dens[[("a", s) for s in states]].to_numpy()
    bR = dens[[("b", s) for s in states]].to_numpy()

    from_area = float(np.sum((aA - bA) * (aR + bR) / 2 * scale))
    from_placement = float(np.sum((aA + bA) / 2 * (aR - bR) * scale))

    area.columns = pd.MultiIndex.from_tuples(area.columns, names=["side", "state"])
    dens.columns = pd.MultiIndex.from_tuples(dens.columns, names=["side", "state"])
    return Decomposition(area=area, mean_density=dens, from_area=from_area,
                         from_placement=from_placement,
                         total=from_area + from_placement)


def sensitivity(yearly: list[xr.Dataset], static: xr.Dataset, years: np.ndarray,
                spread: float = 0.2) -> pd.DataFrame:
    """How much each prior moves the implied flux, by perturbing it alone.

    ``yearly`` is one single-year dataset per entry of ``years``. Each
    fraction is moved up and down by ``spread`` of its own value - the others
    held - and the change in the flux over the whole period recorded. A result
    that survives this is a result about the land; one that does not is a
    result about :data:`POTENTIAL_FRACTION`.
    """
    def period(fractions):
        stocks = np.array([float(total_stock(y, static, fractions).sum()) for y in yearly])
        return float(flux(stocks[[0, -1]], years[[0, -1]])[0])

    base = period(None)
    rows = []
    for name, value in POTENTIAL_FRACTION.items():
        if name not in yearly[0].data_vars:
            continue
        moved = []
        for sign in (-1, 1):
            perturbed = dict(POTENTIAL_FRACTION)
            perturbed[name] = float(np.clip(value * (1 + sign * spread), 0.0, 1.0))
            moved.append(period(perturbed))
        rows.append({"state": name, "fraction": value,
                     "flux_low": moved[0], "flux_high": moved[1],
                     "range": abs(moved[1] - moved[0])})
    out = pd.DataFrame(rows).sort_values("range", ascending=False)
    out.attrs["base_flux"] = base
    return out


def potential_secondary(states: xr.Dataset, static: xr.Dataset) -> float:
    """Carbon the secondary land would hold at its cells' full potential, Pg C."""
    secondary = sum(np.nan_to_num(np.asarray(states[s].values, dtype="float64"))
                    for s in SECONDARY if s in states.data_vars)
    return float(np.nansum(secondary * density(static) * cell_area_m2(static) * KG_TO_PG))


def effective_fraction(states: xr.Dataset, static: xr.Dataset) -> float:
    """What fraction of its potential the secondary land actually holds."""
    total = stock(states, static)
    held = total.get("secondary")
    if held is None:
        held = sum(v for k, v in total.items() if k in SECONDARY)
    return float(held) / potential_secondary(states, static)


def implied_regrowth_fraction(yearly: list[xr.Dataset], static: xr.Dataset,
                              years: np.ndarray, reported_pg_c: float) -> float:
    """The regrowth prior that would reconcile the forcing with the scenario.

    The honest way to compare a coarse bookkeeping with an IAM's own land
    carbon model is not to ask whether the numbers agree - they will not - but
    how far the least certain part has to move before they do. That part is
    how much carbon secondary land holds, so replace whatever the stock used
    there, LUH's ``secmb`` or the prior, with a single fraction ``f`` of each
    cell's potential biomass and solve:

        reported = d(everything else) + f * d(secondary at full potential)

    ``reported_pg_c`` is the scenario's own cumulative land carbon change over
    ``years``, positive for a gain by the land. The return is directly
    comparable with 1: above it, secondary land would have to hold more carbon
    than the potential vegetation it regrows towards, so no admissible prior
    reconciles the two and the disagreement is in the land rather than in the
    bookkeeping. :func:`effective_fraction` gives what the forcing itself says,
    for scale.
    """
    ends = (yearly[0], yearly[-1])
    potential = np.array([potential_secondary(y, static) for y in ends])
    totals = np.array([float(total_stock(y, static).sum()) for y in ends])
    secondary = []
    for y in ends:
        parts = stock(y, static)
        secondary.append(float(parts.get("secondary",
                                         sum(v for k, v in parts.items() if k in SECONDARY))))
    other = (totals[1] - secondary[1]) - (totals[0] - secondary[0])
    spread = potential[1] - potential[0]
    if abs(spread) < 1e-9:
        return float("nan")
    return float((reported_pg_c - other) / spread)


def cumulative(reported: pd.Series, years: np.ndarray) -> float:
    """Cumulative land carbon change, Pg C, from a flux in Mt CO2 per year.

    Trapezoidal over the reported years, and sign-flipped: a scenario whose
    AFOLU emissions are negative is one whose land gains carbon.
    """
    values = np.asarray(reported, dtype="float64")
    return float(-np.trapezoid(values, years) / 1e3 / C_TO_CO2)
