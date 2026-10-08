"""LUH3 (CMIP7, UofMD-landState-3-1) vocabulary.

The single source of truth for what the LUH3 variables *are*, kept separate from
the readers so the crosswalk (``graft.categories``, later) and the validator
share one definition. Values verified against the native 0.25 deg input4MIPs files.
"""

from __future__ import annotations

# --- State (land) fractions -------------------------------------------------
# The 13 variables in the states file that are genuine grid-cell area fractions.
# Their per-cell sum should close to (1 - icwtr).
AREA_STATES: tuple[str, ...] = (
    "primf",   # forested primary
    "primn",   # non-forested primary
    "secdf",   # forested secondary
    "secdn",   # non-forested secondary
    "pltns",   # planted (forest plantations)
    "c3ann",   # C3 annual crops
    "c3nfx",   # C3 nitrogen-fixing crops
    "c3per",   # C3 perennial crops
    "c4ann",   # C4 annual crops
    "c4per",   # C4 perennial crops
    "pastr",   # managed pasture
    "range",   # rangeland
    "urban",   # urban land
)

# Present in the *states* file but NOT area fractions: secondary mean age
# (secma, years) and secondary mean biomass (secmb, kg C m-2). Excluded from
# area closure; carried because the bookkeeping model uses them.
SECONDARY_DIAGNOSTICS: tuple[str, ...] = ("secma", "secmb")

# Coarse groupings used by the bookkeeping and (later) the crosswalk.
PRIMARY_STATES: tuple[str, ...] = ("primf", "primn")
SECONDARY_STATES: tuple[str, ...] = ("secdf", "secdn")
CROP_STATES: tuple[str, ...] = ("c3ann", "c3nfx", "c3per", "c4ann", "c4per")
NATURAL_STATES: tuple[str, ...] = PRIMARY_STATES + SECONDARY_STATES
# States that can only lose area (no LUH transition creates them):
IRRECOVERABLE_STATES: tuple[str, ...] = PRIMARY_STATES


def transition_name(src: str, dst: str) -> str:
    """LUH transition variable name for area moving from ``src`` to ``dst``."""
    return f"{src}_to_{dst}"


def parse_transition(var: str) -> tuple[str, str] | None:
    """Split a ``X_to_Y`` transition variable into ``(src, dst)``.

    Returns ``None`` for names that are not state-to-state transitions between
    two known area states (so management or diagnostic variables are ignored).
    """
    if "_to_" not in var:
        return None
    src, _, dst = var.partition("_to_")
    if src in AREA_STATES and dst in AREA_STATES:
        return src, dst
    return None


def inflow_vars(dst: str, available: object) -> list[str]:
    """Transition variables in ``available`` that add area to state ``dst``."""
    return sorted(
        v for v in available
        if (p := parse_transition(v)) is not None and p[1] == dst
    )


def outflow_vars(src: str, available: object) -> list[str]:
    """Transition variables in ``available`` that remove area from state ``src``."""
    return sorted(
        v for v in available
        if (p := parse_transition(v)) is not None and p[0] == src
    )


# --- Wood harvest ------------------------------------------------------------
# The transitions file also carries 12 wood-harvest fields: ``*_harv`` (area,
# grid-cell fraction) and ``*_bioh`` (harvested biomass carbon). Crucially,
# primary->secondary forest conversion is NOT a state-to-state transition -- there
# is no ``primf_to_secdf`` variable. Instead the harvested primary area moves to
# secondary via ``primf_harv`` (potential-forest land) / ``primn_harv`` (potential
# non-forest land). Any area-conservation check on transitions alone is therefore
# incomplete for the forest states; the harvest area closes it (verified on
# LUH3-VL: residual drops from ~0.25 to ~6e-8 fraction once folded in).
WOOD_HARVEST_AREA_VARS: tuple[str, ...] = (
    "primf_harv", "primn_harv", "secmf_harv", "secnf_harv", "secyf_harv", "pltns_harv",
)

# Harvests that MOVE area between states: source_state -> (harvest_var, dest_state).
# The rest (secmf/secnf/secyf/pltns) reset stand age within a state; no area change.
AREA_MOVING_HARVEST: dict[str, tuple[str, str]] = {
    "primf": ("primf_harv", "secdf"),
    "primn": ("primn_harv", "secdn"),
}


# --- Static / management field names we rely on -----------------------------
STATIC_CELL_AREA = "carea"     # grid-cell area, km^2
STATIC_ICE_WATER = "icwtr"     # ice + water fraction of cell
STATIC_POT_BIOMASS = "ptbio"   # potential biomass density
