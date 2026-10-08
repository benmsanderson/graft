"""Conservation checks on LUH3 forcing.

Three checks, run by streaming a scenario one year-step at a time:

1. **Area closure** -- per cell the 13 land-state fractions should sum to
   ``1 - icwtr`` (the ice/water complement). Reports the worst cell deviation
   and the implied global land area, which is an external sanity yardstick.
2. **State/transition consistency** -- for every state ``s`` the year-on-year
   area change ``states[t+1] - states[t]`` should equal inflows minus outflows
   read from the transitions file. This is the identity the whole bookkeeping
   rests on; a non-zero residual means the gridded product does not close.
3. **Negative fractions** -- surfaces the "clamp negative fractions, absorb the
   deficit into secondary forest" fix rather than inheriting it
   silently: reports the most-negative land-state fraction seen.

This module is the pipeline's permanent verification stage. Its first real output is a conservation report on LUH3 itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from graft import luh_schema as S
from graft.io.luh import LUHScenario

# fraction tolerance below which a residual is "closed". LUH3 is stored float32,
# so an exactly-closing file still shows ~1e-5 rounding; 1e-4 passes clean data
# while catching real breaks (which are O(1e-2) or larger).
DEFAULT_TOL = 1e-4


@dataclass
class YearRecord:
    """Conservation metrics for a single year-step."""

    year: int
    # area closure (fraction units unless noted)
    closure_max_abs: float
    closure_area_wt_mean_abs: float
    land_area_km2: float
    # state/transition consistency
    consistency_max_abs: float
    consistency_worst_state: str
    consistency_by_state: dict[str, float]  # area-weighted mean abs residual
    consistency_residual_area_km2: float  # signed net residual, all states
    # negative fractions
    neg_state_min: float
    neg_state_count: int
    n_valid_cells: int


@dataclass
class ConservationReport:
    """Result of :func:`check_scenario`."""

    scenario: str
    tol: float
    records: list[YearRecord] = field(default_factory=list)

    # -- aggregates ----------------------------------------------------------
    @property
    def worst_closure(self) -> float:
        return max((r.closure_max_abs for r in self.records), default=0.0)

    @property
    def worst_consistency(self) -> float:
        return max((r.consistency_max_abs for r in self.records), default=0.0)

    @property
    def most_negative(self) -> float:
        return min((r.neg_state_min for r in self.records), default=0.0)

    @property
    def passed(self) -> bool:
        return (
            self.worst_closure <= self.tol
            and self.worst_consistency <= self.tol
            and self.most_negative >= -self.tol
        )


def _land_mask_and_area(
    static: xr.Dataset | None, shape: tuple[int, int], lat: np.ndarray
) -> np.ndarray:
    """Per-cell area weights (km^2). Uses ``carea`` if a static grid is given,
    else a cos(lat) proxy broadcast across longitude."""
    if static is not None and S.STATIC_CELL_AREA in static:
        return np.asarray(static[S.STATIC_CELL_AREA].values, dtype="float64")
    w = np.cos(np.deg2rad(lat))
    return np.broadcast_to(w[:, None], shape).astype("float64")


def check_scenario(
    scen: LUHScenario,
    name: str = "scenario",
    *,
    tol: float = DEFAULT_TOL,
    year_stride: int = 1,
    max_steps: int | None = None,
    verbose: bool = False,
) -> ConservationReport:
    """Stream ``scen`` and return a :class:`ConservationReport`.

    ``year_stride``/``max_steps`` subsample the time axis for a quick first pass;
    leave both at their defaults for the full record.
    """
    report = ConservationReport(scenario=name, tol=tol)

    static = scen.static
    icwtr = None
    if static is not None and S.STATIC_ICE_WATER in static:
        icwtr = np.asarray(static[S.STATIC_ICE_WATER].values, dtype="float64")

    # a product that folds plantations into secdf (graft's ESM product) has no
    # pltns state; check the area states it carries
    states = tuple(s for s in S.AREA_STATES if s in scen.states)
    trans_vars = set(scen.transitions.data_vars)
    inflow = {s: S.inflow_vars(s, trans_vars) for s in states}
    outflow = {s: S.outflow_vars(s, trans_vars) for s in states}
    # wood-harvest area that moves between states (primary -> secondary); folded
    # into the consistency identity so forest states close (see luh_schema).
    harvest_move = {
        src: (hv, dst)
        for src, (hv, dst) in S.AREA_MOVING_HARVEST.items()
        if hv in trans_vars
    }

    lat = np.asarray(scen.states["lat"].values, dtype="float64")

    for year, s0, s1, tr in scen.iter_state_steps(stride=year_stride, max_steps=max_steps):
        shape = s0[states[0]].shape
        area = _land_mask_and_area(static, shape, lat)

        # stack of the area-state fractions at t
        state_stack = np.stack(
            [np.asarray(s0[v].values, dtype="float64") for v in states]
        )
        sum_states = np.nansum(state_stack, axis=0)
        finite_any = np.isfinite(state_stack).any(axis=0)

        # --- 1. area closure ---
        target = (1.0 - icwtr) if icwtr is not None else 1.0
        closure_dev = np.where(finite_any, sum_states - target, np.nan)
        valid = np.isfinite(closure_dev)
        n_valid = int(valid.sum())
        w = area[valid]
        wsum = w.sum() if w.size else 1.0
        abs_dev = np.abs(closure_dev[valid])
        closure_max = float(abs_dev.max()) if abs_dev.size else 0.0
        closure_awm = float((w * abs_dev).sum() / wsum) if abs_dev.size else 0.0
        land_area = float((area * np.where(finite_any, sum_states, 0.0)).sum())

        # harvest area moving primary -> secondary this year-step
        harvest_gain = {s: np.zeros(shape) for s in states}
        harvest_loss = {s: np.zeros(shape) for s in states}
        for src, (hv, dst) in harvest_move.items():
            arr = np.nan_to_num(np.asarray(tr[hv].values, dtype="float64"))
            harvest_loss[src] = harvest_loss[src] + arr
            harvest_gain[dst] = harvest_gain[dst] + arr

        # --- 2. state/transition consistency ---
        by_state: dict[str, float] = {}
        worst_state, worst_val = "", 0.0
        signed_area_residual = 0.0
        for s in states:
            delta = np.asarray(s1[s].values, "float64") - np.asarray(s0[s].values, "float64")
            inflw = _sum_vars(tr, inflow[s], shape) + harvest_gain[s]
            outflw = _sum_vars(tr, outflow[s], shape) + harvest_loss[s]
            residual = delta - (inflw - outflw)
            m = np.isfinite(residual)
            if not m.any():
                by_state[s] = 0.0
                continue
            ares = np.abs(residual[m])
            ww = area[m]
            awm = float((ww * ares).sum() / ww.sum())
            by_state[s] = awm
            mx = float(ares.max())
            if mx > worst_val:
                worst_val, worst_state = mx, s
            signed_area_residual += float(np.nansum(area * residual))

        # --- 3. negative fractions ---
        neg_min = float(np.nanmin(state_stack)) if np.isfinite(state_stack).any() else 0.0
        neg_count = int(np.nansum(state_stack < -tol))

        rec = YearRecord(
            year=year,
            closure_max_abs=closure_max,
            closure_area_wt_mean_abs=closure_awm,
            land_area_km2=land_area,
            consistency_max_abs=worst_val,
            consistency_worst_state=worst_state,
            consistency_by_state=by_state,
            consistency_residual_area_km2=signed_area_residual,
            neg_state_min=neg_min,
            neg_state_count=neg_count,
            n_valid_cells=n_valid,
        )
        report.records.append(rec)
        if verbose:
            print(_format_year(rec, tol))

    return report


def _sum_vars(ds: xr.Dataset, names: list[str], shape: tuple[int, int]) -> np.ndarray:
    """Sum a list of transition variables, treating NaN as 0 (no flux)."""
    if not names:
        return np.zeros(shape, dtype="float64")
    acc = np.zeros(shape, dtype="float64")
    for v in names:
        acc += np.nan_to_num(np.asarray(ds[v].values, dtype="float64"))
    return acc


# --- reporting --------------------------------------------------------------
def _format_year(r: YearRecord, tol: float) -> str:
    flag = "ok " if (r.closure_max_abs <= tol and r.consistency_max_abs <= tol) else "!! "
    return (
        f"{flag}{r.year}  closure max={r.closure_max_abs:.2e} awm={r.closure_area_wt_mean_abs:.2e}  "
        f"consist max={r.consistency_max_abs:.2e} ({r.consistency_worst_state})  "
        f"land={r.land_area_km2/1e6:.2f}e6 km2  negmin={r.neg_state_min:.2e}"
    )


def format_report(report: ConservationReport) -> str:
    lines = [
        f"Conservation report: {report.scenario}",
        f"  years checked : {len(report.records)}"
        + (f" ({report.records[0].year}-{report.records[-1].year})" if report.records else ""),
        f"  tolerance     : {report.tol:.1e} (fraction)",
        "",
    ]
    lines += [_format_year(r, report.tol) for r in report.records]
    lines += [
        "",
        f"  worst area-closure residual   : {report.worst_closure:.3e}",
        f"  worst transition residual     : {report.worst_consistency:.3e}",
        f"  most-negative state fraction  : {report.most_negative:.3e}",
        f"  VERDICT: {'PASS' if report.passed else 'FAIL (see flagged years)'}",
    ]
    return "\n".join(lines)
