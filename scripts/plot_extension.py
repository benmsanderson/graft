"""Global primary forest and wood harvest 2020-2500, seven markers and LUH3 (history, vl/h, vl-ext/h-ext); official ScenarioMIP colours, LUH3 dashed."""
import glob, os
from pathlib import Path
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graft import paths
st = xr.open_dataset(paths.STATIC)
cell = st.carea.values.astype("f8") / 1e4
def ys(ds):
    u = ds.time.attrs["units"]; ep = int(u.split("since")[1].strip().split("-")[0]); t = ds.time.values
    return np.round(ep + (t / 365 if u.startswith("days") else t)).astype(int)
def series(files, var, step=5, kind="area"):
    out = {}
    for f in files:
        ds = xr.open_dataset(f, decode_times=False); y = ys(ds)
        for k, yr in enumerate(y):
            if yr % step or yr in out: continue
            v = np.nan_to_num(ds[var].isel(time=k).values.astype("f8"))
            out[yr] = float((v * cell).sum()) if kind == "area" else float(v.sum()) / 1e12
    yrs = sorted(out); return np.array(yrs), np.array([out[y] for y in yrs])
M = ["vl", "l", "ln", "m", "ml", "h", "hl"]
label = {"vl": "VL", "l": "L", "ln": "LN", "m": "M", "ml": "ML", "h": "H", "hl": "HL"}
color = {"h": "#a41212", "hl": "#E744F6", "m": "#fc7b03", "ml": "#dec820", "l": "#20A359", "ln": "#22e5db", "vl": "#16188F"}
O = os.environ.get("GRAFT_OUTPUT", os.path.expanduser("~/madrat/output"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for k in M:
    files = [glob.glob(f"{O}/{k}x7_iamc/annual/multiple-states_*.nc")[0], glob.glob(f"{O}/{k}x7_iamc/extension/multiple-states_*.nc")[0]]
    y, v = series(files, "primf"); axes[0].plot(y, v, color=color[k], lw=1.8, label=label[k])
    tf = [glob.glob(f"{O}/{k}x7_iamc/annual/multiple-transitions_*.nc")[0], glob.glob(f"{O}/{k}x7_iamc/extension/multiple-transitions_*.nc")[0]]
    yb, b = None, None
    acc = {}
    for f in tf:
        ds = xr.open_dataset(f, decode_times=False); yy = ys(ds)
        for i, yr in enumerate(yy):
            if yr % 5 or yr in acc or yr == 2150 and "annual" in f: continue
            acc[yr] = sum(float(np.nansum(ds[p].isel(time=i).values)) for p in ds.data_vars if p.endswith("_bioh")) / 1e12
    yy = sorted(acc); axes[1].plot(yy, [acc[x] for x in yy], color=color[k], lw=1.8)
from graft import paths
L = str(paths.LUH3_SCENARIOS); E = str(paths.LUH3_EXT)
for k, ls in (("vl", (0, (4, 2))), ("h", (0, (4, 2)))):
    files = [glob.glob(f"{L}/UofMD-landState-{k}-3-1/multiple-states_*.nc")[0], glob.glob(f"{E}/UofMD-landState-{k}-ext-3-1/multiple-states_*.nc")[0]]
    y, v = series(files, "primf"); axes[0].plot(y, v, color=color[k], ls=ls, lw=1.3, label=f"LUH3 {label[k]}")
    acc = {}
    for f in [glob.glob(f"{L}/UofMD-landState-{k}-3-1/multiple-transitions_*.nc")[0], glob.glob(f"{E}/UofMD-landState-{k}-ext-3-1/multiple-transitions_*.nc")[0]]:
        ds = xr.open_dataset(f, decode_times=False); yy = ys(ds)
        for i, yr in enumerate(yy):
            if yr % 5 or yr in acc: continue
            acc[yr] = sum(float(np.nansum(ds[p].isel(time=i).values)) for p in ds.data_vars if p.endswith("_bioh")) / 1e12
    yy = sorted(acc); axes[1].plot(yy, [acc[x] for x in yy], color=color[k], ls=ls, lw=1.3)
for ax in axes:
    ax.axvspan(2100, 2150, color="0.9", zorder=0); ax.grid(alpha=0.3); ax.set_xlim(2020, 2500)
axes[0].set_title("Primary forest"); axes[0].set_ylabel("Mha")
axes[1].set_title("Wood harvest"); axes[1].set_ylabel("Pg C / yr")
axes[0].legend(ncol=3, fontsize=8, frameon=False)
axes[0].text(2105, axes[0].get_ylim()[0] + 20, "ramp", fontsize=8, color="0.4")
fig.tight_layout()
fig.savefig(os.environ.get("GRAFT_FIGURE", str(Path(__file__).parents[1] / "docs/slides/img/extension_2500.png")), dpi=150)
print("saved")
