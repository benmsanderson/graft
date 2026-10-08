#!/usr/bin/env bash
# Everything after the mrdownscale run, for one marker's output folder:
#
#   1. annualise the five- and ten-yearly product (scripts/annualise.py)
#   2. check states against transitions under LUH3's rule, every 10th year
#   3. track secondary age and biomass from LUH3 2024 (scripts/age_track.py,
#      regrowth ceiling 0.75, GLM's aboveground share)
#   4. write the static period to 2500 (scripts/extend_product.py)
#   5. summarise: land, harvest across the 2150 join, plausibility
#
#   scripts/extend_marker.sh ~/madrat/output/vlr01_iamc
#
# Paths to LUH3 come from graft.paths (GRAFT_LUH3 and friends).
set -euo pipefail
cd "$(dirname "$0")/.."
D=$(realpath "$1")
PY=${PY:-.venv/bin/python}
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
LUH3=$($PY -c "from graft import paths; print(paths.LUH3)")
STATIC=$($PY -c "from graft import paths; print(paths.STATIC)")
HISTORY=$(ls "$LUH3"/multiple-states_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn_*.nc)
one() { local f=("$D"/$1); [ ${#f[@]} -eq 1 ] && [ -e "${f[0]}" ] || { echo "expected one $1 in $D" >&2; exit 1; }; echo "${f[0]}"; }

echo "== annualise"
$PY scripts/annualise.py "$D/annual" --states "$(one 'multiple-states_*.nc')" \
    --transitions "$(one 'multiple-transitions_*.nc')" --management "$(one 'multiple-management_*.nc')"
echo "== closure, every 10th year"
$PY scripts/check_closure.py "$D/annual" --stride 10
echo "== secondary age and biomass"
$PY scripts/age_track.py --static "$STATIC" --states "$(one 'annual/multiple-states_*.nc')" \
    --transitions "$(one 'annual/multiple-transitions_*.nc')" --from 2024 --to 2150 \
    --seed-from "$HISTORY" --seed-year 2024 --asymptote 0.75 --out "$D/annual/secondary_2024-2150.nc"
echo "== static period to 2500"
rm -rf "${D:?}/extension"
$PY scripts/extend_product.py "$D/annual" "$D/extension" --secondary "$D/annual/secondary_2024-2150.nc"
$PY scripts/check_closure.py "$D/extension" --stride 115
echo "== summary"
$PY scripts/extend_summary.py "$D"
