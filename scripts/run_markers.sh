#!/usr/bin/env bash
# Stage and run one ScenarioMIP product per marker.
#
# Each marker needs its own region mapping, because an R10 region name covers
# different countries in different models, and its own staged source.
# This pairs the two and runs them in turn, keeping going if one fails so a
# single bad marker does not cost the rest of the set.
#
#   scripts/run_markers.sh                 # all seven, in turn
#   scripts/run_markers.sh m ml            # just those tags
#   PERMARKER=1 scripts/run_markers.sh h   # own source folder, shared cache,
#                                          # so several can run side by side
#
# Needs land_r10.csv from iamc_coverage.py --export, the mappings written by
# region_masks.py --write, and a country mask from luh_country_mask.py.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
EXPORT=${EXPORT:-land_r10.csv}
MAINFOLDER=${MADRAT_MAINFOLDER:-$HOME/madrat}
MASK=${MASK:-data/country_cell.csv.gz}
# Extra run_scenariomip.R options (e.g. "gross,fold" for the ESM-input product)
# and the mrdownscale clone to load; empty OPTS keeps the plain ScenarioMIP
# deliverable, which is the default.
OPTS=${OPTS:-}
MRDOWNSCALE=${MRDOWNSCALE:-$HOME/GitHub/mrdownscale}
# appended to each tag, so a new product set can sit beside an earlier one
TAGSUFFIX=${TAGSUFFIX:-}
# One source folder per marker, so several can run at once; everything except
# the staged IAMC source is shared by symlink rather than copied.
PERMARKER=${PERMARKER:-0}

# tag | mapping | scenario, in the order the markers are named
MARKERS=(
  "vl|REMIND-MAgPIE_3.5-4.11_R10|Very Low - SSP1 (Marker)"
  "l|MESSAGEix-GLOBIOM-GAINS_2.1-M-R12_R10|Low - SSP2 (Marker)"
  "ln|AIM_3.0_R10|Low-to-Negative - SSP2 (Marker)"
  "m|IMAGE_3.4_R10|Medium - SSP2 (Marker)"
  "ml|COFFEE_1.6_R10|Medium-to-Low - SSP2 (Marker)"
  "h|GCAM_8s_R10|High - SSP3 (Marker)"
  "hl|WITCH_6.0_R10|High-to-Low - SSP5 (Marker)"
)

wanted=("$@")
failed=()
for entry in "${MARKERS[@]}"; do
  IFS='|' read -r tag mapping scenario <<< "$entry"
  if [ ${#wanted[@]} -gt 0 ] && ! printf '%s\n' "${wanted[@]}" | grep -qx "$tag"; then
    continue
  fi
  echo "=== $tag: $scenario ==="
  if [ "$PERMARKER" = 1 ]; then
    source_dir="$MAINFOLDER/sources_$tag"
    mkdir -p "$source_dir"
    for d in "$MAINFOLDER"/sources/*/; do
      name=$(basename "$d")
      [ "$name" = IAMC ] && continue
      [ -e "$source_dir/$name" ] || ln -s "$d" "$source_dir/$name"
    done
    export MADRAT_SOURCEFOLDER="$source_dir"
    export MADRAT_CACHEFOLDER=${MADRAT_CACHEFOLDER:-$MAINFOLDER/cache}
    staged="$source_dir/IAMC"
  else
    staged="$MAINFOLDER/sources/IAMC"
  fi
  if ! "$PY" scripts/prepare_iamc_source.py "$EXPORT" \
        "data/region_mappings/${mapping}.csv" "$staged" \
        --country-cell "$MASK" --scenario "$scenario"; then
    echo "!!! $tag: staging failed"; failed+=("$tag"); continue
  fi
  # the writer puts the run date in each file name, so a rerun into the same
  # tag would leave two files per kind and later globs would pick either; set
  # earlier outputs aside first
  outdir="$MAINFOLDER/output/${tag}${TAGSUFFIX}_iamc"
  for d in "$outdir" "$outdir/annual"; do
    if ls "$d"/multiple-*.nc >/dev/null 2>&1; then
      mkdir -p "$d/superseded" && mv "$d"/multiple-*.nc "$d/superseded/"
    fi
  done
  if ! Rscript scripts/run_scenariomip.R "$scenario" "${tag}${TAGSUFFIX}-iamc" "$OPTS" "$MRDOWNSCALE"; then
    echo "!!! $tag: run failed"; failed+=("$tag"); continue
  fi
  echo "=== $tag done ==="
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}"; exit 1
fi
echo "all requested markers written"
