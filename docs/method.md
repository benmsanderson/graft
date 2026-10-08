# Method in brief

The full description, evaluation and limitations are in the accompanying
paper. The changes made to mrdownscale, each with its reason and evidence, are
listed in the fork's `README.md`.

## Inputs

- **Scenarios.** The IIASA *ScenarioMIP/CMIP7 Ensemble - Data at the R10 region
  level*, release v0.1: per model and R10 region, land cover (cropland and its
  energy crops, pasture, forest and its primary/secondary/planted parts where
  reported, other natural land, built-up area), roundwood, nitrogen
  fertilizer. Five-yearly, or ten-yearly after 2060 for some models.
- **History.** LUH3 (`UofMD-landState-3-1-1`, 850-2024): states, transitions,
  harvest and management, and the static fields.
- **Regions.** Each model's R10 regions as countries (`data/region_mappings`),
  each grid cell's country from LUH3's `ccode` (`data/country_cell.csv.gz`).

## 2020-2100: mrdownscale

1. The IAM categories are mapped onto LUH's land states. Pasture is split into
   managed pasture and rangeland, and other natural land into primary and
   secondary, by LUH3's own 2024 composition.
2. Forest and urban land are put on LUH's definitions at 2025 by a crosswalk
   that keeps each model's changes.
3. The regional trajectories are harmonized to LUH3 history over 2025-2050;
   primary forest declines with the primary harvest area each scenario's
   roundwood implies.
4. The harmonized regional states are downscaled to the 0.25 degree grid from
   LUH3's 2024 pattern; gross transitions are derived from the states; wood
   harvest comes from roundwood calibrated to LUH3's harvest history.
5. Harvest is booked under LUH3's rule: primary land becomes secondary through
   `primf_harv`/`primn_harv`, not a transition, and each cell's primary
   harvest is its primary decline.

## Post-processing (graft)

- **Annual output** (`annualise.py`): states and management interpolated
  between reporting years, each interval's transitions held, so states and
  transitions agree by construction.
- **Secondary age and biomass** (`graft.age`): mean age and biomass carried from
  LUH3's 2024 fields through the product's own transitions and harvest;
  regrowth towards 0.75 of potential biomass, timescales 62 years (forest) and
  7 years (non-forest).
- **Checks** (`check_closure.py`, `extend_summary.py`): every state change
  against its transitions and harvest; plausibility.

## 2100-2500: the extensions

- **2100-2150** (`extend_iamc.py`, then mrdownscale to 2150): each region's
  land categories continue at their 2060-2100 mean rate, scaled by a
  multiplier falling linearly from one in 2100 to zero in 2149, within the
  region's land total; wood demand falls linearly to half its 2100 value.
- **2150-2500** (`extend_product.py`, `graft.extend`): land use static. Wood
  harvest is handed over, by a linear crossfade over 49 years, from the last
  ramp year's harvest to a maintenance harvest of half the gross recovery of
  all live biomass, allocated between primary and secondary forest by GLM's
  rule with primary forest weighed as a mature stand.

## Packaging

`package_release.py` splits each marker at 2100, as LUH publishes, adds
`secma`/`secmb` to the states files, time and coordinate bounds, and global
attributes stating the inputs, their licence and attribution, and that the
product is not LUH3.
