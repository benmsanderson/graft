# graft

Rapid land-use scenario inputs for Earth system models from regionally
aggregated public IAM data: gridded land-use forcing in the format of the
Land-Use Harmonization dataset (LUH3), 2020-2500, produced here for the seven
CMIP7 ScenarioMIP marker scenarios.

graft is the Python half of the workflow: input preparation, post-processing
(annual output, secondary-land age and biomass, the extensions to 2500),
packaging, and diagnostics against LUH3. Harmonization and downscaling are done
by a fork of PIK's [mrdownscale](https://github.com/pik-piam/mrdownscale)
([benmsanderson/mrdownscale](https://github.com/benmsanderson/mrdownscale),
branch `graft/primary-forest-harvest`), whose `release/` folder holds the
single entry point that builds everything.

The product is independent of LUH3 and not part of the official CMIP7 forcing
datasets. It uses the same method for every scenario and is evaluated against
LUH3 where LUH3 scenarios exist.

## The dataset (v0.1)

| | |
|---|---|
| markers | VL (REMIND-MAgPIE), L (MESSAGEix-GLOBIOM-GAINS), LN (AIM), M (IMAGE), ML (COFFEE), H (GCAM), HL (WITCH) |
| period | 2020-2100 (`CICERO-graft-landState-<marker>-0-1`) and 2100-2500 (`...-<marker>-ext-0-1`) |
| content | LUH-format states (with secondary mean age and biomass), gross transitions and wood harvest, management; annual, 0.25 degrees |
| size | 14 datasets, 97 GB |
| access | NIRD Research Data Archive, DOI to come |
| licence | CC BY 4.0 |

Each product is harmonized to LUH3 history at 2025, closes its land budget
under LUH3's accounting rule to 0.0001 Mha in every year, and joins LUH3
history without a step. Against the LUH3 scenarios published on input4MIPs
(VL, M, H and HL) at 2100:

| | VL | M | H | HL |
|---|---|---|---|---|
| forest | -4.1 % | +1.7 % | -4.1 % | +0.5 % |
| primary forest | -10.3 % | -2.4 % | -0.7 % | -7.0 % |
| cropland | +0.1 % | +1.0 % | -0.3 % | -4.1 % |
| pasture | +6.9 % | +2.5 % | +2.5 % | +7.5 % |

Gap relative to LUH3; grid-cell correlations 0.86-1.00. Almost all of the
grid-cell difference is where land use is placed within regions rather than
how much each region has. The method, evaluation and limitations are
described in the accompanying paper (Sanderson et al., in preparation).

**Inputs and attribution.** The scenarios are those of the ScenarioMIP
modelling teams, from the IIASA *ScenarioMIP/CMIP7 Ensemble - Data at the R10
region level*, release v0.1 (September 2026,
<https://scenariomip.apps.ece.iiasa.ac.at>), used under its licence and not
redistributed here. History is LUH3 (`UofMD-landState-3-1-1`, University of
Maryland; Hurtt et al. 2020, Chini et al. 2021).

## How it is built

```
IIASA R10 release ──> iamc_coverage.py ──> extend_iamc.py ─┐   (input, carried to 2150)
                                                           v
                       prepare_iamc_source.py ──> mrdownscale fork (run_scenariomip.R)
                                                           │   (harmonize to LUH3, downscale, 2020-2150)
                                                           v
            annualise.py ─> check_closure.py ─> age_track.py ─> extend_product.py
                                                           │   (annual; secma/secmb; static period to 2500)
                                                           v
                                        package_release.py ──> release datasets
```

Everything runs from the fork: `release/build.sh` (all seven markers, from an
empty cache, about ten hours) and `release/package.sh`. A from-scratch build
reproduces the released files bit for bit; see the fork's
`release/README.md` for inputs, environment and verification. The changes made
to mrdownscale, each with its reason, are listed in the fork's `README.md`.

## Repository

**`src/graft/`**: the library.

| module | |
|---|---|
| `io.luh`, `io.mrdownscale`, `luh_schema` | LUH3 and mrdownscale readers, LUH vocabulary |
| `paths` | where the LUH3 data live (`GRAFT_LUH3`, `GRAFT_LUH3_SCENARIOS`, `GRAFT_LUH3_EXT`) |
| `validate` | area closure and state/transition consistency |
| `age` | secondary-land mean age and biomass, propagated from LUH3 2024 |
| `extend` | wood harvest after 2150: maintenance demand, GLM's allocation rule |
| `compare`, `carbon` | scoring against LUH3; carbon consistency |

**`scripts/`**, by stage:

| stage | scripts |
|---|---|
| inputs | `fetch_luh3.py`, `iamc_coverage.py`, `region_masks.py`, `luh_country_mask.py`, `prepare_iamc_source.py`, `extend_iamc.py` |
| mrdownscale | `run_markers.sh`, `run_scenariomip.R` |
| post-processing | `annualise.py`, `check_closure.py`, `age_track.py`, `extend_product.py`, `extend_summary.py`, `extend_marker.sh` (all of these, for one marker) |
| release | `package_release.py`, `compare_products.py` |
| figures | `figure_data.py` (released data to small tables), `plot_extension.py`, `run_notebook.sh` |
| diagnostics | `score_luh.py`, `extend_compare.py`, `carbon_check.py`, `carbon_summary.py`, `luh_as_iamc.py`, `luh_provenance.py`, `output_gap.py`, `fetch_luh3_harvest.py`, `export_mrdownscale_states.R`, `run_testthat.sh` |

**`data/`**: R10 region membership per model, and the country of each grid cell
(from LUH3's country codes, pre-1990 blocks split with Natural Earth).
`src/graft/data/` holds GLM's harvest-probability table (Hurtt et al. 2020).

**`notebooks/reconstruction_diagnostics.ipynb`**: the evaluation against LUH3.

**`docs/method.md`**: the method in brief.

## Installing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e .
pytest
export GRAFT_LUH3=/path/to/UofMD-landState-3-1-1   # LUH3 history, from input4MIPs
```

LUH3 history can be fetched checksum-verified with `scripts/fetch_luh3.py`.
The R side (mrdownscale and its dependencies) is set up from the fork's
`release/renv.lock`.

## Citing, licence, contact

Cite the dataset (DOI to come) and the paper (in preparation). The code is
under the BSD 3-Clause licence (`LICENSE`); the fork of mrdownscale stays under
mrdownscale's LGPL-3. Contact: Benjamin Sanderson, CICERO
(benjamin.sanderson@cicero.oslo.no).

Supported by the NextGenCarbon project (European Union Horizon Europe, grant
101184989). Harmonization and downscaling build on mrdownscale (Sauer and
Dietrich, PIK, <https://doi.org/10.5281/zenodo.11244475>).
