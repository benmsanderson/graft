#!/usr/bin/env bash
# Run the mrdownscale fork's testthat suite with the cluster's R + geospatial
# modules loaded (terra links against these at runtime).
set -uo pipefail
source /etc/profile.d/z00_lmod.sh 2>/dev/null
module load R/4.4.2-gfbf-2024a
module load GDAL/3.10.0-foss-2024a GEOS/3.12.2-GCC-13.3.0 PROJ/9.4.1-GCCcore-13.3.0 netCDF/4.9.2-gompi-2024a
export LC_ALL=C.UTF-8
FORK=${1:?usage: run_testthat.sh <path to the mrdownscale fork>}
cd "$FORK"
Rscript -e 'for (p in c("testthat","pkgload")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org")' 2>&1 | tail -2
Rscript -e 'library(testthat); pkgload::load_all(".", quiet=TRUE); test_dir("tests/testthat")' 2>&1 | tail -45
