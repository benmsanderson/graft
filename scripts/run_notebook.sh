#!/usr/bin/env bash
# Execute notebooks/reconstruction_diagnostics.ipynb in place with the venv kernel
# (graft-venv), the cluster modules loaded and threads capped.
source /etc/profile.d/z00_lmod.sh 2>/dev/null
module load R/4.4.2-gfbf-2024a
module load GDAL/3.10.0-foss-2024a GEOS/3.12.2-GCC-13.3.0 PROJ/9.4.1-GCCcore-13.3.0 netCDF/4.9.2-gompi-2024a 2>/dev/null
export LC_ALL=C.UTF-8 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$(dirname "$0")/.."
NB=notebooks/reconstruction_diagnostics.ipynb
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=graft-venv "$NB" 2>&1 | tail -15
