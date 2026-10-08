"""Where graft finds the data it does not ship, set in one place.

Each location can be overridden by an environment variable, so the pipeline
runs anywhere the data are; the defaults are where they sit on CICERO's
cluster, so set the variables elsewhere.

- ``GRAFT_LUH3``: LUH3 history, the ``UofMD-landState-3-1-1`` files, which
  the products are harmonized to (states, transitions, management, static).
- ``GRAFT_LUH3_SCENARIOS``: the folder holding the LUH3 scenarios published on
  input4MIPs (``UofMD-landState-vl-3-1-1``, ``...-h-3-1-1``, ``...-m-3-1``,
  ``...-hl-3-1``); benchmarks only.
- ``GRAFT_LUH3_EXT``: the folder holding LUH3's published extensions
  (``UofMD-landState-vl-ext-3-1``, ``...-h-ext-3-1``, ``...-m-ext-3-1``);
  benchmarks only.
"""
from __future__ import annotations

import os
from pathlib import Path

LUH3 = Path(os.environ.get("GRAFT_LUH3", "/storage/no-backup-nac/LUH2/cmip7/UofMD-landState-3-1-1"))
STATIC = LUH3 / "multiple-static_input4MIPs_landState_CMIP_UofMD-landState-3-1-1_gn.nc"
LUH3_SCENARIOS = Path(os.environ.get("GRAFT_LUH3_SCENARIOS", "/storage/no-backup-nac/LUH2/cmip7"))
LUH3_EXT = Path(os.environ.get("GRAFT_LUH3_EXT", "/storage/no-backup-nac/users/bensan/LUH3-ext"))
#: shipped with graft
DATA = Path(__file__).parent / "data"
