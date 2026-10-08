"""graft: IAMC land-use scenarios to gridded ESM land-use transition forcing.

- ``graft.luh_schema``  -- LUH3 (CMIP7) state/transition/management vocabulary
- ``graft.io``          -- readers for LUH3 and mrdownscale output
- ``graft.paths``       -- where the LUH3 data live (configurable)
- ``graft.validate``    -- area closure + state/transition consistency report
- ``graft.compare``     -- scoring against LUH3: plantation fold, cell scores,
                           skill by resolution, potential-forest mask
- ``graft.carbon``      -- carbon consistency of a gridded product
- ``graft.age``         -- secondary-land age and biomass
- ``graft.extend``      -- wood harvest after 2150, for the extensions to 2500
"""

from graft import luh_schema

__all__ = ["luh_schema"]
__version__ = "0.1"
