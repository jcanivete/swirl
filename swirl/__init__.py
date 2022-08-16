"""
    _/_/_/  _/          _/  _/   _/_/_/    _/
  _/         _/        _/   _/   _/    _/  _/
    _/_/      _/      _/    _/   _/_/_/    _/
        _/     _/ _/ _/     _/   _/  _/    _/
  _/_/_/        _/  _/      _/   _/   _/   _/_/_/


              IRSOL, 11.04.2022

Author: *José Roberto Canivete Cuissa*
Email: *jcanivete@ics.uzh.ch*

The SWirl Identification by Rotation centers Localization (SWIRL)
is an automated vortex identification algorithm written in python
based on the Estimated Vortex Center (EVC) method
(Canivete Cuissa & Steiner, 2022, A&A, submitted).

The only required inputs are a two-dimensional velocity time istance
defined on a Cartesian grid, and the grid cell size.
It returns the list of the identified vortical structures.

If you use the SWIRL code in your research, please cite the paper:
    Canivete Cuissa & Steiner, 2022, A&A (submitted)
where more details about the method and the code can be found.
"""
# Imports
from .main import SWIRL
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import radius, direction
from .utils import vector2D, create_U
from .cluster import compute_delta, compute_rho, prepare_data, compute_dc, clustering
from .vortex import detection
from .plots import plot_decision, plot_evcmap, plot_vortices
