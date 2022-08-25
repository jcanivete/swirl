"""
    _/_/_/  _/          _/  _/   _/_/_/    _/
  _/         _/        _/   _/   _/    _/  _/
    _/_/      _/      _/    _/   _/_/_/    _/
        _/     _/ _/ _/     _/   _/  _/    _/
  _/_/_/        _/  _/      _/   _/   _/   _/_/_/


             (c) IRSOL, 11.04.2022

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
    Canivete Cuissa & Steiner, 2022, A&A (submitted).
where more details about the method and the code can be found.

-----------------------------------------------

Copyright (c) 2022 José Roberto Canivete Cuissa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Imports
from .main import Identification
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import radius, direction
from .utils import vector2D, create_U, read_params
from .cluster import compute_delta, compute_rho, prepare_data, compute_dc, clustering
from .vortex import detection
from .plots import plot_rortex, plot_gevc_map, plot_decision, plot_vortices
