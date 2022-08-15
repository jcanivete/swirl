"""
SWIRL Code
    __init__.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This is the __init__ file of the module.
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