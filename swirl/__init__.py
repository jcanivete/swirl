from .main import SWIRL

from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import radius, direction
from .utils import vector2D, create_U
from .cluster import compute_delta, compute_rho, prepare_data, compute_dc, clustering
from .vortex import detection 
from .plots import plot_decision, plot_evcmap, plot_vortices
