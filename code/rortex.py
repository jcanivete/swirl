#######################################################
###
###             rortex.py
###
#######################################################
##
##  J. R. Canivete Cuissa
##  10.02.2021
##
#########################
##
##
##  This code contains the routines needed to compute
##  the rortex criterion
##
#########################
#
## Imports
#
import numpy as np
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .utilities import create_U
#
#
#######################################################
##
##
def compute_rortex(S, W, v, dl, l, param):
#
#********************
# Computes rortex arrays with l stencils
#******************** 
# 
##  INPUTS
#    
#    - S : list of arrays
#       - swirling strength arrays
#    - W : list of arrays
#       - vorticity arrays
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    - l : list
#       - list of stencils to use
#    - param : list of float
#       - list with the enhanced swirling strength
#         parameters [eps, delta, kappa]
#
##  OUTPUTS
#
#     - R : list of arrays
#       - Rortex arrays
#     - S : list of arrays
#       - Swirling strength arrays
#     - U : list of arrays
#       - Velocity gradient tensor arrays
#     - W : list of arrays
#       - Vorticity arrays
###############
  
  # Check if need to compute vorticity:
  if isinstance(W[0],float): 
    W, _ = compute_vorticity(v, dl, l)
  
  # Check if need to compute swirling strength:
  if isinstance(S[0],float):
    S, _ = compute_swirlingstrength(v, dl, l, param)
  
  # prepare R,U
  R = []
  U = []
  nx = v.x.shape[0]
  ny = v.x.shape[1]

  # loop over stencils:
  for i in np.arange(len(l)):
    
    # Prepare velocity gradient tensor instance
    Ui = np.zeros((nx,ny))

    # fill velocity gradient tensor:
    Ui = create_U(v,dl,l[i]) 

    # get sign of rortex
    signR = np.sign(S[i])
  
    # compute total shear part
    R1 = W[i]**2 - S[i]**2
    mask = R1 < 0.0
    R1[mask] = 0.0
    
    # compute rortex
    R2 = np.abs((W[i]*signR) + np.abs(np.sqrt(R1)))
    Ri = R2*signR
    
    # Clean velocity gradient tensors 
    mask = (Ri==0.0)
    Ui[mask] = np.array([[1.,0.],[0.,1.]])

    # append result
    R.append(Ri)
    U.append(Ui)
    
  return R, S, U, W
##
##
#######################################################
##
##
