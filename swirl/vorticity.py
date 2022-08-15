#######################################################
###
###             vorticity.py
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
##  the vorticity criterion
##
#########################
#
## Imports
#
import numpy as np
from .utils import create_U
#
#
#######################################################
##
##
def compute_vorticity(v, dl, l):  
#
#********************
# Computes vorticity arrays with l stencils
#******************** 
#
##  INPUTS
#    
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    - l : list
#       - list of stencils to use
#
##  OUTPUTS
#
#     - W : list of arrays
#       - Vorticity arrays
#     - U : list of arrays
#       - Velocity gradient tensor arrays
#
###############
    
  # Prepare the W,U instance
  W = []
  U = []
  nx = v.x.shape[0]
  ny = v.x.shape[1]
  
  # loop over stencils
  for il in l:
    # initialize arrays
    Wi = np.zeros((nx,ny))
    Ui = np.zeros((nx,ny))

    # fill velocity gradient tensor:
    Ui = create_U(v,dl,il) 

    # compute vorticity
    Wi[il:-il, il:-il] = (   (v.y[2*il:, il:-il] - v.y[:-2*il, il:-il])/(2.*dl.x*il)
                           - (v.x[il:-il, 2*il:] - v.x[il:-il, :-2*il])/(2.*dl.y*il) )

    
    # Clean velocity gradient tensors 
    mask = (Wi==0.0)
    Ui[mask] = np.array([[1.,0.],[0.,1.]])
      
    # add to W,U
    W.append(Wi)
    U.append(Ui)

  return W, U
##
##
#######################################################
##
##