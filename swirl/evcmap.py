#######################################################
###
###             evcmap.py
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
##  the evcmap based on the estimated vortex center 
##  principle.
##
#########################
#
## Imports
#
import numpy as np
#
#
#######################################################
##
##
def compute_evcmap(X, U, v, dl):
#
#********************
# Computes the EVC map arrays with l stencils
#******************** 
# 
##  INPUTS
#    
#    - X : list of arrays
#       - criterion arrays
#    - U : list of arrays
#       - velocity gradient tensor arrays
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    
##  OUTPUTS
#
#     - M : list of arrays
#       - center maps
#     - dataCells : list of arrays
#       - array containing original coordinates, centers,
#         and criteria values 
###############

  # Initialize empty lists
  dataCells = []

  # get size of grid
  nx = X[0].shape[0]
  ny = X[0].shape[1]

  # Initialize the cardinality of EVC map
  S = np.zeros((nx,ny))

  # loop over stencils:
  for n in np.arange(len(X)):

    # build grids
    x,y = np.meshgrid(np.arange(0,nx), np.arange(0,ny), indexing='ij')

    # compute estimated radius (in units)
    r = radius(X[n], v, dl)

    # compute estimated center direction
    ex, ey = direction(U[n], v, dl)

    # compute jump in pixels to reach 
    # center from coordinates
    px = r*ex/dl.x
    py = r*ey/dl.y

    # Build center maps
    xc = x + px
    yc = y + py

    # Treat outside points
    # Flag ecvs outside of domain
    xc = np.where(xc > nx, -1., xc)
    xc = np.where(xc < 0., -1., xc)
    yc = np.where(yc > ny, -1., yc)
    yc = np.where(yc < 0., -1., yc)
    
    # Flag ecvs where criterion is null
    xc = np.where(X[n] == 0., -1., xc)
    yc = np.where(X[n] == 0., -1., yc)

    # Flatten all quantities
    x = x.flatten()
    y = y.flatten()
    xc = xc.flatten()
    yc = yc.flatten()
    r = r.flatten()
    Xi = X[n].flatten()

    # Remove the flagged points
    mask = np.where(xc==-1.)
    xc = np.delete(xc, mask)
    yc = np.delete(yc, mask)
    x = np.delete(x, mask)
    y = np.delete(y, mask)
    r = np.delete(r, mask)
    Xi = np.delete(Xi, mask)

    mask = np.where(yc==-1.)
    xc = np.delete(xc, mask)
    yc = np.delete(yc, mask)
    x = np.delete(x, mask)
    y = np.delete(y, mask)
    r = np.delete(r, mask)
    Xi = np.delete(Xi, mask)

    # Define coordinate of cell centers and weights
    xcell = np.round(xc)
    ycell = np.round(yc) 
    w = np.sign(Xi)   

    # Compute number of points per cell with np.histogram
    Si, _, _ = np.histogram2d(xcell, ycell, weights=w, bins=[np.arange(0,nx+1), np.arange(0,ny+1)])
    
    # Remove single counters
    mask = np.where(np.abs(Si) < 2)
    Si[mask] = 0.

    # Add to cardinality map
    S = S + Si

    # Fill dataCells
    dataCells_n = np.zeros((7,xc.shape[0]))

    dataCells_n[0] = xc # EVC coord x
    dataCells_n[1] = yc # EVC coord y
    dataCells_n[2] = x  # Cell coord x
    dataCells_n[3] = y  # Cell coord y
    dataCells_n[4] = Xi # Criterium value
    dataCells_n[5] = r  # Estimated distance from vortex center
    dataCells_n[6] = n*np.ones((xc.shape[0])) # Stencil index

    # Append
    dataCells.append(dataCells_n)
  
  # New grid of coordinates 
  xgrid, ygrid = np.meshgrid(np.arange(0,nx), np.arange(0,ny), indexing='ij')

  # Flatten
  S = S.flatten()
  xgrid = xgrid.flatten()
  ygrid = ygrid.flatten()

  # # Find coordinates and number of centers
  # # Remove point where only one or less ecv are found 
  mask = np.where(np.abs(S) < 2.)
  S = np.delete(S, mask)
  xgrid = np.delete(xgrid, mask)
  ygrid = np.delete(ygrid, mask)

  # Assembling the output data arranged in ascending order
  M = np.zeros((3,S.shape[0]))
  inds = S.argsort()  
  M[0] = xgrid[inds]
  M[1] = ygrid[inds]
  M[2] = S[inds]

  # Make array
  dataCells = np.hstack(dataCells)

  return M, dataCells

#######################################################
##
##
def radius(X, v, dl):
#
#********************
# Computes the radius for all "rotating" cells
#******************** 
# 
##  INPUTS
#    
#    - X : array
#       - criterion array
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    
##  OUTPUTS
#
#     - r : array
#       - the radius distance in units
###############

  # compute vnorm 
  vnorm = np.sqrt(v.x**2 + v.y**2)

  # radius in units
  # ! avoid X = 0.0 points
  with np.errstate(divide='ignore'):
    r = np.where(np.abs(X) == 0.,  0., 2.*vnorm/np.abs(X))

  return r



#######################################################
##
##
def direction(U, v, dl):
#
#********************
# Computes the radial direction for all "rotating" cells
#******************** 
# 
##  INPUTS
#    
#    - U : array
#       - velocity gradient tensor array
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    
##  OUTPUTS
#
#     - ex : array
#       - the x direction (normalized)
#     - ey : array
#       - the y direction (normalized)
###############

  # invert U
  # ! all non invertible points already treated in swirling strength
  Uinv = np.linalg.inv(U)
  
  # compute directions
  ex = -(Uinv[:,:,0,0]*v.x + Uinv[:,:,0,1]*v.y)
  ey = -(Uinv[:,:,1,0]*v.x + Uinv[:,:,1,1]*v.y)

  # normalize
  enorm = np.sqrt(ex**2 + ey**2)
  ex = np.where(enorm>0., ex/enorm, 0.)
  ey = np.where(enorm>0., ey/enorm, 0.)

  return ex,ey