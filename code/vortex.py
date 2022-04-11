#######################################################
###
###             vortex.py
###
#######################################################
##
##  J. R. Canivete Cuissa
##  10.02.2021
##
#########################
##
##
##  This code contain the vortex class and the routines
##  needed to organize clustered data into vortices
##
#########################
#
## Imports
#
import numpy as np
import pandas as pd
from .utilities import vector2D
import warnings
#
#
#######################################################
##
##
class Vortex:
#
##  CLASS
#
######################
# This class is contains the data and the information
# of each vortex detected by the algorithm
######################
#
#   ATTRIBUTES:
#   
#     cluster_center : list
#       - coordinates of the cluster center [x,y]
#     cells : array
#       - coordinates of cells belonging to 
#         vortex
#     evc : array
#       - array with EVC's coordinates for each cell
#     ecr : array
#       - array with curvature radius of each cell
#     X : array
#       - vortex detection criteria values for each cell
#     stencils = array
#       - array of stencils used per cell
#     id : int
#       - identification number
#     dl : vector of float
#       - the grid spacing [dx,dy] 
#   
#   METHODS:
#    
#   compute_radius()
#
###############################################
#
  def __init__(self, cluster_center, cells, evc, ecr, X, stencils, id, dl):
  #
  #********************
  # Initialization
  #********************
  #  
  ##  INPUTS
  #       
  #     cluster_center : list
  #       - coordinates of the cluster center [x,y]
  #     cells : array
  #       - coordinates of cells belonging to 
  #         vortex
  #     evc : array
  #       - array with EVC's coordinates for each cell
  #     ecr : array
  #       - array with curvature radius for each cell
  #     X : array
  #       - vortex detection criteria values for each cell
  #     stencils = array
  #       - array of stencils used per cell
  #     id : int
  #       - identification number
  #     dl : vector of float
  #       - the grid spacing [dx,dy] 
  #
  ##  OUTPUTS
  #
  #     - Initialized Vortex class
  #
  ###############

    # initialize center
    self.cluster_center = cluster_center

    # initialize cells
    self.cells = cells

    # initialize EVCs
    self.evc = evc

    # initialize ecr
    self.ecr = ecr

    # initialize crit
    self.X = X

    # initialize stencils
    self.stencils = stencils

    # initialize id 
    self.id = id

    # initialize dl
    self.dl = dl

    #### Compute other properties
    # initialize unique_cells
    self.unique_cells = self.compute_uniquecells()

    # Initialize N_tot and N_unique
    self.N_tot, self.N_unique = self.compute_numbercells()

    # Compute radius
    self.r = self.compute_radius()

    # Center of the vortex. Initialize to cluster_center
    self.center = self.cluster_center
  #
  ###############################################
  #
  # METHODS
  #
  ###########################
  #
  def compute_uniquecells(self):
  #
  #********************
  # Prepare an array of cell locations
  # where doubles coming from multiple stencil
  # calculations have been neglected.
  #********************
  #
  ##  INPUTS
  #    
  ##  OUTPUTS
  #   - unique_cells.T : array
  #      list of cells without repetitions
  #
  ############### 
    # Initialize to "void" array
    unique_cells = np.array([[0,0]])
    
    if self.cells.shape[1]>0:
      # cells coordinates
      cells_all = self.cells.T

      # Remove duplicates
      unique_cells = np.unique(cells_all, axis=0)

    # return unique cells
    return unique_cells.T
###########################
#
#
  def compute_numbercells(self):
  #
  #********************  
  # Just get the number of cells in total and 
  # unique cells
  #********************
  #
  ##  INPUTS
  #    
  ##  OUTPUTS
  #   - N_tot : int
  #      Number of cells
  #   - N_unique : int
  #      Number of unique cells
  #
  ############### 
    N_tot = self.cells.shape[1]
    N_unique = self.unique_cells.shape[1]

    return N_tot, N_unique
  #
  #
  ###########################
  #
  #
  def compute_radius(self):
  #
  #********************  
  # Radius computed according to number of unique cells
  #********************
  #
  ##  INPUTS
  #    
  ##  OUTPUTS
  #   - r : float
  #     - Estimated radius of vortex
  #
  ###############  

    # If number of cells = Area, then sqrt(N/pi) is the
    # radius of the vortex (supposing circular vortex) 
    r = np.sqrt(self.N_unique/np.pi)

    return r
  #
  #
  ###########################
  #
  #
  def compute_center(self):
    
    # Compute center of the vortex coordinates
    # according to weighted average of EVCs

    # evc coordinates
    evcx = np.array(self.evc[0],dtype=int)
    evcy = np.array(self.evc[1],dtype=int)

    if evcx.shape[0] > 0:
      
      # Bins
      bx = np.max(evcx)-np.min(evcx)
      by = np.max(evcy)-np.min(evcy)
      if bx < 1:
        bx = 1
      if by < 1:
        by = 1
      
      # 2D histogram to find weights and average evcs 
      H = np.histogram2d(evcx, evcy, bins=[bx, by])

      # Weighted average method
      Hx = np.histogram(evcx, bins=bx)
      Hy = np.histogram(evcy, bins=by)
      xc = np.sum((Hx[1][1:]+Hx[1][:-1])/2.*Hx[0])/np.sum(Hx[0])
      yc = np.sum((Hy[1][1:]+Hy[1][:-1])/2.*Hy[0])/np.sum(Hy[0])

      # Center of the cluster method
      #xc = self.cluster_center[0]
      #yc = self.cluster_center[1]

      return [xc,yc]
    else:
      return [0,0]
  #
  ###########################
  #
  #
  def update(self, empty):
  #
  #********************  
  # Update properties of vortex
  #********************
  #
  ##  INPUTS
  #   - empty : bool
  #     - Just a bool to know if the vortex is empty   
  ##  OUTPUTS
  #
  ###############    

    if not empty:
      # compute unique_cells
      self.unique_cells = self.compute_uniquecells()

      # compute N_tot and N_unique
      self.N_tot, self.N_unique = self.compute_numbercells()

      # Compute radius
      self.r = self.compute_radius()

      # Center of the vortex 
      self.center = self.compute_center()
    else:
      self.N_tot = 0
    
    return
  #
  #
  ###########################
  #
  def detect_noise(self, noise_f):
  #
  #********************  
  # Idea: find where 1. the distance between EVC and 
  #   the center of the vortex is larger that the 
  #   estimated radius of the vortex or 2. where the 
  #   cell distance from the center of the vortex is
  #   larger than the estimated radius of the vortex.
  #********************
  #
  ##  INPUTS
  #   - noise_f : float
  #     - factor stiffening or relaxing conditions
  #       1 & 2.
  ##  OUTPUTS
  #   - noise : array (7 x Nnoise)
  #     - array structure of noise datacells.
  ###############    

    # Initialize estimated center radius and real radius 
    r = self.r

    # load necessary quantities
    evc_x = self.evc[0]
    evc_y = self.evc[1]
    cell_x = self.cells[0]
    cell_y = self.cells[1]
    cx = self.center[0]
    cy = self.center[1]
    

    # Selection 1: 
    # The EVC position is too far away from the center of the vortex
    # Too far away = noise_f*r
    d1 = np.sqrt((evc_x-cx)**2 + (evc_y-cy)**2)
    mask1, = np.where(d1 > noise_f*r)

    # Selection 2:
    # The cell position is too far away from the center of the vortex
    # Too far away = noise_f*r
    d2 = np.sqrt((cell_x-cx)**2 + (cell_y-cy)**2)
    mask2, = np.where(d2 > noise_f*r)

    # Merge two selections and create a mask
    mask = np.concatenate((mask1,mask2))
    _, i = np.unique(mask, return_index=True)
    mask = mask[i]

    # Create noise array of the type dataCells
    noise = np.zeros((7,mask.shape[0]))
    noise[0] = self.evc[0][mask]   # EVC coord x
    noise[1] = self.evc[1][mask]   # EVC coord y
    noise[2] = self.cells[0][mask] # Cells coord x
    noise[3] = self.cells[1][mask] # Cells coord y
    noise[4] = self.X[mask]        # Criteria
    noise[5] = self.ecr[mask]      # Estimated radius
    noise[6] = self.stencils[mask] # Stencils indices 
    
    # Remove these points from the vortex data points
    # Cells
    cells_tmp0 = np.delete(self.cells[0], mask)
    cells_tmp1 = np.delete(self.cells[1], mask)
    self.cells = np.array([cells_tmp0, cells_tmp1])
    # EVC
    evc_tmp0 = np.delete(self.evc[0], mask)
    evc_tmp1 = np.delete(self.evc[1], mask)
    self.evc = np.array([evc_tmp0, evc_tmp1])
    # Criteria
    self.X = np.delete(self.X, mask)
    # Estimated radius of curvature
    self.ecr = np.delete(self.ecr, mask)
    # Stencils
    self.stencils = np.delete(self.stencils, mask)

    # Update properties of vortex
    self.N_tot, self.N_unique = self.compute_numbercells()

    # Return noise
    return noise
#
#
###########################
#
  def detect_kinks(self, kink_f):
  #
  #********************  
  # Idea: if center of vortex is too distant from center 
  #   of vortex (according to cells), then it's probably a kink. 
  #   Too distant = kink_f*r
  #********************
  #
  ##  INPUTS
  #   - kink_f : float
  #     - factor used to identify kinks
  ##  OUTPUTS
  #   - kinks : array
  #     - array of datacells being kinks
  #   - empty : bool
  #     - bool saying if vortex is now empty
  ###############    
    
    # load necessary quantities
    cx = self.center[0]
    cy = self.center[1]
    r = self.r

    # Compute center of vortex according to cells positions
    # (average position)
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      cx_2 = np.mean(self.cells[0])
      cy_2 = np.mean(self.cells[1])
    
    # Compute distance between two centers
    d = np.sqrt( (cx-cx_2)**2 + (cy-cy_2)**2 )

    # Initialize kinks array
    kinks = np.array([])

    # empty bool to avoid updating vortex if it is empty
    empty = False
    # If d is larger than r, then it is a kink
    if d > kink_f*r:
      # Create kinks array of the type dataCells
      kinks = np.zeros((7,self.N_tot))
      kinks[0] = self.evc[0]   # EVC coord x
      kinks[1] = self.evc[1]   # EVC coord y
      kinks[2] = self.cells[0] # Cells coord x
      kinks[3] = self.cells[1] # Cells coord y
      kinks[4] = self.X        # Criteria
      kinks[5] = self.ecr      # Estimated radius
      kinks[6] = self.stencils # Stencils indices 

      # Update vortex structure
      self.evc = np.array([[0, 0]])
      self.cells = np.array([[0, 0]])
      self.unique_cells = np.array([[0, 0]])
      self.X = 0
      self.ecr = 0
      self.stencils = [0]

      # Empty to true because vortex if empty
      empty = True
    # Return noise
    return kinks, empty
#
#
#######################################################
##
##
def detection(dataCells, M, cluster_id, peaks_ind, fast_clustering, noise_f, kink_f, dl):
#
#********************  
# Main routine that identifies vortices
#********************
#
##  INPUTS
#   
#   dataCells : list of arrays
#     - contains the information about cells with 
#       criterion satisfied
#   M : array
#     - EVC map for fast_clustering
#   cluster_id : array
#     - contains the clustering id for the centers in
#       data
#   peaks_ind : array
#     - contains the index of centers in data
#   fast_clustering : Bool
#     - option to use the fast_clustering method
#   noise_f : float
#     - criteria to remove noise from vortices
#   kink_f : float
#     - criteria to remove kinks from vortices
#   dl : vector of float
#     - the grid spacing [dx,dy] 
#
##  OUTPUTS
#
#   vortices : list
#     - list of vortex structures with all
#       the information about vortices in the
#       data 
#   noise : list
#     - list of noisy cells
#
###############      

  # list of vortices
  vortices = []
  # noise collection
  noise = []

  # Loop over cluster peaks identified
  for index in peaks_ind:
    
    # Identification number of cluster
    id = cluster_id[index]

    # Find coordinates of peaks
    xc = 0 
    yc = 0
    if fast_clustering:
      xc = M[0,index]
      yc = M[1,index]
    else:
      xc = dataCells[0,index]
      yc = dataCells[1,index]

    # Collect data points belonging to cluster
    cluster_data = np.zeros((7,1))

    if fast_clustering:
      # If fast_clustering, first find grid-EVCs 
      # which belong to the cluster 
      mask_id = np.where(cluster_id==id)[0]
      evc_cluster = M[:2, mask_id]
      
      # Then find all EVC that belong to cluster 
      evc_data = np.round(dataCells[:2])
      # Trick to use in1d in 2D: create complex numbers
      arr1 = evc_cluster[0] + evc_cluster[1]*1j
      arr2 = evc_data[0]    + evc_data[1]*1j
      mask_evc = np.where(np.in1d(arr2,arr1))[0]
      # Select dataCells according to this selection
      cluster_data = dataCells[:,mask_evc]
    else:
      # Else, just use cluster_id to select dataCells
      # belonging to cluster
      mask_id = np.where(cluster_id==id)[0]
      cluster_data = dataCells[:,mask_id]

    # Create the vortex structure    
    v = Vortex(cluster_center=[xc,yc],
               cells=cluster_data[2:4],
               evc=cluster_data[:2],
               X=cluster_data[4],
               ecr=cluster_data[5],
               stencils=cluster_data[6],
               id = id,
               dl = dl
               )
    
    # Loop for noise identification:
    Ni = v.N_tot # Number of cells in vortex
    dN = Ni      # Number of cells during last iteration

    while dN > 1 and Ni > 0: # Loop until vortex is void
                             # or no change since last iteration
      # Identify noise
      noise_n = np.array([[]])

      # Identify noise
      if noise_f > 0.0:
        noise_n = v.detect_noise(noise_f)

      # Identify kinks
      empty = False
      if kink_f > 0.0:
        kinks_n, empty = v.detect_kinks(kink_f)
        if kinks_n.shape[0]>0:
          if noise_f > 0.0:
            noise_n = np.concatenate((noise_n, kinks_n), axis=1)
          else:
            noise_n = kinks_n

      # Append noise cells
      if noise_f > 0.0 or kink_f > 0.0:
        if noise_n.shape[0]>1:
          noise.append(noise_n)

      # Update characteristics of vortex
      v.update(empty)

      # Stop if N didn't change or if N=0    
      dN = Ni - v.N_tot
      Ni = v.N_tot

    # Append vortex
    if v.N_tot > 1:
      vortices.append(v)

  # Make array of noise cells
  if (noise_f > 0.0 or kink_f > 0.0) and len(noise)>0:
    noise = np.hstack(noise)
    
  return vortices, noise