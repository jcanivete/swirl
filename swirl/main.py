#######################################################
###
###             main.py
###
#######################################################
##
##  J. R. Canivete Cuissa
##  10.02.2021
##
#########################
##
##
##  This code contain the main structure of the 
##  algorithm, that is the SWIRL class.
##
#########################
#
## Imports
#
import time
from typing import List
import numpy as np
from .utils import timings, vector2D #, prepare_dataframe
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import compute_evcmap
from .cluster import findcluster2D, prepare_data
from .vortex import detection
from .plots import plot_decision
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
#
#
#######################################################
##
##
class SWIRL:
#
##  CLASS
#
#********************
# This class is the main structure of the code
# From this class one can reach all the quantities needed
#********************
#
#   ATTRIBUTES:
#   
#   v : vector of numpy array
#     - the velocity field [vx,vy]  
#   dl : vector of float
#     - the grid spacing [dx,dy] 
#   l : list of int
#     - list of stencils to use [l1,l2,...]
#   W : numpy array
#     - the vorticity array [Wz]
#   S : numpy array
#     - the swirling strength array [Sz] 
#   S_param : list
#     - List of parameters for enhanced swirling strength 
#       [eps, delta, kappa]
#   R : numpy array
#     - the rortex array [Rz]
#   M : vector of numpy array
#     - 2D map containing the center coordinates [ic,jc]
#       for all swirling points [i,j] of the domain
#   crit : string
#     - Name of the criterion choosed to be used:
#       'vorticity', 'swirling strength', 'rortex'
#   dc_coeff : int,float
#     - Percentual coefficient used to compute the critical
#       distance. The critical distance is defined as the 
#       average distance such that dc_coeff % of points inside dc.
#       Preferably, use 2-3
#   fast_clustering : bool
#     - Option to use the fast clustering option in algorithm or 
#       not
#   xi_option : int
#     - Different choices for the xi is the kernel used to compute
#       densities. 
#       Option 1: Heaviside function,
#       Option 2: Gaussian kernel
#   clust_selector : string
#     - How to select clusters in the clustering algorithm
#       'delta-rho' -> uses delta/rho selection
#       'gamma' -> uses gamma
#   noise_f : float
#     - criteria to remove noise from vortices, 
#       usually between [1.0, 2.0]
#   
#******************** 
#  
#   METHODS:
#   
#   vorticity()
#     - Computes vorticity W based on the velocity field, 
#       stencils and grid spacing
#
#   swirlingstrength()
#     - Computes the swirling strength S based on the 
#       velocity field, swirl parameters, stencils and 
#       grid spacing
#
#   rortex()
#     - Computes the rortex criterion R based on the 
#       velocity field, swirl parameters, stencils and 
#       grid spacing
#
#   map()
#     - Computes the map M of the centers and the density map
#       dM of the EVC
#
#   searchvortex()
#     - Creates a collection of VortexCells and
#       group them into VortexStructures according
#       to vortex searching algorithm
#
###############################################
#
#
  def __init__( self, 
                v: list, 
                dl: list=[1.,1.], 
                l: list=[1], 
                S_param: list=[0.,0.,0.], 
                crit: str='rortex', 
                dc_coeff: float=3.,
                dc_adaptive: bool=True, 
                fast_clustering: bool=True, 
                xi_option: int=2, 
                clust_selector: str='delta-rho', 
                clust_options: list=[1.0,0.5,2.0], 
                noise_f: float = 1.0,
                kink_f: float = 1.0,
                verbose: bool = True
              ):
  #
  #********************
  # Builds an instance of the SWIRL Class
  #********************
  #   
  ##  INPUTS
  #    
  #    - Mandatory:
  #       v : list of numpy arrays
  #         - the velocity field
  #    -----------------------------
  #    - Optional   
  #       dl [1,1] : list of float
  #         - the grid spacing
  #       l [1]: list of int
  #         - the list of stencils
  #       S_param=[0.,0.,0.] : list of float
  #         - the enhanced swirling strength param.
  #       crit='rortex' : string
  #         - criterium to use
  #       dc_coeff=3 : int
  #         - parameter of clustering algorithm
  #       dc_adaptive=True : bool
  #         - parameter of clustering algorithm
  #       fast_clustering=False : bool
  #         - parameter of clustering algorithm
  #       xi_option=2 : int
  #         - parameter of clustering algorithm
  #       clust_selector : 'delta-rho'
  #         - choice for clustering selector 
  #       clust_options : [2.0, 1.0, 2.0]
  #         - parameters for the clustering selector
  #       noise_f : 1.0
  #         - parameter to remove noise cells from vortex
  #       kink_f : 1.0
  #         - parameter to remove kinks from vortex detections
  #
  ##  OUTPUTS
  #
  ###############
    #
    ## Safe Initialization of velocity arrays and parameters 
    #
    # Initialize v
    if len(v)==2 and isinstance(v[0], np.ndarray) and isinstance(v[1], np.ndarray):
      self.v = vector2D(v[0],v[1])
    else:
      raise ValueError('Initialization: velocity field is not a list of numpy arrays [vx,vy].')

    # Initialize dl
    if len(dl)==2 and isinstance(dl[0], float) and isinstance(dl[1], float):
      self.dl = vector2D(dl[0],dl[1])
    else:
      raise ValueError('Initialization: grid spacing is not a list of float [dx,dy].') 

    # Initialize l
    if isinstance(l,list): 
      self.l = l
    else:
      raise ValueError('Initialization: stencil list is not a list of int [l1,l2,...].') 

    # Initialize S_param
    if (len(S_param)==3 and isinstance(S_param[0], float) 
                        and isinstance(S_param[1], float) 
                        and isinstance(S_param[2], float)) :
      self.S_param = S_param
    else:
      raise ValueError('Initialization: Swirling strength param. is not a list of float [eps, kappa, delta].') 
    
    # Initialize crit
    if crit in ['rortex','swirling strength','vorticity']:
      self.crit = crit
    else:
      raise ValueError('Initialization: Criterium parameter unknown')

    # Initialize dc_coeff
    if isinstance(dc_coeff, float):
      self.dc_coeff = dc_coeff
    else:
      raise ValueError('Initialization: dc_coeff must be float')

    # Initialize dc_coeff
    if isinstance(dc_adaptive, bool):
      self.dc_adaptive = dc_adaptive
    else:
      raise ValueError('Initialization: dc_adaptive must be bool')

    # Initialize fast_clustering 
    if isinstance(fast_clustering, bool):
      self.fast_clustering = fast_clustering
    else:
      raise ValueError('Initialization: fast_clustering must be a bool')

    # Initialize crit
    if xi_option in [1,2]:
      self.xi_option = xi_option
    else:
      raise ValueError('Initialization: xi_option parameter unknown')

    # Initialize clust_selector
    if clust_selector in ['delta-rho', 'gamma']:
      self.clust_selector = clust_selector
    else:
      raise ValueError('Initialization: clust_selector parameter unknown')

    # Initialize clust_options
    if isinstance(clust_options, list):
      self.clust_options = clust_options
    else:
      raise ValueError('Initialization: clust_options must be a list of parameters')

    # Initialize noise_f
    if isinstance(noise_f, float):
      self.noise_f = noise_f
    else:
      raise ValueError('Initialization: noise_f must be a float')

    # Initialize kink_f
    if isinstance(kink_f, float):
      self.kink_f = kink_f
    else:
      raise ValueError('Initialization: kink_f must be a float')

    # Initialize other quantities:
    self.S = [0.0]
    self.W = [0.0]
    self.U = [0.0]
    self.R = [0.0]
    self.M = [0.0]
    self.dataCells = [0.0]
    self.timings = dict()
    self.cluster_id = None
    self.verbose = verbose

    # Print introduction
    #
    if self.verbose:
      print('---------------------------------------------------------')
      print('---                                                   ---')
      print('---    _/_/_/  _/          _/  _/   _/_/_/    _/      ---')
      print('---  _/         _/        _/   _/   _/    _/  _/      ---') 
      print('---    _/_/      _/      _/    _/   _/_/_/    _/      ---') 
      print('---        _/     _/ _/ _/     _/   _/  _/    _/      ---') 
      print('---  _/_/_/        _/  _/      _/   _/   _/   _/_/_/  ---') 
      print('---                                                   ---')
      print('---------------------------------------------------------')
      print('---------------------------------------------------------')
      print('---                                                   ---')
      print('---                   11.04.2022                      ---')
      print('---                                                   ---')
      print('--- Author: José Roberto Canivete Cuissa              ---')
      print('--- Email: jcanivete@ics.uzh.ch                       ---')
      print('---------------------------------------------------------')
      print('---')
      print('--- Parameters:')
      print('---')
      print('---    dx               :',self.dl.x,', ',self.dl.y)
      print('---    l                :',self.l)
      print('---    S_param          :',self.S_param)
      print('---    crit             :',self.crit)
      print('---    dc_coeff         :',self.dc_coeff)
      print('---    dc_adaptive      :',self.dc_adaptive)
      print('---    fast_clustering  :',self.fast_clustering)
      print('---    xi_option        :',self.xi_option)
      print('---    clust_selector   :',self.clust_selector)
      print('---    clust_options    :',self.clust_options)
      print('---    noise_f          :',self.noise_f)
      print('---    kink_f           :',self.kink_f)
      print('---')
      print('---------------------------------------------------------')
#
#
#
###############################################
#
# METHODS
#
###########################################
#
#
  def vorticity(self):
  #
  #********************
  # Computes the vorticity criterion
  #********************
  # 
  ##  INPUTS
  #
  ##  OUTPUTS
  #
  ###############
    
    # Timing
    t_start = time.process_time()

    # Call external function
    self.W, self.U = compute_vorticity(self.v, 
                                       self.dl, 
                                       self.l
                                       )

    # Timing
    t_total = timings(t_start)
    self.timings['Vorticity'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def swirlingstrength(self):
  # 
  #********************
  # Computes the swirling strength criterion
  #********************
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    
    # Timing
    t_start = time.process_time()

    # Call external function
    self.S, self.U = compute_swirlingstrength(self.v, 
                                              self.dl, 
                                              self.l, 
                                              self.S_param
                                              )

    # Timing
    t_total = timings(t_start)
    self.timings['Swirling strength'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def rortex(self):
  # 
  #********************
  # Computes the rortex criterion
  #********************
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    
    # Timing
    t_start = time.process_time()

    # Call external function
    self.R, self.S, self.U, self.W = compute_rortex(self.S, 
                                                    self.W, 
                                                    self.v, 
                                                    self.dl, 
                                                    self.l, 
                                                    self.S_param
                                                    )

    # Timing
    t_total = timings(t_start)
    self.timings['Rortex'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def compute_criterion(self):
  #
  #********************
  # Computes the chosen criterion
  #******************** 
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    # Print status
    if self.verbose:
      print('--- Computing criterion')
      
    # Compute the criterion chosen
    if self.crit == 'vorticity':
      if isinstance(self.W[0],float):
        self.vorticity()
    elif self.crit == 'swirling strength':
      if isinstance(self.S[0],float):
        self.swirlingstrength()
    elif self.crit == 'rortex':
      if isinstance(self.R[0],float):
        self.rortex()

  ##
  ##
  ###########################################
  #
  #
  def evcmap(self):
  #
  #********************
  # Computes the EVC map
  #******************** 
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    # Print status
    if self.verbose:
      print('--- Computing EVC map')
        
    # Timing
    t_start = time.process_time()

    X = 0
    if self.crit == 'vorticity':
      X = self.W
    elif self.crit == 'swirling strength':
      X = self.S
    elif self.crit == 'rortex':
      X = self.R
    
    # Sanity check
    if isinstance(X[0], float):
      raise ValueError('Centemap: Criteria has not been computed.')

    # Call external function
    self.M, self.dataCells = compute_evcmap(X, 
                                            self.U, 
                                            self.v, 
                                            self.dl
                                            )

    # Timing
    t_total = timings(t_start)
    self.timings['EVC map'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def clustering(self):
  #
  #********************
  # Performs the clustering and groups cells in preliminary
  # vortices
  #******************** 
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    # Print status
    if self.verbose:
      print('--- Clustering')

    # Timing
    t_start = time.process_time()

    # Checking that EVC map has been computed 
    if isinstance(self.M[0], float):
      raise ValueError('Clustering: EVC map has not been computed.')

    # Prepare data for clustering algorithm
    self.data = prepare_data(self.M, self.dataCells, self.fast_clustering)

    # Call clustering algorithm
    self.cluster_id, self.peaks_ind, self.rho, self.delta, self.dc, self.d = findcluster2D( self.data, 
                                                                                            self.dc_coeff, 
                                                                                            self.dc_adaptive,
                                                                                            self.dl,
                                                                                            self.fast_clustering, 
                                                                                            self.xi_option, 
                                                                                            self.clust_selector, 
                                                                                            self.clust_options
                                                                                            ) 
    
    # Create gamma 
    self.gamma = self.rho*self.delta

    # Timing
    t_total = timings(t_start)
    self.timings['Clustering'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def detect_vortices(self):
  #
  #********************
  # Identify vortices based on the clusters found and removes
  # noisy detections
  #******************** 
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    # Print status
    if self.verbose:
      print('--- Detecting vortices')

    # Timing
    t_start = time.process_time()

    # Sanity check
    if self.cluster_id is None:
      raise ValueError('Detection: Clustering has not been done.')

    # Call detection routine for vortex identification
    if self.cluster_id.size > 0:
      self.vortices, self.noise = detection(self.dataCells, 
                                            self.M, 
                                            self.cluster_id, 
                                            self.peaks_ind, 
                                            self.fast_clustering, 
                                            self.noise_f, 
                                            self.kink_f, 
                                            self.dl
                                            )
    else:
      self.vortices = []
      self.noise = []

    # Timing
    t_total = timings(t_start)
    self.timings['Detection'] = t_total
  ##
  ##
  ###########################################
  #
  #
  def run(self):
  # 
  #********************
  # Runs the whole detection algorithm. 
  #********************
  ##  INPUTS
  #    
  ##  OUTPUTS
  #
  ###############
    # Print status
    if self.verbose:
      print('---')
      print('---------------------------------------------------------')
      print('--- Starting identification ')
      print('---------------------------------------------------------')
      print('---')

    # Timing
    t_start = time.process_time()

    # Compute the mathematical criterion
    self.compute_criterion()

    # Compute the EVC map
    self.evcmap()

    # Call clustering routine to get cluster id and peaks
    self.clustering()

    # Call detection routine to identify vortices
    self.detect_vortices()

    # Timing
    t_total = timings(t_start)
    self.timings['Total'] = t_total

    # Print timings
    if self.verbose:
      print('---')
      print('---------------------------------------------------------')
      print('--- Identification completed ')
      print('---------------------------------------------------------')
      print('---')
      print('--- Identified vortices:', len(self.vortices))
      print('---')
      print('--- Timings')
      for t in self.timings:
        print ("{:<6} {:<10} {:<3} {:<10}".format( '---   ', t, ': ', self.timings[t]))
      print('---------------------------------------------------------')
      print('\n')
