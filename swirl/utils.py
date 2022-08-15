#######################################################
###
###             utils.py
###
#######################################################
##
##  J. R. Canivete Cuissa
##  10.02.2021
##
#########################
##
##
##  This code contain some utils such as the 
##  definition of the vector2D class, timings, ...
##
#########################
#
## Imports
#
import time
import math
import numpy as np
#
#
#######################################################
# 
#
class vector2D:
#
##  CLASS
#
######################
# This class is a very simple definition of a
# 2D vector.
######################
#
#   ATTRIBUTES:
#   
#   x : numpy array
#     - the x component of the 2D vector
#   y : numpy array
#     - the y component of the 2D vector
#   
#   METHODS:
#
#
###############################################
#
  def __init__(self, vx=0.0, vy=0.0):
    #
    #********************
    # Initialization
    #********************
    #    
    ##  INPUTS
    #
    #   vx : array
    #     - the x component of the 2D vector
    #   vy : array
    #     - the y component of the 2D vector
    #
    ##  OUTPUTS
    #
    ###############

    # If vx,vy are single values
    if (isinstance(vx,(float,int)) and isinstance(vy,(float,int))):
      self.x = vx
      self.y = vy
    # If vx,vy are arrays
    elif vx.shape == vy.shape:
      self.x = vx
      self.y = vy
    else:
      raise ValueError('utils: wrong initialization of vector2D')

    self.norm = np.sqrt(self.x**2 + self.y**2)/np.sqrt(2)
##
##
################################################
#
#
def timings(ti):
#
#******************** 
# This routine computes the timings of running of a 
# function given the starting time and outputs a printable
# string  
#******************** 
#
##  INPUTS
#   - ti : float
#     - initial time
#
##  OUTPUTS
#   - t : float
#     - the printable string with the timing 
#
###############################################
  # final time
  tf = time.process_time()
  # total time
  ts = tf-ti
  # if extra units needed
  larget = False

  if ts < 10.**-6: # nanoseconds
    unit = ' ns'
    tt = ts*10.**9
  elif ts < 10.**-3: # microseconds
    unit = ' mus'
    tt = ts*10.**6
  elif ts < 1.: # milliseconds
    unit = ' ms'
    tt = ts*10.**3
  elif ts < 60.: # seconds
    unit = ' s'
    tt = ts
  elif ts < 3600: # minutes, seconds
    unit = ' min'
    tt = math.floor(ts/60.)
    larget = True
    unit2 = ' s'
    tt2 = math.floor(ts - math.floor(ts/60)*60)
  else: # hours minutes
    unit = ' h'
    tt = math.floor(ts/3600.)
    larget = True
    unit2 = ' min'
    tt2 = math.floor(ts/60. - math.floor(ts/3600.)*60.)

  if larget is False:
    tt = '%.3f' % tt
  
  t = ' '+str(tt)+unit
  if larget is True:
    t = t + ', '+'%.3f' % tt2+unit2

  return t
##
##
#######################################################
##
##
def create_U(v, dl, l):
#
#******************** 
# This routine creates the velocity gradient tensor
# needed for swirling strength and evcmap routines
#******************** 
#  
##  INPUTS
#    
#    - v : 2D vector
#       - velocity field
#    - dl : 2D vector
#       - grid spacing
#    - l : int
#       - stencil to use
#
##  OUTPUTS
#
#     - U : numpy array
#       - velocity gradient tensor
###############

  # prepare U array
  nx = v.x.shape[0]
  ny = v.x.shape[1]
  U = np.zeros((2,2,nx,ny))

  # Fill the velocity gradient tensor matrix.
  U[:,:,l:-l,l:-l] =  np.array( [ [   ( v.x[2*l:, l:-l]   - v.x[:-2*l, l:-l] )/(2*dl.x*l), 
                                      ( v.x[l:-l, 2*l:]   - v.x[l:-l, :-2*l] )/(2*dl.y*l)],

                                  [   ( v.y[2*l:, l:-l]   - v.y[:-2*l, l:-l] )/(2*dl.x*l),
                                      ( v.y[l:-l, 2*l:]   - v.y[l:-l, :-2*l] )/(2*dl.y*l)]
                                ], dtype = np.float32)

  # Reorder axis to give it to linalg.eig
  U = np.moveaxis(U, [0,1,2,3], [2,3,0,1])

  return U
##
##
#######################################################
##
## TODO
##
# def prepare_dataframe(vortices):
# #
# #******************** 
# # This routine arranges the data for the clustering 
# # algorithm.
# #******************** 
# #  
# ##  INPUTS
# #    
# #    - vortices : list of Vortex
# #       - list of vortex instances with all the info about the swirls   
# ##  OUTPUTS
# #
# #    - df: pandas DataFrame
# #       - dataframe containing 
# #           0 - center
# #           1 - radius
# #           2 - N
# #           3 - EVCs
# #           4 - crit
# #           5 - cells
# # ###############

#   # prepare data
#   data = []

#   # loop over vortices
#   for v in vortices:
#     # append to data: center x, center y, radius, N, EVC, crit, cells
#     data.append([v.center[0], v.center[1], v.r, v.N, v.EVC, v.crit, v.cells])

#   # put everything in a DataFrame:
#   df = pd.DataFrame(data, columns=['x','y','radius','N','evc','criteria','cells'])

#   return df