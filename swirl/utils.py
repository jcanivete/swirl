"""
SWIRL Code
    utils.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains different functions that are useful
in the SWIRL code.
"""
# Imports
from email.policy import default
import time
import math
import warnings
import numpy as np
import configparser
import ast
import warnings
# ----------------


class vector2D:
    """
    A simple 2D vector class.

    Input Attributes
    ----------------
    vx : array, float, int
        The x component of the vector.
    vy : array, float, int
        The y component of the vector.

    Derived Attributes
    ------------------
    norm : array, float
        The norm of the vector.

    """

    def __init__(self, vx=0.0, vy=0.0):

        # If vx,vy are single values
        if (isinstance(vx, (float, int)) and isinstance(vy, (float, int))):
            self.x = vx
            self.y = vy
        # If vx,vy are arrays
        elif vx.shape == vy.shape:
            self.x = vx
            self.y = vy
        else:
            raise ValueError('Wrong initialization of vector2D.')

        self.norm = np.sqrt(self.x**2 + self.y**2)
    # --------------------------------------------


def timings(ti):
    """
    A simple routine that returns the time interval in a formatted string.

    Parameters
    ----------
    ti : float
        Initial time

    Returns
    -------
    t : string
        A printable string with the time interval

    Raises
    ------
    """
    # final time
    tf = time.process_time()
    # total time
    ts = tf-ti
    # if extra units needed
    larget = False

    if ts < 10.**-6:  # nanoseconds
        unit = ' ns'
        tt = ts*10.**9
    elif ts < 10.**-3:  # microseconds
        unit = ' mus'
        tt = ts*10.**6
    elif ts < 1.:  # milliseconds
        unit = ' ms'
        tt = ts*10.**3
    elif ts < 60.:  # seconds
        unit = ' s'
        tt = ts
    elif ts < 3600:  # minutes, seconds
        unit = ' min'
        tt = math.floor(ts/60.)
        larget = True
        unit2 = ' s'
        tt2 = math.floor(ts - math.floor(ts/60)*60)
    else:  # hours minutes
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
    # ------


def create_U(v, dl, l):
    """
    This routine computes the velocity gradient tensor given
    a velocity field, the size of the grid cells and the stencil
    to be used.

    Parameters
    ----------
    v : 2D vector
        Velocity field
    dl : 2D vector
        Grid cell size
    l : int
        Stencil to use

    Returns
    -------
    U : array
        The array containing the velocity gradient tensors.

    Raises
    ------
    """
    # prepare U array
    nx = v.x.shape[0]
    ny = v.x.shape[1]
    U = np.zeros((2, 2, nx, ny))

    # Fill the velocity gradient tensor matrix.
    U[:, :, l:-l, l:-l] = np.array([[(v.x[2*l:, l:-l] - v.x[:-2*l, l:-l])/(2*dl.x*l),
                                     (v.x[l:-l, 2*l:] - v.x[l:-l, :-2*l])/(2*dl.y*l)],

                                    [(v.y[2*l:, l:-l] - v.y[:-2*l, l:-l])/(2*dl.x*l),
                                     (v.y[l:-l, 2*l:] - v.y[l:-l, :-2*l])/(2*dl.y*l)]
                                    ], dtype=np.float32)

    # Reorder axis to give it to linalg.eig
    U = np.moveaxis(U, [0, 1, 2, 3], [2, 3, 0, 1])

    return U
# ----------


def read_params(param_file = ''):
    """
    This routine reads a parameter file with the configparams module
    and returns a dictionary with all the parameters needed for the 
    SWIRL Identification class. If no parameter file is given, then it returns the 
    default values. 
    
    Parameters
    ----------
    param_file - string
        The file containing the parameters. The structure is given.
    
    Returns
    -------
    param_dict - dictionary
        A dictionary containing all the parameters.

    Raises
    ------
    ValueError
        If parameters do not met the requirements.
    """
    # Set default parameters
    params = dict()
    # Read config file
    config = configparser.ConfigParser()
    config.read(param_file)
    # Sanity check if the file is empty
    if len(config.sections())==0 and param_file != '':
        warnings.warn('The parameter file is empty or it does not exist. The default values will be used.')
    # Read params and use the default ones in case they 
    # are not specified
    params['stencils']=ast.literal_eval(config.get('Criterion',
                                                   'stencils',
                                                   fallback='[1]'))
    params['swirlstr_params']=ast.literal_eval(config.get('Criterion',
                                                          'swirlstr_params',
                                                          fallback='[0., 0., 0.]'))
    params['dc_param']=config.getfloat('Clustering',
                                       'dc_param',
                                       fallback=3.)
    params['dc_adaptive']=config.getboolean('Clustering',
                                            'dc_adaptive',
                                            fallback=True)
    params['cluster_fast']=config.getboolean('Clustering',
                                             'cluster_fast',
                                             fallback=True)
    params['cluster_kernel']=config.get('Clustering',
                                        'cluster_kernel',
                                        fallback='Gaussian')
    params['cluster_decision']=config.get('Clustering',
                                          'cluster_decision',
                                          fallback='delta-rho')
    params['cluster_params']=ast.literal_eval(config.get('Clustering',
                                                         'cluster_params',
                                                         fallback='[1.0, 0.5, 2.0]'))
    params['noise_param']=config.getfloat('Noise',
                                           'noise_param',
                                           fallback=1.0)
    params['kink_param']=config.getfloat('Noise',
                                         'kink_param',
                                         fallback=1.0)

    # Make checks
    if len(params['stencils']) < 1:
        raise ValueError("The 'stencils' parameters must have at least one entry.")
    if len(params['swirlstr_params']) != 3:
        raise ValueError("The 'swirlistr_params' must have exactly three entries: [eps, delta, kappa].")
    if len(params['cluster_params']) != 3:
        raise ValueError("The 'cluster_params' must have exactly three entries: [rho_p, delta_p, gamma_p].")
    if params['cluster_kernel'] not in ['Gaussian', 'Heaviside']:
        raise ValueError("The 'cluster_kernel' parameter must be either 'Gaussian' (default) or 'Heaviside'.")
    if params['cluster_decision'] not in ['delta-rho', 'gamma']:
        raise ValueError("The 'cluster_decision' parameter must be either 'delta-rho' (default) or 'gamma'.")
    return params
