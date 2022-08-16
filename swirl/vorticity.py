"""
SWIRL Code
    vorticity.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the routines needed to compute the
vorticity criterion.
"""
# Imports
import numpy as np
from .utils import create_U
# -------------------------


def compute_vorticity(v, dl, l):
    """
    Computes the vorticity criterion.

    Parameters
    ----------
    v : arrays
        The velocity field [vx, vy].
    dl : list
        Grid cells sizes [dx, dy].
    l : list
        Stencil sizes.

    Returns
    -------
    W : list of arrays
        List of vorticity arrays
    U : list of arrays
        List of velocity gradient tensor arrays

    Raises
    ------
    """
    # Prepare the W,U instance
    W = []
    U = []
    nx = v.x.shape[0]
    ny = v.x.shape[1]

    # loop over stencils
    for il in l:
        # initialize arrays
        Wi = np.zeros((nx, ny))
        Ui = np.zeros((nx, ny))

        # fill velocity gradient tensor:
        Ui = create_U(v, dl, il)

        # compute vorticity
        Wi[il:-il, il:-il] = ( (v.y[2*il:, il:-il] - v.y[:-2*il, il:-il])/(2.*dl.x*il)
                              -(v.x[il:-il, 2*il:] - v.x[il:-il, :-2*il])/(2.*dl.y*il))

        # Clean velocity gradient tensors
        mask = (Wi == 0.0)
        Ui[mask] = np.array([[1., 0.], [0., 1.]])

        # add to W,U
        W.append(Wi)
        U.append(Ui)

    return W, U
