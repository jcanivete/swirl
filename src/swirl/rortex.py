"""
SWIRL Code
    rortex.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the routines needed to compute the
rortex criterion.
"""
# Imports
import numpy as np
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .utils import create_U
# ----------------


def compute_rortex(S, W, v, dl, l, param):
    """
    Computes the rortex criterion given the swirling strength and
    the vorticity.

    Parameters
    ----------
    S : list of arrays
        List of swirling strength arrays
    W : list of arrays
        List of vorticity arrays

    Returns
    -------
    R : list of arrays
        List of rortex arrays

    Raises
    ------
    """

    # Check if need to compute vorticity:
    if W is None:
        W, _ = compute_vorticity(v, dl, l)

    # Check if need to compute swirling strength:
    if S is None:
        S, _ = compute_swirlingstrength(v, dl, l, param)

    # prepare R,U
    R = []
    U = []
    nx = v.x.shape[0]
    ny = v.x.shape[1]

    # loop over stencils:
    for i in np.arange(len(l)):

        # Prepare velocity gradient tensor instance
        Ui = np.zeros((nx, ny))

        # fill velocity gradient tensor:
        Ui = create_U(v, dl, l[i])

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
        mask = (Ri == 0.0)
        Ui[mask] = np.array([[1., 0.], [0., 1.]])

        # append result
        R.append(Ri)
        U.append(Ui)

    return R, S, U, W
