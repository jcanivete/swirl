"""
SWIRL Code
    swirlingstrength.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the routines needed to compute the
swirling strength criterion.
"""
# Imports
import numpy as np
from .utils import create_U
# -------------------------


def compute_swirlingstrength(v, dl, l, param):
    """
    Computes the swirling strength criterion.

    Parameters
    ----------
    v : arrays
        The velocity field [vx, vy].
    dl : list
        Grid cells sizes [dx, dy].
    l : list
        Stencil sizes.
    param : list
        List of swirling strength parameters.

    Returns
    -------
    S : list of arrays
        List of swirling strength arrays
    U : list of arrays
        List of velocity gradient tensor arrays

    Raises
    ------
    """
    # To avoid annoying numpy warnings.
    np.seterr(invalid='ignore')
    # Prepare the S,U instances
    S = []
    U = []
    nx = v.x.shape[0]
    ny = v.x.shape[1]

    # Prepare param
    eps = param[0]
    delta = param[1]
    kappa = param[2]

    # loop over stencils
    for il in l:
        # create instances of Ui, Si
        Si = np.zeros((nx, ny))
        Ui = np.zeros((nx, ny))

        # fill velocity gradient tensor:
        Ui = create_U(v, dl, il)

        # Compute eigen vectors and eigenvalues
        # wi -> eigenvalues
        # vi -> eigevectors
        wi, vi = np.linalg.eig(Ui)

        # index:
        # 0 -> im(wi) < 0
        # 1 -> im(wi) > 1
        w_index = np.argsort(np.imag(wi))

        i, j = np.meshgrid(np.arange(np.shape(vi)[0]),
                           np.arange(np.shape(vi)[1]),
                           indexing='ij'
                           )

        # Reorder eigenvectors to form P matrix
        # P = [u+, u-]
        P = 0*vi*1j

        # u+
        P[:, :, 0, 0] = vi[i, j, 0, w_index[i, j, 1]]
        P[:, :, 1, 0] = vi[i, j, 1, w_index[i, j, 1]]
        # u-
        P[:, :, 0, 1] = vi[i, j, 0, w_index[i, j, 0]]
        P[:, :, 1, 1] = vi[i, j, 1, w_index[i, j, 0]]

        # Determinant of P needed to determine orientation
        # of the swirl (see Canivete&Steiner,2020)
        detP = np.linalg.det(P)
        detP = np.sign(np.imag(detP))

        # imaginary and real components of the eigenvalues
        # lci (lambda_ci) -> imaginary part (lci > 0)
        # lcr (lambda_ci) -> real part
        lci = np.imag(wi[i, j, w_index[i, j, 1]])
        lcr = np.real(wi[i, j, w_index[i, j, 1]])

        # Enhanced swirling strength criteria:
        # eps
        if eps != 0.0:
            lci = np.where((2.*np.abs(lci) <= eps), lci, 0.0)

        # delta
        if delta != 0:
            with np.errstate(divide='ignore'):
                crit = np.where(np.abs(lci) == 0.0, 0.0, lcr/np.abs(lci))
                lci = np.where(crit >= delta, 0.0, lci)

        # kappa
        if kappa != 0.:
            with np.errstate(divide='ignore'):
                crit = np.where(np.abs(lci) == 0.0, 0.0, lcr/np.abs(lci))
                lci = np.where(crit < -(kappa), 0.0, lci)

        # Swirling strength = 2*[sign(Im[det(P)])]*lambda_ci
        Si = 2.*detP*lci

        # Clean velocity gradient tensors
        mask = (Si == 0.0)
        Ui[mask] = np.array([[1., 0.], [0., 1.]])

        # Append Si, Ui
        S.append(Si)
        U.append(Ui)

    return S, U
