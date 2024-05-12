"""
SWIRL Code
    cluster.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

Contains the implementation of the grid and vortex adapted
clustering by fast search and find of density peaks (CFSFDP)
algorithm (Rodriguez & Laio, 2014).

Reference: Alex Rodriguez, Alessandro Laio,
           "Clustering by fast search and find of density peaks", 2014,
           Science 344 (6191), 1492-1496.
"""
# Imports
import numpy as np
from scipy import spatial
# -----------------------


def findcluster2D(gevc_map,
                  dc_coeff,
                  dc_adaptive,
                  dx_grid,
                  fast_clustering,
                  xi_option,
                  clust_selector,
                  clust_options
                  ):
    """
    This routine uses a grid and vortex adapted version of the CFSFDP
    algorithm to cluster EVC points and group them into clusters and noise.
    For each cluster, the cluster center is provided which is then used as
    the center of the vortex by the main algorithm.

    Parameters
    ----------
    gevc_map : array
        Coordinates of the (G-)EVC points.
        If fast_clustering is True, a third column with the cardinality
        of each point.
    dc_coeff : float
        If dc_adaptive=True: percentual coefficient used to compute the
        critical distance. The critical distance is defined as the
        average distance such that dc_coeff % of points inside dc.
        Preferably, use 2-3.
        If dc_adaptive=False: Size of dc in physical units.
    dc_adaptive : bool
        Option: to use the adaptive critical distance (dc) computed on the
        % of points inside dc (True) or to use a fixed distance defined
        by dc_coeff*dx_grid.
    dx_grid : float
        Size of the grid cells to translate the dc_coeff in case of
        dc_adaptive=False to number of cells.
    fast_clustering : bool
        Option to use the grid adapted version of the clustering algorithm.
    xi_option : str
        Kernel used to compute densities.
        'Heaviside' : Heaviside function.
        'Gaussian': Gaussian kernel.
    clust_selector : string
        Cluster centers selection process:
        'delta-rho' or 'gamma'.
    clust_options : list
        List of parameters for cluster centers selection.

    Returns
    -------
    cluster_id : array
        Cluster indeces for all EVC points given as input.
        0 means that the point does not belong to any cluster and is
        therefore noise.
    peaks : array
        Indices of the peaks, i.e. the centers of the clusters.
        The position i in the array corresponds to the i+1 index
        of cluster.
    rho : array
        Array with the computed densities for each point.
    delta : array
        Array with the computed deltas for each point.
    dc : float
        Effective critical distance parameter used in the
        clustering algorithm.
    d : 2D array
        Matrix containing all distances between EVC points

    Raises
    ------
    """
    # check size of gevc_map:
    N = gevc_map.shape[0]

    if N > 1:
        # Compute density array rho
        rho, d, dc = compute_rho(gevc_map,
                                 dc_coeff,
                                 dc_adaptive,
                                 dx_grid,
                                 fast_clustering,
                                 xi_option
                                 )
        # Compute delta array
        delta = compute_delta(rho, d)
        if len(rho) > 0:
            # Cluster points given rho and delta
            clusters, peaks, rho, delta = clustering(d,
                                                     rho,
                                                     delta,
                                                     gevc_map,
                                                     fast_clustering,
                                                     clust_selector,
                                                     clust_options
                                                     )
        else:
            clusters = np.array([])
            peaks = np.array([])
            rho = np.array([1.])
            delta = np.array([1.])
    else:
        clusters = np.array([1.])
        peaks = np.array([1.])
        rho = np.array([1.])
        delta = np.array([1.])
        d = np.array([[1.]])

    return clusters, peaks, rho, delta, dc, d
# -------------------------------------------


def compute_rho(gevc_map, dc_coeff, dc_adaptive, dx_grid, fast_clustering, xi_option):
    """
    Computes the density coefficient rho used in the CFSFDP algorithm.

    Parameters
    ----------
    gevc_map : array
        Coordinates of the (G-)EVC points.
        If fast_clustering is True, a third column with the cardinality
        of each point.
    dc_coeff : float
        If dc_adaptive=True: percentual coefficient used to compute the
        critical distance. The critical distance is defined as the
        average distance such that dc_coeff % of points inside dc.
        Preferably, use 2-3.
        If dc_adaptive=False: Size of dc in physical units.
    dc_adaptive : bool
        Option: to use the adaptive critical distance (dc) computed on the
        % of points inside dc (True) or to use a fixed distance defined
        by dc_coeff*dx_grid.
    dx_grid : float
        Size of the grid cells to translate the dc_coeff in case of
        dc_adaptive=False to number of cells.
    fast_clustering : bool
        Option to use the grid adapted version of the clustering algorithm.
    xi_option : str
        Kernel used to compute densities.
        'Heaviside' : Heaviside function.
        'Gaussian': Gaussian kernel.

    Returns
    -------
    rho : array
        Array with the computed densities for each point.
    d : 2D array
        Matrix containing all distances between EVC points
    dc : float
        Effective critical distance parameter used in the
        clustering algorithm.

    Raises
    ------
    """
    # compute distance between points:
    d = spatial.distance.cdist(gevc_map[:2, :].T, gevc_map[:2, :].T)
    # Adjust critical distance (dc)
    dc = compute_dc(d, dc_coeff, dc_adaptive, dx_grid)
    # Divide gevc_map in 2: positive and negative cardinality
    p, = np.where(gevc_map[2] > 0.0)
    n, = np.where(gevc_map[2] < 0.0)
    Q = [q for q in [n, p] if q.shape[0] > 0]

    # Prepare arrays
    rho = []
    d = []
    # Do first positive cardinality gevc_map and then negative
    for q in Q:
        # compute distance between points
        dq = spatial.distance.cdist(gevc_map[:2, q].T, gevc_map[:2, q].T)
        # compute Xi matrix
        if xi_option == 'Heaviside':
            xij = np.where(dq < dc, 1.0, 0.0)
        elif xi_option == 'Gaussian':
            xij = np.exp(-(dq/dc)**2)

        # If fast_clustering is true, apply modified formula (see paper):
        if fast_clustering:
            s = np.abs(gevc_map[2, q]).T
            rhoq = np.einsum('j,ij', s, xij)
        else:
            # rho is the sum of rows - 1 given by distance with itself
            rhoq = np.sum(xij, axis=0) - 1.0

        # append rho and d
        rho.append(rhoq)
        d.append(dq)
    return rho, d, dc
# -------------------


def compute_dc(d, dc_coeff, dc_adaptive, dx_grid):
    """
    Computes the critical distance (dc) used in the kernel for the
    computation of rho. The idea is that only dc_coeff % of all
    points should be considered "neighbors" for each point.
    dc is set using this request.

    Parameters
    ----------
    d : 2D array
        Matrix containing all distances between EVC points
    dc_coeff : float
        If dc_adaptive=True: percentual coefficient used to compute the
        critical distance. The critical distance is defined as the
        average distance such that dc_coeff % of points inside dc.
        Preferably, use 2-3.
        If dc_adaptive=False: Size of dc in physical units.
    dc_adaptive : bool
        Option: to use the adaptive critical distance (dc) computed on the
        % of points inside dc (True) or to use a fixed distance defined
        by dc_coeff*dx_grid.
    dx_grid : float
        Size of the grid cells to translate the dc_coeff in case of
        dc_adaptive=False to number of cells.

    Returns
    -------
    dc : float
        Effective critical distance parameter used in the
        clustering algorithm.

    Raises
    ------
    """
    # Size of arrays
    n = d.shape[0]
    # If dc_adaptive is True, compute it according to % value
    if dc_adaptive is True:
        if n <= 1:
            return 1.0
        else:
            nc = int(n*dc_coeff/100)
            if nc == 0:
                nc = 1

            # Reorder d from smaller to higher in each row
            dord = np.sort(d, axis=1)
            # dc for each row is at nc index, average over them
            dc = np.mean(dord[:, nc])
    else:
        # The critical distance corresponds to the dc_coeff
        dc = dc_coeff

    return dc
# -----------


def compute_delta(rho, d):
    """
    This routine computes the delta coefficient for the CFSFDP algorithm.
    Delta is defined as the minimal distance from a point with an
    higher density.

    Parameters
    ----------
    rho : array
        Array with the computed densities for each point.
    d : 2D array
        Matrix containing all distances between EVC points

    Returns
    -------
    delta : array
        Array with the computed deltas for each point.

    Raises
    ------
    """
    # prepare array
    delta = []

    # Compute maximal distance in both arrays
    dmax = [np.max(di) for di in d]
    if len(dmax) > 0:
        dmax = np.max(dmax)
    else:
        dmax = 0.0

    for rhoq, dq in zip(rho, d):
        # create matrix of difference of rho, rhoij
        one = np.ones(rhoq.shape[0])
        rhoij = np.outer(rhoq, one) - np.outer(one, rhoq)

        # deltaij is d with dmax where rhoij is positive, i.e. where
        # rho_i is larger than rho_j
        if dmax == 0.0:
            dmax = 1.0
        deltaij = np.where(rhoij < 0.0, dq, dmax)

        # delta is given by min of deltaij
        deltaq = np.min(deltaij, axis=1)

        # Set to dmax delta of highest rho
        i_rhomax = np.where(rhoq == np.max(rhoq))[0][0]
        deltaq[i_rhomax] = dmax

        # append
        delta.append(deltaq)

    return delta
# --------------


def clustering(d,
               rho,
               delta,
               gevc_map,
               fast_clustering,
               clust_selector,
               clust_options
               ):
    """
    This routine clusters the gevc_map points according to the CFSFDP algorithm.
    If the fast_clustering is activated (True), it takes into account the
    sign of the criterium. It returns an array with the index of the cluster
    to which each point belongs. If 0 the gevc_map point is considered noise.
    The peak of each cluster is given in the array peaks.

    Parameters
    ----------
    d : 2D array
        Matrix containing all distances between EVC points
    rho : array
        Array with the computed densities for each point.
    delta : array
        Array with the computed deltas for each point.
    gevc_map : array
        Coordinates of the (G-)EVC points.
        If fast_clustering is True, a third column with the cardinality
        of each point.
    fast_clustering : bool
        Option to use the grid adapted version of the clustering algorithm.
    clust_selector : string
        Cluster centers selection process:
        'delta-rho' or 'gamma'.
    clust_options : list
        List of parameters for cluster centers selection.

    Returns
    -------
    cluster_id : array
        Cluster indeces for all EVC points given as input.
        0 means that the point does not belong to any cluster and
        is therefore noise.
    peaks : array
        Indices of the peaks, i.e. the centers of the clusters.
        The position i in the array corresponds to the i+1 index of cluster.
    rho_fc : array
        updated version of rho criterion.
    delta_fc : array
        updated version of delta criterion.

    Raises
    ------
    """
    # Form rho_tot and delta_tot
    if len(rho) > 0:
        rho_tot = np.hstack(rho)
        delta_tot = np.hstack(delta)
    else:
        rho_tot = np.array([])
        delta_tot = np.array([])

    # get total number of points
    N = rho_tot.shape[0]

    delta_fc = np.copy(delta_tot)
    rho_fc = np.copy(rho_tot)
    # If fast_clustering, add dummy points to delta and rho
    if fast_clustering:
        s = np.abs(gevc_map[2])
        stot = np.sum(s)

        # Prepare delta
        delta_fc = np.concatenate((delta_tot, np.ones((int(stot-N)))/2.))

        # Prepare rho
        for i in np.arange(N):
            r = np.ones((int(s[i]-1)))*rho_tot[i]
            rho_fc = np.concatenate((rho_fc, r))

    # Find peaks
    peaks = rho_tot*0
    gamma = rho_tot*delta_tot
    delta_opt = clust_options[0]
    rho_opt = clust_options[1]

    if clust_selector == 'delta-rho':
        peaks = np.where(rho_tot >= rho_opt*np.mean(rho_fc), 1, 0)
        peaks = peaks*np.where(delta_tot >= delta_opt*np.std(delta_fc), 1, 0)
    elif clust_selector == 'gamma':
        gamma_opt = clust_options[2]
        # New
        # Idea: threshold should be slighlty larger than delta_min at rho_max
        # in the gamma approach
        delta_min = np.min(delta_tot)
        rho_max = np.max(rho_tot)
        gamma_param = 2.*gamma_opt*delta_min*rho_max
        peaks = np.where(gamma >= gamma_param, 1, 0)

    # number of peaks (clusters)
    Npeaks = np.sum(peaks)

    # Remove peaks with low gamma:
    peaks = np.argsort(peaks)[::-1]
    peaks = peaks[:Npeaks]

    # Compute maximal distance in both arrays
    dmax = [np.max(di) for di in d]
    dmax = np.max(dmax)

    # k index for cluster id
    k = 1

    # adjust peak indices
    Nq = rho[0].shape[0]

    peaks_n = np.array([p for p in peaks if p < Nq])
    peaks_p = np.array([p-Nq for p in peaks if p >= Nq])
    peaks_np = [peaks_n, peaks_p]

    # prepare array
    cluster_id = []

    for rhoq, dq, peaksq in zip(rho, d, peaks_np):

        # find ordering of rho from large to small
        ordrho = np.argsort(rhoq)[::-1]

        # Reorder d in such a way that highest rho is in
        # row and column 0
        d_ord = dq[ordrho, :]
        d_ord = d_ord[:, ordrho]

        # add max(d) to triangular up part of d_ord to
        # "eliminate" these values from ordering after
        tr = np.tril(d_ord)
        tr = np.where(tr == 0.0, dmax, 0.0)
        d_ord = d_ord + tr

        # Neighbour with higher value of rho is given
        # by the index of the smaller value per row.
        neigh = np.argsort(d_ord, axis=1)[:, 0]

        # Set cluster index for peaks:
        Nq = rhoq.shape[0]
        cluster_idq = np.zeros(Nq, dtype=int)

        for i in peaksq:
            cluster_idq[i] = k
            # Collect indices of peaks too
            k = k + 1

        # Find indices from real to rho ordening
        ordreal = np.argsort(ordrho)

        # Add points to cluster
        for i in ordrho:
            if i not in peaksq:
                # index of neighbor with higher rho
                j = ordrho[neigh[ordreal[i]]]
                cluster_idq[i] = cluster_idq[j]

        # append
        cluster_id.append(cluster_idq)
    # Transform into numpy array
    cluster_id = np.hstack(cluster_id)

    return cluster_id, peaks, rho_fc, delta_fc
# --------------------------------------------


def prepare_data(gevc_map, data_cells, fast_clustering):
    """
    This routine arranges the data for the clustering algorithm.

    Parameters
    ----------
    gevc_map : list of arrays
        arrays containing the G-EVC map
    data_cells : list of arrays
        arrays containing all the info about EVCs
    fast_clustering: bool
        Option to use M instead of data_cells for clustering

    Returns
    -------
    data : array
        Coordinates of the (G-)EVC points.
        If fast_clustering is True, a third column with the cardinality
        of each point.

    Raises
    ------
    """
    # If fast_clustering is True: use M
    if fast_clustering:
        # Achtung! ncols of data = 3!
        data = gevc_map

    # Else use EVC precise coordinates from data_cells
    else:
        # Achtung! ncols of data = 2!
        data = data_cells[:2]

    return data
