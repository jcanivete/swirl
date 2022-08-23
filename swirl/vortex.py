"""
SWIRL Code
    vortex.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the Vortex class, which is the object
describing every single vortex identified by the code.
"""
# Imports
import warnings
import numpy as np
# -------------------------


class Vortex:
    """
    This class is contains the data and the information
    of each vortex detected by the algorithm.

    Input Attributes
    ----------------
    cluster_center : list
        coordinates of the cluster center [x,y]
    cells : array
        coordinates of cells belonging to
        vortex
    evc : array
        array with EVC's coordinates for each cell
    ecr : array
        array with curvature radius of each cell
    rortex : array
        vortex detection criteria values for each cell
    stencils : array
        array of stencils used per cell
    cluster_id : int
        cluster identification number
    dl : vector of float
        the grid spacing [dx,dy]


    Derived Attributes
    ------------------
    vortex_cells : list
        List of unique cells forming the vortex
        (i.e. without doubles because of different stencils computations).
    n_all_cells : int
        Total number of cells forming the vortex.
    n_vortex_cells : int
        Number of unique cells forming the vortex.
    radius : float
        Effective radius of the vortex.
    center : list
        Coordinates of center of vortex.
    orientation : float
        Orientation of the swirl. +1 means anti-clockwise, -1 clockwise.


    Methods
    -------
    update_vortex_cells(self)
        Finds vortex cells (i.e. unique cells).
    update_n_vortex_cells(self)
        Computes number of vortex cells.
    update_n_all_cells(self)
        Computes number of all cells.
    update_radius(self)
        Computes the effective radius of the vortex in grid units.
    update_center(self)
        Computes the center coordinates of the vortex in grid units.
    detect_noise(self, noise_f)
        Identifies noisy cells in the vortex based on the noise_f parameter.
    detect_kinks(self, kink_f)
        Identifies kinks in the vortex based on the kink_f parameter.
    """

    def __init__(self,
                 cluster_center,
                 cells,
                 evc,
                 ecr,
                 rortex,
                 stencils,
                 cluster_id,
                 dl
                 ):
        """
        Class initalization

        Parameters
        ----------
        cluster_center : list
            coordinates of the cluster center [x,y]
        cells : array
            coordinates of cells belonging to vortex
        evc : array
            array with EVC's coordinates for all cells
        ecr : array
            array with curvature radius of all cells
        rortex : array
            Rortex values for all cells
        stencils : array
            array of stencils used per cell
        cluster_id : int
            cluster identification number
        dl : vector of float
            the grid spacing [dx,dy]

        Returns
        -------

        Raises
        ------
        """
        # Initialize attributes
        self.cluster_center = cluster_center
        self.all_cells = cells
        self.evc = evc
        self.rortex = rortex
        self.stencils = stencils
        # Private attributes
        self._ecr = ecr # To remove
        self._cluster_id = cluster_id # To remove
        self._dl = dl # To remove
        # Derived attributes
        self.update_n_all_cells()
        self.update_vortex_cells()
        self.update_radius()
        self.center = self.cluster_center
        # initialize orientation
        # + 1 : anti-clockwise
        # - 1 : clockwise
        self.orientation = np.sign(np.mean(self.rortex))
    # --------------------------------------------------


    def __str__(self):
        """
        Magic method for printing the class as a string.

        Parameters
        ----------

        Returns
        -------
        
        Raises
        ------
        """
        if self.orientation == 1.0:
            text = 'Counter-clockwise vortex,\n'
        elif self.orientation == -1.0:
            text = 'Clockwise vortex,\n'
        text += '---        Center : '+f'{self.center[0]:.2f}'+', '+f'{self.center[0]:.2f}'+',\n'
        text += '---        Radius : '+f'{self.radius:.2f}'+';'
        return text
    # -------------


    def __len__(self):
        """
        Magic method for returning the length of the object, which we define as the
        number of unique cells, i.e. _n_vortex_cells

        Parameters
        ----------

        Returns
        self.n_vortex_cells : int
            number of unique vortex cells

        Raises
        ------
        """
        return self.n_vortex_cells
    # ----------------------------
    

    def update_n_all_cells(self):
        """
        Updates the total number of cells.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        self.n_all_cells = self.all_cells.shape[1]
    # --------------------------------------------


    def update_vortex_cells(self):
        """
        Updates the vortex cells.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        self.vortex_cells = np.array([[0, 0]])
        if self.all_cells.shape[1] > 0:
            # Remove duplicates
            self.vortex_cells = np.unique(self.all_cells.T, axis=0).T
        # Update number of vortex cells
        self.update_n_vortex_cells()
    # ------------------------------


    def update_n_vortex_cells(self):
        """
        Updates the number of vortex cells.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        self.n_vortex_cells = self.vortex_cells.shape[1]
    # --------------------------------------------------


    def update_radius(self):
        """
        Updates the effective radius of the vortex in grid units according to the
        formula:
            r_eff = sqrt(n_vortex_cells/pi),
        where n_vortex_cells in the number of cells forming the vortex.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        self.radius = np.sqrt(self.n_vortex_cells/np.pi)
    # --------------------------------------------------


    def update_center(self):
        """
        Updates the center of the vortex according to all EVCs. To do that, 
        a weighted mean is used to find the center accordint to G-EVCs
        cardinality value.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        # evc coordinates
        evcx = np.array(self.evc[0], dtype=int)
        evcy = np.array(self.evc[1], dtype=int)
        if evcx.shape[0] > 0:
            # Bins
            bx = np.max(evcx)-np.min(evcx)
            by = np.max(evcy)-np.min(evcy)
            if bx < 1:
                bx = 1
            if by < 1:
                by = 1
            # Weighted average method
            Hx = np.histogram(evcx, bins=bx)
            Hy = np.histogram(evcy, bins=by)
            xc = np.sum((Hx[1][1:]+Hx[1][:-1])/2.*Hx[0])/np.sum(Hx[0])
            yc = np.sum((Hy[1][1:]+Hy[1][:-1])/2.*Hy[0])/np.sum(Hy[0])
            self.center = [xc, yc]
        else:
            self.center = [0, 0]
    # --------------------------


    def detect_noise(self, noise_f):
        """
        Removes cells labeled as noise from the vortex.

        Idea: find where 1. the distance between EVC and
        the center of the vortex is larger that the
        estimated radius of the vortex or 2. where the
        cell distance from the center of the vortex is
        larger than the estimated radius of the vortex.

        Parameters
        ----------
        noise_f : float
            factor stiffening or relaxing conditions 1 & 2.

        Returns
        -------
        noise : array
            Array structure of noise cells.

        Raises
        ------
        """
        # Initialize estimated center radius and real radius
        r = self.radius

        # load necessary quantities
        evc_x = self.evc[0]
        evc_y = self.evc[1]
        cell_x = self.all_cells[0]
        cell_y = self.all_cells[1]
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
        mask = np.concatenate((mask1, mask2))
        _, i = np.unique(mask, return_index=True)
        mask = mask[i]

        # Create noise array of the type dataCells
        noise = np.zeros((7, mask.shape[0]))
        noise[0] = self.evc[0][mask]   # EVC coord x
        noise[1] = self.evc[1][mask]   # EVC coord y
        noise[2] = self.all_cells[0][mask]  # Cells coord x
        noise[3] = self.all_cells[1][mask]  # Cells coord y
        noise[4] = self.rortex[mask]        # Criteria
        noise[5] = self._ecr[mask]      # Estimated radius
        noise[6] = self.stencils[mask]  # Stencils indices

        # Remove these points from the vortex data points
        # Cells
        cells_tmp0 = np.delete(self.all_cells[0], mask)
        cells_tmp1 = np.delete(self.all_cells[1], mask)
        self.all_cells = np.array([cells_tmp0, cells_tmp1])
        # EVC
        evc_tmp0 = np.delete(self.evc[0], mask)
        evc_tmp1 = np.delete(self.evc[1], mask)
        self.evc = np.array([evc_tmp0, evc_tmp1])
        # Criteria
        self.rortex = np.delete(self.rortex, mask)
        # Estimated radius of curvature
        self._ecr = np.delete(self._ecr, mask)
        # Stencils
        self.stencils = np.delete(self.stencils, mask)

        # Update number of cells
        self.update_n_all_cells()
        self.update_n_vortex_cells()

        # Return noise
        return noise
    # --------------

    def detect_kinks(self, kink_f):
        """
        Removes vortices labeled as kinks.

        Idea: if center of vortex is too distant from center
        of vortex (according to cells), then it's probably a kink.
        Too distant = kink_f*r

        Parameters
        ----------
        kink_f : float
            Factor used to identify kinks.

        Returns
        -------
        kinks : array
            Array of datacells being labeled as kinks.
        empty : bool
            Bool saying if vortex is now empty (i.e it was a kink).

        Raises
        ------
        """

        # load necessary quantities
        cx = self.center[0]
        cy = self.center[1]
        r = self.radius

        # Compute center of vortex according to cells positions
        # (average position)
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cx_2 = np.mean(self.all_cells[0])
            cy_2 = np.mean(self.all_cells[1])

        # Compute distance between two centers
        d = np.sqrt((cx-cx_2)**2 + (cy-cy_2)**2)

        # Initialize kinks array
        kinks = np.array([])

        # empty bool to avoid updating vortex if it is empty
        empty = False
        # If d is larger than r, then it is a kink
        if d > kink_f*r:
            # Create kinks array of the type dataCells
            kinks = np.zeros((7, self.n_all_cells))
            kinks[0] = self.evc[0]   # EVC coord x
            kinks[1] = self.evc[1]   # EVC coord y
            kinks[2] = self.all_cells[0]  # Cells coord x
            kinks[3] = self.all_cells[1]  # Cells coord y
            kinks[4] = self.rortex        # Criteria
            kinks[5] = self._ecr      # Estimated radius
            kinks[6] = self.stencils  # Stencils indices

            # Update vortex structure
            self.evc = np.array([[0, 0]])
            self.all_cells = np.array([[0, 0]])
            self.n_vortex_cells = np.array([[0, 0]])
            self.rortex = 0
            self._ecr = 0
            self.stencils = [0]

            # Empty to true because vortex if empty
            empty = True
        # Return noise
        return kinks, empty
    # ---------------------
# -------------------------


def detection(dataCells,
              M,
              cluster_id,
              peaks_ind,
              fast_clustering,
              noise_f,
              kink_f,
              dl):
    """
    Vortex detection routine based on the clustering results.

    Parameters
    ----------
    dataCells : list of arrays
        Contains the information about cells with criterion satisfied
    M : array
        G-EVC map for fast_clustering.
    cluster_id : array
        Contains the clustering id for the centers in data.
    peaks_ind : array
        Contains the index of centers in data.
    fast_clustering : Bool
        Option to use the fast_clustering method.
    noise_f : float
        Criteria to remove noise from vortices.
    kink_f : float
        Criteria to remove kinks from vortices.
    dl : vector of float
        The grid spacing [dx,dy].

    Returns
    -------
    vortices : list
        List of vortex structures with all the information
        about identified vortices in the data.
    noise : list
        List of noisy cells.

    Raises
    ------
    """
    # list of vortices
    vortices = []
    # noise collection
    noise = []

    # Loop over cluster peaks identified
    for index in peaks_ind:

        # Identification number of cluster
        idx = cluster_id[index]

        # Find coordinates of peaks
        xc = 0
        yc = 0
        if fast_clustering:
            xc = M[0, index]
            yc = M[1, index]
        else:
            xc = dataCells[0, index]
            yc = dataCells[1, index]

        # Collect data points belonging to cluster
        cluster_data = np.zeros((7, 1))

        if fast_clustering:
            # If fast_clustering, first find grid-EVCs
            # which belong to the cluster
            mask_id = np.where(cluster_id == idx)[0]
            evc_cluster = M[:2, mask_id]

            # Then find all EVC that belong to cluster
            evc_data = np.round(dataCells[:2])
            # Trick to use in1d in 2D: create complex numbers
            arr1 = evc_cluster[0] + evc_cluster[1]*1j
            arr2 = evc_data[0] + evc_data[1]*1j
            mask_evc = np.where(np.in1d(arr2, arr1))[0]
            # Select dataCells according to this selection
            cluster_data = dataCells[:, mask_evc]
        else:
            # Else, just use cluster_id to select dataCells
            # belonging to cluster
            mask_id = np.where(cluster_id == idx)[0]
            cluster_data = dataCells[:, mask_id]

        # Create the vortex structure
        v = Vortex(cluster_center=[xc, yc],
                   cells=cluster_data[2:4],
                   evc=cluster_data[:2],
                   rortex=cluster_data[4],
                   ecr=cluster_data[5],
                   stencils=cluster_data[6],
                   cluster_id=idx,
                   dl=dl
                   )

        # Loop for noise identification:
        Ni = v.n_all_cells  # Number of cells in vortex
        dN = Ni      # Number of cells during last iteration

        while dN > 1 and Ni > 0:  # Loop until vortex is void
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
                if kinks_n.shape[0] > 0:
                    if noise_f > 0.0:
                        noise_n = np.concatenate((noise_n, kinks_n), axis=1)
                    else:
                        noise_n = kinks_n

            # Append noise cells
            if noise_f > 0.0 or kink_f > 0.0:
                if noise_n.shape[0] > 1:
                    noise.append(noise_n)

            # Update characteristics of vortex
            if empty:
                v.n_all_cells = 0
            else:
                v.update_vortex_cells()
                v.update_n_all_cells()
                v.update_radius()
                v.update_center()


            # Stop if N didn't change or if N=0
            dN = Ni - v.n_all_cells
            Ni = v.n_all_cells

        # Append vortex
        if v.n_all_cells > 1:
            vortices.append(v)

    # Make array of noise cells
    if (noise_f > 0.0 or kink_f > 0.0) and len(noise) > 0:
        noise = np.hstack(noise)

    return vortices, noise
