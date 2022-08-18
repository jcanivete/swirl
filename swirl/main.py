"""
SWIRL Code
    swirl_test.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the main structure of the SWIRL code,
that is the SWIRL class.
"""
# Imports
import time
import h5py
import numpy as np
from .utils import timings, vector2D  # , prepare_dataframe
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import compute_evcmap
from .cluster import findcluster2D, prepare_data
from .vortex import detection
# ---------------------------


class SWIRL:
    """
    Given a two-dimensional velocity field and the size of the
    grid cells, the SWIRL class allows to compute different
    mathematical criteria and to perform an automated identification
    of vortices following the estimated vortex center (EVC) method.

    Input Attributes
    ----------------
    v : 2D arrays
        The velocity field
    dl : array
        The grid cell sizes
    l : list
        List of different stencil sizes
    S_param : list
        List of parameters for enhanced swirling strength
        [eps, delta, kappa]
    crit: string
        Which mathematical criterion to use in the identification
        process.
    dc_coeff : float
        If dc_adaptive=True: percentual coefficient used to compute the
        critical distance. The critical distance is defined as the
        average distance such that dc_coeff % of points inside dc.
        Preferably, use 2-3.
        If dc_adaptive=False: Size of dc in physical units.
    dc_adaptive : bool
        Option: to use the adaptive critical distance (dc) computed on the
        % of points inside dc (True) or to use a fixed distance defined
        by dc_coeff*dl.
    fast_clustering : bool
        Option to use the grid adapted version of the clustering algorithm.
    xi_option : int
        Kernel used to compute densities.
        Option 1: Heaviside function.
        Option 2: Gaussian kernel.
    clust_selector : string
        Cluster centers selection process:
        'delta-rho' or 'gamma'.
    clust_options : list
        List of parameters for cluster centers selection.
    noise_f: float
        Parameter to remove noisy cells from identification process.
    kink_f: float
        Parameter to remove "kink" curves from identified vortices.
    verbose: bool
        Parameter to have a "verbose" report of the SWIRL object and of the
        identification process.

    Derived Attributes
    ------------------
    W : list of arrays
        List of the vorticity arrays computed from the velocity field.
    S : list of arrays
        List of the swirling strength arrays computed from the velocity field.
    R : list of arrays
        List of the Rortex arrays computed from the velocity field.
    M : list of arrays
        G-EVC maps.
    dataCells : list
        List containing all the cells presenting curvature in their flow and
        their properties (coordinates, evc center, criterium value, ...).
    timings : dict
        Dictionary with the timings for each one of the processes of the
        automated identification.
    cluster_id : array
        Cluster indeces for all EVC points given as input.
        0 means that the point does not belong to any cluster and is
        therefore noise.
    noise : arrays
        List of all grid cells that are classified as noise
    vortices : list
        List with all the identified Vortex objects.
    radii : array
        List of radii of the identified vortices.
    centers : array
        List of the centers of the identified vortices.
    orientations : array
        List of the orientations of the identified vortices.
    n_vortices : int
        Number of identified vortices

    Methods
    -------
    vorticity(self)
        Computes the vorticity W based on the velocity field,
        stencils and grid spacing.
    swirlingstrength(self)
        Computes the swirling strength S based on the velocity field,
        swirl parameters, stencils and grid spacing.
    rortex(self)
        Computes the rortex criterion R based on the velocity field,
        swirl parameters, stencils and grid spacing.
    compute_criterion(self)
        Computes the criterion choosen as parameter.
    evcmap(self)
        Computes the (G-)EVC map.
    clustering(self)
        Performs the clustering with the adapted CFSFDP algorithm.
    detect_vortics(self)
        Based on the clustering output, creates a collection of
        Vortex objects which contains the final result of the identification.
    run(self)
        Runs the whole identification process based on the parameters
        given as input.
    save(self, file_name)
        It saves the properties of the identified vortices in a h5py file.
    """

    def __init__(self,
                 v: list,
                 dl: list,
                 l: list = [1],
                 S_param: list = [0., 0., 0.],
                 crit: str = 'rortex',
                 dc_coeff: float = 3.,
                 dc_adaptive: bool = True,
                 fast_clustering: bool = True,
                 xi_option: int = 2,
                 clust_selector: str = 'delta-rho',
                 clust_options: list = [1.0, 0.5, 2.0],
                 noise_f: float = 1.0,
                 kink_f: float = 1.0,
                 verbose: bool = True
                 ):
        """
        Class initalization

        Parameters
        ----------
        v : list of arrays [vx,vy]
            The velocity field.
        dl : array [dx,dy]
            The grid cell sizes.
        l : list
            List of different stencil sizes.
        S_param : list [eps, delta, kappa]
            List of parameters for enhanced swirling strength.
        crit: string
            Which mathematical criterion to use in the identification
            process.
        dc_coeff : float
            If dc_adaptive=True: percentual coefficient used to compute the
            critical distance. The critical distance is defined as the
            average distance such that dc_coeff % of points inside dc.
            Preferably, use 2-3.
            If dc_adaptive=False: Size of dc in physical units.
        dc_adaptive : bool
            Option: to use the adaptive critical distance (dc) computed on the
            % of points inside dc (True) or to use a fixed distance defined
            by dc_coeff*dl.
        fast_clustering : bool
            Option to use the grid adapted version of the clustering algorithm.
        xi_option : int
            Kernel used to compute densities.
            Option 1: Heaviside function.
            Option 2: Gaussian kernel.
        clust_selector : string
            Cluster centers selection process:
            'delta-rho' or 'gamma'.
        clust_options : list
            List of parameters for cluster centers selection.
        noise_f: float
            Parameter to remove noisy cells from identification process.
        kink_f: float
            Parameter to remove "kink" curves from identified vortices.
        verbose: bool
            Parameter to have a "verbose" report of the SWIRL object and of the
            identification process.

        Returns
        -------

        Raises
        ------
        """
        # Safe Initialization of velocity arrays and parameters
        #
        # Initialize v
        if len(v) == 2 and isinstance(v[0], np.ndarray) and isinstance(v[1], np.ndarray):
            self.v = vector2D(v[0], v[1])
        else:
            raise ValueError(
                'Initialization: velocity field is not a list of numpy arrays [vx,vy].')

        # Initialize dl
        if len(dl) == 2 and isinstance(dl[0], float) and isinstance(dl[1], float):
            self.dl = vector2D(dl[0], dl[1])
        else:
            raise ValueError(
                'Initialization: grid spacing is not a list of float [dx,dy].')

        # Initialize l
        if isinstance(l, list):
            self.l = l
        else:
            raise ValueError(
                'Initialization: stencil list is not a list of int [l1,l2,...].')

        # Initialize S_param
        if (len(S_param) == 3 and
            isinstance(S_param[0], float) and
            isinstance(S_param[1], float) and
            isinstance(S_param[2], float)
            ):
            self.S_param = S_param
        else:
            raise ValueError(
                'Initialization: Swirling strength param. is not a list of float [eps, kappa, delta].')

        # Initialize crit
        if crit in ['rortex', 'swirling strength', 'vorticity']:
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
        if xi_option in [1, 2]:
            self.xi_option = xi_option
        else:
            raise ValueError('Initialization: xi_option parameter unknown')

        # Initialize clust_selector
        if clust_selector in ['delta-rho', 'gamma']:
            self.clust_selector = clust_selector
        else:
            raise ValueError(
                'Initialization: clust_selector parameter unknown')

        # Initialize clust_options
        if isinstance(clust_options, list):
            self.clust_options = clust_options
        else:
            raise ValueError(
                'Initialization: clust_options must be a list of parameters')

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

        # Initialize verbose
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise ValueError('Initialization: verbose must be a bool')

        # Initialize other quantities:
        self.S = [0.0]
        self.W = [0.0]
        self.U = [0.0]
        self.R = [0.0]
        self.M = [0.0]
        self.dataCells = [0.0]
        self.timings = dict()
        self.cluster_id = None

        # Print verbose
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
            print('--- Author:      José Roberto Canivete Cuissa         ---')
            print('--- Affiliation: IRSOL                                ---')
            print('--- Email:       jcanivete@ics.uzh.ch                 ---')
            print('---------------------------------------------------------')
            print('---')
            print('--- Parameters:')
            print('---------------')
            print('---    dl               :', self.dl.x, ', ', self.dl.y)
            print('---    l                :', self.l)
            print('---    S_param          :', self.S_param)
            print('---    crit             :', self.crit)
            print('---    dc_coeff         :', self.dc_coeff)
            print('---    dc_adaptive      :', self.dc_adaptive)
            print('---    fast_clustering  :', self.fast_clustering)
            print('---    xi_option        :', self.xi_option)
            print('---    clust_selector   :', self.clust_selector)
            print('---    clust_options    :', self.clust_options)
            print('---    noise_f          :', self.noise_f)
            print('---    kink_f           :', self.kink_f)
            print('---')
            print('---------------------------------------------------------')
    # ------------------------------------------------------------------------

    def vorticity(self):
        """
        It computes the vorticity for every cell in the 2D grid and saves it
        as an attribute.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """

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
    # -------------------------------------

    def swirlingstrength(self):
        """
        It computes the swirling strength for every cell in the 2D grid
        and saves it as an attribute.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """

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
    # ---------------------------------------------

    def rortex(self):
        """
        It computes the rortex for every cell in the 2D grid and saves it
        as an attribute.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
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
    # ----------------------------------

    def compute_criterion(self):
        """
        It computes the the choosen criterion for every cell in the
        2D grid and saves it as an attribute.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
        if self.verbose:
            print('--- Computing criterion')

        # Compute the criterion chosen
        if self.crit == 'vorticity':
            if isinstance(self.W[0], float):
                self.vorticity()
        elif self.crit == 'swirling strength':
            if isinstance(self.S[0], float):
                self.swirlingstrength()
        elif self.crit == 'rortex':
            if isinstance(self.R[0], float):
                self.rortex()
    # -----------------------

    def evcmap(self):
        """
        It computes the (G-)EVC map with the choosen criterion

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
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
            raise ValueError('evcmap: Criteria has not been computed.')

        # Call external function
        self.M, self.dataCells = compute_evcmap(X,
                                                self.U,
                                                self.v,
                                                self.dl
                                                )

        # Timing
        t_total = timings(t_start)
        self.timings['EVC map'] = t_total
    # -----------------------------------

    def clustering(self):
        """
        It performs the clustering process necessary for the automated
        identification of vortices.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
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
        self.cluster_id, self.peaks_ind, self.rho, self.delta, self.dc, self.d = findcluster2D(self.data,
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
    # --------------------------------------

    def detect_vortices(self):
        """
        Given the identified clusters, this routine creates a list of vortex objects
        which correspond to the identified vortices.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
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

        # Store main quantities
        self.radii = np.array([v.r for v in self.vortices])
        self.centers = np.array([v.center for v in self.vortices])
        self.orientations = np.array([v.orientation for v in self.vortices])
        self.n_vortices = len(self.vortices)

        # Timing
        t_total = timings(t_start)
        self.timings['Detection'] = t_total
    # -------------------------------------

    def run(self):
        """
        It runs the full automated identification based on the input
        parameters.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        """
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
                print("{:<6} {:<10} {:<3} {:<10}".format(
                    '---   ', t, ': ', self.timings[t]))
            print('---------------------------------------------------------')
            print('\n')
        
    # -------------------------------------

    def save(self, file_name='SWIRL_vortices'):
        """
        It saves the properties of the identified vortices in a h5py file.

        Parameters
        ----------
        file_name : string
            The name of the h5py file where the data will be saved.

        Returns
        -------

        Raises
        ------
        """
        # Create the file
        hf = h5py.File(file_name+'.h5','w')

        
