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
from dataclasses import dataclass
import time
from typing import Type
import h5py
import numpy as np
from .utils import timings, vector2D, read_params  # , prepare_dataframe
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
    v : list OR string
        A list containing the velocity fields, [vx, vy]. The velocity
        fields must be two-dimensional arrays.
        OR
        A string with the path to a h5py file or numpy files (to be implemented)

    grid_dx : list
        A list containing the grid cell sizes, [dx, dy]. The grid cell
        sizes must be float numbers.

    param_file [optional]: str
        The path and name of the parameter file. The parameter file 
        must follow the given format. Check the example file in the /example 
        directory of the github SWIRL repository. If not given, the default
        values of the parameters will be used.
        The parameter file contains the following parameters:
        stencils : list
            List of different stencil sizes. The stencils sizes must be ints.
            - Default: [1]
        swirlstr_params : list
            List of parameters for the enhanced swirling strength criterion,
            [eps, delta, kappa]. Each one of these values must be a float.  
            - Default: [0., 0., 0.]
        dc_param : float
            Parameter used to compute the critical distance used in the 
            clustering algorithm. Depending on the value of the dc_adaptive 
            parameter, it can represent the percentual number of data points that,
            in average, are considered as neighbours, i.e. inside the critical 
            distance (False), or to define the critical distance dc = dc_param*grid_dx,
            where grid_dx is the grid cell size.
            - Default: [3.]
        dc_adaptive : bool
            Option to use the adaptive critical distance evaluation or to use the 
            fixed one based on the value of dc_param.
            - Default: True
        cluster_fast : bool
            Option to use the grid adapted version of the clustering algorithm, which
            accelerates greatly the computation without sacrificing accuracy.
            - Default: True
        cluster_kernel : string
            Kernel used to compute densities in the clustering algorithm.
            'Gaussian': Gaussian kernel.
            'Heaviside': Heaviside function.
            - Default : 'Gaussian'
        cluster_decision : string
            The method used to select the cluster centers in the clustering process.
            'delta-rho' : Use the delta and rho criteria to select the cluster centers.
            'gamma' : Use the gamma criterion to select the cluster centers.
            - Default : 'delta-rho'
        cluster_params : list
            List of parameters for the selection of cluster centers in the clustering
            process. The list must contain three entries, all floats: 
            [rho_p, delta_p, gamma_p]
            - Default : [1.0, 0.5, 2.0]
        noise_param : float
            Parameter to remove noisy cells from identification process.
            - Default : 1.0
        kink_param : float
            Parameter to remove misidentified "kinks" from identified vortices.
            - Default : 1.0
    
    verbose [optional]: bool
        Parameter to have a "verbose" report of the SWIRL object and of the
        identification process.
        - Default : True

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
                 grid_dx: list,
                 param_file: str = '',
                 verbose: bool = True
                 ):
        """
        Class initalization

        Parameters
        ----------
        v : list OR string
            A list containing the velocity fields, [vx, vy]. The velocity
            fields must be two-dimensional arrays.
            OR
            A string with the path to a h5py file or numpy files (to be implemented)
        grid_dx : list
            A list containing the grid cell sizes, [dx, dy]. The grid cell
            sizes must be float numbers.
        param_file [optional]: str
            The path and name of the parameter file. The parameter file 
            must follow the given format. Check the example file in the /example 
            directory of the github SWIRL repository. If not given, the default
            values of the parameters will be used.
            Default : ''
        verbose [optional]: bool
            Parameter to have a "verbose" report of the SWIRL object and of the
            identification process.
            Default : True

        Returns
        -------

        Raises
        ------
        """
        # Safe Initialization of velocity arrays ...
        if isinstance(v, list):
            self.v = vector2D(v[0], v[1])
        elif isinstance(v, str):
            raise NotImplementedError('Reading the velocity field from a file is not yet implemented.')
        else:
            raise TypeError('The velocity field [vx, vy] must be given as a list of arrays or as a file path.')
        # and grid size spacing
        if isinstance(grid_dx, list):
            self.grid_dx = vector2D(grid_dx[0], grid_dx[1])
            self.dx = grid_dx[0]
            self.dy = grid_dx[1]
        else:
            raise TypeError('The grid cell sizes grid_dx = [dx,dy] must be given as a list of floats.')
        # Initialize parameters
        self.params = read_params(param_file)
        # Initialize verbose
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError('Verbose must be a bool.')

        # Initialize other quantities:
        self.S = [0.0]
        self.W = [0.0]
        self.U = [0.0]
        self.R = [0.0]
        self.M = [0.0]
        self.dataCells = [0.0]
        self.timings = dict()
        self.cluster_id = None
        self.radii = None
        self.centers = None
        self.orientations = None
        self.n_vortices = None

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
            print('---               (c) IRSOL, 11.04.2022               ---')
            print('---                                                   ---')
            print('--- Author:      José Roberto Canivete Cuissa         ---')
            print('--- Email:       jcanivete@ics.uzh.ch                 ---')
            print('---------------------------------------------------------')
            print('---')
            print('--- Parameters:')
            print('---------------')
            print('---    grid_dx          :', self.dx, ', ', self.dy)
            print('---    stencil          :', self.params['stencils'])
            print('---    swirlstr_params  :', self.params['swirlstr_params'])
            print('---    dc_param         :', self.params['dc_param'])
            print('---    dc_adaptive      :', self.params['dc_adaptive'])
            print('---    cluster_fast     :', self.params['cluster_fast'])
            print('---    cluster_kernel   :', self.params['cluster_kernel'])
            print('---    cluster_decision :', self.params['cluster_decision'])
            print('---    cluster_params   :', self.params['cluster_params'])
            print('---    noise_param      :', self.params['noise_param'])
            print('---    kink_param       :', self.params['kink_param'])
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
                                           self.grid_dx,
                                           self.params['stencils']
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
                                                  self.grid_dx,
                                                  self.params['stencils'],
                                                  self.params['swirlstr_params']
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
                                                        self.grid_dx,
                                                        self.params['stencils'],
                                                        self.params['swirlstr_params']
                                                        )
        # Timing
        t_total = timings(t_start)
        self.timings['Rortex'] = t_total
    # ----------------------------------


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
        # Sanity check
        if isinstance(self.R[0], float):
            raise RuntimeError('evcmap: Rortex has not been computed yet.')
        # Call external function
        self.M, self.dataCells = compute_evcmap(self.R,
                                                self.U,
                                                self.v,
                                                self.grid_dx
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
        self.data = prepare_data(self.M, self.dataCells, self.params['cluster_fast'])
        # Call clustering algorithm
        self.cluster_id, self.peaks_ind, self.rho, self.delta, self.dc, self.d = findcluster2D(self.data,
                                                                                               self.params['dc_param'],
                                                                                               self.params['dc_adaptive'],
                                                                                               self.grid_dx,
                                                                                               self.params['cluster_fast'],
                                                                                               self.params['cluster_kernel'],
                                                                                               self.params['cluster_decision'],
                                                                                               self.params['cluster_params']
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
                                                  self.params['cluster_fast'],
                                                  self.params['noise_param'],
                                                  self.params['kink_param'],
                                                  self.grid_dx
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
        # Compute the mathematical criterion needed (i.e rortex)
        self.rortex()
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
        # Check if there are vortices identified
        if self.n_vortices is None:
            raise RuntimeError("No vortices have been identified. The file won't be created.")

        # Create the file
        hf = h5py.File(file_name+'.h5','w')
        # Set attributes
        hf.attrs.__setitem__("n_vortices", self.n_vortices)
        hf.attrs.__setitem__("radii", self.radii)
        hf.attrs.__setitem__("centers", self.centers)
        hf.attrs.__setitem__("orientations", self.orientations)
        # Create params dataset
        params_group = hf.create_group('params')
        params_group.create_dataset('grid_dx', data=np.array([self.grid_dx.x, self.grid_dx.y]))
        for key in self.params.keys():
            params_group.create_dataset(key, data=self.params[key])
        # Create vortices datasets
        hf.create_group('vortices')
        for n in np.arange(self.n_vortices):
            vortex_group = hf.create_group('vortices/'+str(n).zfill(5))
            # Save radius, center, orientation, cells, evc, rortex
            vortex_group.create_dataset("r", (1,), data=self.vortices[n].r)
            vortex_group.create_dataset("center", (2,), data=self.vortices[n].center)
            vortex_group.create_dataset("orientation", (1,), data=self.vortices[n].orientation)
            vortex_group.create_dataset("cells" , data=self.vortices[n].cells)
            vortex_group.create_dataset("evc", data=self.vortices[n].evc)
            vortex_group.create_dataset("rortex", data=self.vortices[n].X)
        # Close file
        hf.close()

