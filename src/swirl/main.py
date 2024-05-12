"""
SWIRL Code
    main.py

José Roberto Canivete Cuissa
IRSOL, 10.02.2021

----------------

This code contains the main structure of the SWIRL code,
that is the SWIRL Identification class.
"""
# Imports
# from dataclasses import dataclass
import time
from typing import Type
import h5py
import numpy as np
import warnings
from .utils import timings, vector2D, read_params  # , prepare_dataframe
from .vorticity import compute_vorticity
from .swirlingstrength import compute_swirlingstrength
from .rortex import compute_rortex
from .evcmap import compute_evcmap
from .cluster import findcluster2D, prepare_data
from .vortex import detection
# ---------------------------


class Identification:
    """
    Given a two-dimensional velocity field and the size of the
    grid cells, the Identification class allows to compute different
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
            distance (True), or to define the critical distance dc = dc_param in units
            of grid cells (False).
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
        Parameter to have a "verbose" report of the Identification object and of the
        identification process.
        - Default : True

    Properties
    ----------
    vorticity : list of arrays
        List of the vorticity arrays computed from the velocity field. If not yet
        computed, it computes the quantity.

    swirling_str : list of arrays
        List of the swirling strength arrays computed from the velocity field. If
        not yet computed, it computes the quantity.
    
    rortex : list of arrays
        List of the Rortex arrays computed from the velocity field. If not yet
        computed, it computes the quantity.
    
    gevc_map : list of arrays
        The G-EVC map. If not yet computed, it computes the quantity.
    

    Other attributes
    ----------------
    timings : dict
        Dictionary with the timings for each one of the processes of the
        automated identification.
    
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
    clustering(self)
        Performs the clustering with the adapted CFSFDP algorithm.
    
    detect_vortices(self)
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
            Parameter to have a "verbose" report of the Identification object and of the
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
        self.param_file = param_file
        self.params = read_params(param_file)
        # Initialize verbose
        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError('Verbose must be a bool.')

        # Initialize other quantities:
        self._swirling_str = None
        self._vorticity = None
        self._vgt = None
        self._rortex = None
        self._gevc_map = None
        self._data_cells = [0.0]
        self._cluster_id = None
        self.timings = dict()
        self._vortices_list = None
        self.radii = None
        self.centers = None
        self.orientations = None
        self.n_vortices = None

        # Print verbose
        if self.verbose:
            self._text = ''
            self._text+= '---------------------------------------------------------\n'
            self._text+= '---                                                   ---\n'
            self._text+= '---    _/_/_/  _/          _/  _/   _/_/_/    _/      ---\n'
            self._text+= '---  _/         _/        _/   _/   _/    _/  _/      ---\n'
            self._text+= '---    _/_/      _/      _/    _/   _/_/_/    _/      ---\n'
            self._text+= '---        _/     _/ _/ _/     _/   _/  _/    _/      ---\n'
            self._text+= '---  _/_/_/        _/  _/      _/   _/   _/   _/_/_/  ---\n'
            self._text+= '---                                                   ---\n'
            self._text+= '---------------------------------------------------------\n'
            self._text+= '---------------------------------------------------------\n'
            self._text+= '---                                                   ---\n'
            self._text+= '---               (c) IRSOL, 11.04.2022               ---\n'
            self._text+= '---                                                   ---\n'
            self._text+= '--- Author:      José Roberto Canivete Cuissa         ---\n'
            self._text+= '--- Email:       jcanivete@ics.uzh.ch                 ---\n'
            self._text+= '---------------------------------------------------------\n'
            self._text+= '---\n'
            self._text+= '--- Parameters:\n'
            self._text+= '---------------\n'
            self._text+= '---    grid_dx          : '+str(self.dx)+', '+str(self.dy)+'\n'
            self._text+= '---    stencil          : '+str(self.params['stencils'])+'\n'
            self._text+= '---    swirlstr_params  : '+str(self.params['swirlstr_params'])+'\n'
            self._text+= '---    dc_param         : '+str(self.params['dc_param'])+'\n'
            self._text+= '---    dc_adaptive      : '+str(self.params['dc_adaptive'])+'\n'
            self._text+= '---    cluster_fast     : '+str(self.params['cluster_fast'])+'\n'
            self._text+= '---    cluster_kernel   : '+str(self.params['cluster_kernel'])+'\n'
            self._text+= '---    cluster_decision : '+str(self.params['cluster_decision'])+'\n'
            self._text+= '---    cluster_params   : '+str(self.params['cluster_params'])+'\n'
            self._text+= '---    noise_param      : '+str(self.params['noise_param'])+'\n'
            self._text+= '---    kink_param       : '+str(self.params['kink_param'])+'\n'
            self._text+= '---\n'
            self._text+= '---------------------------------------------------------\n'
            print(self._text)
    # ------------------------------------------------------------------------


    def __len__(self):
        """
        Magic method for the length of the object. It returns the number of swirls 
        identified.

        Parameters
        ----------

        Returns
        -------
        self.n_vortices:
            The number of vortices identified
        
        Raises
        ------
        Warning
            If the identification process has not been done yet, it reminds it 
            to the user.
        """
        if self._vortices_list is None:
            warnings.warn("\n Warning: the identification has not been carried out yet.")
            return 0
        else:
            return self.n_vortices
    # ----------------------------


    def __repr__(self):
        """
        Magic method for the representation of the class in a print statement.

        Parameters
        ----------

        Returns
        -------
        
        Raises
        ------
        """
        if self._vortices_list:
            return 'SWIRL Identification Class object. Vortices identified : '+str(self.n_vortices)
        else:
            return 'SWIRL Identification Class object. Identification not yet performed'
    
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
        print(self._text[:-59])
        if self._vortices_list:
            print('--- Identification:')
            print('-------------------')
            print('---    Number of identified vortices :', self.n_vortices)
            print('---    Details:')
            print('---------------')
            n = 0
            for vortex in self._vortices_list:
                print('--- ',n,'.  ',vortex)
                print('---------------')
                n+=1
        print('---')
        print('---------------------------------------------------------')
        return '\n'
    # -------------


    def __getitem__(self, i):
        """
        Magic method for indexation of Identification objects. It returns the vortex in the 
        vortex list 'vortices' with the index i.

        Parameters
        ----------
        i : int
            The index of the vortex

        Returns
        -------
        v : Vortex object
            The vortex with index i
        
        Raises
        ------
        Warning
            If the identification process has not been done yet, it reminds it 
            to the user.
        """
        if self._vortices_list is None:
            warnings.warn("\n Warning: the identification has not been carried out yet.")
            return
        else:
            return self._vortices_list[i]
    # -----------------------------


    @property
    def vorticity(self):
        """
        It computes the vorticity for every cell in the 2D grid, saves it
        as an attribute, and returns it.

        Parameters
        ----------

        Returns
        -------
        _vorticity : list
            List of vorticity arrays computed with different stencils

        Raises
        ------
        """
        if self._vorticity is None:
            # Timing
            t_start = time.process_time()
            # Call external function
            (self._vorticity,
             self._vgt) = compute_vorticity(self.v,
                                            self.grid_dx,
                                            self.params['stencils']
                                            )
            # Timing
            t_total = timings(t_start)
            self.timings['Vorticity'] = t_total
        return self._vorticity
    # -------------------------------------


    @property
    def swirling_str(self):
        """
        It computes the swirling strength for every cell in the 2D grid,
        saves it as an attribute, and returns it.

        Parameters
        ----------

        Returns
        -------
        _swirling_str : list
            List of swirling strength arrays computed with different stencils

        Raises
        ------
        """
        if self._swirling_str is None:
            # Timing
            t_start = time.process_time()
            # Call external function
            (self._swirling_str,
             self._vgt) = compute_swirlingstrength(self.v,
                                                   self.grid_dx,
                                                   self.params['stencils'],
                                                   self.params['swirlstr_params']
                                                   )
            # Timing
            t_total = timings(t_start)
            self.timings['Swirling strength'] = t_total
        return self._swirling_str
    # ---------------------------------------------


    @property
    def rortex(self):
        """
        It computes the rortex for every cell in the 2D grid,
        saves it as an attribute, and returns it.

        Parameters
        ----------

        Returns
        -------
        _rortex : list
            List of rortex arrays computed with different stencils

        Raises
        ------
        """
        if self._rortex is None:
            # Timing
            t_start = time.process_time()
            # Call external function
            (self._rortex,
             self._swirling_str,
             self._vgt,
             self._vorticity) = compute_rortex(self._swirling_str,
                                               self._vorticity,
                                               self.v,
                                               self.grid_dx,
                                               self.params['stencils'],
                                               self.params['swirlstr_params']
                                               )
            # Timing
            t_total = timings(t_start)
            self.timings['Rortex'] = t_total
        return self._rortex
    # ----------------------------------


    @property
    def gevc_map(self):
        """
        It computes the (G-)EVC map with the choosen criterion

        Parameters
        ----------

        Returns
        -------
        The G-EVC map

        Raises
        ------
        """
        if self._gevc_map is None:
            # Print status
            if self.verbose:
                print('--- Computing EVC map')
            # Timing
            t_start = time.process_time()
            # Sanity check
            if self._rortex is None:
                self.rortex
            # Call external function
            self._gevc_map, self._data_cells = compute_evcmap(self._rortex,
                                                              self._vgt,
                                                              self.v,
                                                              self.grid_dx
                                                              )
            # Timing
            t_total = timings(t_start)
            self.timings['EVC map'] = t_total
        return self._gevc_map
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
        if self._gevc_map is None:
            self.gevc_map
        # Prepare data for clustering algorithm
        self._data = prepare_data(self._gevc_map, self._data_cells, self.params['cluster_fast'])
        # Call clustering algorithm
        (self._cluster_id,
         self._peaks_ind,
         self.rho,
         self.delta,
         self._dc,
         self._d) = findcluster2D(self._data,
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
        if self._cluster_id is None:
            raise RuntimeError('Detection: Clustering has not been done.')
        # Call detection routine for vortex identification
        if self._cluster_id.size > 0:
            self._vortices_list, self.noise = detection(self._data_cells,
                                                        self._gevc_map,
                                                        self._cluster_id,
                                                        self._peaks_ind,
                                                        self.params['cluster_fast'],
                                                        self.params['noise_param'],
                                                        self.params['kink_param'],
                                                        self.grid_dx
                                                        )
        else:
            self._vortices_list = []
            self.noise = []
        # Store main quantities
        self.radii = np.array([v.radius for v in self._vortices_list])
        self.centers = np.array([v.center for v in self._vortices_list])
        self.orientations = np.array([v.orientation for v in self._vortices_list])
        self.n_vortices = len(self._vortices_list)
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
            print('---------------------------------------------------------')
            print('--- Starting identification ')
            print('---------------------------------------------------------')
            print('---')
        # Timing
        t_start = time.process_time()
        # Compute the mathematical criterion needed (i.e rortex)
        self.rortex
        # Compute the EVC map
        self.gevc_map
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
            print('--- Identified vortices:', self.n_vortices)
            print('---')
            print('--- Timings')
            for t in self.timings:
                print("{:<6} {:<10} {:<3} {:<10}".format(
                    '---   ', t, ': ', self.timings[t]))
            print('---------------------------------------------------------')
            print('\n')
    # -------------------------------------


    def save(self, file_name='swirl_vortices'):
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
        hf.attrs.__setitem__("param_file", self.param_file)
        # Create short data datasets
        data_group = hf.create_group('data')
        data_group.create_dataset("radii", data=self.radii)
        data_group.create_dataset("centers", data=self.centers)
        data_group.create_dataset("orientations", data=self.orientations)
        data_group.create_dataset("gevc_map", data=self.gevc_map)
        data_group.create_dataset("rho", data=self.rho)
        data_group.create_dataset("delta", data=self.delta)
        data_group.create_dataset("gamma", data=self.gamma)
        data_group.create_dataset("rortex", data=self.rortex)
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
            vortex_group.create_dataset("radius", (1,), data=self[n].radius)
            vortex_group.create_dataset("center", (2,), data=self[n].center)
            vortex_group.create_dataset("orientation", (1,), data=self[n].orientation)
            vortex_group.create_dataset("all_cells" , data=self[n].all_cells)
            vortex_group.create_dataset("vortex_cells", data=self[n].vortex_cells)
            vortex_group.create_dataset("evc", data=self[n].evc)
            vortex_group.create_dataset("rortex", data=self[n].rortex)
            vortex_group.create_dataset("cluster_center", data=self[n].cluster_center)
        # Close file
        hf.close()
