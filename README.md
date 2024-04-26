```                         
    _/_/_/  _/          _/  _/   _/_/_/    _/
  _/         _/        _/   _/   _/    _/  _/
    _/_/      _/      _/    _/   _/_/_/    _/
        _/     _/ _/ _/     _/   _/  _/    _/
  _/_/_/        _/  _/      _/   _/   _/   _/_/_/


             (c) IRSOL, 11.04.2022
```

Author: _José Roberto Canivete Cuissa_   
Email: _jose.canivete@irsol.usi.ch_
 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10016647.svg)](https://doi.org/10.5281/zenodo.10016647)


----

# SWIRL code
The SWirl Identification by Rotation centers Localization (SWIRL) is an automated vortex identification algorithm written in python and based on the _Estimated Vortex Center_ (EVC) method [Canivete Cuissa & Steiner, 2022](#references). 
Given a two-dimensional velocity field defined on a Cartesian grid and the grid cell size in physical units, the SWIRL code returns a list with the identified vortical structures and their main properties.

----

## Table of contents
- [Abstract](#abstract)
- [Dependencies](#dependencies)
- [How to install](#how-to-install)
- [How to run](#how-to-run)
- [Parameters](#parameters)
- [References](#references)

----

## Abstract
Vortices are one of the fundamental features of turbulent fluid dynamics, yet it is extremely difficult to accurately and automatically identify vortical motions in, for example, simulations and observations of turbulent flows. We propose a new method for an accurate and reliable identification of vortices which requires the knowledge of the velocity field. The advantage of this method is that it accounts for both the local and global properties of the flow, which makes it robust to noise and reliable even in highly turbulent scenarios.  
In our method, vortical motions are identified in the given velocity field by clustering estimated centers of rotation computed from grid points where the flow is characterized by some degree of curvature (rotation). The estimated center points are dubbed _estimated vortex center_ (EVC) points. Since a vortex can be intuitively defined as a collection of fluid particles coherently rotating around a common axis, clusters of EVC points will appear in the center of coherently rotating flows.  
To accurately estimate EVC points, and therefore allow for a robust identification of vortices, we employ the Rortex mathematical criterion ([Tian et al., 2018](#references); [Liu et al., 2018](#references)) and the properties of the input velocity field. The clustering is then performed with a grid- and vortex-adapted version of the Clustering by fast search and find of density peaks (CFSFDP) algorithm ([Rodriguez & Laio, 2014](#references)).  
The algorithm is implemented in the SWIRL code and it has been tested on (noisy) artificial vortex flows and on (magneto-)hydrodynamical and turbulent numerical simulations with excellent results. More details on the EVC method and on the implementation of the code can be found in [Canivete Cuissa & Steiner, 2022](#references).

> If you find the SWIRL code useful in your research, we would really appreciate if you could cite the paper [Canivete Cuissa & Steiner, 2022](#references) in your published work.

----

## Dependencies
The SWIRL code requires Python 3 and makes use of the following Python libraries:
- numpy      (tested with _v. 1.21.6_)
- scipy      (tested with _v. 1.4.1_)
- h5py       (tested with _v. 2.10.0_)
- matplotlib (tested with _v. 3.3.2_)

----

## How to install
Download or clone this github repository into your machine. 

The easiest way to install the code is to use pip:
```
python3 -m pip install . 
```

If you're a developer and want to install the code in editable mode, you can run:
```
python3 -m pip install -e .
```

## How to run
A detailed tutorial on how to run the SWIRL code is presented in a Jupyter-Notebook in the _example_ folder. Here, we just give a brief recap.


The core of the swirl module is the `Identification` class, which allows to run the algorithm.
Given a two-dimensional velocity field, `vx` and `vy`, and the size of the grid cells in the two dimensions, `dx` and `dy`, an instance of the `Identification` class can be initialized with
```
>>> vortices = swirl.Identification(v = [vx, vy],
                                    grid_dx = [dx, dy])
```

> Important!   
> The spatial units of the velocity field and of the grid cell sizes must be the same, otherwise the identification algorithm won't work correctly. For example, if the velocity field is given in units of $cm/s$, then the grid cells sizes must be given in units of $cm$.  

Once the object has been initialized, one can run the algorithm with
```
>>> vortices.run()
```

The identified vortices are stored in the `vortices` object. To access them, one uses the `vortices` object as if it was a list. This means that the first vortex is store in `vortices[0]`, the second in `vortices[1]`, and so on. The number of identified vortices can therefore be obtained with `len(vortices)`.  
The properties of the identified vortices are saved in a `Vortex` object, which contains the following attributes
- radius (float) : The effective radius of the vortex
- center (list) : The coordinates of the center of the vortex
- orientation (float) : The orientation of the vortex: +1.0 = Counterclock-wise, -1.0 = Clockwise
- vortex_cells (array) : The list of the coordinates of the cells forming the vortex
- and more ...

Therefore, to access the radius of the first vortex identified, one runs the following command:
```
>>> r = vortices[0].radius
```

The list of the radii, centers, or orientations of all the vortices identified can be obtained directly from the `vortices` object with
```
>>> radii = vortices.radii
>>> centers = vortices.centers
>>> orientations = vortices.orientations
```

To save the properties of the identified vortices, one can run the following method
```
>>> vortices.save('name_of_the_file')
```
which will save all the information relative to the identification process in a structured hdf5 file.

----

## Parameters
The `Identification` class accepts a number of different parameters that one might need to tweak in order to optimize the results. These parameters must be given in a parameter file, which format is shown in the file _default_swirl.param_. If a parameter file is not given as an input in the initialization of the `Identification` class, or if some parameters are omitted, the SWIRL code will use the default values (i.e. the ones shown in the _default_swirl.param_ file). The parameters are defined as follows:
- `stencils` [list] = [1]  
    A list of integers that correspond the stencils of grid cells used for the computation of numerical derivatives. 
- `swirlstr_params` [list] = [0., 0., 0.]  
    The parameters used in the computation of the swirling strength. The first one is the threshold value $\epsilon_{\lambda}$, 
    we suggest this value to be always $0.0$. The second and thirs values correspond to the $\kappa_{\zeta}$ and $\delta_{\zeta}$ 
    parameters used to compute the enhanced swirling strength. We suggest these values to be either $0.0$ or around $1.0$.  
- `dc_param` [float] = 3.  
    Parameter used to compute the critical distance used in the 
    clustering algorithm. Depending on the value of the dc_adaptive 
    parameter, it can represent the percentual number of data points that,
    in average, are considered as neighbours, i.e. inside the critical 
    distance (True), or to define the critical distance dc = dc_param in units
    of grid cells (False).
- `dc_adaptive` [boolean] = True  
    Option to use the adaptive critical distance evaluation or to use the 
    fixed one based on the value of dc_param.  
- `cluster_fast` [boolean] = True  
    Option to use the grid adapted version of the clustering algorithm, which
    accelerates greatly the computation without sacrificing accuracy. We suggest to keep it True
- `cluster_kernel` [string] = Gaussian  
    Kernel used to compute densities in the clustering algorithm.
    'Gaussian': Gaussian kernel.
    'Heaviside': Heaviside function.
- `cluster_decision` [string] = delta-rho  
    The method used to select the cluster centers in the clustering process.
    'delta-rho' : Use the delta and rho criteria to select the cluster centers.
    'gamma' : Use the gamma criterion to select the cluster centers.
- `cluster_params` [list] = [1.0, 0.5, 2.0]  
    List of parameters for the selection of cluster centers in the clustering
    process. The list must contain three entries, which correspond to the parameters 
    $delta_p$, $rho_p$, $gamma_p$.
- `noise_param` [float] = 1.0  
    Parameter to remove noisy cells from the identification process. It correspond to the parameter
    $r_p$. We recommend values $\gtrsim 1.0$.
- `kink_param` [float] = 1.0  
    Parameter to remove 'non-spiraling' coherent curvatures identified as vortices. 
    It also corresponds to the parameter $r_p$. We recommend values $\sim 1.0$.
    
Finally, one can also select the option `verbose=False` in the initialization of the `Identification` object, to not print any output when running the code.

----

## References
1. [Canivete Cuissa, J. R., Steiner, O., 2022, A&A 668, A118](https://ui.adsabs.harvard.edu/abs/2022A%26A...668A.118C/abstract)  
2. [Liu, C., Gao, Y., Tian, S., & Dong, X. 2018, Physics of Fluids, 30, 035103](https://ui.adsabs.harvard.edu/abs/2018PhFl...30c5103L/abstract)  
3. [Rodriguez, A. & Laio, A. 2014, Science, 344, 1492](https://ui.adsabs.harvard.edu/abs/2014Sci...344.1492R/abstract)
4. [Tian, S., Gao, Y., Dong, X., & Liu, C. 2018, Journal of Fluid Mechanics, 849, 312](https://ui.adsabs.harvard.edu/abs/2018JFM...849..312T/abstract)
----
