```                         
    _/_/_/  _/          _/  _/   _/_/_/    _/      
  _/         _/        _/   _/   _/    _/  _/      
    _/_/      _/      _/    _/   _/_/_/    _/      
        _/     _/ _/ _/     _/   _/  _/    _/      
  _/_/_/        _/  _/      _/   _/   _/   _/_/_/  


              IRSOL, 11.04.2022                      
                                                   
 Author: *José Roberto Canivete Cuissa*              
 Email: *jcanivete@ics.uzh.ch*                       
```  


# SWIRL code
The SWirl Identification by Rotation centers Localization (`SWIRL`) is an automated vortex identification algorithm written in python based on the _Estimated Vortex Center_ (EVC) method (*Canivete Cuissa & Steiner, 2022, A&A (submitted)*). The only required input is a two-dimensional velocity time istance defined on a Cartesian grid, and it returns the list of the identified vortical structures.

# _Work in progress_
This repository is still work in progress. Code examples, tests, and licence will be added soon, as well as improvements concerning the handling of inputs and outputs of the code. For any question, please contact me at *jcanivete@ics.uzh.ch*.

# Abstract
Vortices are one of the fundamental features of turbulent fluid dynamics, yet it is extremely difficult to accurately and automatically identify vortical motions in, for example, simulations and observations of turbulent flows. We propose a new method for an accurate and reliable identification of vortices which requires the knowledge of the velocity field. The advantage of this method is that it accounts for both the local and global properties of the flow, which makes it robust to noise and reliable even in highly turbulent scenarios.  
In our method, vortical motions are identified in the given velocity field by clustering estimated centers of rotation computed from grid points where the flow is characterized by some degree of curvature (rotation). The estimated center points are dubbed _estimated vortex center_ (EVC) points. Since a vortex can be intuitively defined as a collection of fluid particles coherently rotating around a common axis, clusters of EVC points will appear in the center of coherently rotating flows.  
To accurately estimate EVC points, and therefore allow for a robust identification of vortices, we employ the Rortex mathematical criterion ([Tian et al., 2018](https://ui.adsabs.harvard.edu/abs/2018JFM...849..312T/abstract); [Liu et al., 2018](https://ui.adsabs.harvard.edu/abs/2018PhFl...30c5103L/abstract)) and the properties of the input velocity field. The clustering is then performed with a grid- and vortex-adapted version of the _Clustering by fast search and find of density peaks_ (CFSFDP) algorithm [Rodriguez and Laio, 2014](https://ui.adsabs.harvard.edu/abs/2014Sci...344.1492R/abstract).  
The algorithm is implemented in the python `SWIRL` code and it has been tested on (noisy) artificial vortex flows and on (magneto-)hydrodynamical and turbulent numerical simulations with excellent results. If you are interested in using the algorithm, feel free to contact me at *jcanivete@ics.uzh.ch*.

If you use the SWIRL code in your research, please cite the paper [Canivete Cuissa & Steiner, 2022, A&A (submitted)](...), where more details about the method and the code can be found.

# Dependencies
The `SWIRL` code makes use of the following python libraries:
- numpy (tested with _v. 1.21.6_)
- scipy (tested with _v. 1.4.1_)
- (matplotlib) (tested with _v. 3.3.2_)

# How to install
Download or clone this github repository into your machine. Then, to import the module, run the following lines

```
>>> import sys
>>> sys.path.append(r'\path\to\the\folder')
>>> from swirl import SWIRL
```

Alternatively, you can add the folder where you saved the code to your PYTHONPATH and simply import the SWIRL module. 

# How to run
The imported `SWIRL` object is a python class that contains the methods to run the algorithm.
Given a 2D velocity field saved as a list of two-dimensional numpy arrays, `[vx, vy]`, a `SWIRL` class instance can be initialized as

```
>>> v = SWIRL([vx, vy])
```

and the code can be successively be run with

```
>>> v.run()
```

The output, that is the identified vortices, are stored in a list called `vortices` which is an attribute of the `SWIRL` class. Each element of this list is an identified vortex and it is also an instance of another class object, called `Vortex`. An `Vortex` object contains all the informations relative to an identified vortex, such as:
- the center coordinates `center`, 
- the list of grid cells forming the vortex structure `cells`, 
- the effective radius of the vortex `r`,
- and more ...

# How to run (advanced)

The `SWIRL` class accepts a number of different parameters that one might need to tweak in order to optimize the results. These parameters are
- the grid cell size for results in physical units `dl`,
- the list of multiple grid stencils to be used `l`,
- parameters to use the enhanced version of the swirling strength criterion `S_param`,
- the mathematical criterion to be used in the EVC computation `crit`,   
- the critical distance to be used in the clustering `dc_coeff`,  
- to use or not an adaptive critical distance `dc_adaptive`,  
- to use or not a fast clustering process `fast_clustering`, 
- which kernel to use in the clustering `xi_option`,
- which selection criteria to use in the clustering `clust_selector`, 
- cluster centers selection parameters `clust_options`, 
- noise removal parameter `noise_f`,
- partial rotations removal parameter `kink_f`,
- option to have a verbose report `verbose`

The parameters and their purpose are described in [Canivete Cuissa & Steiner, 2022, A&A (submitted)](...).
