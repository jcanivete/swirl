"""
test_criteria.py

José Roberto Canivete Cuissa
IRSOL, 15.08.2022

----------------

Unit test module for testing the implementation of mathematical criteria
in the SWIRL code.
"""
# Imports
import unittest
import numpy as np
from swirl import SWIRL

class CriteriaTests(unittest.TestCase):
    """
    Test cases to check the correctness of the mathematical criteria
    involved in the SWIRL algorithm.
    """
    @staticmethod
    def test_zero_velocity():
        """
        Test vorticity, swirling strength, and rortex with a uniform
        zero velocity field
        """
        # Size of grid
        nx, ny = 10, 10
        # Velocity field
        vx, vy = np.zeros((nx,ny)), np.zeros((nx,ny))
        # Expected results
        sstr_true = np.zeros((nx,ny))
        vort_true = np.zeros((nx,ny))
        rort_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], sstr_true)
        np.testing.assert_array_equal(swirl.W[0], vort_true)
        np.testing.assert_array_equal(swirl.R[0], rort_true)

    #----------------------------------------------
    @staticmethod
    def test_const_velocity():
        """
        Test vorticity, swirling strength, and rortex with a uniform
        non-zero velocity field
        """
        # Size of grid
        nx, ny = 10, 10
        # Velocity field
        v_norm = 5.0
        vx = v_norm*np.ones((nx,ny))
        vy = v_norm*np.ones((nx,ny))
        # Expected results
        sstr_true = np.zeros((nx,ny))
        vort_true = np.zeros((nx,ny))
        rort_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], sstr_true)
        np.testing.assert_array_equal(swirl.W[0], vort_true)
        np.testing.assert_array_equal(swirl.R[0], rort_true)

    #----------------------------------------------
    @staticmethod
    def test_shear_velocity():
        """
        Test vorticity, swirling strength and rortex with a shear
        velocity field.
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx),np.arange(ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        v_norm = 5.0
        vx = v_norm*ygrid
        vy = np.zeros((nx,ny))
        # Expected results
        vort_true = np.zeros((nx,ny))
        vort_true[1:-1,1:-1] = -v_norm  # Only internal part has non-zero value
        sstr_true = np.zeros((nx,ny))
        rort_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], sstr_true)
        np.testing.assert_array_equal(swirl.W[0], vort_true)
        np.testing.assert_array_equal(swirl.R[0], rort_true)

    #----------------------------------------------
    @staticmethod
    def test_rotational_vortex():
        """
        Test vorticity, swirling strength and rortex with a rotational
        vortex.
        Rotational vortex:  vx = -a*y
                            vy =  a*x
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx),np.arange(ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        alpha = 5.0
        vx = -alpha*ygrid
        vy = alpha*xgrid
        # Expected results
        vort_true = np.zeros((nx,ny))
        vort_true[1:-1,1:-1] = 2.*alpha  # Only internal part has non-zero value
        sstr_true = vort_true
        rort_true = vort_true
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_almost_equal(swirl.S[0], sstr_true)
        np.testing.assert_array_almost_equal(swirl.W[0], vort_true)
        np.testing.assert_array_almost_equal(swirl.R[0], rort_true)

    #----------------------------------------------
    @staticmethod
    def test_lamb_oseen_vortex():
        """
        Test vorticity, swirling strength and rortex with a Lamb Oseen
        vortex.
        Lamb Oseen vortex:  vx = -y*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
                            vy =  x*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        The expected values are stored in the file lamb_oseen.npy as an [W, S, R]
        """
        # Size of grid
        nx, ny = 20, 20
        # Grid
        dx = 1./nx
        dy = 1./ny
        xrange = np.arange(-.5,.5+dx,dx)
        yrange = np.arange(-.5,.5+dy,dy)
        xgrid, ygrid = np.meshgrid(xrange,yrange)
        xgrid = xgrid.T
        ygrid = ygrid.T
        # Velocity field
        alpha = 1.234
        vmax = 1.0
        rmax = 0.3
        # Lamb-Oleen
        r = np.sqrt(xgrid**2 + ygrid**2)
        vx = -ygrid*rmax*vmax/r**2*(1 + 0.5/alpha)*(1 - np.exp(-alpha*r**2/(rmax**2)))
        vy = xgrid*rmax*vmax/r**2*(1 + 0.5/alpha)*(1 - np.exp(-alpha*r**2/(rmax**2)))
        # Expected results
        true_arrays = np.load('lamb_oseen.npy')
        vort_true = true_arrays[0]
        sstr_true = true_arrays[1]
        rort_true = true_arrays[2]
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [dx, dy],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_almost_equal(swirl.S[0], sstr_true)
        np.testing.assert_array_almost_equal(swirl.W[0], vort_true)
        np.testing.assert_array_almost_equal(swirl.R[0], rort_true)
