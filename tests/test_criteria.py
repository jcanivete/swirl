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
        swirling_str_true = np.zeros((nx,ny))
        vorticity_true = np.zeros((nx,ny))
        rortex_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], swirling_str_true)
        np.testing.assert_array_equal(swirl.W[0], vorticity_true)
        np.testing.assert_array_equal(swirl.R[0], rortex_true)

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
        swirling_str_true = np.zeros((nx,ny))
        vorticity_true = np.zeros((nx,ny))
        rortex_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], swirling_str_true)
        np.testing.assert_array_equal(swirl.W[0], vorticity_true)
        np.testing.assert_array_equal(swirl.R[0], rortex_true)

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
        vorticity_true = np.zeros((nx,ny))
        vorticity_true[1:-1,1:-1] = -v_norm  # Only internal part has non-zero value
        swirling_str_true = np.zeros((nx,ny))
        rortex_true = np.zeros((nx,ny))
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities
        swirl.vorticity()
        swirl.swirlingstrength()
        swirl.rortex()
        # Testing
        np.testing.assert_array_equal(swirl.S[0], swirling_str_true)
        np.testing.assert_array_equal(swirl.W[0], vorticity_true)
        np.testing.assert_array_equal(swirl.R[0], rortex_true)
