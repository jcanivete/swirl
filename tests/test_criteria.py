# Imports
import numpy as np
import unittest
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
        Nx, Ny = 10, 10  # Size of grid
        vx, vy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny)) # Velocity field
        S_true = np.zeros((Nx,Ny))
        W_true = np.zeros((Nx,Ny))
        R_true = np.zeros((Nx,Ny))

        s = SWIRL(v = [vx, vy], 
                  dl = [1.0, 1.0],
                  verbose=False)
            
        s.vorticity()
        s.swirlingstrength()
        s.rortex()

        np.testing.assert_array_equal(s.S[0], S_true)
        np.testing.assert_array_equal(s.W[0], W_true)
        np.testing.assert_array_equal(s.R[0], R_true)