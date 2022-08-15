"""
test_identification.py

José Roberto Canivete Cuissa
IRSOL, 15.08.2022

----------------

Unit test module for testing the implementation of automated
vortex identification algorithm in the SWIRL code.
"""
# Imports
import unittest
import numpy as np
from swirl import SWIRL

class CriteriaTests(unittest.TestCase):
    """
    Test cases to check the correctness of the identification algorithm
    involved in the SWIRL algorithm.
    """
    def test_zero_velocity(self):
        """
        Test identification algorithm with a uniform zero velocity field
        """
        # Size of grid
        nx, ny = 10, 10
        # Velocity field
        vx, vy = np.zeros((nx,ny)), np.zeros((nx,ny))
        # Expected results
        n_vortices_true = 0
        # SWIRL instance
        swirl = SWIRL(v = [vx, vy],
                  dl = [1.0, 1.0],
                  verbose=False)
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)
