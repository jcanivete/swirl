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
        # Size of grid
        Nx, Ny = 10, 10  
        # Velocity field
        vx, vy = np.zeros((Nx,Ny)), np.zeros((Nx,Ny)) 
        # Expected results
        S_true = np.zeros((Nx,Ny))
        W_true = np.zeros((Nx,Ny))
        R_true = np.zeros((Nx,Ny))
        # SWIRL instance
        s = SWIRL(v = [vx, vy], 
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities 
        s.vorticity()
        s.swirlingstrength()
        s.rortex()
        # Testing
        np.testing.assert_array_equal(s.S[0], S_true)
        np.testing.assert_array_equal(s.W[0], W_true)
        np.testing.assert_array_equal(s.R[0], R_true)

    #----------------------------------------------
    @staticmethod
    def test_const_velocity():
        """
        Test vorticity, swirling strength, and rortex with a uniform
        non-zero velocity field
        """
        # Size of grid
        Nx, Ny = 10, 10  
        # Velocity field
        v_norm = 5.0
        vx = v_norm*np.ones((Nx,Ny)) 
        vy = v_norm*np.ones((Nx,Ny)) 
        # Expected results
        S_true = np.zeros((Nx,Ny)) 
        W_true = np.zeros((Nx,Ny))
        R_true = np.zeros((Nx,Ny))
        # SWIRL instance
        s = SWIRL(v = [vx, vy], 
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities 
        s.vorticity()
        s.swirlingstrength()
        s.rortex()
        # Testing
        np.testing.assert_array_equal(s.S[0], S_true)
        np.testing.assert_array_equal(s.W[0], W_true)
        np.testing.assert_array_equal(s.R[0], R_true)

    #----------------------------------------------
    @staticmethod
    def test_shear_velocity():
        """
        Test vorticity, swirling strength and rortex with a shear
        velocity field.
        """
        # Size of grid
        Nx, Ny = 10, 10  
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(Nx),np.arange(Ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        v_norm = 5.0
        vx = v_norm*ygrid 
        vy = np.zeros((Nx,Ny)) 
        # Expected results
        W_true = np.zeros((Nx,Ny))
        W_true[1:-1,1:-1] = -v_norm  # Only internal part has non-zero value
        S_true = np.zeros((Nx,Ny)) 
        R_true = np.zeros((Nx,Ny))
        # SWIRL instance
        s = SWIRL(v = [vx, vy], 
                  dl = [1.0, 1.0],
                  verbose=False)
        # Computing quantities 
        s.vorticity()
        s.swirlingstrength()
        s.rortex()
        # Testing
        np.testing.assert_array_equal(s.S[0], S_true)
        np.testing.assert_array_equal(s.W[0], W_true)
        np.testing.assert_array_equal(s.R[0], R_true)