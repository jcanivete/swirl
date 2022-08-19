"""
SWIRL Code
    swirl_test.py

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
    Test cases to check the correctness of the mathematical criteria involved
    in the SWIRL algorithm.
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
        vx, vy = np.zeros((nx, ny)), np.zeros((nx, ny))
        # Expected results
        sstr_true = np.zeros((nx, ny))
        vort_true = np.zeros((nx, ny))
        rort_true = np.zeros((nx, ny))
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Computing quantities
        swirl.vorticity
        swirl.swirling_str
        swirl.rortex
        # Testing
        np.testing.assert_array_equal(swirl.swirling_str[0], sstr_true)
        np.testing.assert_array_equal(swirl.vorticity[0], vort_true)
        np.testing.assert_array_equal(swirl.rortex[0], rort_true)

    # ----------------------------------------------
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
        vx = v_norm*np.ones((nx, ny))
        vy = v_norm*np.ones((nx, ny))
        # Expected results
        sstr_true = np.zeros((nx, ny))
        vort_true = np.zeros((nx, ny))
        rort_true = np.zeros((nx, ny))
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Computing quantities
        swirl.vorticity
        swirl.swirling_str
        swirl.rortex
        # Testing
        np.testing.assert_array_equal(swirl.swirling_str[0], sstr_true)
        np.testing.assert_array_equal(swirl.vorticity[0], vort_true)
        np.testing.assert_array_equal(swirl.rortex[0], rort_true)

    # ----------------------------------------------
    @staticmethod
    def test_test_shear_flow():
        """
        Test vorticity, swirling strength and rortex with a shear flow.
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        v_norm = 5.0
        vx = v_norm*ygrid
        vy = np.zeros((nx, ny))
        # Expected results
        vort_true = np.zeros((nx, ny))
        # Only internal part has non-zero value
        vort_true[1:-1, 1:-1] = -v_norm
        sstr_true = np.zeros((nx, ny))
        rort_true = np.zeros((nx, ny))
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Computing quantities
        swirl.vorticity
        swirl.swirling_str
        swirl.rortex
        # Testing
        np.testing.assert_array_equal(swirl.swirling_str[0], sstr_true)
        np.testing.assert_array_equal(swirl.vorticity[0], vort_true)
        np.testing.assert_array_equal(swirl.rortex[0], rort_true)

    # ----------------------------------------------
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
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        alpha = 5.0
        vx = -alpha*ygrid
        vy = alpha*xgrid
        # Expected results
        vort_true = np.zeros((nx, ny))
        # Only internal part has non-zero value
        vort_true[1:-1, 1:-1] = 2.*alpha
        sstr_true = vort_true
        rort_true = vort_true
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Computing quantities
        swirl.vorticity
        swirl.swirling_str
        swirl.rortex
        # Testing
        np.testing.assert_array_almost_equal(swirl.swirling_str[0], sstr_true)
        np.testing.assert_array_almost_equal(swirl.vorticity[0], vort_true)
        np.testing.assert_array_almost_equal(swirl.rortex[0], rort_true)

    # ----------------------------------------------
    @staticmethod
    def test_lamb_oseen_vortex():
        """
        Test vorticity, swirling strength and rortex with a Lamb Oseen
        vortex.
        Lamb Oseen vortex:
        vx = -y*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        vy =  x*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        The expected values are stored in the file lamb_oseen.npy as:
        array = [W, S, R]
        """
        # Size of grid
        nx, ny = 20, 20
        # Grid
        dx = 1./nx
        dy = 1./ny
        xrange = np.arange(-.5, .5+dx, dx)
        yrange = np.arange(-.5, .5+dy, dy)
        xgrid, ygrid = np.meshgrid(xrange, yrange)
        xgrid = xgrid.T
        ygrid = ygrid.T
        # Velocity field
        alpha = 1.234
        vmax = 1.0
        rmax = 0.3
        # Lamb-Oleen
        r = np.sqrt(xgrid**2 + ygrid**2)
        vx = -ygrid*rmax*vmax/r**2 * \
            (1 + 0.5/alpha)*(1 - np.exp(-alpha*r**2/(rmax**2)))
        vy = xgrid*rmax*vmax/r**2*(1 + 0.5/alpha) * \
            (1 - np.exp(-alpha*r**2/(rmax**2)))
        # Expected results
        true_arrays = np.load('lamb_oseen.npy')
        vort_true = true_arrays[0]
        sstr_true = true_arrays[1]
        rort_true = true_arrays[2]
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx=[dx, dy],
                      verbose=False
                      )
        # Computing quantities
        swirl.vorticity
        swirl.swirling_str
        swirl.rortex
        # Testing
        np.testing.assert_array_almost_equal(swirl.swirling_str[0], sstr_true)
        np.testing.assert_array_almost_equal(swirl.vorticity[0], vort_true)
        np.testing.assert_array_almost_equal(swirl.rortex[0], rort_true)

# ----------------------------------------------


class IdentificationTests(unittest.TestCase):
    """
    Test cases to check the correctness of the identification process
    involved in the SWIRL algorithm.
    """

    def test_zero_velocity(self):
        """
        Test identification algorithm with a uniform zero velocity field
        """
        # Size of grid
        nx, ny = 10, 10
        # Velocity field
        vx, vy = np.zeros((nx, ny)), np.zeros((nx, ny))
        # Expected results
        n_vortices_true = 0
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_const_velocity(self):
        """
        Test identification algorithm with a uniform constant velocity field
        """
        # Size of grid
        nx, ny = 10, 10
        # Velocity field
        alpha = 5.0
        vx = alpha*np.ones((nx, ny))
        vy = alpha*np.ones((nx, ny))
        # Expected results
        n_vortices_true = 0
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_shear_flow(self):
        """
        Test identification algorithm with a shear flow
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        xgrid[...] = xgrid[...].T
        ygrid[...] = ygrid[...].T
        # Velocity field
        v_norm = 5.0
        vx = v_norm*ygrid
        vy = np.zeros((nx, ny))
        # Expected results
        n_vortices_true = 0
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_rotational_vortex(self):
        """
        Test identification algorithm with a with a rotational
        vortex.
        Rotational vortex:  vx = -a*y
                            vy =  a*x
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx), np.arange(ny))
        xgrid[...] = xgrid[...].T - nx/2
        ygrid[...] = ygrid[...].T - ny/2
        # Velocity field
        alpha = 5.0
        vx = -alpha*ygrid
        vy = alpha*xgrid
        # Expected results
        n_vortices_true = 1
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_lamb_oseen_vortex(self):
        """
        Test identification algorithm with a with a Lamb Oseen
        vortex.
        Lamb Oseen vortex:
        vx = -y*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        vy =  x*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        """
        # Size of grid
        nx, ny = 20, 20
        # Grid
        dx = 1./nx
        dy = 1./ny
        xrange = np.arange(-.5, .5+dx, dx)
        yrange = np.arange(-.5, .5+dy, dy)
        xgrid, ygrid = np.meshgrid(xrange, yrange)
        xgrid = xgrid.T
        ygrid = ygrid.T
        # Velocity field
        alpha = 1.234
        vmax = 1.0
        rmax = 0.3
        # Lamb-Oleen
        r = np.sqrt(xgrid**2 + ygrid**2)
        vx = -ygrid*rmax*vmax/r**2 * \
            (1 + 0.5/alpha)*(1 - np.exp(-alpha*r**2/(rmax**2)))
        vy = xgrid*rmax*vmax/r**2*(1 + 0.5/alpha) * \
            (1 - np.exp(-alpha*r**2/(rmax**2)))
        # Expected results
        n_vortices_true = 1
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx=[dx, dy],
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_double_lamb_oseen_vortex(self):
        """
        Test identification algorithm with the double Lamb Oseen
        vortex.
        Lamb Oseen vortex:
        vx = -y*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        vy =  x*rmax*vmax/(r^2*(1 + 0.5/alpha))*(1 - exp(-alpha*r^2/rmax^2))
        """
        # Load velocity fields
        data_dir = '../data/lamb_oseen/'
        vx = np.load(data_dir+'vx.npy')
        vy = np.load(data_dir+'vy.npy')
        # Expected results
        n_vortices_true = 2
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      param_file=data_dir+'lamb_oseen.param',
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

    # ----------------------------------------------
    def test_multiple_vortices(self):
        """
        Test the identification on multiple swirls
        """
        # Load velocity fields
        data_dir = '../data/multiple_vortices/'
        vx = np.load(data_dir+'vx.npy')
        vy = np.load(data_dir+'vy.npy')
        dx = 1.0/200.
        # Expected results
        n_vortices_true = 9
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx=[dx, dx],
                      param_file=data_dir+'multiple_vortices.param',
                      verbose=False
                      )
        # Run algorithm
        swirl.run()
        # Testing
        self.assertEqual(len(swirl.vortices), n_vortices_true)

# ----------------------------------------------


class EVCMapTests(unittest.TestCase):
    """
    Test cases to check the correctness of the EVC map in the SWIRL algorithm.
    """
    @staticmethod
    def test_rotational_vortex():
        """
        Test the EVC map resulting from a simple rotational swirl
        """
        # Size of grid
        nx, ny = 10, 10
        # Grid
        xgrid, ygrid = np.meshgrid(np.arange(nx+1), np.arange(ny+1))
        xgrid[...] = xgrid[...].T - 5.0
        ygrid[...] = ygrid[...].T - 5.0
        # Velocity field
        alpha = 5.0
        vx = -alpha*ygrid
        vy = alpha*xgrid
        # Expected results
        evc_map_true = np.array([[5.], [5.], [81.]])
        # SWIRL instance
        swirl = SWIRL(v=[vx, vy],
                      grid_dx = [1.0, 1.0],
                      verbose=False
                      )
        # Testing
        swirl.run()
        np.testing.assert_array_equal(swirl.M, evc_map_true)


# ---------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
