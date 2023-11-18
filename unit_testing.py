import unittest

import numpy as np

import constants
import kepler
import util
import propagation

class TestUtil(unittest.TestCase):
    def test_angle_constraints(self):
        self.assertAlmostEqual(util.constrain_angle_0_360(-0.1*np.pi),2*np.pi - 0.1*np.pi)
        self.assertAlmostEqual(util.constrain_angle_0_360(-2.1*np.pi),2*np.pi - 0.1*np.pi)
        self.assertAlmostEqual(util.constrain_angle_0_360(0.1*np.pi),0.1*np.pi)
        self.assertAlmostEqual(util.constrain_angle_0_360(15.1*np.pi),1.1*np.pi)

class TestKepler(unittest.TestCase):
    def test_kepler_solver(self):

        # Circular orbit test - eccentric anomaly should be the same as the mean anomaly.
        solver = kepler.BasicKeplersEquationSolver()
        self.assertAlmostEqual(solver(0.0, 0.0), 0.0, places=14)
        self.assertAlmostEqual(solver(0.5*np.pi, 0.0), 0.5*np.pi, places=14)
        self.assertAlmostEqual(solver(np.pi, 0.0), np.pi, places=14)
        self.assertAlmostEqual(solver(1.5*np.pi, 0.0), 1.5*np.pi, places=14)
        self.assertAlmostEqual(solver(2.0*np.pi, 0.0), 2.0*np.pi, places=14)

        # Eccentric anomaly - calculate the mean anomaly for a range of eccentric anomalies
        # then solve to try to recover the eccentric anomaly.
        for eccentricity in np.linspace(0,0.9,9):
            for eccentric_anomaly in np.linspace(0,2*np.pi,18):
                mean_anomaly = eccentric_anomaly - eccentricity*np.sin(eccentric_anomaly)
                solved_eccentric_anomaly = solver(mean_anomaly, eccentricity)
                self.assertAlmostEqual(eccentric_anomaly, solved_eccentric_anomaly, 6)

        # Test for failed convergence.
        solver = kepler.BasicKeplersEquationSolver(relative_change_epsilon=1e-300, max_iter=2)
        self.assertRaises(kepler.KeplersSolverDidNotConverge, lambda: solver(1.23311313, 0.9))


class TestPropagator(unittest.TestCase):
    def test_simple_circular_orbit(self):
        # Circular orbit with zero inclination.
        r = 100000.0
        v = np.sqrt(constants.MU_JUPITER/r)
        state = [r, 0.0, 0.0, 0.0, v, 0.0]      # Orbit starts out at +X moving in +Y.
        a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = propagation.state_to_elements(state, constants.MU_JUPITER)
        self.assertAlmostEqual(a, r, 8)
        self.assertAlmostEqual(eccentricity, 0.0, 8)
        self.assertAlmostEqual(inclination, 0.0, 8)

    def test_ganymede(self):
        # Test Ganymede orbit - these all need adjustment so that they can cope with the ranges of angles - e.g., the mean anomaly.
        ganymede = kepler.KeplerianEllipticalOrbit(constants.A_GANYMEDE, constants.E_GANYMEDE, constants.I_GANYMEDE, constants.LAN_GANYMEDE, constants.ARGPERI_GANYMEDE, constants.M0_GANYMEDE, constants.EPOCH, constants.MU_JUPITER, kepler_solver=kepler.BasicKeplersEquationSolver(relative_change_epsilon=1e-12))
        state = ganymede(constants.EPOCH)
        a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = propagation.state_to_elements(state, constants.MU_JUPITER)
        self.assertAlmostEqual(a, constants.A_GANYMEDE, 8)
        self.assertAlmostEqual(eccentricity, constants.E_GANYMEDE, 8)
        self.assertAlmostEqual(inclination, np.radians(constants.I_GANYMEDE), 8)
        self.assertAlmostEqual(long_asc_node, np.radians(constants.LAN_GANYMEDE), 8)
        self.assertAlmostEqual(arg_periapsis, np.radians(constants.ARGPERI_GANYMEDE), 8)
        self.assertAlmostEqual(true_anomaly0, ganymede.true_anomaly, 8)
        self.assertAlmostEqual(eccentric_anomaly0, ganymede.eccentric_anomaly, 8)
        self.assertAlmostEqual(mean_anomaly0, ganymede.mean_anomaly, 8)

        state = ganymede(constants.EPOCH+86400.0)
        a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = propagation.state_to_elements(state, constants.MU_JUPITER)
        self.assertAlmostEqual(a, constants.A_GANYMEDE, 8)
        self.assertAlmostEqual(eccentricity, constants.E_GANYMEDE, 8)
        self.assertAlmostEqual(inclination, np.radians(constants.I_GANYMEDE), 8)
        self.assertAlmostEqual(long_asc_node, np.radians(constants.LAN_GANYMEDE), 8)
        self.assertAlmostEqual(arg_periapsis, np.radians(constants.ARGPERI_GANYMEDE), 8)
        self.assertAlmostEqual(true_anomaly0, ganymede.true_anomaly, 8)
        self.assertAlmostEqual(eccentric_anomaly0, ganymede.eccentric_anomaly, 8)
        self.assertAlmostEqual(mean_anomaly0, ganymede.mean_anomaly, 8)

        state = ganymede(constants.EPOCH+2*86400.0)
        a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = propagation.state_to_elements(state, constants.MU_JUPITER)
        self.assertAlmostEqual(a, constants.A_GANYMEDE, 8)
        self.assertAlmostEqual(eccentricity, constants.E_GANYMEDE, 8)
        self.assertAlmostEqual(inclination, np.radians(constants.I_GANYMEDE), 8)
        self.assertAlmostEqual(long_asc_node, np.radians(constants.LAN_GANYMEDE), 8)
        self.assertAlmostEqual(arg_periapsis, np.radians(constants.ARGPERI_GANYMEDE), 8)
        self.assertAlmostEqual(true_anomaly0, ganymede.true_anomaly, 8)
        self.assertAlmostEqual(eccentric_anomaly0, ganymede.eccentric_anomaly, 8)
        self.assertAlmostEqual(mean_anomaly0, ganymede.mean_anomaly, 8)

if __name__=="__main__":
    unittest.main()
