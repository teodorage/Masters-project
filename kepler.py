"""Basic code for Keplerian orbits"""
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

import util

class KeplersSolverDidNotConverge(Exception):
    """Exception to flag when a Kepler's equation solver didn't converge.
    """

    def __init__(self, num_iter: int, delta_ecc_anomaly: float, relative_change: float, mean_anomaly: float, eccentricity: float, message: str=None):
        """Constructor.

        Parameters
        ----------
        num_iter : int
            Number of iterations reached.
        delta_ecc_anomaly : float
            Last change in the eccentric anomaly [radians].
        relative_change : float
            Relative change at the last step.
        mean_anomaly : float
            Mean anomaly [radians].
        eccentricity : float
            Eccentricity.
        message : str, optional
            Optional additional message, by default None
        """
        super().__init__("Convergence failed at {} iterations with dE={}, relative change={} (called with M={} ecc={}){}".format(
             num_iter, delta_ecc_anomaly, relative_change, mean_anomaly, eccentricity, " ["+message+"]" if message is not None else ""))


class BasicKeplersEquationSolver:
    """Basic class for solving Kepler's equation.

        Uses Newton-Raphson with a simple iteration scheme.
    """

    def __init__(self, relative_change_epsilon: float=1e-6, max_iter: int=100):
        """Constructor

        Parameters
        ----------
        relative_change_epsilon : float, optional
            Threshold for relative change at which point the solution is deemed to be found.  Default is 1e-6.
        max_iter : int, optional
            Maximum number of iterations permitted before reporting a failure to converge, by default 100
        """
        self.relative_change_epsilon = relative_change_epsilon
        self.max_iter = max_iter

    def __call__(self, mean_anomaly: float, eccentricity: float) -> float:
        """Solver Kepler's equation.

        Parameters
        ----------
        mean_anomaly : float
            Mean anomaly at the point we want to solve Kepler's equation for [radians].
        eccentricity : float
            Eccentricity of the orbit.

        Returns
        -------
        float
            Eccentric anomaly [radians]

        Raises
        ------
        KeplersSolverDidNotConverge
            If the solver didn't converge.
        """

        # Initial starting guess for the eccentric anomaly.
        ecc_anomaly = 0.0
        if eccentricity>=0.8:
            ecc_anomaly = np.pi

        # Iterate using Newton-Raphson.
        iter = 0
        relative_change = 1e6
        while (relative_change>self.relative_change_epsilon) and (iter<self.max_iter):

            # Compute the delta to add onto the eccentric anomaly for this step, 
            # work out the new eccentric anomaly estimate, work out the relative
            # change and increment the iteration variable.
            delta_ecc_anomaly = (mean_anomaly + eccentricity*np.sin(ecc_anomaly) - ecc_anomaly)/(1-eccentricity*np.cos(ecc_anomaly))
            ecc_anomaly += delta_ecc_anomaly
            relative_change = np.abs(delta_ecc_anomaly/(ecc_anomaly+1e-300))
            iter += 1

        if iter==self.max_iter:
            raise KeplersSolverDidNotConverge(iter, delta_ecc_anomaly, relative_change, mean_anomaly, eccentricity)

        return ecc_anomaly


class KeplerianEllipticalOrbit:
    """Define Keplerian (elliptical) orbits and use to calculate positions.
    """
    def __init__(self, a: float, eccentricity: float, inclination: float, long_asc_node: float, arg_periapsis: float, mean_anomaly0: float, epoch: float, mu: float, kepler_solver = BasicKeplersEquationSolver(1e-6,100)):
        """Constructor

        The semi-major axis and the standard gravitational parameter must have the same length units.  For example,
        if [a] = km, then [mu]=km^3/s^2.  If [a]=AU, then [mu]=AU^3/s^2.

        Parameters
        ----------
        a : float
            Semi-major axis.  Note that output positions will be in the same units as a.  Must have same length units as mu.
        eccentricity : float
            Eccentricity of the orbit 0 <= e < 1
        inclination : float
            Inclination of the orbit (degrees).
        long_asc_node : float
            Longitude of the ascending node (degrees).
        arg_periapsis : float
            Argument of periapsis (degrees).
        mean_anomaly0 : float
            Mean anomaly at epoch (degrees).
        epoch : float
            Epoch for the elements (seconds).
        mu : float
            Standard Gravitational Parameter of central body.  Must have same length units as a.
        kepler_solver
            Object that provides a call method to solve Kepler's equation.
        """
        self.a = a
        self.eccentricity = eccentricity
        self.inclination = np.radians(inclination)
        self.long_asc_node = np.radians(long_asc_node)
        self.arg_periapsis = np.radians(arg_periapsis)
        self.period = util.period(mu, a)
        self.mean_motion = 2*np.pi/self.period
        self.mean_anomaly0 = np.radians(mean_anomaly0)
        self.epoch = epoch
        self.mu = mu
        self.kepler_solver = kepler_solver


    def calculate_state(self, t: float) -> np.ndarray:
        """Calculate the state at a given time

        Based on section 6.2 in GTOC 6 Problem Description.

        Parameters
        ----------
        t : float
            Time (in seconds) at which we want to calculate the state.

        Returns
        -------
        np.ndarray
            State as a 6 element array containing [x,y,z,vx,vy,vz] in the same
            units as a and mu.
        """

        # Compute the angles.
        self.mean_anomaly = util.constrain_angle_0_360(self.mean_motion*(t-self.epoch) + self.mean_anomaly0)
        self.eccentric_anomaly = self.kepler_solver(self.mean_anomaly, self.eccentricity)
        self.true_anomaly = 2.0*np.arctan(np.sqrt((1+self.eccentricity)/(1-self.eccentricity))*np.tan(self.eccentric_anomaly*0.5))
        self.flight_path_angle = np.arctan2(self.eccentricity*np.sin(self.true_anomaly), 1.0+self.eccentricity*np.cos(self.true_anomaly))

        # Compute the distance from the focus and the speed.
        r = self.a*(1-self.eccentricity*np.cos(self.eccentric_anomaly))
        v = util.vis_viva(self.mu, r, self.a)

        # Two angles used in computing the position and velocity.
        pos_angle = self.true_anomaly + self.arg_periapsis
        vel_angle = pos_angle - self.flight_path_angle

        # Figure out the state.
        s = [r*(np.cos(pos_angle)*np.cos(self.long_asc_node) - np.sin(pos_angle)*np.cos(self.inclination)*np.sin(self.long_asc_node)),
            r*(np.cos(pos_angle)*np.sin(self.long_asc_node) + np.sin(pos_angle)*np.cos(self.inclination)*np.cos(self.long_asc_node)),
            r*np.sin(pos_angle)*np.sin(self.inclination),
            v*(-np.sin(vel_angle)*np.cos(self.long_asc_node) - np.cos(vel_angle)*np.cos(self.inclination)*np.sin(self.long_asc_node)),
            v*(-np.sin(vel_angle)*np.sin(self.long_asc_node) + np.cos(vel_angle)*np.cos(self.inclination)*np.cos(self.long_asc_node)),
            v*np.cos(vel_angle)*np.sin(self.inclination)]

        return s


    def __call__(self, time: Union[float, list, np.ndarray, tuple]) -> np.ndarray:
        """Method to calculate the state for a given time or set of times.

        Parameters
        ----------
        time : Union[float, list, np.ndarray, tuple]
            Time (in seconds) for the time(s) we wish to calculate the state for.

        Returns
        -------
        np.ndarray
            State array with either six elements [x,y,z,vx,vy,vz] or state array
            with N-by-6 elements where N is the number of times given.
        """
        if isinstance(time, (list,np.ndarray,tuple)):
            s = np.zeros((len(time),6))
            for i,this_time in enumerate(time):
                s[i,:] = self.calculate_state(this_time)
        elif isinstance(time, float):
            s = self.calculate_state(time)
        else:
            return TypeError("Expected time to be a float, array, list or tuple, not a {}".format(type(time)))

        return s

