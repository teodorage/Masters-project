"""Propagators
"""
from typing import Union, Tuple

import numpy as np
import matplotlib.pyplot as plt

import kepler
import constants
import util

def state_to_elements(state: Union[list,tuple,np.ndarray], mu: float) -> (float, float, float, float, float, float):
    """"""

    # https://en.wikipedia.org/wiki/Orbit_determination
    r = np.array(state[0:3])
    v = np.array(state[3:])

    # Compute specific angular momentum.
    v_specific_angular_momentum = np.cross(r, v)

    # Compute ascending node vector - this points in the direction of the ascending node.
    v_ascending_node = np.cross([0,0,1.0], v_specific_angular_momentum)

    # Compute the eccentricity vector - the magnitude is the eccentricity
    # and the direction points in the direction of periapsis.
    v_eccentricity = np.cross(v, v_specific_angular_momentum)/mu - r/np.sqrt(r.dot(r))

    # Compute semi-latus rectum, p=h^2/mu=a(1-e^2) and hence the semi-major axis since we can
    # already calculate eccentricity = |v_eccentricity|
    # NOTE: there is no checking that |v_eccentricity|==1, if it does then we'll get a divide-by-zero.
    semilatus = v_specific_angular_momentum.dot(v_specific_angular_momentum)/mu
    a = semilatus/(1-np.dot(v_eccentricity,v_eccentricity))

    # Compute inclination of the orbital plane from the angle between the specific angular momentum
    # and the z-axis (zero inclination => they will be parallel).
    # NOTE: there is no checking that the inclination is 0<=i<=180
    inclination = np.arccos(np.dot([0,0,1.0], v_specific_angular_momentum)/np.sqrt(v_specific_angular_momentum.dot(v_specific_angular_momentum)))

    # Compute the longitude of the ascending node which is the angle between v_ascending node
    # and the x-axis.
    long_asc_node = np.arctan2(v_ascending_node[1], v_ascending_node[0])

    # Calculate the argument of periapsis which is the angle between the ascending node vector
    # and the eccentricity vector.  We check that the eccentricity vector is not zero before
    # trying to calculate it explicitly, otherwise we set to zero.
    if v_eccentricity.dot(v_eccentricity)<1e-13:
        arg_periapsis = 0.0
    else:
        arg_periapsis = np.arccos(np.dot(v_ascending_node, v_eccentricity)/(np.sqrt(v_ascending_node.dot(v_ascending_node)*v_eccentricity.dot(v_eccentricity))))
        if v_eccentricity[2]<0.0:
            arg_periapsis = -arg_periapsis

    # Calculate the true anomaly at epoch, the angle between the position vector and periapsis at this time.
    eccentricity = np.sqrt(v_eccentricity.dot(v_eccentricity))
    true_anomaly0 = np.arccos(np.dot(v_eccentricity, r)/(np.sqrt(v_eccentricity.dot(v_eccentricity)*r.dot(r))))*np.sign(r.dot(v))
    eccentric_anomaly0 = util.constrain_angle_0_360(np.arctan(np.tan(0.5*true_anomaly0)*np.sqrt((1-eccentricity)/(1+eccentricity)))*2.0)
    mean_anomaly0 = util.constrain_angle_0_360(eccentric_anomaly0 - eccentricity*np.sin(eccentric_anomaly0))

    return a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0



def simple_propagate(state: Union[list,tuple,np.ndarray], t0: float, t: float, mu: float, kepler_solver=kepler.BasicKeplersEquationSolver(relative_change_epsilon=1e-8)):
    # Get the orbital elements.
    a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = state_to_elements(state, mu)

    # Compute the mean anomaly
    period = util.period(mu, a)
    mean_motion = 2*np.pi/period
    mean_anomaly = util.constrain_angle_0_360(mean_motion*(t-t0) + mean_anomaly0)

    # Solve Kepler's equation and compute the true anomaly and flight path angle.
    eccentric_anomaly = kepler_solver(mean_anomaly, eccentricity)
    true_anomaly = 2.0*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(eccentric_anomaly*0.5))
    flight_path_angle = np.arctan2(eccentricity*np.sin(true_anomaly), 1.0+eccentricity*np.cos(true_anomaly))

    # Compute the distance from the focus and the speed.
    r = a*(1-eccentricity*np.cos(eccentric_anomaly))
    v = util.vis_viva(mu, r, a)

    # Two angles used in computing the position and velocity.
    pos_angle = true_anomaly + arg_periapsis
    vel_angle = pos_angle - flight_path_angle

    # Figure out the new state
    new_state = [r*(np.cos(pos_angle)*np.cos(long_asc_node) - np.sin(pos_angle)*np.cos(inclination)*np.sin(long_asc_node)),
        r*(np.cos(pos_angle)*np.sin(long_asc_node) + np.sin(pos_angle)*np.cos(inclination)*np.cos(long_asc_node)),
        r*np.sin(pos_angle)*np.sin(inclination),
        v*(-np.sin(vel_angle)*np.cos(long_asc_node) - np.cos(vel_angle)*np.cos(inclination)*np.sin(long_asc_node)),
        v*(-np.sin(vel_angle)*np.sin(long_asc_node) + np.cos(vel_angle)*np.cos(inclination)*np.cos(long_asc_node)),
        v*np.cos(vel_angle)*np.sin(inclination)]

    return np.array(new_state)

if __name__=="__main__":
    ganymede = kepler.KeplerianEllipticalOrbit(constants.A_GANYMEDE, constants.E_GANYMEDE, constants.I_GANYMEDE, constants.LAN_GANYMEDE, constants.ARGPERI_GANYMEDE, constants.M0_GANYMEDE, constants.EPOCH, constants.MU_JUPITER)
    state0 = ganymede(constants.EPOCH+1.0*86400.0)
    a, eccentricity, inclination, long_asc_node, arg_periapsis, true_anomaly0, eccentric_anomaly0, mean_anomaly0 = state_to_elements(state0, constants.MU_JUPITER)
    print(a, constants.A_GANYMEDE)
    print(eccentricity, constants.E_GANYMEDE)
    print(inclination, np.radians(constants.I_GANYMEDE))
    print(long_asc_node, np.radians(constants.LAN_GANYMEDE))
    print(arg_periapsis, np.radians(constants.ARGPERI_GANYMEDE))
    print(true_anomaly0, ganymede.true_anomaly)
    print(eccentric_anomaly0, ganymede.eccentric_anomaly)
    print(mean_anomaly0, ganymede.mean_anomaly)

    state1 = ganymede(constants.EPOCH+14.0*86400.0)
    prop_state1 = simple_propagate(state0, constants.EPOCH+1.0*86400.0, constants.EPOCH+14.0*86400.0, constants.MU_JUPITER)
    print(state1-prop_state1)
