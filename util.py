"""Various functions for orbital mechanics."""
import numpy as np

def r_periapsis(a, eccentricity):
    """Calculate radius of periapsis from semimajor axis and eccentricity.
    """
    return a*(1-eccentricity)

def r_apoapsis(a, eccentricty):
    """Calculate radius of apoapsis from semimajor axis and eccentricity.
    """
    return a*(1+eccentricty)

def eccentricity_from_rarp(ra, rp):
    """Calculate eccentricity from radius of apoapsis and periapsis.
    """
    return (ra-rp)/(ra+rp)

def semimajoraxis_from_rarp(ra, rp):
    """Calculate semimajor axis from radius of apoapsis and periapsis.
    """
    return 0.5*(ra+rp)

def period(mu, a):
    """Calculate the period from Kepler's 3rd Law

    Parameters
    ----------
    mu
        Standard gravitational parameter: the units must match those for a, for example, if a is in [km] then mu must be [km^3/s^2].
    a
        Semimajor axis: the units must match those for mu, for example, if mu is in [km^3/s^2] then a must be in [km].
    """
    return 2*np.pi*np.sqrt(a*a*a/mu)

def vis_viva(mu, r, a):
    """Calculate orbital speed from gravitational parameter, orbital radius and semimajor axis.
    """
    return np.sqrt(mu*(2.0/r - 1.0/a))

def constrain_angle_0_360(theta: float) -> float:
    """Constrains and angle to lie within 0 and 2pi radians.

    Parameters
    ----------
    theta : float
        Angle to constrain

    Returns
    -------
    float
        New angle.
    """
    if theta<0.0:
        return (np.floor_divide(np.abs(theta),2*np.pi) + 1)*2*np.pi + theta
    elif theta>2*np.pi:
        return np.remainder(theta, 2*np.pi)
    else:
        return theta

def semimajor_axis(v, r, mu):
    
    return 1/(2/r-(v**2/mu))