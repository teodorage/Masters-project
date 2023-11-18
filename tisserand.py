"""Code to make Tisserand graphs
"""

from typing import Union
import numpy as np
import matplotlib.pyplot as plt

import util

def tisserand_parameter(a_sc: Union[np.ndarray, float, list],
                        ecc_sc: Union[np.ndarray, float, list],
                        inc_sc: Union[np.ndarray, float, list],
                        a_perturber):
    """Calculate Tisserand's parameter

    Parameters
    ----------
    a_sc : Union[np.ndarray, float, list]
        Semi-major axis of the small body [km].
    ecc_sc : Union[np.ndarray, float, list]
        Eccentricity of the small body.
    inc_sc : Union[np.ndarray, float, list]
        Inclination of the small body's orbit [deg].
    a_perturber : _type_
        Semi-major axis of the perturbing larger body.
    """

    _a = np.array(a_sc)
    _e = np.array(ecc_sc)
    _i = np.array(inc_sc)

    tp = (a_perturber/_a) + 2.0*np.cos(_i)*np.sqrt(_a*(1-_e**2)/a_perturber)

    if len(a_sc)==0:
        return tp[0]
    else:
        return tp



def tisserand_graph_apsides(a_perturber, inclination, vinf_min=0.0, vinf_max=1.0, dvinf=0.1, pump_min=0.0, pump_max=180.0, dpump=15.0, length_units=1.0, **kwargs):
    """Generate a Tisserand graph in the space of apoapsis and periapsis.

    Parameters
    ----------
    a_perturber
        Semimajor axis of the perturbing satellite.
    inclination
        Orbital inclination of the spacecraft.
    vinf_min : float, optional
        Minimum v-infinity in units of the orbital speed of the perturbing satellite, by default 0.0
    vinf_max : float, optional
        Maximum v-infinity in units of the orbital speed of the perturbing satellite, by default 1.0
    dvinf : float, optional
        Delta used to plot vinfinity curves, by default 0.1
    pump_min : float, optional
        Minimum pump angle, by default 0.0 deg
    pump_max : float, optional
        Maximum pump angle, by default 180.0 deg
    dpump : float, optional
        Delta used to plot pump angle curves, by default 15.0 degrees.
    length_units : float, optional
        Units to scale plots into [km].
    """

    # Plot vinfinity curves first.
    pump = np.linspace(0,np.pi,180)
    for vinf in np.arange(vinf_min, vinf_max+dvinf, dvinf):
        a_sc = 1.0/(1.0 - vinf**2 - 2.0*vinf*np.cos(pump))      # eq. 3.12 in Palma
        ecc_sc = np.sqrt(1.0 - (1.0/a_sc)*(0.5*(3.0 - (1.0/a_sc) - vinf**2)/np.cos(np.radians(inclination)))**2)
        ra = util.r_apoapsis(a_sc, ecc_sc)*a_perturber/length_units
        rp = util.r_periapsis(a_sc, ecc_sc)*a_perturber/length_units
        plt.plot(ra[a_sc>0], rp[a_sc>0], '-', **kwargs)

    # Plot pump angle curves next.
    vinf = np.linspace(0, 1.0, 200)
    for pump in np.radians(np.arange(pump_min, pump_max+dpump, dpump)):
        a_sc = 1.0/(1.0 - vinf**2 - 2.0*vinf*np.cos(pump))      # eq. 3.12 in Palma
        ecc_sc = np.sqrt(1.0 - (1.0/a_sc)*(0.5*(3.0 - (1.0/a_sc) - vinf**2)/np.cos(np.radians(inclination)))**2)
        ra = util.r_apoapsis(a_sc, ecc_sc)*a_perturber/length_units
        rp = util.r_periapsis(a_sc, ecc_sc)*a_perturber/length_units
        plt.plot(ra[a_sc>0], rp[a_sc>0], '--', **kwargs)

    # Plot box edges.
    for pump in [0,np.pi]:
        a_sc = 1.0/(1.0 - vinf**2 - 2.0*vinf*np.cos(pump))      # eq. 3.12 in Palma
        ecc_sc = np.sqrt(1.0 - (1.0/a_sc)*(0.5*(3.0 - (1.0/a_sc) - vinf**2)/np.cos(np.radians(inclination)))**2)
        ra = util.r_apoapsis(a_sc, ecc_sc)*a_perturber/length_units
        rp = util.r_periapsis(a_sc, ecc_sc)*a_perturber/length_units
        plt.plot(ra[a_sc>0], rp[a_sc>0], '-', **kwargs)
