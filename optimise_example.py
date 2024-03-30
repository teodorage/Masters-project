import constants

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

from general_functions_chris import determine_orbit

def cost_fun_only_eccentricity(param):
    """This cost function only returns the eccentricity of the last arc.

    The param argument is a list with the deg_r and deg_v angles in degrees.
    It should return the cost.
    """
    t_start = 58849.0*86400.0

    delta = np.radians(param[0])
    theta = np.radians(param[1])
    r = [1000*constants.R_JUPITER  * np.cos(delta), 1000*constants.R_JUPITER  * np.sin(delta), 0]
    v = [3.4 * np.cos(delta+ theta), 3.4 * np.sin(delta + theta), 0]

    arcs = determine_orbit(r,v,t_start,3,0,{})

    cost = arcs[max(arcs.keys())]["eccentricity"]

    print("deg_r={:8.6f} deg_v={:8.6f} cal_ga={} cal_alt={:7f} a={:8f} ecc={:4.3f} cost={}".format(param[0], param[1], arcs[0]["Callisto GA"], arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO, arcs[max(arcs.keys())]["semimajor axis"], arcs[max(arcs.keys())]["eccentricity"],cost))

    return cost

def cost_fun_only_cal_altitude(param):
    """This cost function only returns the altitude of the callisto closest approach for the first arc.

    The param argument is a list with the deg_r and deg_v angles in degrees.
    It should return the cost.
    """
    t_start = 58849.0*86400.0

    delta = np.radians(param[0])
    theta = np.radians(param[1])
    r = [1000*constants.R_JUPITER  * np.cos(delta), 1000*constants.R_JUPITER  * np.sin(delta), 0]
    v = [3.4 * np.cos(delta+ theta), 3.4 * np.sin(delta + theta), 0]

    arcs = determine_orbit(r,v,t_start,3,0,{})

    cost = arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO

    print("deg_r={:8.6f} deg_v={:8.6f} cal_ga={} cal_alt={:7f} a={:8f} ecc={:4.3f} cost={}".format(param[0], param[1], arcs[0]["Callisto GA"], arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO, arcs[max(arcs.keys())]["semimajor axis"], arcs[max(arcs.keys())]["eccentricity"],cost))

    return cost


def cost_fun_cal_altitude_with_penalties(param):
    """This cost function is calculated from the callisto closest approach altitude.

    This tries to account for the fact that we want a flyby >50 km and <2000 km altitude.

    The param argument is a list with the deg_r and deg_v angles in degrees.
    It should return the cost.
    """
    t_start = 58849.0*86400.0

    delta = np.radians(param[0])
    theta = np.radians(param[1])
    r = [1000*constants.R_JUPITER  * np.cos(delta), 1000*constants.R_JUPITER  * np.sin(delta), 0]
    v = [3.4 * np.cos(delta+ theta), 3.4 * np.sin(delta + theta), 0]

    arcs = determine_orbit(r,v,t_start,3,0,{})

    cost = np.exp((arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO)/2000) + \
            np.exp(-(arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO)/50)

    print("deg_r={:8.6f} deg_v={:8.6f} cal_ga={} cal_alt={:7f} a={:8f} ecc={:4.3f} cost={}".format(param[0], param[1], arcs[0]["Callisto GA"], arcs[0]["closest distance to callisto [km]"]-constants.R_CALLISTO, arcs[max(arcs.keys())]["semimajor axis"], arcs[max(arcs.keys())]["eccentricity"],cost))

    return cost




#scipy.optimize.minimize(cost_fun_only_cal_altitude, [49.0, 175.25], method="powell", bounds=((49, 55), (175, 177)))
scipy.optimize.minimize(cost_fun_cal_altitude_with_penalties, [49.0, 175.25], method="powell", bounds=((49, 50), (175, 175.5)))
