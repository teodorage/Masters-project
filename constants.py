"""These are all the orbital and physics constants for the GTOC6 problem
"""
import numpy as np

##
# Moon radii [km] and standard gravitational parameters [km^3/s^2]
R_IO = 1826.5
MU_IO = 5959.916
R_EUROPA = 1561.0
MU_EUROPA = 3202.739
R_GANYMEDE = 2634.0
MU_GANYMEDE = 9887.834
R_CALLISTO = 2408.0
MU_CALLISTO = 7179.289


##
# Other constants
MU_JUPITER = 126686534.92180        # [km^3/s^2]
R_JUPITER = 71492.0                 # [km]
LITTLE_G = 9.80665                  # [m/s^2]
DAY_IN_SECONDS = 86400.0            # [s]
YEAR_IN_DAYS = 365.25               # [days]

##
# Keplerian orbital elements for the moons and the epoch.  Angles are all in
# [degrees] and semi-major axes are in [km].
EPOCH_MJD0 = 58849.0
EPOCH = EPOCH_MJD0*DAY_IN_SECONDS

# Semimajor axis [km]
A_IO = 422029.68714001
A_EUROPA = 671224.23712681
A_GANYMEDE = 1070587.4692374
A_CALLISTO = 1883136.6167305

# Eccentricity
E_IO = 4.308524661773E-03
E_EUROPA = 9.384699662601E-03
E_GANYMEDE = 1.953365822716E-03
E_CALLISTO = 7.337063799028E-03

# Inclination [deg]
I_IO = 40.11548686966E-03
I_EUROPA = 0.46530284284480
I_GANYMEDE = 0.13543966756582
# I_CALLISTO = 0.25354332731555
I_CALLISTO = 0

# Longitude of the ascending node [deg]
LAN_IO = -79.640061742992
LAN_EUROPA = -132.15817268686
LAN_GANYMEDE = -50.793372416917
LAN_CALLISTO = 86.723916616548

# Argument of periapsis [deg]
ARGPERI_IO = 37.991267683987
ARGPERI_EUROPA = -79.571640035051
ARGPERI_GANYMEDE = -42.876495018307
ARGPERI_CALLISTO = -160.76003434076

# Mean anomaly [deg]
M0_IO = 286.85240405645
M0_EUROPA = 318.00776678240
M0_GANYMEDE = 220.59841030407
M0_CALLISTO = 321.07650614246


##
# Radii of spheres of influence
R_SOI_CALLISTO = A_CALLISTO*np.power(MU_CALLISTO/MU_JUPITER, 2.0/5.0)
R_SOI_GANYMEDE = A_GANYMEDE*np.power(MU_GANYMEDE/MU_JUPITER, 2.0/5.0)
