import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import constants
import bodies
import kepler
import util

# Satellites to plot.
io = bodies.get_io()  #kepler.KeplerianEllipticalOrbit(constants.A_IO, constants.E_IO, constants.I_IO, constants.LAN_IO, constants.ARGPERI_IO, constants.M0_IO, constants.EPOCH, constants.MU_JUPITER)
europa = bodies.get_europa()  #kepler.KeplerianEllipticalOrbit(constants.A_EUROPA, constants.E_EUROPA, constants.I_EUROPA, constants.LAN_EUROPA, constants.ARGPERI_EUROPA, constants.M0_EUROPA, constants.EPOCH, constants.MU_JUPITER)
ganymede = bodies.get_ganymede()  #kepler.KeplerianEllipticalOrbit(constants.A_GANYMEDE, constants.E_GANYMEDE, constants.I_GANYMEDE, constants.LAN_GANYMEDE, constants.ARGPERI_GANYMEDE, constants.EPOCH, constants.M0_GANYMEDE, constants.MU_JUPITER)
callisto = bodies.get_callisto()  #kepler.KeplerianEllipticalOrbit(constants.A_CALLISTO, constants.E_CALLISTO, constants.I_CALLISTO, constants.LAN_CALLISTO, constants.ARGPERI_CALLISTO, constants.EPOCH, constants.M0_CALLISTO, constants.MU_JUPITER)

# Uncomment to get some test elliptical orbits - note the arguments of periapsis are 180 degrees apart.
test_a = kepler.KeplerianEllipticalOrbit(util.semimajoraxis_from_rarp(20*constants.R_JUPITER, 2*constants.R_JUPITER),
                                         util.eccentricity_from_rarp(20*constants.R_JUPITER, 2*constants.R_JUPITER),
                                         0.0, 0.0, 0.0, 0.0, constants.EPOCH, constants.MU_JUPITER)
# test_b = kepler.KeplerianEllipticalOrbit(util.semimajoraxis_from_rarp(20*constants.R_JUPITER, 2*constants.R_JUPITER),
#                                          util.eccentricity_from_rarp(20*constants.R_JUPITER, 2*constants.R_JUPITER),
#                                          0.0, 0.0, 180.0, 0.0, constants.EPOCH, constants.MU_JUPITER)

# This variable controls what fraction of the satellite's orbit is shown as a tail.
orbit_tail_fraction = 0.25

# The three numbers in np.arange are the start time, the stop time, and the time step
# for the animation.  In each moon case we work out a set of times, going from some
# fraction of the orbit period forward to the current time we are drawing.  Then
# calculate positions.  Then draw the tail and a circle for the current moon position.
fig = plt.Figure(figsize=(10,10))
for mjd_centre in np.arange(constants.EPOCH_MJD0, constants.EPOCH_MJD0+10.0, 3600.0/constants.DAY_IN_SECONDS):

    # Clear the plot.
    plt.cla()

    # Plot Io.
    t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-io.period*orbit_tail_fraction, 0.0, 50)
    state = io(t)
    plt.plot(state[:,0], state[:,1], '-', c="tab:orange")
    plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:orange", markerfacecolor="tab:orange")

    # Plot Europa.
    t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-europa.period*orbit_tail_fraction, 0.0, 50)
    state = europa(t)
    plt.plot(state[:,0], state[:,1], '-', c="tab:blue")
    plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:blue", markerfacecolor="tab:blue")

    # Plot Ganymede.
    t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-ganymede.period*orbit_tail_fraction, 0.0, 50)
    state = ganymede(t)
    plt.plot(state[:,0], state[:,1], '-', c="tab:green")
    plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:green", markerfacecolor="tab:green")

    # Plot Callisto.
    t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-callisto.period*orbit_tail_fraction, 0.0, 50)
    state = callisto(t)
    plt.plot(state[:,0], state[:,1], '-', c="tab:purple")
    plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:purple", markerfacecolor="tab:purple")

    # Uncomment to get some test elliptical orbits - note the arguments of periapsis are 180 degrees apart.
    # Plot test a.
    t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-test_a.period*orbit_tail_fraction, 0.0, 50)
    state = test_a(t)
    plt.plot(state[:,0], state[:,1], '-', c="tab:red")
    plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:red", markerfacecolor="tab:red")

    # # Plot test b.
    # t = mjd_centre*constants.DAY_IN_SECONDS + np.linspace(-test_b.period*orbit_tail_fraction, 0.0, 50)
    # state = test_b(t)
    # plt.plot(state[:,0], state[:,1], '--', c="tab:red")
    # plt.plot([state[-1,0]], [state[-1,1]], 'o', c="tab:red", markerfacecolor="tab:red")

    # Plot Jupiter
    plt.gca().add_artist(matplotlib.patches.Circle((0,0), radius=constants.R_JUPITER, edgecolor="black", facecolor="black"))

    # Finish setting up the plot.
    plt.xlim([-40*constants.R_JUPITER,40*constants.R_JUPITER])
    plt.ylim([-40*constants.R_JUPITER,40*constants.R_JUPITER])
    plt.gca().set_aspect('equal')
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.title("MJD: {:7.2f} (max={})".format(mjd_centre, constants.EPOCH_MJD0+10.0))

    plt.draw()
    plt.pause(0.00001)
