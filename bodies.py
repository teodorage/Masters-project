"""Factory functions to return KeplerianEllipticalOrbit objects for different Galilean moons.
"""

import constants
import kepler

def get_io():
    """Return KeplerianEllipticalOrbit object setup for Io.
    """
    return kepler.KeplerianEllipticalOrbit(constants.A_IO, constants.E_IO, constants.I_IO, constants.LAN_IO, constants.ARGPERI_IO, constants.M0_IO, constants.EPOCH, constants.MU_JUPITER)

def get_europa():
    """Return KeplerianEllipticalOrbit object setup for Europa.
    """
    return kepler.KeplerianEllipticalOrbit(constants.A_EUROPA, constants.E_EUROPA, constants.I_EUROPA, constants.LAN_EUROPA, constants.ARGPERI_EUROPA, constants.M0_EUROPA, constants.EPOCH, constants.MU_JUPITER)

def get_ganymede():
    """Return KeplerianEllipticalOrbit object setup for Ganymede.
    """
    return kepler.KeplerianEllipticalOrbit(constants.A_GANYMEDE, constants.E_GANYMEDE, constants.I_GANYMEDE, constants.LAN_GANYMEDE, constants.ARGPERI_GANYMEDE, constants.EPOCH, constants.M0_GANYMEDE, constants.MU_JUPITER)

def get_callisto():
    """Return KeplerianEllipticalOrbit object setup for Callisto.
    """
    return kepler.KeplerianEllipticalOrbit(constants.A_CALLISTO, constants.E_CALLISTO, constants.I_CALLISTO, constants.LAN_CALLISTO, constants.ARGPERI_CALLISTO, constants.EPOCH, constants.M0_CALLISTO, constants.MU_JUPITER)

