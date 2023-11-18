"""Example Tisserand plots - reproduces figure 1 in Calasurdo et al. (2014) (the winning GTOC6 paper)
"""
import numpy as np
import matplotlib.pyplot as plt

import tisserand
import constants

# This demo reproduces figure 1 in Calasurdo et al. (2014) Tour of Jupiter Galilean moons: Winning solution of GTOC6
tisserand.tisserand_graph_apsides(constants.A_CALLISTO, 0.0, color='tab:purple', length_units=constants.R_JUPITER)
tisserand.tisserand_graph_apsides(constants.A_GANYMEDE, 0.0, color='tab:green', length_units=constants.R_JUPITER)
tisserand.tisserand_graph_apsides(constants.A_EUROPA, 0.0, color='tab:blue', length_units=constants.R_JUPITER)
tisserand.tisserand_graph_apsides(constants.A_IO, 0.0, color='tab:orange', length_units=constants.R_JUPITER)
plt.ylim([0,30])
plt.xlim([0,50])
plt.xlabel("Apoapsis, $r_a$ [$\mathrm{R}_\mathrm{J}$]")
plt.ylabel("Periapsis, $r_p$ [$\mathrm{R}_\mathrm{J}$]")
plt.gca().set_aspect("equal")
plt.show()
