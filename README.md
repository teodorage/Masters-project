Here is some example code to start off and that we can improve.

* `constants.py` contains all the constants from the GTOC6 problem description.
* `util.py` contains a series of utility functions for calculating various things, like the period of an orbit from Kepler's 3rd law.
* `tisserand.py` contains some code to plot Tisserand graphs.  You can see it in action in `example_tisserand.py` which reproduces FIgure 1 from the winning GTOC6 solution.
* `kepler.py` contains a class (KeplerianEllipticalOrbit) which can compute the position and velocity for a body in an elliptical Keplerian orbit.  You can see it in action in `example_galilean_orbits.py` which makes a movie of the Galilean satellites orbiting Jupiter.  You can use this as a basis for experimentation.
* `bodies.py` contains four functions which make it easy to make KeplerianEllipticalOrbit objects for the four Galilean satellites -I basically got bored writing the actual code out all the time.
* `propagation.py` contains some in-progress code for propagating orbits.
* `util_testing.py` testing a few of the routines.
* `hyperbolic.py` determines the spacecraft positions and velocites and performs the gravity assist
* `gravity_assist.py` contains functions used to perform the gravity assist.