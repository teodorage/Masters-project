
import numpy as np
import matplotlib.pyplot as plt
import constants
import matplotlib.patches
import kepler
import bodies
import datetime

def r_v_vectors(r0,v0, degrees_r,degrees_v):
    """Calculate a grid of starting position and velocity vectors and orbital parameters

    Parameters
    ----------
    r0 : float
        Jovicentric starting distance [km]
    v0 : float
        Jovicentric starting velocity [km]
    degrees_r : np.ndarray
        Array of position angles [degrees]
    degrees_v : np.ndarray
        Array of velocity angles [degrees]
    """

    pos = np.zeros((len(degrees_r), len(degrees_v), 3))     #eccentricity, semimajor axis and mean motion of the spacecraft
    vel = np.zeros((len(degrees_r), len(degrees_v), 3))
    mom = np.zeros((len(degrees_r), len(degrees_v), 3))
    ecc = np.zeros((len(degrees_r), len(degrees_v), 3))

    for i in range(len(degrees_r)):
        delta = np.radians(degrees_r[i])
        r = [r0 * np.cos(delta), r0 * np.sin(delta), 0]
        pos[i, :, :] = r

        for j in range(len(degrees_v)):
            theta = np.radians(degrees_v[j])
            v = [v0 * np.cos(delta + theta), v0 * np.sin(delta + theta), 0]
            vel[i, j, :] = v

            ang_mom = np.cross(r, v)
            mom[i, j, :] = ang_mom
            e = np.subtract((np.cross(v, ang_mom)/constants.MU_JUPITER),(np.divide(r,r0)))
            ecc[i,j,:] = e

    a = 1.0/(2.0/np.linalg.norm(pos,axis=2) - np.linalg.norm(vel,axis=2)**2/constants.MU_JUPITER)
    mean = np.sqrt(np.divide(-constants.MU_JUPITER,(a**3)))

    return pos, vel, mom, ecc, a, mean


def basicKeplerEquationSolver( mean_anomaly: float, eccentricity: float) -> float: # Solving kepler's equation for the hyperbolic anomaly
    relative_change_epsilon: float=1e-6
    max_iter = 100
    # Initial starting guess for the hyperbolic anomaly.
    hyper_anomaly = 0.0
    if np.linalg.norm(eccentricity) >1:
        hyper_anomaly = np.pi

    # Iterate using Newton-Raphson.
    iter = 0
    relative_change = 1e6
    while np.any(relative_change>relative_change_epsilon) and (iter< max_iter):

        delta_hyper_anomaly = (mean_anomaly - np.linalg.norm(eccentricity)*np.sinh(hyper_anomaly) + hyper_anomaly)/(1-np.linalg.norm(eccentricity)*np.cosh(hyper_anomaly))
        hyper_anomaly -= delta_hyper_anomaly
        relative_change = np.abs(delta_hyper_anomaly/(hyper_anomaly+1e-300))
        iter += 1
        # print(iter, hyper_anomaly, delta_hyper_anomaly)

    return hyper_anomaly




def array_basicKeplerEquationSolver( mean_anomaly: np.ndarray, eccentricity: float) -> float: # Solving kepler's equation for the hyperbolic anomaly
    """Optimised hyperbolic solver for Kepler's problem
    
    Rather than solving Kepler's problem for each mean anomaly at a time, this version
    solves for a whole set of them at once.  So if we have 10 mean anomalies that we
    want to find the hyperbolic anomalies for, and if each one requires 5 iterations,
    then than requires us to go 50 times through the while loop below.  But if we do them
    all at once, with the delta_hyper_anomaly and hyper_anomaly stored as arrays
    then we only have to go through the loop 5 times and it's much faster.
    """
    relative_change_epsilon: float=1e-8
    max_iter = 1000
    # Initial starting guess for the hyperbolic anomaly.
    hyper_anomaly = np.zeros_like(mean_anomaly)
    if np.linalg.norm(eccentricity) >1:
        hyper_anomaly += np.pi

    # Iterate using Newton-Raphson.
    iter = 0
    relative_change = 1e6
    while np.any(relative_change>relative_change_epsilon) and (iter< max_iter):

        delta_hyper_anomaly = (mean_anomaly - np.linalg.norm(eccentricity)*np.sinh(hyper_anomaly) + hyper_anomaly)/(1-np.linalg.norm(eccentricity)*np.cosh(hyper_anomaly))
        hyper_anomaly -= delta_hyper_anomaly
        relative_change = np.abs(delta_hyper_anomaly/(hyper_anomaly+1e-300))
        iter += 1
        # print(iter, hyper_anomaly, delta_hyper_anomaly)

    if iter==max_iter:
        raise kepler.KeplersSolverDidNotConverge(iter, delta_hyper_anomaly, relative_change, mean_anomaly, eccentricity)

    return hyper_anomaly





r0 = 1000*constants.R_JUPITER # distance from jupiter to the spacecraft [km]
v0 = 3.4 # speed [km/s]
# t_start = constants.EPOCH 
t_start = 58849.0*86400.0 #[seconds]

# By trial and error I found that if the velocity vector was pointed within tan^-1(16/1000) degrees
# of Jupiter then it would collide with Jupiter, hence 180- this angle as the upper limit.
degrees_r = np.arange(0,360,30)
degrees_v = np.arange(175,180-np.degrees(np.arctan2(16*constants.R_JUPITER,r0)),0.75)

position, velocity, angular, eccentricity, semimajor_axis, mean_motion =r_v_vectors(r0,v0,degrees_r,degrees_v)

# plt.pcolormesh(degrees_r, degrees_v, np.linalg.norm(eccentricity, axis=2).T)
# h=plt.colorbar()
# h.set_label("Eccentricity")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()

# plt.pcolormesh(degrees_r, degrees_v, semimajor_axis.T)
# h=plt.colorbar()
# h.set_label("Semi-major axis [km]")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()


# Calculate the starting hyperbolic anomaly.
H_start = -np.arccosh(np.multiply((1/np.linalg.norm(eccentricity, axis = 2)),(1-r0/semimajor_axis)))  #hyperbolic anomaly at the start of the mission
# plt.pcolormesh(degrees_r, degrees_v, np.degrees(H_start).T)
# h=plt.colorbar()
# h.set_label("Starting hyperbolic anomaly [deg]")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()

# Calculate the starting mean anomaly.
M_start = np.linalg.norm(eccentricity, axis = 2)*np.sinh(H_start)-H_start  #mean anomaly at the beginning of the mission
# plt.pcolormesh(degrees_r, degrees_v, np.degrees(M_start).T)
# h=plt.colorbar()
# h.set_label("Starting mean anomaly [deg]")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()

# Calculate the time it takes to get to periapsis
t_periapsis = -M_start/mean_motion
# plt.pcolormesh(degrees_r, degrees_v, (t_periapsis.T)/constants.DAY_IN_SECONDS)
# h=plt.colorbar()
# h.set_label("Time to periapsis [days]")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()

# Each position/velocity calculated in the r_v_vectors() function corresponds
# to a separate orbit.  So for each one, we:
# 1. Get a set of times from the start to periapsis.
# 2. Calculate the mean anomaly for all the times to the periapsis.
# 3. Work out the hyperbolic anomaly by solving Kepler's equation for all the times.
# 4. Work out the true anomaly for all the times.
# 5. Work out the radial distance to the spacecraft.
# 6. Work out the position and then plot it.
eccentricity_scalar = np.linalg.norm(eccentricity, axis=2)
r_periapsis = np.zeros_like(semimajor_axis)
callisto = bodies.get_callisto()
# distance = np.zeros()
def hyperbolicOrbit(degrees_r,degrees_v, t_periapsis,mean_motion,M_start, eccentricity_scalar, semimajor_axis, eccentricity,t_start):
    D = np.zeros((len(degrees_r),len(degrees_v)))
    for i in range(len(degrees_r)):
        for j in range(len(degrees_v)):
            print("Computing trajectory: deg_r {}/{}, deg_v {}/{}".format(i+1, len(degrees_r), j+1, len(degrees_v)))

        # t_start is a datetime.datetime object, but mean_anomaly just needs
        # times from t_start, so we can just make an array of times
        # from t_start

            times_from_start = np.arange(0, 1.5*t_periapsis[i,j], 3600.0)
            # times = [datetime.datetime.utcfromtimestamp(t_start) + datetime.timedelta(seconds=t) for t in times_from_start]
            times = np.array([t_start + t for t in times_from_start])

            mean_anomaly = mean_motion[i,j]*(times_from_start) + M_start[i,j]
            hyper_anomaly = array_basicKeplerEquationSolver(mean_anomaly, eccentricity_scalar[i,j])     # Using the faster array version - we can just pass all the mean anomalies and it will return all the hyperbolic anomalies
            true_anomaly = 2.0*np.arctan(np.sqrt((eccentricity_scalar[i,j] +1)/(eccentricity_scalar[i,j] - 1))*np.tan(hyper_anomaly*0.5)) # true anomaly
            radial = semimajor_axis[i,j]*(1-eccentricity_scalar[i,j] *np.cosh(hyper_anomaly)) # determines the radius which is then used for the x-y coordinates

            arg_periapsis = np.arccos(eccentricity[i,j,0]/(eccentricity_scalar[i,j]))
            if eccentricity[i,j,1]<0.0:
                arg_periapsis = -arg_periapsis

            x1 = radial*np.cos(true_anomaly+arg_periapsis) # x coordinate
            y1 = radial*np.sin(true_anomaly+arg_periapsis) # y coordinate
            r_periapsis[i,j] = np.min(np.sqrt(x1*x1 + y1*y1))
            # plt.plot(x1,y1)
            # print(times,times_from_start)
            callisto_state = callisto(times)
            callisto_position = callisto_state[:,0:2]
            # print(len(callisto_state),len(callisto_position))
            # print(len(x1),len(callisto_position))


            d = np.sqrt((x1-callisto_position[:,0])**2+(y1-callisto_position[:,1])**2)
            distances = np.sqrt((x1-callisto_position[:,0])**2+(y1-callisto_position[:,1])**2)
            D[i,j] = np.min(distances)
            r_soi = constants.A_CALLISTO*np.power(constants.MU_CALLISTO/constants.MU_JUPITER, 2.0/5.0)
            # plt.semilogy(times, d/r_soi)
            plt.pcolormesh(degrees_r, degrees_v, D.T/r_soi, norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100.0))


hyperbolicOrbit(degrees_r,degrees_v, t_periapsis,mean_motion,M_start, eccentricity_scalar, semimajor_axis, eccentricity,t_start)



# To get the Callisto positions.  We need to create a callisto object outside the loop.
# Inside the loop we need to make a whole load of times to pass into the callisto object.
# We can use times_from_start but need to add on t_start.  Then we can get the position
# and velocity of Callisto for all the times that we have worked out the spacecraft position.
# Now we have the spacecraft position and the Callisto position we just need to work out the
# distance between them and store it.


# plt.gca().add_artist(matplotlib.patches.Circle((0,0), 25*constants.R_JUPITER, facecolor="none", edgecolor="r"))
# plt.gca().add_artist(matplotlib.patches.Circle((0,0), r0, facecolor="none", edgecolor="k"))
# plt.gca().set_aspect("equal")


# plt.ylabel(r"Distance to Callisto $d/r_{SOI}$")
# plt.xlabel("Time")
# plt.legend()
# plt.show()

# plt.pcolormesh(degrees_r, degrees_v, (r_periapsis.T)/constants.R_JUPITER)
# h.set_label("Periapsis [Rjupiter]")

h=plt.colorbar()
h.set_label(r"$\log_{10}(D/r_{SOI})$")
plt.xlabel("Position angle [degrees]")
plt.ylabel("Velocity angle [degrees]")
plt.show()


