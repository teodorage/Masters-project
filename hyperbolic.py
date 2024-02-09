
import numpy as np
import matplotlib.pyplot as plt
import constants
import matplotlib.patches
import kepler
import bodies
import datetime
import gravity_assist


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
    # all of these represent values that the spacecraft has at the beginning and they only depend on where on a circle
    # the spacecraft starts from and where the velocity vector points
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
# degrees_r = np.arange(0,360,10)
degrees_r = np.arange(209,211,1)
# degrees_v = np.arange(175,180-np.degrees(np.arctan2(16*constants.R_JUPITER,r0)),0.25)
degrees_v = np.arange(177.5,178.0,0.25)


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


# def hyperbolicOrbit(degrees_r,degrees_v, t_periapsis,mean_motion,M_start, eccentricity_scalar, semimajor_axis, eccentricity,t_start):
range_degrees_inSOI = []
degree_r_map = {}

D = np.zeros((len(degrees_r),len(degrees_v)))
for i in range(len(degrees_r)): # these for loops actually calculate the spacecraft's position and velocity based on the time 
    for j in range(len(degrees_v)):
            print("Computing trajectory: deg_r {}/{}, deg_v {}/{}".format(i+1, len(degrees_r), j+1, len(degrees_v)))

        # t_start is a datetime.datetime object, but mean_anomaly just needs
        # times from t_start, so we can just make an array of times
        # from t_start

            times_from_start = np.arange(0, 1.5*t_periapsis[i,j], 3600.0)
            # times = [datetime.datetime.utcfromtimestamp(t_start) + datetime.timedelta(seconds=t) for t in times_from_start]
            times = np.array([t_start + t for t in times_from_start])

            assert(len(times_from_start)==len(times))

            mean_anomaly = mean_motion[i,j]*(times_from_start) + M_start[i,j]
            hyper_anomaly = array_basicKeplerEquationSolver(mean_anomaly, eccentricity_scalar[i,j])     # Using the faster array version - we can just pass all the mean anomalies and it will return all the hyperbolic anomalies
            true_anomaly = 2.0*np.arctan(np.sqrt((eccentricity_scalar[i,j] +1)/(eccentricity_scalar[i,j] - 1))*np.tan(hyper_anomaly*0.5)) # true anomaly
            radial = semimajor_axis[i,j]*(1-eccentricity_scalar[i,j] *np.cosh(hyper_anomaly)) # determines the radius which is then used for the x-y coordinates
            gamma  = np.arctan(eccentricity_scalar[i,j]*np.sin(true_anomaly)/(1+eccentricity_scalar[i,j]*np.cos(true_anomaly))) #flight path angle

            arg_periapsis = np.arccos(eccentricity[i,j,0]/(eccentricity_scalar[i,j]))
            if eccentricity[i,j,1]<0.0:
                arg_periapsis = -arg_periapsis

            x1 = radial*np.cos(true_anomaly+arg_periapsis) # x coordinate
            y1 = radial*np.sin(true_anomaly+arg_periapsis) # y coordinate
            z1 = np.zeros(len(x1))
             
            vx = v0*(-np.sin(true_anomaly+arg_periapsis-gamma))
            vy = v0*(np.cos(true_anomaly+arg_periapsis-gamma))
            vz = np.zeros(len(vx))

            r_periapsis[i,j] = np.min(np.sqrt(x1*x1 + y1*y1))

            spacecraft_position = np.array(list(zip(x1,y1,z1)))
            spacecraft_velocity = np.array(list(zip(vx,vy,vz)))

            key = str(degrees_r[i]) + "-" + str(degrees_v[j])
        
            callisto_state = callisto(times) #calculate callisto's state for all the times
            callisto_position = callisto_state[:,0:3] #position vector for callisto
            callisto_velocity = callisto_state[:,3:6] #velocity vecotr of callisto

            degree_r_map[key] = {"spacecraft_position":spacecraft_position, "spacecraft_velocity":spacecraft_velocity,
                                  "callisto_position":callisto_position, "callisto_velocity":callisto_velocity}

            
            # plt.plot(x1-callisto_position[:,0],y1-callisto_position[:,1]) #plot the trajectory of spacecraft
            # plt.plot(x1,y1)
            # print(callisto_position[:,0], callisto_position[:,1])

            distances = np.sqrt((x1-callisto_position[:,0])**2+(y1-callisto_position[:,1])**2) #determine distance between callisto and spacecraft
            D[i,j] = np.min(distances) 
            r_soi = constants.A_CALLISTO*np.power(constants.MU_CALLISTO/constants.MU_JUPITER, 2.0/5.0) #sphere of infl
  
            # # plt.semilogy(times, distances/r_soi)
            # plt.pcolormesh(degrees_r, degrees_v, D.T/r_soi, norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=100.0))
   
            for k in range(len(distances)):
                if distances[k] <= r_soi: # checks whether the  spacecraft is inside the sphere of influence and saves those angles combinations
                # print("Spacecraft is inside Callisto's SOI at deg_r {}, deg_v {}.".format(degrees_r[i], degrees_v[j]))
                    range_degrees_inSOI.append((degrees_r[i], degrees_v[j]))
                    # print(spacecraft_velocity[i,:])
                    


for key,value in degree_r_map.items():
    b1,b2,rx,ry, vx, vy, pos_in_moon_frame = gravity_assist.change_to_moon_frame(value["callisto_position"], value["callisto_velocity"], value["spacecraft_position"], value["spacecraft_velocity"])
    delta,b = gravity_assist.gravity_assist(value["spacecraft_velocity"], rx[0],ry[0], value["spacecraft_velocity"][:,0],value["spacecraft_velocity"][:,1])

    k1,k2, x_j, y_j, z_j= gravity_assist.change_to_jupiter_frame(b1,b2, pos_in_moon_frame)
 
    # for i in range(len(b)):
    #     if b[i]<100 and b[i]>0:
    #         print(b[i])
    # plt.plot(rx,ry)
# plt.show()


# plt.plot(new_position[:,0],new_position[:,1])
# plt.plot(callistopos[:,0],callistopos[:,1])
# print(new_position)

# plt.plot(rx[:,0],ry[:,0])
# plt.plot(rx,ry) #plots all column values on the first row
# for i in range(len(rx[0])):
#     plt.plot(rx[i,:], ry[i,:])

# plt.plot(x[0,:],y[0,:])


# jupiter_frame_pos, k1, k2 =  change_to_jupiter_frame(b1,b2, new_position)
# print(jupiter_frame_pos, spacecraft_position)
# print(callistopos)


# hyperbolicOrbit(degrees_r,degrees_v, t_periapsis,mean_motion,M_start, eccentricity_scalar, semimajor_axis, eccentricity,t_start)
# plt.scatter(callisto_position[:,0], callisto_position[:,1], marker = 'o', c="tab:red")
# plt.plot([callisto_position[-1,0]], [callisto_position[-1,1]], 'o', c="tab:red", markerfacecolor="tab:red")

# for trajectory

# plt.gca().add_artist(matplotlib.patches.Circle((0,0), 25*constants.R_JUPITER, facecolor="none", edgecolor="r"))
# plt.gca().add_artist(matplotlib.patches.Circle((0,0), 2410.3, facecolor="none", edgecolor="r"))
# # plt.gca().add_artist(matplotlib.patches.Circle((0,0), r0, facecolor="none", edgecolor="k"))
# plt.gca().set_aspect("equal")
# plt.gca().add_artist(matplotlib.patches.Circle((0,0), r_soi, facecolor="none", edgecolor="r"))
# plt.show()


# for the curve line plot

# plt.ylabel(r"Distance to Callisto $d/r_{SOI}$")
# plt.xlabel("Time")
# plt.show()


# plt.pcolormesh(degrees_r, degrees_v, (r_periapsis.T)/constants.R_JUPITER)
# h.set_label("Periapsis [Rjupiter]")

# for the colour plot

# h=plt.colorbar()
# h.set_label(r"$\log_{10}(D/r_{SOI})$")
# plt.xlabel("Position angle [degrees]")
# plt.ylabel("Velocity angle [degrees]")
# plt.show()


