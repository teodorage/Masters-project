
import numpy as np
import matplotlib.pyplot as plt
import constants
import matplotlib.patches


r0 = 1000*constants.R_JUPITER # distance from jupiter to the spacecraft [km]
v0 = 3.4 # speed [km/s]
t_start = constants.EPOCH 

degrees_r = np.arange(0,360,36)
degrees_v = np.arange(91,180,45)

def r_v_vectors(r0,v0, degrees_r,degrees_v):      #determines the position, velocity, specific angular momentum, 
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
position, velocity, angular, eccentricity, semimajor_axis, mean_motion =r_v_vectors(r0,v0,degrees_r,degrees_v)
# print(semimajor_axis)

H_start = -np.arccosh(np.multiply((1/np.linalg.norm(eccentricity, axis = 2)),(1-r0/semimajor_axis)))  #hyperbolic anomaly at the start of the mission
M_start = np.linalg.norm(eccentricity, axis = 2)*np.sinh(H_start)-H_start  #mean anomaly at the beginning of the mission
t_periapsis = -M_start/mean_motion # time it takes to get to periapsis



def calculate_time(M_start, t_periapsis, mean_motion, t_start): #Calculates the mean anomaly for all the times to the periapsis
    list = np.array(0, len(t_periapsis), 600.0) +t_start
    Mean = []
    time  = []
    for i in range(0,len(list)):
        t = list[i]
        M = mean_motion*(t-t_start)+M_start
        Mean.append(M)
        time.append(t)

            
    # time_step = 600.0
    # list = len(t_periapsis)
    # time = np.arange(0, list*time_step,time_step) +t_start
    # Mean = mean_motion*(time-t_start)+M_start
    return Mean, time

mean_anomaly, time = calculate_time(M_start, t_periapsis, mean_motion, t_start)
print(time)
       
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

def determine_HyperbolicAnomaly(mean_anomaly, eccentricity,arg_periapsis, semimajor_axis):
    hyper = []
    true = []
    x = []
    y = []
    position = []
    for i in range(len(mean_anomaly)):
            hyperbolic = basicKeplerEquationSolver(mean_anomaly[i],np.linalg.norm(eccentricity))  # determines the hyperbolic anomaly 
            anomaly = 2.0*np.arctan(np.sqrt((np.linalg.norm(eccentricity) +1)/(np.linalg.norm(eccentricity) -1))*np.tan(hyperbolic*0.5)) # true anomaly
            radial = semimajor_axis*(1-np.linalg.norm(eccentricity) *np.cosh(hyperbolic)) # determines the radius which is then used for the x-y coordinates
            x1 = radial*np.cos(anomaly+arg_periapsis) # x coordinate
            y1 = radial*np.sin(anomaly+arg_periapsis) # y coordinate
            pos = np.sqrt(x1**2+y1**2) # spacecraft's position

            hyper.append(hyperbolic)
            true.append(anomaly)
            x.append(x1)
            y.append(y1)
            position.append(pos)
            
    return hyper, true, x, y, position

hyperbolic_anomaly, true_anomaly, x_coord, y_coord, position = determine_HyperbolicAnomaly(mean_anomaly, eccentricity,0,semimajor_axis)
# print(hyperbolic_anomaly)
# print(t_periapsis)


# def plot(hyperbolic_anomaly, true_anomaly, t_periapsis):   # Method to plot the hyperbolic trajectory
#     plt.figure()
#     hyperbolic_anomaly = np.array(hyperbolic_anomaly).reshape(10,2)
#     true_anomaly = np.array(true_anomaly).reshape(10,2)
#     for i in range(len(hyperbolic_anomaly)):
#         for j in range(len(true_anomaly[i])):
#         # for j in range(len(np.array(t_periapsis)[i])):
#             plt.plot(np.array(np.degrees(hyperbolic_anomaly))[i][j], np.array(np.degrees(true_anomaly))[i][j])
#     plt.show()

# #             plt.plot(np.degrees(hyperbolic_anomaly[0][i][j]), t_periapsis[0][i][j])
# # # plt.plot(x_coord,y_coord)
# #     # plt.gca().add_artist(matplotlib.patches.Circle((0,0), radius=40*constants.R_JUPITER, edgecolor="black", facecolor="black"))
# #     # plt.xlim([-1010*constants.R_JUPITER,1010*constants.R_JUPITER])
# #     # plt.ylim([-1010*constants.R_JUPITER,1010*constants.R_JUPITER])
#  # plt.xlabel("Time since start of mission [seconds]")
#  # plt.ylabel("Hyperbolic anomaly [degrees]")
# print(np.array(hyperbolic_anomaly).shape)

# plot(hyperbolic_anomaly,true_anomaly,t_periapsis)
#print(np.array(t_periapsis).shape)
# plt.plot(np.reshape(np.array(np.degrees(hyperbolic_anomaly)),-1), np.reshape(np.array(np.degrees(true_anomaly)),-1))
