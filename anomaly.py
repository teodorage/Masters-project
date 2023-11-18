
import numpy as np
import matplotlib.pyplot as plt
import constants


r_start = 1000*constants.R_JUPITER # distance from jupiter to the spacecraft [km]
v = 3.4 # speed [km/s]
eccentricity = 1.4
a = 1/(2/r_start-(v**2/constants.MU_JUPITER)) #semimajor axis
mean_motion = np.sqrt(-constants.MU_JUPITER/a**3) #mean motion

H_start = -np.arccosh(1/eccentricity*(1-r_start/a))
M_start = eccentricity*np.sinh(H_start)-H_start
t_periapsis = -M_start/mean_motion
t_start = 235786.0 

def calculate_time(M_start, t_periapsis, mean_motion, t_start):
    list = np.arange(0, t_periapsis, 600.0) +t_start
    Mean = []
    time  = []
    for i in range(0,len(list)):
        t = list[i]
        M = mean_motion*(t-t_start)+M_start
        Mean.append(M)
        time.append(t)
    return Mean, time

mean_anomlay, time = calculate_time(M_start, t_periapsis, mean_motion, t_start)
# print(mean_anomlay)

       
def basicKeplerEquationSolver( mean_anomaly: float, eccentricity: float) -> float:
            relative_change_epsilon: float=1e-6
            max_iter = 100
            # Initial starting guess for the hyperbolic anomaly.
            hyper_anomaly = 0.0
            if eccentricity >1:
                hyper_anomaly = np.pi

            # Iterate using Newton-Raphson.
            iter = 0
            relative_change = 1e6
            while (relative_change>relative_change_epsilon) and (iter< max_iter):

                delta_hyper_anomaly = (mean_anomaly - eccentricity*np.sinh(hyper_anomaly) + hyper_anomaly)/(1-eccentricity*np.cosh(hyper_anomaly))
                hyper_anomaly -= delta_hyper_anomaly
                relative_change = np.abs(delta_hyper_anomaly/(hyper_anomaly+1e-300))
                iter += 1
                # print(iter, hyper_anomaly, delta_hyper_anomaly)

            return hyper_anomaly
def determine_HyperbolicAnomaly(mean_anomlay, eccentricity):
    hyper = []
    true = []
    x = []
    y = []
    position = []
    for i in range(len(mean_anomlay)):
            hyperbolic = basicKeplerEquationSolver(mean_anomlay[i],1.4)
            anomaly = 2.0*np.arctan(np.sqrt((eccentricity+1)/(eccentricity-1))*np.tan(hyperbolic*0.5))
            radial = a*(1-eccentricity*np.cosh(hyperbolic))
            x1 = radial*np.cos(anomaly)
            y1 = radial*np.sin(anomaly)
            pos = np.sqrt(x1**2+y1**2)

            hyper.append(hyperbolic)
            true.append(anomaly)
            x.append(x1)
            y.append(y1)
            position.append(pos)
            

    return hyper, true, x, y, position

hyperbolic_anomaly, true_anomaly, x_coord, y_coord, position = determine_HyperbolicAnomaly(mean_anomlay, eccentricity)


# print(hyperbolic_anomaly)
# plt.plot(np.degrees(hyperbolic_anomaly), np.degrees(true_anomaly))
# plt.plot(position, np.degrees(hyperbolic_anomaly))
plt.xlabel("Time since start of mission [seconds]")
plt.ylabel("Hyperbolic anomaly [degrees]")
plt.show()
