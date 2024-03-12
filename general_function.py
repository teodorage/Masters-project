import numpy as np
import kepler
import constants
import bodies
import matplotlib.pyplot as plt
import util
import gravity_assist

def array_basicKeplerEquationSolver( mean_anomaly: np.ndarray, eccentricity: float) -> float: # Solving kepler's equation for the hyperbolic anomaly
 
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



def hyperbolic_trajectory_calculator(r,eccentricity,semimajor_axis, arg_periapsis):

    """Determines the spacecraft position and velocity in a hyprebolic trajectory
    by calculating first the mean anomaly (M_start) and hyperbolic anomaly at start (H_start)
    then calculates the mean anomaly, hyperbolic anomaly and true anomaly for a series of times """

    H_start = -np.arccosh(np.multiply((1/np.linalg.norm(eccentricity)),(1-np.linalg.norm(r)/semimajor_axis))) 
    M_start = np.linalg.norm(eccentricity)*np.sinh(H_start)-H_start
    mean_motion = np.sqrt(np.divide(-constants.MU_JUPITER,(semimajor_axis**3)))

    t_periapsis = -M_start/mean_motion
    times =  np.arange(0,1.5*t_periapsis,60)

    meananomaly = mean_motion*(times) + M_start
    hyperbolic_anomaly = array_basicKeplerEquationSolver(meananomaly, np.linalg.norm(eccentricity))     # Using the faster array version - we can just pass all the mean anomalies and it will return all the hyperbolic anomalies
    true_anomaly = 2.0*np.arctan(np.sqrt((np.linalg.norm(eccentricity) +1)/(np.linalg.norm(eccentricity) - 1))*np.tanh(hyperbolic_anomaly*0.5)) # true anomaly
    radial = semimajor_axis*(1-np.linalg.norm(eccentricity) *np.cosh(hyperbolic_anomaly)) # determines the radius which is then used for the x-y coordinates
    gamma  = np.arctan2(np.linalg.norm(eccentricity)*np.sin(true_anomaly),(1+np.linalg.norm(eccentricity)*np.cos(true_anomaly))) #flight path angle
    
    if eccentricity[1]<0.0:
        arg_periapsis = -arg_periapsis

    x = radial*np.cos(true_anomaly+arg_periapsis) # x coordinate
    y = radial*np.sin(true_anomaly+arg_periapsis) # y coordinate
    z = np.zeros(len(x))

    v = util.vis_viva(constants.MU_JUPITER, radial, semimajor_axis)
    vx = v*(-np.sin(true_anomaly+arg_periapsis-gamma))
    vy = v*(np.cos(true_anomaly+arg_periapsis-gamma))
    vz = np.zeros(len(vx))

    spacecraft_position = np.array(list(zip(x,y,z)))
    spacecraft_velocity = np.array(list(zip(vx,vy,vz)))
  

    return spacecraft_position,spacecraft_velocity, times

def elliptical_trajectory_calculator(r,eccentricity, semimajor_axis, arg_periapsis):
    """Determines the spacecraft position and velocity in an elliptical orbit; first it calculates the
    eccentric anomlay (E_start) and mean anomaly at start (M_start)"""

    E_start = np.multiply(1/np.linalg.norm(eccentricity),(1-np.linalg.norm(r)/semimajor_axis))
    M_start = E_start - np.linalg.norm(eccentricity)*np.sin(E_start)
    mean_motion = np.sqrt(np.divide(constants.MU_JUPITER,(semimajor_axis**3)))

    t_periapsis = -M_start/mean_motion
    times = np.arange(58849.0*86400.0,(58849.0*86400.0+ t_periapsis), 1800)

    meananomaly = mean_motion*(times) - M_start
    eccentricanomaly = gravity_assist.keplersolver(meananomaly, np.linalg.norm(eccentricity))     # Using the faster array version - we can just pass all the mean anomalies and it will return all the hyperbolic anomalies
    trueanomaly = 2.0*np.arctan(np.sqrt((np.linalg.norm(eccentricity)+1)/(1-np.linalg.norm(eccentricity)))*np.tan(eccentricanomaly*0.5)) # true anomaly
    radial = semimajor_axis*(1-np.linalg.norm(eccentricity) *np.cos(eccentricanomaly)) # determines the radius which is then used for the x-y coordinates
    gamma  = np.arctan(np.linalg.norm(eccentricity)*np.sin(trueanomaly)/(1+np.linalg.norm(eccentricity)*np.cos(trueanomaly))) #flight path angle         
 

    x = radial*np.cos(trueanomaly+arg_periapsis) # position x coordinate
    y = radial*np.sin(trueanomaly+arg_periapsis) # position y coordinate
    z = np.zeros(len(x)) # position z coordinate, always 0

    v = util.vis_viva(constants.MU_JUPITER,radial, semimajor_axis)
    vx = v*(-np.sin(trueanomaly+arg_periapsis-gamma)) # velocity x coordinate
    vy = v*(np.cos(trueanomaly+arg_periapsis-gamma)) # velocity y coordinate
    vz = np.zeros(len(vx)) # velocity z coordinate, always 0

    spacecraft_position = np.array(list(zip(x,y,z))) # position vector
    spacecraft_velocity = np.array(list(zip(vx,vy,vz))) # velocity vector

    return spacecraft_position,spacecraft_velocity, times


def check_forGA(position,moon_position,mu_moon,semimajor_moon):
    for i in range(len(position)):
        distance = np.sqrt((position[i,0]-moon_position[i,0])**2+(position[i,1]-moon_position[i,1])**2) 
        r_soi = semimajor_moon*np.power(mu_moon/constants.MU_JUPITER, 2.0/5.0) 
        if distance < r_soi:
        # if distance < 2000 and distance>50:
            return 1, i
    
    return 0,1e10


def GravityAssist(position,velocity,moon_position,moon_velocity,moon_radius):
                
            b1,b2,rx,ry,vx,vy, pos_moon_frame, vel_moon_frame = gravity_assist.change_moon_frame(moon_position,moon_velocity,position,velocity)
            delta,b, v_inf_out = gravity_assist.gravity_assist(vel_moon_frame, pos_moon_frame[0],pos_moon_frame[1], vel_moon_frame[0],vel_moon_frame[1])

            # print("Flyby altitude {} km".format(np.min(np.linalg.norm(pos_moon_frame))-moon_radius))
            k1,k2, pos_jupiter_frame, vel_jupiter_frame = gravity_assist.change_jupiter_frame(b1,b2, pos_moon_frame,vel_moon_frame, moon_position,moon_velocity)
            t_rsoi = 2*np.linalg.norm(pos_moon_frame)/(np.linalg.norm(vel_moon_frame)) #time it takes the spacecraft to go from one end of SOI to the other 
            v_inf_out_jupiter = np.array([np.dot(k1,v_inf_out),np.dot(k2,v_inf_out),0])+moon_velocity #change v_inf_out back into Jupiter frame
            return v_inf_out_jupiter, pos_jupiter_frame, delta


results = {}

def determine_orbit(r0,v0,time_start,arc_max,arc_num,results):
    key = arc_num
    results[key] = {"starting position":r0, "starting velocity":v0, "starting time":time_start, "bending angle":0}
    if arc_num<arc_max:

        angular_momentum = np.cross(r0, v0)
        eccentricity = np.subtract((np.cross(v0, angular_momentum)/constants.MU_JUPITER),(np.divide(r0,np.linalg.norm(r0))))
        arg_periapsis = np.arccos(eccentricity[0]/np.linalg.norm(eccentricity))
        semimajor_axis = 1.0/(2.0/np.linalg.norm(r0) - np.linalg.norm(v0)**2/constants.MU_JUPITER)
        apoapsis = util.r_apoapsis(semimajor_axis,np.linalg.norm(eccentricity))
        # print('Apoapsis distance {}'.format(apoapsis))

        if semimajor_axis < 0.0: #hyporbolic trajecotry
            position,velocity,times = hyperbolic_trajectory_calculator(r0,eccentricity,semimajor_axis,arg_periapsis)

        else: #elliptical trajectory
            position,velocity,times = elliptical_trajectory_calculator(r0,eccentricity,semimajor_axis, arg_periapsis)

        time = np.array([time_start+ t for t in times])
        callisto = bodies.get_callisto()
        ganymede = bodies.get_ganymede()
        callisto_state = callisto(time)
        ganymede_state = ganymede(time)

        ca_flyby, ca_index = check_forGA(position,callisto_state[:,0:3], constants.MU_CALLISTO,constants.A_CALLISTO)
        ga_flyby, ga_index = check_forGA(position,ganymede_state[:,0:3],constants.MU_GANYMEDE,constants.A_GANYMEDE)


        if ca_flyby == 1:
            if ga_flyby == 1:
                if ga_index < ca_index:
                    index = ga_index
                    v_inf, new_position, delta = GravityAssist(position[ga_index], velocity[ga_index],ganymede_state[ga_index,0:3],ganymede_state[ga_index,3:6],constants.R_GANYMEDE)
                else:
                    index = ca_index
                    v_inf, new_position, delta = GravityAssist(position[ca_index],velocity[ca_index],callisto_state[ca_index,0:3], callisto_state[ca_index,3:6],constants.R_CALLISTO)
            else:
                    index = ca_index
                    v_inf, new_position, delta = GravityAssist(position[ca_index],velocity[ca_index],callisto_state[ca_index,0:3], callisto_state[ca_index,3:6],constants.R_CALLISTO)

            print(v_inf,new_position)
            arc_num += 1
            determine_orbit(new_position,v_inf,time[index],arc_max,arc_num,results)
            results[key] = {"starting position":new_position, "starting velocity":v_inf, "starting time":time[index], "bending angle": delta}


        else:
            print('No gravity assist found for arc number {}'.format(arc_num+1))
            return
            

    results[key] = {"starting position":r0, "starting velocity":v0, "starting time":time_start, "bending angle": delta}
    return results





# r = np.array([46902972.10058936, 53955697.30928642     ,   0.        ])
# v = np.array([-2.44575532 ,-2.36183846  ,0.        ])

t_start = 58849.0*86400.0 #[seconds]
r = np.array([46335387.96680537, 54443896.68241637 , 0.        ])
r2 = np.array([46145061.01076064 ,54605305.67914784    ,    0.        ])
v = np.array([-2.42088856 ,-2.38732038 , 0.        ])
v2 = np.array([-2.4125405 , -2.39575631,  0.        ])

result = determine_orbit(r,v,t_start,3,0,results)
print(result)

# rcal= np.array([-1749122.32275296  , 720344.74662463  ,      0.        ])
# spacecraft_position,spacecraft_velocity, times = elliptical_trajectory_calculator(rcal,7.337063799028E-03,1883136.6167305,-160.76003434076)
# print(spacecraft_position,spacecraft_velocity)