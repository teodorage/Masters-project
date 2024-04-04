import numpy as np
import kepler
import constants
import bodies
import matplotlib.pyplot as plt
import util
import gravity_assist
import matplotlib

def array_basicKeplerEquationSolver( mean_anomaly: np.ndarray, eccentricity: float) -> float: 
    """Solving Kepler's equation for the hyperbolic anomaly"""
 
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

    """ Determines the spacecraft position and velocity in a hyperbolic trajectory.

    Parameters
    ----------
    mean_motion : 
        Mean motion [radians/s].
    times : np.ndarray
        All the times to compute the trajectory for [s]
    M_start : 
        Mean anomaly at start [radians]
    H_start:
        hyperbolic anomaly at start
    eccentricity : 
        Eccentricity.
    semimajor_axis : 
        Semimajor axis [km].

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        State for the spacecraft (x, y, z, vx, vy, vz) and the times
    """

    H_start = -np.arccosh(np.multiply((1/np.linalg.norm(eccentricity)),(1-np.linalg.norm(r)/semimajor_axis))) 
    M_start = np.linalg.norm(eccentricity)*np.sinh(H_start)-H_start 
    mean_motion = np.sqrt(np.divide(-constants.MU_JUPITER,(semimajor_axis**3)))

    t_periapsis = -M_start/mean_motion
    times =  np.arange(0,1.5*t_periapsis,1000)

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
    """ Determines the spacecraft position and velocity in an elliptical trajectory.

    Parameters
    ----------
    mean_motion : 
        Mean motion [radians/s].
    times : np.ndarray
        All the times to compute the trajectory for [s]
    M_start : 
        Mean anomaly at start [radians]
    E_start:
        Eccentric anomaly at start
    eccentricity : 
        Eccentricity.
    semimajor_axis : 
        Semimajor axis [km].

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.darray
        State for the spacecraft (x, y, z, vx, vy, vz) and the times
    """
    E_start = np.arccos(np.multiply(1/np.linalg.norm(eccentricity),(1-np.linalg.norm(r)/semimajor_axis)))
    M_start = E_start - np.linalg.norm(eccentricity)*np.sin(E_start)
    mean_motion = np.sqrt(np.divide(constants.MU_JUPITER,(semimajor_axis**3)))

    t_periapsis = M_start/mean_motion
    times = np.arange(0,t_periapsis, 1800)

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




def check_forGA(position,moon_position,mu_moon,semimajor_moon,r_moon):
    """Determines the distance between the spacecraft and a second object at each time step
    and checks whether the spacecraft is inside the sphere of influence.
    It returns 1 if a flyby was found and the index of that position or 0 if there 
    isn't a flyby and a really high number for the index
    """
    distance = util.distances(position,moon_position)
    r_soi = semimajor_moon*np.power(mu_moon/constants.MU_JUPITER, 2.0/5.0) 
    for i in range(len(distance)):
        if distance[i] < r_soi:
        # if (distance[i]-r_moon) < 2000 and (distance[i]-r_moon)>50:
            return 1, i
    
    return 0,1e10


def GravityAssist(position,velocity,moon_position,moon_velocity,moon_radius, mu_moon, r_soi):
            """Performs the gravity assist
            It returns the new velocity, position and the bending angle"""
                
            b1,b2,rx,ry,vx,vy, pos_moon_frame, vel_moon_frame = gravity_assist.change_moon_frame(moon_position,moon_velocity,position,velocity)
            delta,b, v_inf_out = gravity_assist.gravity_assist(vel_moon_frame, pos_moon_frame[0],pos_moon_frame[1], vel_moon_frame[0],vel_moon_frame[1],mu_moon)

            # print("Flyby altitude {} km".format(np.min(np.linalg.norm(pos_moon_frame))-moon_radius))
            k1,k2, pos_jupiter_frame, vel_jupiter_frame = gravity_assist.change_jupiter_frame(b1,b2, pos_moon_frame,vel_moon_frame, moon_position,moon_velocity)
            # t_rsoi = 2*np.linalg.norm(pos_moon_frame)/(np.linalg.norm(vel_moon_frame)) #time it takes the spacecraft to go from one end of SOI to the other 
            v_inf_out_jupiter = np.array([np.dot(k1,v_inf_out),np.dot(k2,v_inf_out),0])+moon_velocity #change v_inf_out back into Jupiter frame

            # Need to shift position vector to outside the sphere of influence.
            # Work out time taken to move across 2r_soi: delta_pos = v_inf_out_jupiter * time
            time_to_cross_soi = 2*r_soi/np.linalg.norm(v_inf_out_jupiter)
            pos_jupiter_frame += v_inf_out_jupiter*time_to_cross_soi
            return v_inf_out_jupiter, pos_jupiter_frame, delta



def determine_orbit(r0,v0,time_start,arc_max,arc_num,results):
    key = arc_num

    if arc_num<arc_max:

        angular_momentum = np.cross(r0, v0)
        eccentricity = np.subtract((np.cross(v0, angular_momentum)/constants.MU_JUPITER),(np.divide(r0,np.linalg.norm(r0))))
        arg_periapsis = np.arccos(eccentricity[0]/np.linalg.norm(eccentricity))
        semimajor_axis = 1.0/(2.0/np.linalg.norm(r0) - (np.linalg.norm(v0)**2)/constants.MU_JUPITER)
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

        ca_flyby, ca_index = check_forGA(position,callisto_state[:,0:3], constants.MU_CALLISTO,constants.A_CALLISTO,constants.R_CALLISTO)
        ga_flyby, ga_index = check_forGA(position,ganymede_state[:,0:3],constants.MU_GANYMEDE,constants.A_GANYMEDE,constants.R_GANYMEDE)

        callisto_closest_dist = np.min(util.distances(position,callisto_state[:,0:3]))
        ganymede_closest_dist = np.min(util.distances(position,ganymede_state[:,0:3]))
        t_to_min_callisto_distance = times[np.argmin(util.distances(position,callisto_state[:,0:3]))]
        t_to_min_ganymede_distance = times[np.argmin(util.distances(position,ganymede_state[:,0:3]))]
        jupiter_closest_dist = util.r_periapsis(semimajor_axis,np.linalg.norm(eccentricity))/71492.0  #in Jupter radii
        
        results[key] = {"starting position [km]":r0, "starting velocity [km\s]":v0,"semimajor axis": semimajor_axis, 
                        "eccentricity":np.linalg.norm(eccentricity) ,"starting time [s]":time_start,
                        "closest distance to Jupiter [R_J]":jupiter_closest_dist, "closest distance to callisto [km]": callisto_closest_dist, "time to min distance to Callisto [s]": t_to_min_callisto_distance,
                        "Callisto GA": ca_flyby, "closest distance to ganymede [km]":ganymede_closest_dist, "time to min distance to Ganymede [s]":t_to_min_ganymede_distance, "Ganymede GA":ga_flyby}

        if ca_flyby == 1:
            if ga_flyby == 1:
                if ga_index < ca_index:
                    index = ga_index
                    v_inf, new_position, delta = GravityAssist(position[ga_index], velocity[ga_index],ganymede_state[ga_index,0:3],ganymede_state[ga_index,3:6],constants.R_GANYMEDE,constants.MU_GANYMEDE, constants.R_SOI_GANYMEDE)
                else:
                    index = ca_index
                    v_inf, new_position, delta = GravityAssist(position[ca_index],velocity[ca_index],callisto_state[ca_index,0:3], callisto_state[ca_index,3:6],constants.R_CALLISTO, constants.MU_CALLISTO, constants.R_SOI_CALLISTO)
            else:
                    index = ca_index
                    v_inf, new_position, delta = GravityAssist(position[ca_index],velocity[ca_index],callisto_state[ca_index,0:3], callisto_state[ca_index,3:6],constants.R_CALLISTO,constants.MU_CALLISTO, constants.R_SOI_CALLISTO)

            # print(v_inf,new_position)
            arc_num += 1
            results = determine_orbit(new_position,v_inf,time[index],arc_max,arc_num,results)



        else:
            print('No gravity assist found for arc number {}'.format(arc_num))
            return results
            

  
    return results



# r = np.array([46902972.10058936, 53955697.30928642     ,   0.        ])
# v = np.array([-2.44575532 ,-2.36183846  ,0.        ])
results = {}
t_start = 58849.0*86400.0 #[seconds]
r = np.array([46335387.96680537, 54443896.68241637 , 0.        ])
r2 = np.array([46145061.01076064 ,54605305.67914784    ,    0.        ])
v = np.array([-2.42088856 ,-2.38732038 , 0.        ])
v2 = np.array([-2.4125405 , -2.39575631,  0.        ])

result = determine_orbit(r,v,t_start,3,0,results)
# for key, value in result.items():
#     print(f"Arc {key}:")
#     for k, v in value.items():
#         print(f"{k}: {v}")


def grid_search(min_deg_r=49.0, max_deg_r=55.0, delta_deg_r=0.25,
                min_deg_v=175.0, max_deg_v=177.5, delta_deg_v=0.25):
    degrees_r = np.arange(min_deg_r,max_deg_r,delta_deg_r)
    degrees_v = np.arange(min_deg_v,max_deg_v,delta_deg_v)

    for i in range(len(degrees_r)):  
        for j in range(len(degrees_v)):
     
            print("Computing trajectory: deg_r {}/{}, deg_v {}/{}".format(i+1, len(degrees_r), j+1, len(degrees_v)))
            print(degrees_r[i], degrees_v[j])
            delta = np.radians(degrees_r[i])
            theta = np.radians(degrees_v[j])

            r = [1000*constants.R_JUPITER  * np.cos(delta), 1000*constants.R_JUPITER  * np.sin(delta), 0]
            v = [3.4 * np.cos(delta+ theta), 3.4 * np.sin(delta + theta), 0]

            result = determine_orbit(r,v,t_start,3,0,{})
            if 1 in result:
                for arc_num in result.keys():
                    print(arc_num, result[arc_num])

# What we had before.
#grid_search(max_deg_v=176.5)


def cost_fun_only_eccentricity(start_deg_r, start_deg_v, t_start):
    delta = np.radians(start_deg_r)
    theta = np.radians(start_deg_v)
    r = [1000*constants.R_JUPITER  * np.cos(delta), 1000*constants.R_JUPITER  * np.sin(delta), 0]
    v = [3.4 * np.cos(delta+ theta), 3.4 * np.sin(delta + theta), 0]

    arcs = determine_orbit(r,v,t_start,3,0,{})

    cost = arcs[max(arcs.keys())]["eccentricity"]
    costs = []
    closest_distances_to_jupiter = []
    # print(cost)
    # cost = 0.0

    for key, value in arcs.items():
        # cost1 = cost + np.exp(-value["closest distance to Jupiter [R_J]"]/2.0)
        cost2 =cost + np.exp(-value["closest distance to callisto [km]"]/50.0)
        cost1 = cost2+np.exp(-2000/value["closest distance to callisto [km]"])
    costs.append(cost1)
    closest_distances_to_jupiter.append(value["closest distance to callisto [km]"])
    # print(cost1,value["closest distance to callisto [km]"])

    return closest_distances_to_jupiter, costs

    # return cost1,cost2,cost3, arcs


# Plot each cost

# deg_r = np.random.uniform(49.0,50.0, size=4)
# deg_v = np.random.uniform(175.0,176.0, size=4)
# for i in range(4):
#     print(cost_fun_only_eccentricity(deg_r[i], deg_v[i], 58849.0*86400.0))


    
deg_r = np.arange(0,60,1.25)
deg_v = np.arange(175.0,179.0,0.1)

start_deg_r, start_deg_v = np.meshgrid(deg_r, deg_v)
cost_values = np.zeros_like(start_deg_r,dtype=float)

all_costs = []
all_distances = []

for i in range(len(deg_r)):
    for j in range(len(deg_v)):

        print("Computing trajectory: deg_r {}/{}, deg_v {}/{}".format(i+1, len(deg_r), j+1, len(deg_v)))
        # cost_fun_only_eccentricity(deg_r[i], deg_v[j], 58849.0 * 86400.0)
        distances, costs = cost_fun_only_eccentricity(deg_r[i], deg_v[j], 58849.0 * 86400.0)
        all_distances.extend(distances)
        all_costs.extend(costs)
        cost_values[j,i] = costs[0]

# plt.plot(all_distances, all_costs, 'o',color='blue')
# plt.xlabel('Closest Distance to Callisto [km]')
# plt.ylabel('Cost')


# plt.show()


# plt.plot(cost_values)

# Plot the cost values using pcolormesh
plt.pcolormesh(start_deg_r, start_deg_v, cost_values,norm=matplotlib.colors.Normalize(vmin=cost_values.min(), vmax=cost_values.max()))
plt.colorbar().set_label("Cost Function")
plt.xlabel("Position angle [degrees]")
plt.ylabel("Velocity angle [degrees]")
plt.show()

# 49.0 175.25 no GA
# 49.0 175.0 Callisto GA