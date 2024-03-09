import numpy as np
import constants
import matplotlib.patches
import kepler
import math


def dot_product(a,b): #function to calculate the dot product of two vectors

    result = a[:,0]*b[:,0] + a[:,1]* b[:,1] +a[:,2]* b[:,2]
    return result


def change_moon_frame(position, velocity, spacecraft_position, spacecraft_velocity): 
    """This function is for one position and velocity vector"""

    b1 = -np.divide(position, np.linalg.norm(position))  #set of coordinates in Moon frame
    b3_unorm = np.cross(position, velocity)
    b3 = b3_unorm/np.linalg.norm(b3_unorm)
    b2 = np.cross(b3, b1)

    rx = np.dot(b1, spacecraft_position-position) # spacecraft x coord in Moon frame
    ry = np.dot(b2, spacecraft_position-position) # spacecraft y coord in Moon frame

    vx = np.dot(b1, spacecraft_velocity-velocity) # spacecraft velocity x coord in moon frame
    vy = np.dot(b2, spacecraft_velocity-velocity) # spacecraft velocity y coord in moon frame

    new_position = np.array([rx,ry])
    new_velocity = np.array([vx,vy])

    return b1,b2,rx,ry, vx,vy, new_position, new_velocity

def change_jupiter_frame(b1,b2,pos_in_moon_frame,vel_in_moon_frame, position, velocity ):
    """This funciton is for only one position and velocity vector"""
    k1 = np.array([b1[0],b2[0]]).T
    k2 = np.array([b1[1],b2[1]]).T

    x_j = np.dot(k1,pos_in_moon_frame[0:2])
    y_j = np.dot(k2, pos_in_moon_frame[0:2])
    z_j = 0
    pos_in_jupiter_frame = np.array([x_j,y_j,z_j]) + position[0:3]

    vx_j = np.dot(k1,vel_in_moon_frame)
    vy_j = np.dot(k2, vel_in_moon_frame)
    vz_j = 0
    vel_in_jupiter_frame = np.array([vx_j,vy_j,vz_j])+velocity[0:3]
   
    
    return k1, k2, pos_in_jupiter_frame,vel_in_jupiter_frame







def change_to_moon_frame(position, velocity, spacecraft_position, spacecraft_velocity):
    """This function is for an array of position and velocity vectors"""

    b1 = -np.divide(position, np.linalg.norm(position, axis = 1)[:,None])  #set of coordinates in Moon frame
    b3_unorm = np.cross(position, velocity)
    b3 = b3_unorm/np.linalg.norm(b3_unorm, axis=1)[:,None]
    b2 = np.cross(b3, b1)

    rx = dot_product(b1, spacecraft_position-position) # spacecraft x coord in Moon frame
    ry = dot_product(b2, spacecraft_position-position) # spacecraft y coord in Moon frame
    rz = np.zeros(len(rx))

    vx = dot_product(b1, spacecraft_velocity-velocity) # spacecraft velocity x coord in moon frame
    vy = dot_product(b2, spacecraft_velocity-velocity) # spacecraft velocity y coord in moon frame

    new_position = np.array(list(zip(rx,ry)))
    new_velocity = np.array(list(zip(vx,vy)))

    return b1,b2,rx,ry, vx,vy, new_position, new_velocity

def dot(a,b):
     result = a[:,0]*b[:,0] + a[:,1]* b[:,1]
     return result
 
def change_to_jupiter_frame(b1,b2,pos_in_moon_frame,vel_in_moon_frame, position, velocity ):
    """This function is for an array of position and velocity vectors"""
    
    k1 = np.array([b1[:,0],b2[:,0]]).T
    k2 = np.array([b1[:,1],b2[:,1]]).T

    x_j = dot(k1,pos_in_moon_frame[:,0:2])
    y_j = dot(k2, pos_in_moon_frame[:,0:2])
    z_j = np.zeros(len(x_j))
    pos_in_jupiter_frame = np.array(list(zip(x_j,y_j,z_j))) + position[:,0:3]

    vx_j = dot(k1,vel_in_moon_frame)
    vy_j   = dot(k2, vel_in_moon_frame)
    vz_j = np.zeros(len(vx_j))
    vel_in_jupiter_frame = np.array(list(zip(vx_j,vy_j,vz_j)))+velocity[:,0:3]
   
    
    return k1, k2, pos_in_jupiter_frame,vel_in_jupiter_frame



def gravity_assist(v_inf,r0x,r0y,vx,vy):

    b = (r0y*vx-r0x*vy)/(np.sqrt(vx**2+vy**2)) #impact parameter
    closest_distance = (constants.MU_CALLISTO/(np.linalg.norm(v_inf)**2))*(np.sqrt(1+(b*(np.linalg.norm(v_inf)**2)/constants.MU_CALLISTO)**2)-1)
    frac = constants.MU_CALLISTO/(closest_distance)

    delta = 2.0 * np.arcsin(frac/((np.linalg.norm(v_inf)**2)+frac)) #bending angle

    rotation_matrix = np.array([[math.cos(delta), -math.sin(delta)],
                                [math.sin(delta), math.cos(delta)]])

   
    v_inf_out = rotation_matrix @ v_inf[0:2]

    return delta ,b, v_inf_out


def keplersolver(mean_anomaly: float, eccentricity: float):
    
    """Solving Keplers equation for an elliptical orbit"""

    relative_change_epsilon: float=1e-6
    max_iter = 100
    # Iterate using Newton-Raphson.
    iter = 0
    relative_change = 1e6
        # Initial starting guess for the eccentric anomaly.
    ecc_anomaly = 0.0
    if eccentricity>=0.8:
        ecc_anomaly = np.pi

        # Iterate using Newton-Raphson.
    iter = 0
    relative_change = 1e6
        
    while np.any(relative_change>relative_change_epsilon) and (iter< max_iter):
            # Compute the delta to add onto the eccentric anomaly for this step, 
            # work out the new eccentric anomaly estimate, work out the relative
            # change and increment the iteration variable.
        delta_ecc_anomaly = (mean_anomaly + eccentricity*np.sin(ecc_anomaly) - ecc_anomaly)/(1-eccentricity*np.cos(ecc_anomaly))
        ecc_anomaly += delta_ecc_anomaly
        relative_change = np.abs(delta_ecc_anomaly/(ecc_anomaly+1e-300))
        iter += 1
        return ecc_anomaly