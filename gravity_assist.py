import numpy as np
import constants
import matplotlib.patches
import kepler
import math


def dot_product(a,b): #function to calculate the dot product of two vectors

    result = a[:,0]*b[:,0] + a[:,1]* b[:,1] +a[:,2]* b[:,2]
    return result


def change_to_moon_frame(position, velocity, spacecraft_position, spacecraft_velocity):

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
    
    k1 = np.array([b1[:,0],b2[:,0]]).T
    k2 = np.array([b1[:,1],b2[:,1]]).T

    x_j = dot(k1,pos_in_moon_frame[:,0:2])
    y_j = dot(k2, pos_in_moon_frame[:,0:2])
    z_j = np.zeros(len(x_j))
    pos_in_jupiter_frame = np.array(list(zip(x_j,y_j,z_j))) + position[:,0:3]

    vx_j = dot(k1,vel_in_moon_frame)
    vy_j   = dot(k2, vel_in_moon_frame)
    vz_j = np.zeros(len(vx_j))
    vel_in_jupiter_frame = np.array(list(zip(vx_j,vy_j,vz_j)))
   
    
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


