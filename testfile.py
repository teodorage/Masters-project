import numpy as np
import gravity_assist


def dot_product(a,b): #function to calculate the dot product of two vectors

    result = a[0]*b[0] + a[1]* b[1] +a[2]* b[2]
    return result
def change_to_moon_frame(position, velocity, spacecraft_position, spacecraft_velocity):

    b1 = -np.divide(position, np.linalg.norm(position))  #set of coordinates in Moon frame
    b3_unorm = np.cross(position, velocity)
    b3 = b3_unorm/np.linalg.norm(b3_unorm)
    b2 = np.cross(b3, b1)

    rx = dot_product(b1, spacecraft_position-position) # spacecraft x coord in Moon frame
    ry = dot_product(b2, spacecraft_position-position) # spacecraft y coord in Moon frame


    vx = dot_product(b1, spacecraft_velocity-velocity) # spacecraft velocity x coord in moon frame
    vy = dot_product(b2, spacecraft_velocity-velocity) # spacecraft velocity y coord in moon frame

  

    return b1,b2,rx,ry, vx,vy

def dot(a,b):
     result = a[0]*b[0] + a[1]* b[1]
     return result
 
def change_to_jupiter_frame(b1,b2,pos_in_moon_frame):
    
    k1 = np.array([b1,b2]).T
    k2 = np.array([b1,b2]).T

    x_j = dot(k1,pos_in_moon_frame)
    y_j = dot(k2, pos_in_moon_frame)
    z_j = np.zeros(len(x_j))
    
    return k1, k2, x_j, y_j

pos = np.array([13200,0,0])
vel = np.array([3,2,0])
spacepos = np.array([3200,0,0])
spacevel = np.array([2,0,0])
b1,b2,rx,ry, vx,vy = change_to_moon_frame(pos,vel,spacepos,spacevel)
newpos = np.array([rx,ry])
k1, k2, x,y = change_to_jupiter_frame(b1,b2,newpos)
print(rx,ry)
print(x,y)