
import numpy as np
import matplotlib.pyplot as plt
import util
import constants
import random
import matplotlib.patches


r = 1000*constants.R_JUPITER # distance from jupiter to the spacecraft [km]
v = 3.4 # speed [km/s]
eccentricity = 1.4
#H = np.random(0, 2*np.pi) #hyperbolic anomaly
a = 1/(2/r-(v**2/constants.MU_JUPITER)) #semimajor axis
N = np.sqrt(-constants.MU_JUPITER/a**3) #mean motion
#N*time = eccentricity*np.sinh(H)-H
def calculate_time(eccentricity,a, N):
    num_list = np.arange(0,2*np.pi,0.01)
    time = []
    H_anomaly = []
    x=[]
    y=[]
    pos = []
    M = [] #mean anomaly
    for i in range(0, len(num_list)):
        Hyper = num_list[i]
        t = ((eccentricity*np.sinh(Hyper))-Hyper)/N
        x_coord = a*(np.cosh(Hyper)-eccentricity) #x coordinate
        y_coord = -a*np.sqrt(eccentricity**2-1)*np.sinh(Hyper) #y coordinate
        position = np.sqrt(x_coord**2+y_coord**2)
        mean = eccentricity*np.sinh(Hyper)-Hyper
        x.append(x_coord)
        y.append(y_coord)
        time.append(t)
        H_anomaly.append(Hyper)
        pos.append(position)
        M.append(mean)
    return time, H_anomaly,x,y, pos, M


time, H,x,y, pos, mean =calculate_time(eccentricity,a,N)
# x_coord = a*(np.cosh(H)-eccentricity) #x coordinate
# y_coord = -a*np.sqrt(eccentricity**2-1)*np.sinh(H) #y coordinate

plt.figure()
# plt.plot(mean,np.degrees(H))
# plt.plot(pos,np.degrees(H))
plt.ylabel('H')
plt.xlabel('Time')
# plt.show()
# print(time, np.degrees(H))
# print(x,y)
print(N,a)