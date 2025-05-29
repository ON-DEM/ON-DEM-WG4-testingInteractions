import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from motion_functions import *


R = 1.0

results = my_simulate_motion(
    [0.0,0.0,0.0], [0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos
    [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # inital ori
    0, R/2, 1, 0.0, 0.05, [1.0,0.0,0.0], # normal loading
    0, 0, 0, 0, 0, # twist
    0, 0, 0, 0, 0, # roll
    0, 0, 0, 0, 0, # shear
    [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
    6*np.pi, 6*np.pi/200 # time
)


# Plotting
plt.figure(0)
plt.plot(results['t'], results['x_i'][:,0],'r', label='x_i')
plt.plot(results['t'], results['x_j'][:,0],'b', label='x_j')
plt.plot(results['t'], results['v_i'][:,0],'r--', label='v_i')
plt.plot(results['t'], results['v_j'][:,0],'b--', label='v_j')
plt.xlabel('Time')
plt.ylabel('Position or velocity')
plt.legend()
plt.show(block=False)

