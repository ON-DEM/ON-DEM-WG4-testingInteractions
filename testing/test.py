# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from motion_functions import *
from contact_functions import *
from contact_laws import *


# Generate velocities and motion profile
R = 1.0
motion = my_simulate_motion(
    [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
    [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # initial ori
    0, R/2, 1, np.pi, 0.05, [2*R,0.0,0.0], # normal loading, initial branch
    0, 0, 0, 0, 0, # twist
    0, 0, 0, 0, 0, # roll
    0, 0, 0, 0, 0, # shear
    [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
    6*np.pi, 6*np.pi/200 # time
)


# Simulate contact interaction
contact_params = {'k_n':100, 
                  'k_t':50, 
                  'mu':0.5, 
                  'R_i': R,
                  'R_j': R}

results = my_simulate_contact(
    motion,
    contact_params,
    Fn_linear_elastic,
    Ft_linear_Coloumb)


# Plotting motion
plt.figure(0)
plt.plot(motion['t'], motion['x_i'][:,0],'r', label='x_i')
plt.plot(motion['t'], motion['x_j'][:,0],'b', label='x_j')
plt.plot(motion['t'], motion['v_i'][:,0],'r--', label='v_i')
plt.plot(motion['t'], motion['v_j'][:,0],'b--', label='v_j')
plt.plot(results['t'], results['u_n'],'g', label='u_n')
plt.xlabel('Time')
plt.ylabel('Position or velocity')
plt.legend()
plt.show(block=False)
plt.savefig('test_plot.png')

# Plotting forces and torques
plt.figure(1)
plt.plot(results['t'], results['F_i'][:,0],'r', label='F_i')
plt.plot(results['t'], results['F_j'][:,0],'b', label='F_j')
plt.plot(results['t'], results['T_i'][:,0],'r--', label='T_i')
plt.plot(results['t'], results['T_j'][:,0],'b--', label='T_j')
plt.xlabel('Time')
plt.ylabel('Force or torque')
plt.legend()
plt.show(block=False)

writeDemInput(results, 'test_input.txt')
# End of file


