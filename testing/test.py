# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from motion_functions import *
from contact_functions import *
from contact_laws import *
from helpers import *

# Which test to run?
testID = 4

# Generate velocities and motion profile
R = 1.0
if testID == 1:
    # Tangential elastic response
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [1.0,0,0,0], [1.0,0,0,0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 2:
    # Tangential plastic response
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [1.0,0,0,0], [1.0,0,0,0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.08*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 3:
    # Out-of-plane tangent force rotation
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,1.0,0], # initial pos, vel, ang vel
        [1.0,0,0,0], [1.0,0,0,0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 4:
    # In-plane tangent force rotation
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,1.0], # initial pos, vel, ang vel
        [1.0,0,0,0], [1.0,0,0,0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 5:
    # Purely repulsive viscous forces
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 6:
    # Shear displacement calculated with surface arm
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )
elif testID == 7:
    # Do nothing
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        6*np.pi, 6*np.pi/200 # time
    )

# Make phase pi to start with approach


# Simulate contact interaction
contact_params = {'k_n':    1.0e7, 
                  'k_t':    0.5e7, 
                  'mu':     0.5, 
                  'R_i':    R,
                  'R_j':    R}

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
plt.savefig('test_plot_pos.png')

# Plotting forces and torques x
plt.figure(1)
plt.plot(results['t'], results['F_i'][:,0],'r', label='F_i')
plt.plot(results['t'], results['F_j'][:,0],'b', label='F_j')
plt.plot(results['t'], results['T_i'][:,0],'r--', label='T_i')
plt.plot(results['t'], results['T_j'][:,0],'b--', label='T_j')
plt.xlabel('Time')
plt.ylabel('Force or torque x')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.legend()
plt.show(block=False)
plt.savefig('test_plot_force_x.png')

# Plotting forces and torques y
plt.figure(2)
plt.plot(results['t'], results['F_i'][:,1],'r', label='F_i')
plt.plot(results['t'], results['F_j'][:,1],'b', label='F_j')
plt.plot(results['t'], results['T_i'][:,1],'r--', label='T_i')
plt.plot(results['t'], results['T_j'][:,1],'b--', label='T_j')
plt.xlabel('Time')
plt.ylabel('Force or torque y')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.legend()
plt.show(block=False)
plt.savefig('test_plot_force_y.png')

# Plotting forces and torques z
plt.figure(3)
plt.plot(results['t'], results['F_i'][:,2],'r', label='F_i')
plt.plot(results['t'], results['F_j'][:,2],'b', label='F_j')
plt.plot(results['t'], results['T_i'][:,2],'r--', label='T_i')
plt.plot(results['t'], results['T_j'][:,2],'b--', label='T_j')
plt.xlabel('Time')
plt.ylabel('Force or torque z')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.legend()
plt.show(block=False)
plt.savefig('test_plot_force_z.png')

dict_to_json(results,'theoretical_output.json')
dict_to_csv(results, open('theoretical_output.csv', 'w'))

demInputs = {k: results[k] for k in ['t', 'v_i', 'v_j', 'omega_i', 'omega_j']}
dict_to_json(demInputs,'dem_input.json')
write_DEM_input(results, 'dem_input.csv')

# End of file