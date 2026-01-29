# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from A_motion_functions import *
from B_contact_functions import *
from C_contact_laws import *
from D_helpers import *

# Which test to run?
# Get testID from command line argument
if len(sys.argv) < 2:
    print("Usage: python F_run_test.py <testID>")
    print("Example: python F_run_test.py 1")
    sys.exit(1)

testID = int(sys.argv[1])
testname = 'test_'+str(testID).zfill(2)
doPlot = False

# Generate velocities and motion profile
R_i = 1.0
R_j = 1.0
R = (R_i + R_j)/2.0
tmax = 6.0*np.pi
dt = 6.0*np.pi/1.0e5
if testID == 1:
    # Tangential elastic response
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 2:
    # Tangential plastic response
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.08*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 3:
    # Out-of-plane tangent force rotation
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[1.0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 4:
    # In-plane tangent force rotation
    motion = my_simulate_motion(
        [0,0,0],[0,0,0],[0,0,1.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 5:
    # Carnot cycle: approach-load, shear forward at high load, unload, shear back at low load
    # Period = 3π, two complete cycles in tmax = 6π
    # Normal: oscillate between approach and unload
    # Shear: 90° out of phase - shear at maximum and minimum penetration depths
    w_cycle = 2*np.pi / (3*np.pi)  # frequency for 3π period
    # Place initial branch along z: initial position/branch at [0,0,1.95*R]
    # Choose roll and shear axes orthogonal to branch (x and y)
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.06*R, w_cycle, np.pi/2, 0, [0.0,0.0,1.95*R], # normal: approach-unload cycle (branch along z)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.03*R, w_cycle, 0, 0, # shear: 90° out of phase with normal (peaks at load transitions)
        [1.0,0.0,0.0], [0.0,1.0,0.0], # roll and shear axes (both orthogonal to branch)
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 6:
    # Purely repulsive viscous forces
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 7:
    # Viscous force discontinuity at zero overlap
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 8:
    # Including or excluding viscous force from Coulomb limit
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 9:
    # Particle size effect
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [0.5*1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        0.5*R_i, 0.5*R_j
    )
elif testID == 10:
    # Oblique impact?
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 11:
    # Complex rotation case?
    motion = my_simulate_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 1, 0, 0, [1.95*R,0.0,0.0], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 1, 0, 0, # shear
        [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )

# Set the phase to pi to start with approach


# Simulate contact interaction
contact_params = {'k_n':    1.0e7, 
                  'k_s':    0.5e7, 
                  'k_r':    0.0, 
                  'k_t':    0.0,
                  'mu':     0.5, 
                  'eta_n':  0.0,
                  'eta_s':  0.0,
                  'eta_r':  0.0,
                  'eta_t':  0.0,
                  'R_i':    R_i,
                  'R_j':    R_j}

results = my_simulate_contact(
    motion,
    contact_params,
    Fn_spring_dashpot,
    Fs_spring_dashpot_Coulomb)

# import numpy as np
# # initial position (vector)
# x0 = np.asarray(results['x_j'][0])

# # velocities (Nt, 3)
# v = np.asarray(results['v_j'])

# # integrate velocity to get displacement
# # forward Euler: x_{n+1} = x_n + v_n * dt
# displacements = np.cumsum(v * dt, axis=0)

# # full position history
# x_j = x0 + displacements

# # magnitude of position vector at each timestep
# r = np.linalg.norm(x_j, axis=1)

# # subtract 1.95
# out = abs(r - 1.95)/1.95*100

# # print result
# print(np.mean(out))
# print(np.std(out))

if doPlot:
    # Plotting motion
    plt.figure(0)
    plt.plot(results['t'], results['x_i'][:,0],'r', label='x_i')
    plt.plot(results['t'], results['x_j'][:,0],'b', label='x_j')
    plt.plot(results['t'], results['v_i'][:,0],'r--', label='v_i')
    plt.plot(results['t'], results['v_j'][:,0],'b--', label='v_j')
    plt.plot(results['t'], results['u_n'],'g', label='u_n')
    plt.xlim(0, tmax)
    plt.xlabel('Time')
    plt.ylabel('Position or velocity')
    plt.legend()
    plt.show(block=False)
    plt.savefig('../figures/'+testname+'_plot_pos.png')

    # Plotting forces and torques x
    plt.figure(1)
    plt.plot(results['t'], results['F_i'][:,0],'r', label='F_i')
    plt.plot(results['t'], results['F_j'][:,0],'b', label='F_j')
    plt.plot(results['t'], results['T_i'][:,0],'r--', label='T_i')
    plt.plot(results['t'], results['T_j'][:,0],'b--', label='T_j')
    plt.xlim(0, tmax)
    plt.xlabel('Time')
    plt.ylabel('Force or torque x')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend()
    plt.show(block=False)
    plt.savefig('../figures/'+testname+'_plot_force_x.png')

    # Plotting forces and torques y
    plt.figure(2)
    plt.plot(results['t'], results['F_i'][:,1],'r', label='F_i')
    plt.plot(results['t'], results['F_j'][:,1],'b', label='F_j')
    plt.plot(results['t'], results['T_i'][:,1],'r--', label='T_i')
    plt.plot(results['t'], results['T_j'][:,1],'b--', label='T_j')
    plt.xlim(0, tmax)
    plt.xlabel('Time')
    plt.ylabel('Force or torque y')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend()
    plt.show(block=False)
    plt.savefig('../figures/'+testname+'_plot_force_y.png')

    # Plotting forces and torques z
    plt.figure(3)
    plt.plot(results['t'], results['F_i'][:,2],'r', label='F_i')
    plt.plot(results['t'], results['F_j'][:,2],'b', label='F_j')
    plt.plot(results['t'], results['T_i'][:,2],'r--', label='T_i')
    plt.plot(results['t'], results['T_j'][:,2],'b--', label='T_j')
    plt.xlim(0, tmax)
    plt.xlabel('Time')
    plt.ylabel('Force or torque z')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    plt.legend()
    plt.show(block=False)
    plt.savefig('../figures/'+testname+'_plot_force_z.png')

dict_to_json(results,'../output_ANA/theoretical_output_'+testname+'.json')
dict_to_csv(results, open('../output_ANA/theoretical_output_'+testname+'.csv', 'w'))

demInputs = {k: results[k] for k in ['t', 'v_i', 'v_j', 'omega_i', 'omega_j']}
dict_to_json(demInputs,'../input_DEM/dem_input_'+testname+'.json')
write_DEM_input(results, '../input_DEM/dem_input_'+testname+'.csv')

# End of file