from yade import utils, plot, qt
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./')
from input_functions import *

R = 1.0

t = np.linspace(0,6*np.pi,200)
V_i = 0.0
A_i = R/2
w_i = 1
k_i = 0.05
x0 = 0.5

# Particle i
x_i = my_input_position_i(t, x0, V_i, A_i, w_i, k_i)
y_i, z_i = np.zeros((len(t),)), np.zeros((len(t),))
pos_i = np.stack((x_i,y_i,z_i),axis=1)

vx_i  = my_input_velocity_i(t, V_i, A_i, w_i, k_i)
vy_i, vz_i = np.zeros((len(t),)), np.zeros((len(t),))
vel_i = np.stack((vx_i,vy_i,vz_i),axis=1)

omega_i = np.zeros((len(t), 3))

# Particle j
x_j = my_input_position_i(t, x0-2*R, V_i, 0, w_i, k_i)
y_j, z_j = np.zeros((len(t),)), np.zeros((len(t),))
pos_j = np.stack((x_j, y_j, z_j), axis=1)

vx_j  = my_input_velocity_i(t, V_i, 0, w_i, k_i)
vy_j, vz_j = np.zeros((len(t),)), np.zeros((len(t),))
vel_j = np.stack((vx_j, vy_j, vz_j), axis=1)

omega_j = np.zeros((len(t), 3))

# Plotting
plt.figure(0)
plt.plot(t, x_i,'r', label='x_i')
plt.plot(t, vx_i,'r--', label='vx_i')
plt.plot(t, x_j,'b', label='x_j')
plt.plot(t, vx_j,'b--', label='vx_j')
plt.xlabel('Time')
plt.ylabel('Position or velocity')
plt.legend()
plt.show(block=False)

contact_pen = my_get_contact_pen(pos_i, pos_j, vel_i, vel_j, R_i=R, R_j=R)
contact_rot = my_get_contact_rot(omega_i, omega_j, contact_pen["n_ij"])

plt.figure(1)
plt.plot(t, contact_pen["penetration"],'g', label='u_n')
plt.plot(t, contact_pen["delta_vel_n"][:,0],'r', label='v_n')
plt.plot(t, contact_pen["delta_vel_t"][:,0],'b', label='v_t')
plt.xlabel('Time')
plt.ylabel('Penetration or relative velocity')
plt.legend()
plt.show(block=False)