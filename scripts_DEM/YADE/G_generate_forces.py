# Copyright 2025: Bruno Chareyre <bruno.chareyre@grenoble-inp.fr> and Danny van der Haven <dlhv2@cantab.ac.uk>
# Execution: "yade generateForces.py input.txt output.csv"
# input.json is the analytical output from F_generate_analytical.py, which already contains
# both the trajectory data and the contact_params dict used to generate it.

from yade import plot, sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../scripts_PY')
from D_helpers import *

if len(sys.argv) > 2:
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
else:
	print("Usage: yade G_generate_forces.py <input_file> <output_file>")
	exit()
      
imposePos = True  # Whether to impose positions and orientations, or only velocities

# --- Load imposed state data from text file ---
imposed_data = json_to_dict(inputFile)

# keys to exclude (do not import these)
_exclude = {'omega_b', 'n_ij', 'v_ijn', 'a_ijn', 'l_ij',
            'u_n', 'v_s', 'v_r', 'v_theta','du_s','du_r','du_theta'}

# build selected dict with everything except the excluded keys
selected = {k: v for k, v in imposed_data.items() if k not in _exclude}

# --- Read contact parameters from the JSON (written by F_generate_analytical.py) ---
# Fall back to sensible defaults if an older JSON without contact_params is loaded.
_cp_defaults = {'k_n': 1.0e7, 'k_s': 0.5e7, 'k_r': 0.0, 'k_t': 0.0,
                'mu': 0.5,
                'eta_n': 0.0, 'eta_s': 0.0, 'eta_r': 0.0, 'eta_t': 0.0,
                'R_i': 1.0, 'R_j': 1.0}
_cp = {**_cp_defaults, **(imposed_data.get('contact_params', {}))}
 
kn  = float(_cp['k_n'])   # normal stiffness [N/m]
ks  = float(_cp['k_s'])   # shear stiffness [N/m]
kr = float(_cp['k_r'])   # rolling stiffness [N m/rad]
kt = float(_cp['k_t'])   # twisting stiffness [N m/rad]
mu  = float(_cp['mu'])    # friction coefficient [-]
etan = float(_cp['eta_n']) # normal viscosity [kg/s]
etas = float(_cp['eta_s']) # shear viscosity [kg/s]
etar = float(_cp['eta_r']) # rolling viscosity [kg m^2/s]
etat = float(_cp['eta_t']) # twisting viscosity [kg m^2/s]
R_i = float(_cp['R_i'])   # radius of particle i [m]
R_j = float(_cp['R_j'])   # radius of particle j [m]

# --- Basic validation / conversion helpers ---
def as_np(name):
    val = selected.get(name, None)
    return None if val is None else np.asarray(val)

# Common arrays (None if missing)
t         = as_np('t')                # expected shape (N,)
N = len(imposed_data['t'])
dt        = as_np('dt')               # may be scalar or (N,)
x_i       = as_np('x_i')              # expected shape (N,3)
x_j       = as_np('x_j')              # expected shape (N,3)
v_i       = as_np('v_i')              # expected shape (N,3)
v_j       = as_np('v_j')              # expected shape (N,3)
a_i       = as_np('a_i')              # expected shape (N,3)
a_j       = as_np('a_j')              # expected shape (N,3)
q_i       = as_np('q_i')              # expected shape (N,4) qx,qy,qz,qw
q_j       = as_np('q_j')              # expected shape (N,4)
omega_i   = as_np('omega_i')  		  # expected shape (N,3)
omega_j   = as_np('omega_j')  		  # expected shape (N,3)
F_i       = as_np('F_i')              # analytical forces if present (N,3)
F_j       = as_np('F_j')              # analytical forces if present (N,3)
T_i       = as_np('T_i')              # analytical torques if present (N,3)
T_j       = as_np('T_j')              # analytical torques if present (N,3)

# --- Initialize simulation scene ---
# Add a dummy material
O.materials.append(FrictMat(young=kn, poisson=(ks/kn), frictionAngle=atan(mu)))

sphere1 = sphere(center=(x_i[0,0], x_i[0,1], x_i[0,2]), radius=R_i, fixed=True)
sphere2 = sphere(center=(x_j[0,0], x_j[0,1], x_j[0,2]), radius=R_j, fixed=True)
O.bodies.append([sphere1, sphere2])

O.bodies[0].state.pos = (x_i[0,0], x_i[0,1], x_i[0,2])
O.bodies[1].state.pos = (x_j[0,0], x_j[0,1], x_j[0,2])
O.bodies[0].state.vel = (v_i[0,0], v_i[0,1], v_i[0,2])
O.bodies[1].state.vel = (v_j[0,0], v_j[0,1], v_j[0,2])
O.bodies[0].state.angVel = (omega_i[0,0], omega_i[0,1], omega_i[0,2])
O.bodies[1].state.angVel = (omega_j[0,0], omega_j[0,1], omega_j[0,2])
O.bodies[0].state.ori = Quaternion((q_i[0,0], q_i[0,1], q_i[0,2]), q_i[0,3])
O.bodies[1].state.ori = Quaternion((q_j[0,0], q_j[0,1], q_j[0,2]), q_j[0,3])

# --- Time stepping logic ---
current_index = 0
def imposeState():
    global current_index
    if current_index < N:
        # velocities (arrays must be shape (N,3))
        O.bodies[0].state.vel = (v_i[current_index,0], v_i[current_index,1], v_i[current_index,2])
        O.bodies[1].state.vel = (v_j[current_index,0], v_j[current_index,1], v_j[current_index,2])

        # angular velocities (arrays must be shape (N,3))
        O.bodies[0].state.angVel = (omega_i[current_index,0], omega_i[current_index,1], omega_i[current_index,2])
        O.bodies[1].state.angVel = (omega_j[current_index,0], omega_j[current_index,1], omega_j[current_index,2])

        # positions (only if imposePos is True) -- arrays must be shape (N,3)
        if imposePos:
            O.bodies[0].state.pos = (x_i[current_index,0], x_i[current_index,1], x_i[current_index,2])
            O.bodies[1].state.pos = (x_j[current_index,0], x_j[current_index,1], x_j[current_index,2])

        	# orientations (if imposePos is True) -- shape (N,4) in [qx,qy,qz,qw] order
            O.bodies[0].state.ori = Quaternion((q_i[current_index,0], q_i[current_index,1], q_i[current_index,2]), q_i[current_index,3])
            O.bodies[1].state.ori = Quaternion((q_j[current_index,0], q_j[current_index,1], q_j[current_index,2]), q_j[current_index,3])

    current_index += 1

# --- Persistent velocity state for half-step averaging ---
# Newton integrator uses a leapfrog scheme: velocities live at t+dt/2 (half-steps).
# To recover the on-step value at t we average the bracketing half-step values:
#   v(t) ≈ 0.5 * (v(t-dt/2) + v(t+dt/2))
# On the very first step t-dt/2 does not exist, so we use the current value directly.
_prev_v1 = None   # velocity of body 0 from the previous half-step
_prev_v2 = None   # velocity of body 1 from the previous half-step
_prev_w1 = None   # angular velocity of body 0 from the previous half-step
_prev_w2 = None   # angular velocity of body 1 from the previous half-step
 
# Staging dict: kinematics are recorded here by saveKinematics() and consumed
# together with forces/torques by saveForcesTorques() in a single plot.addData() call.
_kinem = {}

def saveKinematics():
	"""Record positions, orientations, and on-step velocities BEFORE imposeState().
	The bodies still hold the state from the end of the previous step, which is
	the correct kinematic snapshot for the current time t."""
	global _prev_v1, _prev_v2, _prev_w1, _prev_w2, _kinem
	s1 = O.bodies[0].state
	s2 = O.bodies[1].state
 
	# Read raw (half-step) velocities from the integrator
	v1 = np.array(s1.vel)
	v2 = np.array(s2.vel)
	w1 = np.array(s1.angVel)
	w2 = np.array(s2.angVel)
 
	# Average to reconstruct the on-step velocity at t;
	# fall back to the current value for the very first step (no previous half-step exists)
	v1_t = v1 if _prev_v1 is None else 0.5 * (_prev_v1 + v1)
	v2_t = v2 if _prev_v2 is None else 0.5 * (_prev_v2 + v2)
	w1_t = w1 if _prev_w1 is None else 0.5 * (_prev_w1 + w1)
	w2_t = w2 if _prev_w2 is None else 0.5 * (_prev_w2 + w2)
 
	# Advance the stored previous half-step values
	_prev_v1, _prev_v2 = v1.copy(), v2.copy()
	_prev_w1, _prev_w2 = w1.copy(), w2.copy()
 
	# Stage kinematics for saveForcesTorques()
	_kinem = dict(
		t=O.time,
		x1=s1.pos[0],  y1=s1.pos[1],  z1=s1.pos[2],
		x2=s2.pos[0],  y2=s2.pos[1],  z2=s2.pos[2],
		qx1=s1.ori[0], qy1=s1.ori[1], qz1=s1.ori[2], qw1=s1.ori[3],
		qx2=s2.ori[0], qy2=s2.ori[1], qz2=s2.ori[2], qw2=s2.ori[3],
		v1x=v1_t[0], v1y=v1_t[1], v1z=v1_t[2],
		v2x=v2_t[0], v2y=v2_t[1], v2z=v2_t[2],
		w1x=w1_t[0], w1y=w1_t[1], w1z=w1_t[2],
		w2x=w2_t[0], w2y=w2_t[1], w2z=w2_t[2],
	)
 
def saveForcesTorques():
	"""Record forces and torques AFTER InteractionLoop(), then commit the full
	snapshot (kinematics staged by saveKinematics() + forces/torques) to plot."""
	f1, t1 = O.forces.f(0), O.forces.t(0)
	f2, t2 = O.forces.f(1), O.forces.t(1)
	# Store for plotting
	plot.addData(**_kinem,
		f1x=f1[0], f1y=f1[1], f1z=f1[2],
		f2x=f2[0], f2y=f2[1], f2z=f2[2],
		t1x=t1[0], t1y=t1[1], t1z=t1[2],
		t2x=t2[0], t2y=t2[1], t2z=t2[2])

# --- Engines ---
# Engine order matters for temporal consistency:
#   1. saveKinematics  – snapshot pos/ori/vel at time t, before any override
#   2. imposeState     – set prescribed pos/ori/vel for this step
#   3. InteractionLoop – compute contact forces from the newly imposed geometry
#   4. saveForcesTorques – snapshot forces/torques and commit the full record
#   5. NewtonIntegrator – advance the simulation by dt
O.dt = dt[0]
O.engines = [
	ForceResetter(),
	InsertionSortCollider([Bo1_Sphere_Aabb()]),
	PyRunner(command='saveKinematics()',    initRun=True, iterPeriod=1),
	PyRunner(command='imposeState()',       initRun=True, iterPeriod=1),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom(avoidGranularRatcheting=True,exactRotations=True)],
		[Ip2_FrictMat_FrictMat_FrictPhys(
			kn=MatchMaker(algo='val', val=kn),
			ks=MatchMaker(algo='val', val=ks)
		)],
		[Law2_ScGeom_FrictPhys_CundallStrack(sphericalBodies=True)]
	),
	PyRunner(command='saveForcesTorques()', initRun=True, iterPeriod=1),
	NewtonIntegrator(gravity=(0, 0, 0), damping=0)
]

# --- Set up plotting ---
plot.plots = {'x1': ('f1x',)}

## --- Run simulation and save ---
O.run(N, True)

import csv
file = open(outputFile, mode='w', newline='', encoding='utf-8')
file.write('# ')
writer = csv.writer(file, delimiter=' ')
# Header is: # x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2 
# 			 v1x v1y v1z v2x v2y v2z w1x w1y w1z w2x w2y w2z 
# 			 f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
# or in vector notation: pos1 pos2 vel1 vel2 q1 q2 omega1 omega2 F1 F2 T1 T2
writer.writerow(plot.data.keys())
writer.writerows(zip(*plot.data.values()))
file.close()

# plot.plot()
# plt.savefig(baseName+".png")
