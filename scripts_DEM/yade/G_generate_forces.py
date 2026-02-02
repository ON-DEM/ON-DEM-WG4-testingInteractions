# Copyright 2025: Bruno Chareyre <bruno.chareyre@grenoble-inp.fr>
# Execution: "yade generateForces.py input.txt output.csv", where input.txt is a time series of velocities

from yade import plot, sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../scripts_PY')
from D_helpers import *

if len(sys.argv) > 2:
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
else:
	print("Usage: yade generateForces.py <input_file> <output_file>")
	exit()
      
imposePos = True  # Whether to impose positions and orientations, or only velocities

# --- Load imposed state data from text file ---
imposed_data = json_to_dict(inputFile)

# keys to exclude (do not import these)
_exclude = {'omega_b', 'n_ij', 'v_ijn', 'a_ijn', 'l_ij',
            'u_n', 'v_s', 'v_r', 'v_theta','du_s','du_r','du_theta'}

# build selected dict with everything except the excluded keys
selected = {k: v for k, v in imposed_data.items() if k not in _exclude}

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
O.materials.append(FrictMat(young=1.0e7, poisson=0.5, frictionAngle=atan(0.5)))
kn = 1.0e7  # normal stiffness
ks = 0.5e7  # tangential or shear stiffness

sphere1 = sphere(center=(x_i[0,0], x_i[0,1], x_i[0,2]), radius=1.0, fixed=True)
sphere2 = sphere(center=(x_j[0,0], x_j[0,1], x_j[0,2]), radius=1.0, fixed=True)
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
	
def saveData():
	f1,t1 = O.forces.f(0),O.forces.t(0)
	f2,t2 = O.forces.f(1),O.forces.t(1)
	s1=O.bodies[0].state
	s2=O.bodies[1].state
	v1=O.bodies[0].state.vel
	v2=O.bodies[1].state.vel
	w1=O.bodies[0].state.angVel
	w2=O.bodies[1].state.angVel
	# Store for plotting
	plot.addData(t=O.time, 
			x1=s1.pos[0], y1=s1.pos[1], z1=s1.pos[2],
			x2=s2.pos[0], y2=s2.pos[1], z2=s2.pos[2],
			qx1=s1.ori[0], qy1=s1.ori[1], qz1=s1.ori[2], qw1=s1.ori[3],
			qx2=s2.ori[0], qy2=s2.ori[1], qz2=s2.ori[2], qw2=s2.ori[3],
			v1x=v1[0], v1y=v1[1], v1z=v1[2],
			v2x=v2[0], v2y=v2[1], v2z=v2[2],
			w1x=w1[0], w1y=w1[1], w1z=w1[2],
			w2x=w2[0], w2y=w2[1], w2z=w2[2],
			f1x=f1[0], f1y=f1[1], f1z=f1[2],
			f2x=f2[0], f2y=f2[1], f2z=f2[2],
			t1x=t1[0], t1y=t1[1], t1z=t1[2], 
			t2x=t2[0], t2y=t2[1], t2z=t2[2])

# --- Engines ---
O.dt = dt[0]
O.engines = [
	ForceResetter(),
	InsertionSortCollider([Bo1_Sphere_Aabb()]),
	PyRunner(command='imposeState()', initRun=True, iterPeriod=1),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom(avoidGranularRatcheting=True)],
		[Ip2_FrictMat_FrictMat_FrictPhys(
			kn=MatchMaker(algo='val', val=kn),
			ks=MatchMaker(algo='val', val=ks)
		)],
		[Law2_ScGeom_FrictPhys_CundallStrack(sphericalBodies=True, approxTangentRot=False)]
	),
	PyRunner(command='saveData()', initRun=True, iterPeriod=1),
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
