# Copyright 2025: Bruno Chareyre <bruno.chareyre@grenoble-inp.fr>
# Execution: "yade generateForces.py input.txt output.csv", where input.txt is a time series of velocities

from yade import plot, sys, os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
else:
	print("Usage: yade generateForces.py <input_file> <output_file>")
	exit()
	
# --- Load velocity data from text file ---
velocity_data = np.loadtxt(inputFile, comments='#')  # shape (N, 7)

# --- Initialize simulation scene ---
sphere1 = sphere(center=(0, 0, 0), radius=1.0, fixed=True)
sphere2 = sphere(center=(0, 0, 1.95), radius=1.0, fixed=True)
O.bodies.append([sphere1, sphere2])
O.bodies[0].state.ori = Quaternion((0,0,0),1) # Just to be sure
O.bodies[1].state.ori = Quaternion((0,0,0),1) # Just to be sure

# Add a dummy material
O.materials.append(FrictMat(young=1.0e7, poisson=0.5, frictionAngle=atan(0.5)))
kn = 1.0e7  # normal stiffness
ks = 0.5e7  # tangential or shear stiffness

# --- Time stepping logic ---
current_index = 0
def imposeVelocity():
	global current_index
	if current_index < len(velocity_data):
		row = velocity_data[current_index]
		# Format: time, vx1, vy1, vz1, vx2, vy2, vz2, wx1, wy1, wz1, wx2, wy2, wz2
		v1 = (row[1], row[2], row[3])
		v2 = (row[4], row[5], row[6])
		w1 = (row[7], row[8], row[9])
		w2 = (row[10], row[11], row[12])

		O.bodies[0].state.vel = v1
		O.bodies[1].state.vel = v2
		O.bodies[1].state.angVel = w1
		O.bodies[1].state.angVel = w2

	current_index += 1
	
def saveData():
	f1,t1 = O.forces.f(0),O.forces.t(0)
	f2,t2 = O.forces.f(1),O.forces.t(1)
	s1=O.bodies[0].state
	s2=O.bodies[1].state
	# Store for plotting
	plot.addData(t=O.time, x1=s1.pos[0], y1=s1.pos[1], z1=s1.pos[2], x2=s2.pos[0], y2=s2.pos[1], z2=s2.pos[2], qx1=s1.ori[0], qy1=s1.ori[1], qz1=s1.ori[2], qw1=s1.ori[3], qx2=s2.ori[0], qy2=s2.ori[1], qz2=s2.ori[2], qw2=s2.ori[3], f1x=f1[0], f1y=f1[1], f1z=f1[2], f2x=f2[0], f2y=f2[1], f2z=f2[2], t1x=t1[0], t1y=t1[1], t1z=t1[2],  t2x=t2[0], t2y=t2[1], t2z=t2[2])

# --- Engines ---
O.dt = 6.0*np.pi/200.0
O.engines = [
	ForceResetter(),
	InsertionSortCollider([Bo1_Sphere_Aabb()]),
	PyRunner(command='imposeVelocity()', initRun=True, iterPeriod=1),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom(avoidGranularRatcheting=True)],
		[Ip2_FrictMat_FrictMat_FrictPhys(
			kn=MatchMaker(algo='val', val=kn),
			ks=MatchMaker(algo='val', val=ks))],
		[Law2_ScGeom_FrictPhys_CundallStrack()]
	),
	PyRunner(command='saveData()', initRun=True, iterPeriod=1),
	NewtonIntegrator(gravity=(0, 0, 0), damping=0)
]

# --- Set up plotting ---
plot.plots = {'x1': ('f1x',)}

## --- Run simulation and save ---
O.run(len(velocity_data), True)

import csv
file = open(outputFile, mode='w', newline='', encoding='utf-8')
file.write('# ')
writer = csv.writer(file, delimiter=' ')
writer.writerow(plot.data.keys())
writer.writerows(zip(*plot.data.values()))
file.close()

# plot.plot()
# plt.savefig(baseName+".png")
