# Copyright 2025: Bruno Chareyre <bruno.chareyre@grenoble-inp.fr>
# Execution: "yade generateForces.py input.txt", where "input.txt" is a time series of velocities

from yade import pack, plot, sys
import numpy as np

if len(sys.argv)>1:
	inputFile = sys.argv[1]
else:
	print("please specify input file")
	exit()
	
# --- Load velocity data from text file ---
velocity_data = np.loadtxt("velocities.txt", comments='#')  # shape (N, 7)

# --- Initialize simulation scene ---
sphere1 = sphere(center=(0, 0, 0), radius=1, fixed=True)
sphere2 = sphere(center=(2, 0, 0), radius=1, fixed=True)
O.bodies.append([sphere1, sphere2])

# Add a dummy material
O.materials.append(FrictMat(young=1e7, poisson=0.3, frictionAngle=0.5))

# --- Time stepping logic ---
current_index = 0

def applyVelocities():
    global current_index
    if current_index < len(velocity_data):
        row = velocity_data[current_index]
        # Format: time, vx1, vy1, vz1, vx2, vy2, vz2
        v1 = (row[1], row[2], row[3])
        v2 = (row[4], row[5], row[6])

        O.bodies[0].state.vel = v1
        O.bodies[1].state.vel = v2

    current_index += 1
    
    
def saveData():
	# Get total force on particle 2
    f1,t1 = O.forces.f(0),O.forces.t(0)
    f2,t2 = O.forces.f(1),O.forces.t(1)
    # Store for plotting
    plot.addData(f1x=f1[0], f1y=f1[1], f1z=f1[2], t1x=t1[0], t1y=t1[1], t1z=t1[2], f2x=f2[0], f2y=f2[1], f2z=f2[2], t2x=t2[0], t2y=t2[1], t2z=t2[2], t=O.time)

# --- Engines ---
O.dt = 1e-4
O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    PyRunner(command='applyVelocities()', iterPeriod=1),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    PyRunner(command='saveData()', iterPeriod=1),    
    NewtonIntegrator(gravity=(0, 0, 0), damping=0)
]

# --- Set up plotting ---
plot.plots = {'t': ('f1x',)}
plot.plot()
O.dt=1

## --- Run simulation ---
O.run(len(velocity_data), True)
