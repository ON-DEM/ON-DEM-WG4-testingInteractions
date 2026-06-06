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
_exclude = {'omega_f', 'n_ij', 'v_ijn', 'a_ijn', 'l_ij',
            'u_n', 'v_s', 'v_r', 'omega_t','omega_b','du_s','du_r','dtheta_t','dtheta_b'}

# build selected dict with everything except the excluded keys
selected = {k: v for k, v in imposed_data.items() if k not in _exclude}

# --- Read contact parameters from the JSON (written by F_generate_analytical.py) ---
# Fall back to sensible defaults if an older JSON without contact_params is loaded.
_cp_defaults = {'k_n': 1.0e7, 'k_s': 0.5e7, 'k_r': 0.0, 'k_t': 0.0, 'k_b': 0.0,
                'mu_s': 0.5, 'mu_r': 0.5, 'mu_t': 0.5, 'mu_b': 0.5,
                'eta_n': 0.0, 'eta_s': 0.0, 'eta_r': 0.0, 'eta_t': 0.0, 'eta_b': 0.0,
                'R_i': 1.0, 'R_j': 1.0,
                'armKn': [],   'armEtan': [],
                'armKs': [],   'armEtas': [],
                'armKr': [],   'armEtar': [],
                'armKt': [],   'armEtat': [],
                'armKb': [],   'armEtab': []}
_cp = {**_cp_defaults, **(imposed_data.get('contact_params', {}))}
 
kn  = float(_cp['k_n'])   # normal stiffness [N/m]
ks  = float(_cp['k_s'])   # shear stiffness [N/m]
kr = float(_cp['k_r'])   # rolling stiffness [N/m]
kt = float(_cp['k_t'])   # twisting stiffness [N m/rad]
kb = float(_cp['k_b'])   # bending stiffness [N m/rad]
mus = float(_cp['mu_s'])   # shear friction coefficient [-]
mur = float(_cp['mu_r'])   # rolling friction coefficient [-]
mut = float(_cp['mu_t'])   # twisting friction coefficient [-]
mub = float(_cp['mu_b'])   # bending friction coefficient [-]
etan = float(_cp['eta_n']) # normal viscosity [kg/s]
etas = float(_cp['eta_s']) # shear viscosity [kg/s]
etar = float(_cp['eta_r']) # rolling viscosity [kg/s]
etat = float(_cp['eta_t']) # twisting viscosity [kg m^2/s]
etab = float(_cp['eta_b']) # bending viscosity [kg m^2/s]
R_i = float(_cp['R_i'])   # radius of particle i [m]
R_j = float(_cp['R_j'])   # radius of particle j [m]
# Maxwell arm parameters — list type preserved; empty list disables that mode's arms.
armKn   = list(_cp['armKn'])
armEtan = list(_cp['armEtan'])
armKs   = list(_cp['armKs'])
armEtas = list(_cp['armEtas'])
armKr   = list(_cp['armKr'])
armEtar = list(_cp['armEtar'])
armKt   = list(_cp['armKt'])
armEtat = list(_cp['armEtat'])
armKb   = list(_cp['armKb'])
armEtab = list(_cp['armEtab'])

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
v_i_half  = as_np('v_i_half')		  # expected shape (N,3)
v_j_half  = as_np('v_j_half')		  # expected shape (N,3)
a_i       = as_np('a_i')              # expected shape (N,3)
a_j       = as_np('a_j')              # expected shape (N,3)
q_i       = as_np('q_i')              # expected shape (N,4) qx,qy,qz,qw
q_j       = as_np('q_j')              # expected shape (N,4)
omega_i   = as_np('omega_i')  		  # expected shape (N,3)
omega_j   = as_np('omega_j')  		  # expected shape (N,3)
omega_i_half = as_np('omega_i_half')  # expected shape (N,3)
omega_j_half = as_np('omega_j_half')  # expected shape (N,3)
F_i       = as_np('F_i')              # analytical forces if present (N,3)
F_j       = as_np('F_j')              # analytical forces if present (N,3)
T_i       = as_np('T_i')              # analytical torques if present (N,3)
T_j       = as_np('T_j')              # analytical torques if present (N,3)

# --- Initialize simulation scene ---
# Add Maxwell material
O.materials.append(
    MaxwellMat(young=kn, poisson=(ks/kn), etan=etan,
               frictionAngle=atan(mus), mur=mur, mut=mut, mub=mub,
               ks=ks, kr=kr, kt=kt, kb=kb,
               etas=etas, etar=etar, etat=etat, etab=etab,
               armKn=armKn,   armEtan=armEtan,
               armKs=armKs,   armEtas=armEtas,
               armKr=armKr,   armEtar=armEtar,
               armKt=armKt,   armEtat=armEtat,
               armKb=armKb,   armEtab=armEtab)
)

sphere1 = sphere(center=(x_i[0,0], x_i[0,1], x_i[0,2]), radius=R_i, fixed=False)
sphere2 = sphere(center=(x_j[0,0], x_j[0,1], x_j[0,2]), radius=R_j, fixed=False)
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
def imposeKinematics():
    """Impose the prescribed state for the current step: on-step positions and
    orientations, plus the exact analytical HALF-step linear/angular velocities
    (v_*_half, omega_*_half) read straight from the JSON, which is the value the
    leapfrog integrator expects at v(t_k - dt/2)."""
    global current_index
    if current_index < N:
        # velocities -- impose the analytical HALF-step value v(t_k - dt/2) (leapfrog).
        # (arrays must be shape (N,3))
        O.bodies[0].state.vel = (v_i_half[current_index,0], v_i_half[current_index,1], v_i_half[current_index,2])
        O.bodies[1].state.vel = (v_j_half[current_index,0], v_j_half[current_index,1], v_j_half[current_index,2])

        # angular velocities -- impose the HALF-step value as well (arrays must be shape (N,3))
        O.bodies[0].state.angVel = (omega_i_half[current_index,0], omega_i_half[current_index,1], omega_i_half[current_index,2])
        O.bodies[1].state.angVel = (omega_j_half[current_index,0], omega_j_half[current_index,1], omega_j_half[current_index,2])

        # positions (only if imposePos is True) -- arrays must be shape (N,3)
        if imposePos:
            O.bodies[0].state.pos = (x_i[current_index,0], x_i[current_index,1], x_i[current_index,2])
            O.bodies[1].state.pos = (x_j[current_index,0], x_j[current_index,1], x_j[current_index,2])

        	# orientations (if imposePos is True) -- shape (N,4) in [qx,qy,qz,qw] order
            O.bodies[0].state.ori = Quaternion((q_i[current_index,0], q_i[current_index,1], q_i[current_index,2]), q_i[current_index,3])
            O.bodies[1].state.ori = Quaternion((q_j[current_index,0], q_j[current_index,1], q_j[current_index,2]), q_j[current_index,3])

    current_index += 1

# Staging dict: kinematics are recorded here by saveKinematics() and consumed
# together with forces/torques by saveForcesTorques() in a single plot.addData() call.
_kinem = {}

def saveKinematics():
	"""Record positions, orientations, and HALF-step velocities AFTER imposeKinematics().
	YADE's NewtonIntegrator is a leapfrog scheme: state.vel/state.angVel live at the
	half-steps v(t_k - dt/2). We save those raw half-step values directly -- no on-step
	reconstruction (which would cost O(dt^2) accuracy). The velocity columns are suffixed
	'_half' so downstream comparison (I_make_figure.py) can detect that these are
	half-step quantities and compare them against the analytical half-step velocities
	(v_*_half, omega_*_half) instead of the on-step ones. Positions and orientations are
	on-step and are also stored directly."""
	global _kinem
	s1 = O.bodies[0].state
	s2 = O.bodies[1].state

	# Stage kinematics for saveForcesTorques(). Velocities carry the '_half' suffix
	# to mark them as leapfrog half-step values v(t_k - dt/2).
	_kinem = dict(
		t=O.time,
		x1=s1.pos[0],  y1=s1.pos[1],  z1=s1.pos[2],
		x2=s2.pos[0],  y2=s2.pos[1],  z2=s2.pos[2],
		qx1=s1.ori[0], qy1=s1.ori[1], qz1=s1.ori[2], qw1=s1.ori[3],
		qx2=s2.ori[0], qy2=s2.ori[1], qz2=s2.ori[2], qw2=s2.ori[3],
		v1x_half=s1.vel[0], v1y_half=s1.vel[1], v1z_half=s1.vel[2],
		v2x_half=s2.vel[0], v2y_half=s2.vel[1], v2z_half=s2.vel[2],
		w1x_half=s1.angVel[0], w1y_half=s1.angVel[1], w1z_half=s1.angVel[2],
		w2x_half=s2.angVel[0], w2y_half=s2.angVel[1], w2z_half=s2.angVel[2],
	)
 
# --- Force/torque imposition (counterpart to imposeKinematics) ---
# Currently unused (commented out in O.engines below). Provided so the same script
# can drive a FORCE-controlled test: prescribe the analytical contact reaction and
# let NewtonIntegrator produce the kinematics, which saveKinematics() then measures.
# Forces and torques live ON-step in the leapfrog scheme (the acceleration a(t_k) is
# evaluated at t_k), so -- unlike velocities -- they need no half-step conversion.
# Uses its own step counter so it stays in sync whether or not imposeKinematics()
# is also active (each runs once per iteration).
current_index_FT = 0
def imposeForcesTorques():
    """Add the analytical on-step forces and torques to both bodies for this step.
    Place AFTER InteractionLoop in O.engines to override/augment the computed contact
    force; if used as the sole force source, disable InteractionLoop."""
    global current_index_FT
    k = current_index_FT
    if k < N:
        if F_i is not None: O.forces.addF(0, Vector3(F_i[k,0], F_i[k,1], F_i[k,2]))
        if F_j is not None: O.forces.addF(1, Vector3(F_j[k,0], F_j[k,1], F_j[k,2]))
        if T_i is not None: O.forces.addT(0, Vector3(T_i[k,0], T_i[k,1], T_i[k,2]))
        if T_j is not None: O.forces.addT(1, Vector3(T_j[k,0], T_j[k,1], T_j[k,2]))
    current_index_FT += 1

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
#   1. (iff checking) saveKinematics – snapshot pos/ori/vel at time t, BEFORE imposeKinematics
#   2. imposeKinematics – set prescribed pos/ori (on-step) and vel (lagging half-step) for this step
#   3. (iff measuring) saveKinematics  – snapshot the freshly-imposed pos/ori/half-step vel
#   4. InteractionLoop – compute contact forces from the newly imposed geometry
#   5. imposeForcesTorques – set prescribed froces/torques (on-step) for this step
#   6. (iff measuring) saveForcesTorques – snapshot forces/torques and commit the full record
#   7. NewtonIntegrator – advance the simulation by dt
O.dt = dt[0]
O.engines = [										# t, x(t), q(t), v(t-dt/2), a(t-dt), F(t-dt), T(t-dt)
	ForceResetter(),
	PyRunner(command='imposeKinematics()',  initRun=True, iterPeriod=1),
	PyRunner(command='saveKinematics()',    initRun=True, iterPeriod=1),
	InsertionSortCollider([Bo1_Sphere_Aabb()],verletDist=0.5*(R_i+R_j)),
	InteractionLoop(
		[Ig2_Sphere_Sphere_ScGeom6D(avoidGranularRatcheting=True,exactRotations=True)],
		[Ip2_MaxwellMat_MaxwellMat_MaxwellPhys(
			kn=MatchMaker(algo='val', val=kn), etan=MatchMaker(algo='val', val=etan), 
			ks=MatchMaker(algo='val', val=ks), etas=MatchMaker(algo='val', val=etas), frictAngle=MatchMaker(algo='val', val=atan(mus)),
			kr=MatchMaker(algo='val', val=kr), etar=MatchMaker(algo='val', val=etar), mur=MatchMaker(algo='val', val=mur),
			kt=MatchMaker(algo='val', val=kt), etat=MatchMaker(algo='val', val=etat), mut=MatchMaker(algo='val', val=mut),
			kb=MatchMaker(algo='val', val=kb), etab=MatchMaker(algo='val', val=etab), mub=MatchMaker(algo='val', val=mub)
		)],
		[Law2_ScGeom_MaxwellPhys_general(limitViscousPart=True,preserveHistory=True)]
	), 												# t, x(t), q(t), v(t-dt/2), a(t-dt), F(t), T(t)
# 	PyRunner(command='imposeForcesTorques()', initRun=True, iterPeriod=1),
	PyRunner(command='saveForcesTorques()', initRun=True, iterPeriod=1),
	NewtonIntegrator(gravity=(0, 0, 0), damping=0) 	# t+dt, x(t+dt), q(t+dt), v(t+dt/2), a(t), F(t), T(t)
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
# 			 v1x_half v1y_half v1z_half v2x_half v2y_half v2z_half
# 			 w1x_half w1y_half w1z_half w2x_half w2y_half w2z_half
# 			 f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
# or in vector notation: pos1 pos2 q1 q2 vel1_half vel2_half omega1_half omega2_half F1 F2 T1 T2
# Velocities carry the '_half' suffix: they are the leapfrog HALF-step values v(t-dt/2) and omega(t-dt/2).
writer.writerow(plot.data.keys())
writer.writerows(zip(*plot.data.values()))
file.close()

# plot.plot()
# plt.savefig(baseName+".png")
