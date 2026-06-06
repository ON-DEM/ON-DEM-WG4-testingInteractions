# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from A_analytical_motion import *
from B_analytical_contact import *
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
Nsteps = 1.0e4

# Run the faulty / alternative versions of the tests?
doERR = False

# Size parameters
R_i = 1.0
R_j = 1.0
R = (R_i + R_j)/2.0

# Time parameters
tmax = 6.0*np.pi
dt = 6.0*np.pi/Nsteps

# Contact parameters (default values; overridden per-test where needed below).
# These are also embedded in the output JSON so downstream scripts (G*, H*, I*)
# can read them without any hardcoding or separate parameter files.
contact_params = {'k_n':    1.0e7,
                  'k_s':    0.5e7,
                  'k_r':    0.0,
                  'k_t':    0.0,
                  'k_b':    0.0,
                  'mu_s':   0.5,
                  'mu_r':   0.5,
                  'mu_t':   0.5,
                  'mu_b':   0.5,
                  'eta_n':  0.0,
                  'eta_s':  0.0,
                  'eta_r':  0.0,
                  'eta_t':  0.0,
                  'eta_b':  0.0,
                  'R_i':    R_i,
                  'R_j':    R_j,
                  # Maxwell arm parameters — empty list disables that mode's arms.
                  # Each entry i defines the i-th arm: spring stiffness and dashpot viscosity.
                  'armKn':   [], 'armEtan': [],
                  'armKs':   [], 'armEtas': [],
                  'armKr':   [], 'armEtar': [],
                  'armKt':   [], 'armEtat': [],
                  'armKb':   [], 'armEtab': []}


# Generate velocities and motion profile
# Set the phase to pi to start with approach
if testID == 1:
    # Tangential elastic response
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 2:
    # Tangential plastic response
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.08*R, 1.0, 0, 0, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 3:
    # Out-of-plane tangent force rotation
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[1.0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0.02*R, 0, 0, 0, 0, # shear
        [1.0,0,0], [1.0,0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 4:
    # In-plane tangent force rotation
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,1.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch
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
    w_cycle = 4*np.pi / (3*np.pi)  # frequency for 3π period
    # Place initial branch along z: initial position/branch at [0,0,1.95*R]
    # Choose roll and shear axes orthogonal to branch (x and y)
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.03*R, w_cycle, np.pi/2, 0, [0.0,0.0,1.96*R], # normal: approach-unload cycle (branch along z)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.03*R, w_cycle, 0, 0, # shear: 90° out of phase with normal (peaks at load transitions)
        [1.0,0.0,0.0], [0.0,1.0,0.0], # roll and shear axes (both orthogonal to branch)
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 6:
    # Purely repulsive viscous force
    # Normal: oscillate between approach and separation, crossing u_n = 0 at t = 0, π, 2π, ...
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.04*R, 1.0, 0, 0, [-2.02*R,0.0,0.0], # normal: oscillates to clearly lose contact
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0, 0, 0, 0, # shear (none)
        [0.0,0.0,1.0], [0.0,1.0,0.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    contact_params['armKn']   = [4.0e7]
    contact_params['armEtan'] = [1.0e7]
    # contact_params['eta_n'] = 1.0e7 # If dashpot instead of Maxwell arm.
elif testID == 7:
    # Limit force properly instead of velocity.
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.03*R, 1.0, 0, 0, [-1.999*R,0.0,0.0],
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0, 0, 0, 0, # shear (none)
        [0.0,0.0,1.0], [0.0,1.0,0.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    contact_params['armKn']   = [4.0e7]
    contact_params['armEtan'] = [1.0e7]
    # contact_params['eta_n'] = 1.0e7 # If dashpot instead of Maxwell arm.
elif testID == 8:
    # Continuity (C0) of viscous force at u_n = 0
    # A non-C0 implementation produces a force jump of magnitude eta_n*v_max at each crossing:
    # upon approach (v_n < 0): F_n jumps from 0 to +eta_n*v_max as soon as u_n = 0+.
    # upon separation (v_n > 0): F_n would jump from -eta_n*v_max to 0 at u_n = 0+ if uncapped.
    # Either jump causes accelerations that are independent of time-step size (CFL broken).
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.04*R, 1.0, 0, 0, [-2.02*R,0.0,0.0], # maximum speed at contact-boundary crossings
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0, 0, 0, 0, # shear (none)
        [0.0,0.0,1.0], [0.0,1.0,0.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    contact_params['armKn']   = [4.0e7]
    contact_params['armEtan'] = [1.0e7]
    # contact_params['eta_n'] = 1.0e7 # If dashpot instead of Maxwell arm.
elif testID == 9:
    # Preservation of viscoelastic memory upon loss of contact
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.04*R, 2.0, 0, -0.05, [-2.01*R,0.0,0.0], # maximum speed at contact-boundary crossings
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0, 0, 0, 0, # shear (none)
        [0.0,0.0,1.0], [0.0,1.0,0.0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    contact_params['armKn']   = [0.2e7]
    contact_params['armEtan'] = [1.0e7]
elif testID == 10:
    # Independence of shear displacement from viscosity
    # Trajectory is identical to test 1; only eta_s differs (see contact_params below).
    # The shear displacement u_s must accumulate from v_s alone:
    #   u_s(t+dt) = u_s(t) + v_s * dt
    # A common mistake is to include the viscous force in the increment:
    #   u_s(t+dt) = u_s(t) + (k_s*u_s + eta_s*v_s) / k_s * dt   ← WRONG
    # This converts dissipated (viscous) energy into stored elastic energy, causing
    # the shear force to drift away from the analytical prediction over time.
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch (same as test 1)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, -0.1, # shear: same oscillation as test 1
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    # eta_s chosen to give a viscous shear force of eta_s*v_s_max = 1e5*0.02R = 2e3 N,
    # about 1% of the elastic peak. Small enough to keep the test in the elastic regime
    # but large enough for accumulation errors to be visible over many cycles.
    contact_params['eta_s'] = 1.0e6
elif testID == 11:
    # Consistency in application of Coulomb limit
    # Tests whether the Coulomb limit is applied to the total tangential force
    # (elastic + viscous combined) rather than the elastic part alone.
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # constant contact (same as test 2)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear: elastic alone stays below Coulomb; total exceeds it
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    # The elastic component alone does not surpass the Coulomb limit, only with the viscous part.
    contact_params['eta_s'] = 1.0e7
elif testID == 12:
    # Dependence of force on particle size
    # kn = E * R so kn = L*kn, and to keep overlap (strain) similar we have u/R constant, so u -> L*u
    # Fn = L^2*kn is expected for similarity or proper size scaling.
    scale = 0.5
    R_i = scale*R_i
    R_j = scale*R_j
    R = (R_i + R_j)/2.0 # Recompute
    contact_params['R_i'] = R_i
    contact_params['R_j'] = R_j
    contact_params['k_n'] *= scale   # For YADE, if we set E = k_n and it's automatically scaled.
    contact_params['k_s'] *= scale   # Original ratio preserved
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0.04*R, 1.0, np.pi/2, 0, [-2.0*R,0,0], 
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0, 0, 0, 0, 0,                       # no shear
        [0,0,1.0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 13:
    # Distinction between rolling and bending
    R_i = 2.0    # large particle
    R_j = 0.2    # small particle  (size ratio 10:1)
    R = (R_i + R_j)/2.0
    contact_params['k_r'] = 0.25e7
    contact_params['R_i'] = R_i
    contact_params['R_j'] = R_j
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.96*R], # constant contact (~2% initial overlap)
        0, 0, 0, 0, 0,                       # no twist
        0, 0.05*np.pi, 1.0, 0, 0,             # pure oscillating roll
        0, 0, 0, 0, 0,                       # no shear: F_s must remain zero
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 14:
    # Complex shearing motion
    motion = my_analytical_motion(
        [1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.96*R], # normal loading, initial branch
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.8*R, 1.0, 0, -0.1, # shear
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 15:
    # Complex rolling motion
    contact_params['k_r'] = 0.25e7
    motion = my_analytical_motion(
        [1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0],       # tilt rigid-body rotation (ω_f along x)  with 0.2 around the y axis is quite nice
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.96*R],         # constant contact # [0,1.96*R,0],
        0, 0, 0, 0, 0,                       # no twist
        0, np.pi, 1.0, 0, -0.1,          # complicated roll velocity
        0, 0, 0, 0, 0,                       # no shear
        [1.0,0,0], [0,1.0,0],                # roll and shear axes # [0,0,1.0], [1.0,0,0],
        tmax, dt,
        R_i, R_j
    )
elif testID == 16:
    # Complex twisting motion
    contact_params['k_t'] = 0.50e7
    motion = my_analytical_motion(
        [1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [1.96*R,0,0],         # constant contact #0, 0, 0, 0, 0, [0,0,1.96*R],
        0, np.pi, 1.0, 0, -0.1,          # complicated twist velocity
        0, 0, 0, 0, 0,                       # no roll
        0, 0, 0, 0, 0,                       # no shear
        [0,0,1.0], [0,1.0,0],                # roll and shear axes (needed for orientation) # [1.0,0,0], [0,1.0,0],
        tmax, dt,
        R_i, R_j
    )
elif testID == 17:
    # Complex combined motion
    #
    # F_n = k_n * u_n = 1e7 * 0.05 = 5e5 N  →  Coulomb limit = mu * F_n = 2.5e5 N.
    # Each mode amplitude is chosen so that its peak force stays well below Coulomb/3:
    #   Shear: F_s_max = k_s * (B_s/w) = 5e6 * 0.01 = 5.0e4 N  << Coulomb/3
    #   Roll:  F_r_max = k_r * (B_r/w) = 2.5e6 * 0.01 = 2.5e4 N << Coulomb/3
    #   Twist: T_t_max = k_t * (B_t/w) = 2.5e6 * 0.01 = 2.5e4 N·m (torque) << limit
    # Phase offsets (0, pi/2, pi) keep the mode maxima from coinciding, ensuring the
    # combined response stays elastic at all times.
    #
    # Should go in and out of elastic / plastic regime as well as in and out of contact.
    contact_params['k_r'] = 0.25e7
    contact_params['k_t'] = 0.25e7
    motion = my_analytical_motion(
        [1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.96*R],         # constant contact (u_n = 0.05)
        0, 1.0, 1.0, 0, 0,                 # twist: phi=0
        0, 1.0, 1.0, np.pi/2, 0,           # roll: phi=pi/2, 90 deg out of phase
        0, 0.5*R, 1.0, np.pi, 0,           # shear: phi=pi, 180 deg out of phase
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )



# Simulate contact interaction
if not doERR:
    # Tests 6 and 8 use the Maxwell arm normal force (spring + series dashpot, repulsive only).
    # All other tests use the standard parallel spring-dashpot.
    Fn_func = Fn_spring_dashpot_maxwell if testID in (6, 7, 8, 9) else Fn_spring_dashpot
    results = my_analytical_contact(
        motion,
        contact_params,
        Fn_func,
        Fs_spring_dashpot_Coulomb,
        Tr_spring_dashpot_Coulomb,
        Tt_spring_dashpot_Coulomb,
        Tb_spring_dashpot_Coulomb
        )
else:
    if testID == 1:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_1,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 2:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_2,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif (testID == 3) or (testID == 4):
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_3_4,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 5:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_5,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 6:
        #contact_params['armKn']   = [0.0]
        #contact_params['armEtan'] = [0.0]
        #contact_params['eta_n'] = 1.0e7
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_fail_test_6,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 7:
        #contact_params['armKn']   = [0.0]
        #contact_params['armEtan'] = [0.0]
        #contact_params['eta_n'] = 1.0e7
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_fail_test_7,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 8:
        contact_params['armKn']   = [0.0]
        contact_params['armEtan'] = [0.0]
        contact_params['eta_n'] = 1.0e7
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 9:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_fail_test_9,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 10:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_10,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 11:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_11,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 12:
        # Putting these back to their original values
        contact_params['k_n'] /= scale
        contact_params['k_s'] /= scale
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    elif testID == 13:
        Reff = 2.0*(contact_params['R_i'] * contact_params['R_j']) / (contact_params['R_i'] + contact_params['R_j'])
        contact_params['k_b'] = contact_params['k_r'] * (Reff ** 2)
        contact_params['k_r'] = 0.0
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    else:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )


# Writing output
if not doERR:
    dict_to_csv(results, open('../output_ANA/theoretical_output_'+testname+'.csv', 'w'))
else:
    dict_to_csv(results, open('../output_ANA_ERR/theoretical_output_'+testname+'.csv', 'w'))

# Embed contact_params in the results so that downstream scripts (G*, H*, I*) can
# read the material and geometry parameters directly from the JSON without any
# hardcoding or separate parameter files.
results['contact_params'] = contact_params

if not doERR:
    dict_to_json(results,'../output_ANA/theoretical_output_'+testname+'.json')
else:
    dict_to_json(results,'../output_ANA_ERR/theoretical_output_'+testname+'.json')

# Plotting is obsolete here, dealt with in other scripts.
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

# The below is now obsolete, we directly take the analytical results.
#demInputs = {k: results[k] for k in ['t', 'v_i', 'v_j', 'omega_i', 'omega_j']}
#dict_to_json(demInputs,'../input_DEM/dem_input_'+testname+'.json')
#write_DEM_input(results, '../input_DEM/dem_input_'+testname+'.csv')

# End of file