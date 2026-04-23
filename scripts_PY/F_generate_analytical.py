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
                  'R_j':    R_j}

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
    contact_params['eta_n'] = 1.0e7
elif testID == 7:
    # Recovery of viscous force
    # Maintain contact but go back and forth
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
    contact_params['eta_n'] = 1.0e7
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
    contact_params['eta_n'] = 1.0e7
elif testID == 9:
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
elif testID == 10:
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
elif testID == 11:
    # Dependence of force on particle size
    # kn = E * R, so E = k_n if we have R = 1. So if we halve R, we must double kn.
    # Sensible mixing then gives 2 * k_n * 2 * k_n / (k_n + 2*k_n) = 4/3 * k_n
    contact_params['R_j'] = 0.5*R_j
    R_j = contact_params['R_j']
    scale = (2*R/contact_params['R_i']*R/contact_params['R_j'])/(R/contact_params['R_i']+R/contact_params['R_j'])  # = 4/3 for R_i=1, R_j=0.5
    contact_params['k_n'] = scale*contact_params['k_n']   # For YADE, if we set E = k_n and it's automatically scaled.
    contact_params['k_s'] = scale*contact_params['k_s']   # Original ratio preserved
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0.04*R, 1.0, np.pi/2, 0, [-(R_i+R_j),0,0], 
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0, 0, 0, 0, 0,                       # no shear
        [0,0,1.0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 12:
    # Distinction between rolling and bending
    #
    # Verification criterion: rolling and bending are distinct kinematic modes when
    # R_i ≠ R_j (Sec. 4.12, Eq. 26-28). Under pure rolling (ωr ≠ 0, ωs = 0), a
    # bending model incorrectly introduces a shear component proportional to
    # (R_j - R_i)/(R_i + R_j). At the 4:1 size ratio used here that spurious
    # shear component is 60% of the rolling velocity, making the error easily visible.
    # The analytical shear force is zero throughout; any non-zero F_s in the DEM
    # output indicates a bending model was used where a rolling model is required.
    # This is a modelling choice that must be made consciously.
    R_i = 2.0    # large particle
    R_j = 0.5    # small particle  (size ratio 4:1)
    R = (R_i + R_j) / 2.0    # = 1.25
    contact_params['k_r'] = 0.25e7
    contact_params['R_i'] = R_i
    contact_params['R_j'] = R_j
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,0.98*(R_i+R_j)], # constant contact (~2% initial overlap)
        0, 0, 0, 0, 0,                       # no twist
        0, 0.1*np.pi, 1.0, 0, 0,             # pure oscillating roll
        0, 0, 0, 0, 0,                       # no shear: F_s must remain zero
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 13:
    # Complex rolling motion
    contact_params['k_r'] = 0.25e7
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[1.0,0,0],           # tilt rigid-body rotation (ωb along x)
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,1.96*R,0],         # constant contact
        0, 0, 0, 0, 0,                       # no twist
        0.05*np.pi, 0.1*np.pi, 1.0, -0.1, 0,        # constant roll velocity (A term only)
        0, 0, 0, 0, 0,                       # no shear
        [0,0,1.0], [1.0,0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 14:
    # Complex twisting motion
    #
    # Verification criterion: the twisting torque must remain aligned with the current
    # contact normal under rigid-body tilt motion (Sec. 4.14). A constant twist
    # velocity is applied while ωb = [1,0,0] tilts the contact plane. If the twist
    # axis is not updated with the contact normal the torque acquires a spurious
    # off-normal component. Failure indicates a modelling error.
    contact_params['k_t'] = 0.50e7
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[1.0,0,0],           # tilt rigid-body rotation (ωb along x)
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.96*R],         # constant contact
        0.1*np.pi, 0.1*np.pi, 1.0, -0.1, 0,# constant twist velocity (A term only)
        0, 0, 0, 0, 0,                       # no roll
        0, 0, 0, 0, 0,                       # no shear
        [1.0,0,0], [0,1.0,0],                # roll and shear axes (needed for orientation)
        tmax, dt,
        R_i, R_j
    )
elif testID == 15:
    # Complex combined motion
    #
    # F_n = k_n * u_n = 1e7 * 0.05 = 5e5 N  →  Coulomb limit = mu * F_n = 2.5e5 N.
    # Each mode amplitude is chosen so that its peak force stays well below Coulomb/3:
    #   Shear: F_s_max = k_s * (B_s/w) = 5e6 * 0.01 = 5.0e4 N  << Coulomb/3
    #   Roll:  F_r_max = k_r * (B_r/w) = 2.5e6 * 0.01 = 2.5e4 N << Coulomb/3
    #   Twist: T_t_max = k_t * (B_t/w) = 2.5e6 * 0.01 = 2.5e4 N·m (torque) << limit
    # Phase offsets (0, pi/2, pi) keep the mode maxima from coinciding, ensuring the
    # combined response stays elastic at all times.
    contact_params['k_r'] = 0.25e7
    contact_params['k_t'] = 0.25e7
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.95*R],         # constant contact (u_n = 0.05)
        0, 0.010, 1.0, 0, 0,                 # twist: phi=0
        0, 0.010, 1.0, np.pi/2, 0,           # roll: phi=pi/2, 90 deg out of phase
        0, 0.010*R, 1.0, np.pi, 0,           # shear: phi=pi, 180 deg out of phase
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 16:
    # Shear force rotation at large size ratios
    #
    # Verification criterion: for a large size ratio R_i/R_j = 8, standard integration
    # algorithms can cause non-physical sticking of the smaller particle to the larger
    # one because the shear force is not rotated correctly as the contact normal tilts
    # (Sec. 4.16, [11]). A constant shear is imposed under a continuous tilt rigid-body
    # rotation; error in the shear force accumulates progressively and is detectable via
    # MAPE. Failure indicates a modelling error.
    R_i = 4.0    # large particle
    R_j = 0.5    # small particle  (size ratio 8:1)
    R = (R_i + R_j) / 2.0    # = 2.25
    contact_params['R_i'] = R_i
    contact_params['R_j'] = R_j
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0.2,0,0],           # tilt rigid-body rotation (ωb along x)
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,-0.98*(R_i+R_j)],# constant contact (~2% initial overlap)
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0.02*R, 0, 0, 0, 0,                  # constant shear velocity (A term only)
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 17:
    # Oblique impact
    #
    # Verification criterion: during a combined normal approach and simultaneous shear
    # (oblique impact), the normal and shear forces should remain independent and
    # correctly signed throughout the collision (Sec. 4.17). Normal damping is active;
    # the normal force must remain repulsive (cf. test 6), and the shear force must
    # stay within the Coulomb limit. The branch is oriented along x so that impact
    # direction is unambiguous in output files.
    # Shear peaks at maximum contact depth (90 deg phase offset from normal).
    contact_params['eta_n'] = 5.0e5
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0.05*R, 1.0, 0, 0, [-2.05*R,0,0],# normal: oscillates in/out of contact along -x
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0, 0.04*R, 1.0, np.pi/2, 0,          # shear: 90 deg out of phase, peaks at max contact
        [0,0,1.0], [0,1.0,0],                # roll and shear axes perpendicular to x branch
        tmax, dt,
        R_i, R_j
    )
elif testID == 18:
    # Particle rotating on top of another particle
    #
    # Verification criterion: a small particle (R_j=0.5) rolling over the surface of
    # a much larger particle (R_i=4.0) experiences a continuously rotating contact
    # normal (tilt ωb) simultaneously with combined rolling and shearing (Sec. 4.18).
    # This is a compound stress test: large size asymmetry amplifies the bending/rolling
    # distinction error (test 14) while the tilt tests the force rotation algorithm
    # (tests 3 and 15). The roll and shear oscillations are 90 deg out of phase so
    # that the peak forces do not coincide. Failure indicates a modelling error.
    R_i = 4.0    # large base particle
    R_j = 0.5    # small particle rolling over the top
    R = (R_i + R_j) / 2.0    # = 2.25
    contact_params['k_r'] = 0.25e7
    contact_params['R_i'] = R_i
    contact_params['R_j'] = R_j
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0.2,0,0],           # tilt rigid-body rotation (ωb along x)
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,0.98*(R_i+R_j)],# constant contact (~2% initial overlap)
        0, 0, 0, 0, 0,                       # no twist
        0, 0.02*R, 1.0, 0, 0,               # oscillating roll: phi=0
        0, 0.02*R, 1.0, np.pi/2, 0,         # simultaneous shear: 90 deg out of phase
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 19:
    # Combination test: simultaneous normal and shear damping
    #
    # Verification criterion: when both normal and shear viscous dashpots are active
    # simultaneously, neither should contaminate the other's elastic displacement
    # accumulation (combines tests 8 and 9, Sec. 4.19). The analytical force is the
    # superposition of the independently computed normal and shear spring-dashpot
    # responses, which is exact when the initial overlap and shear displacement are
    # both zero at t=0 (u0=0 condition). Failure indicates a modelling error.
    contact_params['eta_n'] = 5.0e5
    contact_params['eta_s'] = 5.0e5
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0],             # no rigid-body motion
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0.03*R, 4.0/3.0, np.pi/2, 0, [0,0,1.95*R],  # normal oscillation (same as tests 5-10)
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0, 0.03*R, 4.0/3.0, 0, 0,           # simultaneous shear (0 deg phase)
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )
elif testID == 20:
    # Initial shear followed by progressive rigid-body rotation
    #
    # Verification criterion: a small shear displacement accumulated under zero
    # rigid-body motion must be correctly preserved and rotated as a large spin
    # (ωb = [0,0,2.0], parallel to the branch vector) is applied continuously
    # (Sec. 4.20). Implementations that do not properly update the shear-force
    # direction in the contact plane will produce an error that grows progressively
    # with the cumulative rotation angle. This is the in-plane (spin) analogue of
    # test 4, but at a much larger rotation rate (2.0 vs 1.0 rad/s) to amplify the
    # accumulated error. Failure indicates a modelling error.
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,1.0,2.0],           # large spin rigid-body rotation (ωb along z)
        [0,0,0,1.0], [0,0,0,1.0],            # initial orientations
        0, 0, 0, 0, 0, [0,0,1.95*R],         # constant contact
        0, 0, 0, 0, 0,                       # no twist
        0, 0, 0, 0, 0,                       # no roll
        0.01*R, 0, 0, 0, 0,                  # small constant shear velocity (A term only)
        [1.0,0,0], [0,1.0,0],                # roll and shear axes
        tmax, dt,
        R_i, R_j
    )

# Simulate contact interaction
if not doERR:
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
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_fail_test_7,
            Fs_spring_dashpot_Coulomb,
            Tr_spring_dashpot_Coulomb,
            Tt_spring_dashpot_Coulomb,
            Tb_spring_dashpot_Coulomb
            )
    #elif testID == 8:
        # Do nothing
    elif testID == 9:
        results = my_analytical_contact(
            motion,
            contact_params,
            Fn_spring_dashpot,
            Fs_fail_test_9,
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
    
    elif testID == 12:
        Reff = 2.0*(contact_params['R_i'] * contact_params['R_j']) / (contact_params['R_i'] + contact_params['R_j'])
        contact_params['k_b'] = contact_params['k_r'] * Reff
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