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
                  'mu':     0.5,
                  'eta_n':  0.0,
                  'eta_s':  0.0,
                  'eta_r':  0.0,
                  'eta_t':  0.0,
                  'R_i':    R_i,
                  'R_j':    R_j}

# Generate velocities and motion profile
# Set the phase to pi to start with approach
if testID == 1:
    # Tangential elastic response
    motion = my_analytical_motion(
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
    motion = my_analytical_motion(
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
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[1.0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch
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
    w_cycle = 4*np.pi / (3*np.pi)  # frequency for 3π period
    # Place initial branch along z: initial position/branch at [0,0,1.95*R]
    # Choose roll and shear axes orthogonal to branch (x and y)
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.03*R, w_cycle, np.pi/2, 0, [0.0,0.0,1.95*R], # normal: approach-unload cycle (branch along z)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.03*R, w_cycle, 0, 0, # shear: 90° out of phase with normal (peaks at load transitions)
        [1.0,0.0,0.0], [0.0,1.0,0.0], # roll and shear axes (both orthogonal to branch)
        tmax, dt, # time
        R_i, R_j
    )
elif testID == 6:
    # Purely repulsive viscous force
    # Normal only: l(t) = l0 + (B/w)*(cos(wt) - 1) = 2.05R + 0.05R*(cos(t)-1)
    #            = 2.0R + 0.05R*cos(t)  →  ranges [1.95R, 2.05R]
    # Contact phase:    t ∈ (π/2,  3π/2) where l < 2R   (max overlap 0.05R)
    # Separation phase: t ∈ [0, π/2) ∪ (3π/2, 2π]       (max gap 0.05R)
    # At the contact-boundary crossing the normal velocity is |v_n| = B = 0.05R.
    # During the separating half of the contact phase (π < t < 3π/2), v_n > 0 while
    # u_n > 0; a non-clipped dashpot yields F_n < 0 (attractive) at those moments.
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.05*R, 1.0, 0, 0, [2.05*R,0.0,0.0], # normal: oscillates to clearly lose contact
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
    # Normal only: l(t) = 2.0R + 0.025R*(cos(t)-1) = 1.975R + 0.025R*cos(t)
    #   → l_max = 2.0R  (u_n grazes exactly zero at t = 0, 2π, 4π, ...)
    #   → l_min = 1.95R (max overlap 0.05R)
    # The contact is maintained throughout (l ≤ 2R always) but u_n = 0 instantaneously
    # once per cycle. A buggy implementation may permanently disable the viscous
    # contribution after that instant; subsequent cycles would then show no damping.
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.025*R, 1.0, 0, 0, [2.0*R,0.0,0.0], # normal: barely grazes u_n = 0 each cycle
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
    # phi = π/2 gives v_n = -B*cos(t): the velocity magnitude is maximum (= B = 0.05R)
    # exactly at the two contact-boundary crossings per cycle:
    #   l(t) = 2.0R - 0.05R*sin(t)  →  ranges [1.95R, 2.05R]
    #   l = 2R  at  t = 0 and t = π  (v_n = -B and +B respectively)
    # A non-C0 implementation produces a force jump of magnitude eta_n*B at each crossing:
    # upon approach (v_n < 0): F_n jumps from 0 to +eta_n*B as soon as u_n = 0+.
    # upon separation (v_n > 0): F_n would jump from -eta_n*B to 0 at u_n = 0+ if uncapped.
    # Either jump causes accelerations that are independent of time-step size (CFL broken).
    motion = my_analytical_motion(
        [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0.05*R, 1.0, np.pi/2, 0, [2.0*R,0.0,0.0], # maximum speed at contact-boundary crossings
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
        0, 0, 0, 0, 0, [0,0,1.95*R], # normal loading, initial branch (same as test 1)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear: same oscillation as test 1
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
    # Constant normal contact (l0 = 1.95R, u_n = 0.05R):
    #   F_n = k_n * u_n = 1e7 * 0.05 = 5e5 N  (constant)
    #   Coulomb limit = mu * F_n = 0.5 * 5e5 = 2.5e5 N
    # Shear with B = 0.02R (lower amplitude than test 2):
    #   F_s_elastic_max = k_s * 2*B = 0.5e7 * 0.04 = 2.0e5 N  < Coulomb  ✓
    # With eta_s = 1e7 the combined peak (elastic + viscous) exceeds Coulomb:
    #   F_s_elastic(t*) ≈ 1.45e5 N  ≈ 58% of Coulomb  ("about half")
    #   F_s_viscous(t*) ≈ 1.78e5 N
    #   F_s_total(t*)   ≈ 3.23e5 N  > Coulomb  ✓
    # Tests whether the Coulomb limit is applied to the total tangential force
    # (elastic + viscous combined) rather than the elastic part alone.
    motion = my_analytical_motion(
        [0,0,0],[0,0,0],[0,0,0], # initial pos, vel, ang vel
        [0,0,0,1.0], [0,0,0,1.0], # initial ori
        0, 0, 0, 0, 0, [0,0,1.95*R], # constant contact (same as test 2)
        0, 0, 0, 0, 0, # twist
        0, 0, 0, 0, 0, # roll
        0, 0.02*R, 1.0, 0, 0, # shear: elastic alone stays below Coulomb; total exceeds it
        [1.0,0,0], [0,1.0,0], # roll and shear axes
        tmax, dt, # time
        R_i, R_j
    )
    # eta_s = 1e7 chosen so that the combined force exceeds the Coulomb limit even
    # though the elastic component alone does not (see force decomposition above).
    contact_params['eta_s'] = 1.0e7
elif testID == 11:
    # Complex combined motion (NOT YET IMPLEMENTED)
    #
    # Verification criterion: within the elastic regime the force from a motion that
    # simultaneously combines twist, roll, and shear should equal the exact sum of the
    # forces produced by each mode acting in isolation (superposition).
    #
    # Suggested parameters:
    #   l0 = [0,0,1.95*R]  (constant contact, same as tests 1-2)
    #   No rigid-body motion (vb=0, ωb=0)
    #   Twist:  A=0, B=0.010, w=1.0, phi=0    → θ_t_max = 0.02 rad
    #   Roll:   A=0, B=0.010, w=1.0, phi=π/2  → 90° out of phase with twist
    #   Shear:  A=0, B=0.010*R, w=1.0, phi=π  → 180° out of phase with twist
    #   Roll and shear axes: [1,0,0] and [0,1,0]
    #   contact_params: k_r and k_t must be set to non-zero values, e.g.
    #     k_r = 0.25e7, k_t = 0.25e7  (quarter of k_s, all well within elastic regime)
    #   All three mode amplitudes chosen so that each force stays below Coulomb/3,
    #   guaranteeing the combined response also stays elastic.
    raise NotImplementedError(
        "Test 11 (complex combined motion) is not yet implemented. "
        "See inline comments for suggested parameters.")
elif testID == 12:
    # Dependence of force on particle size (NOT YET IMPLEMENTED)
    #
    # Verification criterion: for the same normal displacement, smaller particles
    # should yield a higher (Hertz) or at minimum size-proportional (linear) normal
    # force. Stiffness constants that do not scale with particle size violate
    # dimensional consistency with respect to Young's modulus.
    #
    # Suggested parameters (half the radius of tests 1-10):
    #   R_i = R_j = 0.5  →  R = 0.5,  touching distance = 1.0
    #   l0 = [0,0,2.05*R] = [0,0,1.025]   (same relative offset as test 6)
    #   Normal: A=0, B=0.05*R=0.025, w=1.0, phi=0   (same relative amplitude)
    #   No shear/roll/twist
    #   contact_params: k_n and k_s must scale with radius
    #     e.g. k_n = 2*E*R_eff with R_eff = R_i*R_j/(R_i+R_j) for a consistent comparison
    #   Note: R_i and R_j must be updated before calling my_analytical_motion,
    #         and contact_params['R_i'], contact_params['R_j'] updated accordingly.
    raise NotImplementedError(
        "Test 12 (particle size effect) is not yet implemented. "
        "See inline comments for suggested parameters.")
elif testID == 13:
    # Shear force rotation at large size ratios (NOT YET IMPLEMENTED)
    #
    # Verification criterion: for a large size ratio R_i/R_j, a smaller particle
    # rolling over a larger one should roll off naturally. Standard integration
    # algorithms can cause non-physical sticking at large size ratios.
    #
    # Suggested parameters:
    #   R_i = 2.0, R_j = 0.5  →  R = 1.25,  touching distance = 2.5
    #   l0 = [0,0,2.5*0.98]  (2% initial overlap)
    #   Rigid-body rotation: ωb = [1,0,0] (tilt, as in test 3) with magnitude ~0.5 rad/s
    #   Shear: A=0.01*(R_i+R_j)/2, B=0, w=0, phi=0  (constant shear velocity)
    #   Roll and shear axes: [1,0,0] and [0,1,0]
    #   contact_params: same k_n, k_s, mu as tests 1-10;
    #     note that R_i ≠ R_j means bending ≠ rolling (see Eq. 26-28 in the paper).
    raise NotImplementedError(
        "Test 13 (shear force rotation at large size ratios) is not yet implemented. "
        "See inline comments for suggested parameters.")

# Simulate contact interaction
results = my_analytical_contact(
    motion,
    contact_params,
    Fn_spring_dashpot,
    Fs_spring_dashpot_Coulomb)

# Writing output
dict_to_csv(results, open('../output_ANA/theoretical_output_'+testname+'.csv', 'w'))

# Embed contact_params in the results so that downstream scripts (G*, H*, I*) can
# read the material and geometry parameters directly from the JSON without any
# hardcoding or separate parameter files.
results['contact_params'] = contact_params

dict_to_json(results,'../output_ANA/theoretical_output_'+testname+'.json')

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