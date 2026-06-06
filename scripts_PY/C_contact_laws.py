# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import numpy as np
import math

MIN_ANGLE_MAG = 1e-14

#
#   SHARED HELPERS FOR THE FRICTIONAL (RETURN-MAPPING) LAWS
#
# The shear/rolling/twisting/bending laws are all the same incremental,
# co-rotational elasto-plastic recurrence: rotate the stored elastic force into the
# current contact frame, add the elastic increment, apply the Coulomb cap, then add
# the (optionally capped) viscous part. The recurrence is inherently sequential
# (the Coulomb cap is a nonlinear function of the running state), but two things are
# hoisted out of the loop for speed:
#   (1) the co-rotation matrix R(omega_f*dt) is constant here (omega_f and dt are
#       per-step constant), so it is built once instead of every iteration; and
#   (2) all elementwise work (the elastic increment, the viscous force, and the
#       Coulomb cap) is precomputed as arrays, and the per-step math is written with
#       plain scalars to avoid NumPy's large per-call overhead on length-3 vectors.

def _rodrigues_matrix(omega):
    """Rotation matrix for the rotation vector omega (axis*angle); identity for a
    negligible angle. Matches the inline Rodrigues construction used previously."""
    theta = math.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])
    if theta <= MIN_ANGLE_MAG:
        return np.eye(3)
    ax, ay, az = omega[0]/theta, omega[1]/theta, omega[2]/theta
    K = np.array([[0.0, -az, ay],
                  [az, 0.0, -ax],
                  [-ay, ax, 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

def _return_map(mask, R, dinc, vinc, Fmax, clip_viscous=True):
    """Sequential co-rotational return mapping shared by the frictional laws.

    Parameters
    ----------
    mask : (N,) bool        True where contact is lost (force and history reset to 0).
    R    : (3,3)            Constant co-rotation matrix R(omega_f*dt).
    dinc : (N,3)            Elastic increment subtracted each step (e.g. k_s*du_s, or
                            cross(rollArm, k_r*du_r) for rolling).
    vinc : (N,3)            Viscous force subtracted each step (e.g. eta_s*v_s).
    Fmax : (N,)             Coulomb cap magnitude per step.
    clip_viscous : bool     Whether the Coulomb cap is re-applied after the viscous
                            part (True for the standard laws, False for the *_ext
                            variant where the dashpot is excluded from the cap).

    Returns the (N,3) force/torque series. The elastic component carried to the next
    step is stored *after* the elastic cap but *before* the viscous part is added.
    """
    N = dinc.shape[0]
    out = np.zeros((N, 3))
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R.ravel()
    ox = oy = oz = 0.0   # stored elastic component (history)
    for i in range(N):
        if mask[i]:
            ox = oy = oz = 0.0
            continue                       # out[i] stays zero: contact lost
        # Rotate the stored elastic component into the current contact frame
        rx = r00*ox + r01*oy + r02*oz
        ry = r10*ox + r11*oy + r12*oz
        rz = r20*ox + r21*oy + r22*oz
        # Add the elastic increment
        rx -= dinc[i, 0]; ry -= dinc[i, 1]; rz -= dinc[i, 2]
        # Coulomb cap on the elastic component
        fmax = Fmax[i]
        mag = math.sqrt(rx*rx + ry*ry + rz*rz)
        if mag > fmax:
            s = fmax / mag; rx *= s; ry *= s; rz *= s
        # Store the elastic component for the next step (before the viscous part)
        ox, oy, oz = rx, ry, rz
        # Add the viscous part, optionally re-capping the total
        rx -= vinc[i, 0]; ry -= vinc[i, 1]; rz -= vinc[i, 2]
        if clip_viscous:
            mag = math.sqrt(rx*rx + ry*ry + rz*rz)
            if mag > fmax:
                s = fmax / mag; rx *= s; ry *= s; rz *= s
        out[i, 0] = rx; out[i, 1] = ry; out[i, 2] = rz
    return out

#
#   NORMAL FORCE LAWS
#

def Fn_spring_dashpot(contact_params, motions):
    """
    Linear spring-dashpot in parallel
    F_n = - ( k_n * u_n + eta_n * v_n ) * n 
    """
    k_n     = contact_params['k_n']         # (1)
    eta_n   = contact_params['eta_n']       # (1)
    u_n     = motions['u_n'].reshape(-1)    # (N,1)
    v_ijn   = motions['v_ijn']              # (N,3)
    n_ij    = motions['n_ij']               # (N,3)

    # Check if there is contact at all.
    active = (u_n > 0)

    # Normal velocity magnitude (separation +, approach -), swapped so that approach is resistive (separation -, approach +).
    v_n = - np.einsum('ij,ij->i', v_ijn, n_ij)

    # Magnitude
    Fn_mag = np.zeros_like(u_n)
    Fn_mag[active] = k_n * u_n[active] + eta_n * v_n[active]
    Fn_mag = np.maximum(Fn_mag, 0) # No adhesion in this model
    # This clipping should be applied to the total force, not the viscous component only, 
    # because the internal viscous forces can indeed be tensile, they just can't transmit over a gap.

    # Apply contact model
    Fn = - Fn_mag[:, None] * n_ij

    return Fn

def Fn_spring_dashpot_maxwell(contact_params, motions):
    """
    Normal mode: parallel spring k_n plus one Maxwell arm (armKn[0] in series
    with armEtan[0]), both repulsive-only.

    The arm history is NOT reset on loss of contact. Instead it decays freely
    with v_approach = 0 (pure exponential relaxation) so that a particle
    re-entering contact before full decay sees the correct residual arm force.
    The normal force applied to the bodies is still zero while out of contact.
    """
    k_n     = contact_params['k_n']
    k_arm   = contact_params.get('armKn',   [0.0])[0]
    eta_arm = contact_params.get('armEtan', [0.0])[0]
    u_n     = motions['u_n'].reshape(-1)
    v_ijn   = motions['v_ijn']
    n_ij    = motions['n_ij']
    dt      = np.array(motions['dt'], dtype=float)

    has_arm = (k_arm > 0 and eta_arm > 0)
    tau     = eta_arm / k_arm if has_arm else 1.0

    active = (u_n > 0)
    v_n    = -np.einsum('ij,ij->i', v_ijn, n_ij)

    N         = len(u_n)
    Fn_mag    = np.zeros(N)
    arm_force = 0.0

    for i in range(N):
        # Always update the arm, even out of contact: v_approach = 0 gives pure decay.
        # This preserves viscoelastic memory for re-contact before full relaxation.
        if has_arm:
            v_approach = v_n[i] if active[i] else 0.0
            decay      = np.exp(-dt[i] / tau)
            arm_force  = arm_force * decay + eta_arm * v_approach * (1.0 - decay)

        if active[i]:
            Fn_mag[i] = max(0.0, k_n * u_n[i] + arm_force)
        # else: Fn_mag[i] remains 0 — no force applied across a gap

    Fn = -Fn_mag[:, None] * n_ij
    return Fn

#
#   SHEAR FORCE LAWS
#

def Fs_spring_dashpot_Coulomb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fs = - k_s * u_s * n_s - eta_s * v_s, with |Fs| <= mu*|Fn|
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu_s      = contact_params['mu_s']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    du_s    = np.array(motions['du_s'], dtype=float)        # (N,3) - over the last time step
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    # Test for contact (force and history reset where there is none)
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Hoist the constant co-rotation matrix and precompute the per-step elastic
    # increment, viscous force, and Coulomb cap; the sequential return mapping (history
    # + Coulomb cap) is delegated to _return_map (see top of file).
    R = _rodrigues_matrix(omega_f[0] * dt[0])
    return _return_map(mask, R, k_s * du_s, eta_s * v_s, mu_s * Fn_mag, clip_viscous=True)

def Fs_spring_dashpot_Coulomb_ext(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fs = - k_s * u_s * n_s, with |Fs| <= mu*|Fn|
    Fs -= eta_s * v_s
    In other words, the dashpot is excluded from the Coulomb limit
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu_s      = contact_params['mu_s']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    du_s    = np.array(motions['du_s'], dtype=float)        # (N,3) - over the last time step
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    # Test for contact (force and history reset where there is none)
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Same return mapping as Fs_spring_dashpot_Coulomb, but the viscous dashpot is
    # excluded from the Coulomb cap (clip_viscous=False).
    R = _rodrigues_matrix(omega_f[0] * dt[0])
    return _return_map(mask, R, k_s * du_s, eta_s * v_s, mu_s * Fn_mag, clip_viscous=False)

#
#   ROLLING TORQUE LAWS
#

def Tr_spring_dashpot_Coulomb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Tr = - k_r * u_r - eta_r * v_r, with |Tr| <= mu*Reff*|Fn|
    """
    k_r     = contact_params['k_r']                         # (1)
    eta_r   = contact_params['eta_r']                       # (1)
    mu_r    = contact_params['mu_r']                        # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_r     = np.array(motions['v_r'], dtype=float)         # (N,3)
    du_r    = np.array(motions['du_r'], dtype=float)        # (N,3) - over the last time step
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    n_ij    = motions['n_ij']                               # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)
    R_i     = contact_params['R_i']
    R_j     = contact_params['R_j']

    # Effective radius needed to match dimensions on friction check
    R_eff = 2.0*R_i*R_j/(R_i+R_j)

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Roll arm
    rollArm = R_eff * n_ij

    # The rolling torque increments are the moments of the rolling-spring/dashpot
    # forces about the roll arm; precompute them (vectorised cross products) and the
    # Coulomb cap, then run the shared sequential return mapping.
    R = _rodrigues_matrix(omega_f[0] * dt[0])
    dinc = np.cross(rollArm, k_r * du_r)
    vinc = np.cross(rollArm, eta_r * v_r)
    Fmax = mu_r * R_eff * Fn_mag
    return _return_map(mask, R, dinc, vinc, Fmax, clip_viscous=True)


#
#   TWISTING TORQUE LAWS
#

def Tt_spring_dashpot_Coulomb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Tt = - k_t * theta - eta_t * omega_t * dt, with |Tt| <= mu*R_eff*|Fn|
    """
    k_t     = contact_params['k_t']                         # (1)
    eta_t   = contact_params['eta_t']                       # (1)
    mu_t      = contact_params['mu_t']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    n_ij    = np.array(motions['n_ij'], dtype=float)        # (N,3)
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    omega_t = np.asarray(motions['omega_t'], dtype=float)   # (N,3)
    dtheta_t = np.array(motions['dtheta_t'], dtype=float)   # (N,3) - over the last time step
    dt      = np.array(motions['dt'], dtype=float)          # (1)
    R_i     = contact_params['R_i']
    R_j     = contact_params['R_j']

    # Effective radius needed to match dimensions on friction check
    R_eff = 2.0*R_i*R_j/(R_i+R_j)

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Precompute the elastic increment (k_t*dtheta_t), viscous torque (eta_t*omega_t),
    # and Coulomb cap, then run the shared sequential return mapping. Twisting acts
    # directly along the contact normal, so no lever-arm cross product is needed.
    R = _rodrigues_matrix(omega_f[0] * dt[0])
    Fmax = mu_t * R_eff * Fn_mag
    return _return_map(mask, R, k_t * dtheta_t, eta_t * omega_t, Fmax, clip_viscous=True)

#
#   BENDING TORQUE LAWS
#

def Tb_spring_dashpot_Coulomb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Tb = - k_b * theta_b - eta_b * omega_b * dt, with |Tb| <= mu*R_eff*|Fn|
    """
    k_b     = contact_params['k_b']                         # (1)
    eta_b   = contact_params['eta_b']                       # (1)
    mu_b      = contact_params['mu_b']                        # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    n_ij    = np.array(motions['n_ij'], dtype=float)        # (N,3)
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    omega_b = np.asarray(motions['omega_b'], dtype=float)   # (N,3)
    dtheta_b = np.array(motions['dtheta_b'], dtype=float)   # (N,3) - over the last time step
    dt      = np.array(motions['dt'], dtype=float)          # (1)
    R_i     = contact_params['R_i']
    R_j     = contact_params['R_j']

    # Effective radius needed to match dimensions on friction check
    R_eff = 2.0*R_i*R_j/(R_i+R_j)

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Precompute the elastic increment (k_b*dtheta_b), viscous torque (eta_b*omega_b),
    # and Coulomb cap, then run the shared sequential return mapping.
    R = _rodrigues_matrix(omega_f[0] * dt[0])
    Fmax = mu_b * R_eff * Fn_mag
    return _return_map(mask, R, k_b * dtheta_b, eta_b * omega_b, Fmax, clip_viscous=True)

#
#   Mixing rules for mechanical contact properties
#

def my_compute_effective_params(contact_params):
    """
    Compute effective contact parameters for two particles from contact_params dict:
      - E* effective normal modulus
      - G* effective shear modulus
      - R* effective radius
      - m* effective mass

    Expects keys: 'E_i','nu_i','E_j','nu_j','R_i','R_j' (optional 'G_i','G_j','m_i,'m_j').
    """
    E_i, nu_i = contact_params['E_i'], contact_params['nu_i']
    E_j, nu_j = contact_params['E_j'], contact_params['nu_j']
    R_i, R_j = contact_params['R_i'], contact_params['R_j']
    G_i = contact_params.get('G_i', None)
    G_j = contact_params.get('G_j', None)
    m_i = contact_params.get('m_i',None)
    m_j = contact_params.get('m_j',None)

    # Effective normal modulus
    inv_E_star = (1 - nu_i**2) / E_i + (1 - nu_j**2) / E_j
    E_star = 1.0 / inv_E_star

    # Determine shear moduli
    if G_i is None:
        G_i = E_i / (2.0 * (1.0 + nu_i))
    if G_j is None:
        G_j = E_j / (2.0 * (1.0 + nu_j))

    # Effective shear modulus
    inv_G_star = (2.0 - nu_i) / G_i + (2.0 - nu_j) / G_j
    G_star = 1.0 / inv_G_star

    # Effective radius
    R_star = 2.0 * (R_i * R_j) / (R_i + R_j)

    # Effective mass
    if (m_i is not None) and (m_j is not None):
        m_star = (m_i * m_j) / (m_i + m_j)
    else:
        m_star = 1

    return E_star, G_star, R_star, m_star


#
#   Faulty or alternative versions of the above contact models for demonstrative purposes.
#

def Fs_fail_test_1(contact_params, motions, Fn):

    """
    Failure mode for test 1:
    No elastic component; shear force is set directly to mu*|Fn| opposing the sliding velocity.
    There is no spring, so there is no static friction — the contact is always at the Coulomb limit
    whenever sliding occurs. Without a velocity the shear force is zero (no direction to apply it).
    """
    mu_s    = contact_params['mu_s']
    u_n     = np.array(motions['u_n'], dtype=float)
    v_s     = np.array(motions['v_s'], dtype=float)

    # Only act where there is contact
    active  = (u_n.ravel() > 0)
    Fn_mag  = np.linalg.norm(Fn, axis=1)

    N       = v_s.shape[0]
    Fs      = np.zeros((N, 3))
    for i in range(N):
        if not active[i]:
            continue
        v_s_mag = np.linalg.norm(v_s[i])
        if v_s_mag > 1e-12:
            # Always at the Coulomb limit, direction opposes sliding velocity
            Fs[i] = -mu_s * Fn_mag[i] * v_s[i] / v_s_mag
        # else: zero shear force — no elastic spring to hold a direction at rest

    return Fs



def Fs_fail_test_2(contact_params, motions, Fn):

    """
    Failure mode for test 2:
    The elastic spring displacement is accumulated without capping at the Coulomb limit
    (the spring 'winds up' during sustained sliding). The output force is still Coulomb-limited,
    but Fs_old is saved before capping, so the stored state grows without bound.
    This causes a large stored elastic force to snap back when sliding stops.
    """
    k_s     = contact_params['k_s']
    eta_s   = contact_params['eta_s']
    mu_s    = contact_params['mu_s']
    u_n     = np.array(motions['u_n'], dtype=float)
    v_s     = np.array(motions['v_s'], dtype=float)
    du_s    = np.array(motions['du_s'], dtype=float)
    omega_f = np.asarray(motions['omega_f'], dtype=float)
    dt      = np.array(motions['dt'], dtype=float)

    mask    = (u_n.ravel() == 0.0)
    Fn_mag  = np.linalg.norm(Fn, axis=1)
    N, dim  = du_s.shape

    Fs      = np.zeros((N, dim))
    Fs[0]   = 0
    Fs_tmp  = np.zeros(3)
    Fs_old  = np.zeros(3)

    for i in range(N):
        if mask[i]:
            Fs[i]   = 0.0
            Fs_old  = np.zeros(3)
        else:
            Fs_tmp  = Fs_old

            # Small-angle rotation update inside the loop
            omega   = omega_f[i] * dt[i]
            theta   = np.linalg.norm(omega)
            if theta > MIN_ANGLE_MAG:
                axis = omega / theta

                # Rodrigues' rotation formula for rotation matrix
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R       = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
                Fs_tmp  = R @ Fs_tmp

            # Integrate increment
            Fs_tmp -= k_s * du_s[i]

            # Fail: save elastic component BEFORE capping — spring winds up without bound
            Fs_old  = Fs_tmp.copy()

            # Apply Coulomb limit to the output only
            Fs_mag  = np.linalg.norm(Fs_tmp)
            Fs_max  = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Add viscous component
            Fs_tmp -= eta_s * v_s[i]

            # Apply Coulomb limit again (this is a modelling choice!)
            Fs_mag  = np.linalg.norm(Fs_tmp)
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            Fs[i]   = Fs_tmp.copy() # copy to avoid aliasing

    return Fs



def Fs_fail_test_3_4(contact_params, motions, Fn):

    """
    Failure mode for tests 3 and 4:
    The accumulated elastic shear force is never rotated to track the evolving contact frame.
    For contacts where the contact normal rotates (e.g. rolling), the stored force vector drifts
    out of the tangent plane, producing unphysical components in the normal direction.
    """
    k_s     = contact_params['k_s']
    eta_s   = contact_params['eta_s']
    mu_s    = contact_params['mu_s']
    u_n     = np.array(motions['u_n'], dtype=float)
    v_s     = np.array(motions['v_s'], dtype=float)
    du_s    = np.array(motions['du_s'], dtype=float)
    # omega_f and dt are still read (present in motions) but intentionally unused below
    dt      = np.array(motions['dt'], dtype=float)

    mask    = (u_n.ravel() == 0.0)
    Fn_mag  = np.linalg.norm(Fn, axis=1)

    N, dim  = du_s.shape
    Fs      = np.zeros((N, dim))
    Fs[0]   = 0

    Fs_tmp  = np.zeros(3)
    Fs_old  = np.zeros(3)

    for i in range(N):
        if mask[i]:
            Fs[i]   = 0.0
            Fs_old  = np.zeros(3)

        else:
            # Retrieve previous shear force WITHOUT any rotation update (the fail)
            Fs_tmp  = Fs_old

            # Integrate increment
            Fs_tmp -= k_s * du_s[i]

            # Apply Coulomb limit
            Fs_mag  = np.linalg.norm(Fs_tmp)
            Fs_max  = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)
            
            # Save elastic component for next time step
            Fs_old  = Fs_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Fs_tmp -= eta_s * v_s[i]

            # Apply Coulomb limit, again (this is a modelling choice!)
            Fs_mag  = np.linalg.norm(Fs_tmp)
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            Fs[i]   = Fs_tmp.copy() # copy to avoid aliasing

    return Fs



def Fs_fail_test_5(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fs = - k_s * u_s * n_s - eta_s * v_s, with |Fs| <= mu*|Fn|
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu_s      = contact_params['mu_s']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    #v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    #du_s    = np.array(motions['du_s'], dtype=float)        # (N,3) - over the last time step
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)
    R_i     = contact_params['R_i']
    R_j     = contact_params['R_j']
    n_ij    = np.array(motions['n_ij'], dtype=float)        # (N,3)
    v_i     = np.array(motions['v_i'], dtype=float)        # (N,3)
    v_j     = np.array(motions['v_j'], dtype=float)        # (N,3)
    omega_i = np.array(motions['omega_i'], dtype=float)    # (N,3)
    omega_j = np.array(motions['omega_j'], dtype=float)    # (N,3)

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    

    # Accumulate shear force
    N, dim = n_ij.shape
    Fs = np.zeros((N,dim))
    Fs[0] = 0
    Fs_tmp = np.zeros(3)
    Fs_old = np.zeros(3)
    for i in range(N):
        # Shear force and displacement are lost if contact is lost
        if mask[i]:
            Fs[i] = 0.0
            Fs_old = np.zeros(3)
        else:
            # Retrieve elastic component previous shear force
            Fs_tmp = Fs_old

            # Arm lengths for manual shear velocity calculation
            r_i = R_i - 0.5*u_n[i]
            r_j = R_j - 0.5*u_n[i]

            # Manually compute shear displacement and velocity
            v_rel = (v_j[i] + r_j * np.cross(-n_ij[i], omega_j[i])) - (v_i[i] + r_i * np.cross(n_ij[i], omega_i[i]))
            # Above is same as v_j - v_i + r_i * omega_i x n_ij + r_j * omega_j x n_ij
            v_s = v_rel - np.dot(n_ij[i], v_rel) * n_ij[i] # Relative velocity projected to tangent plane
            du_s = v_s * dt[i]

            # Small-angle rotation update inside the loop
            omega = omega_f[i]*dt[i]
            theta = np.linalg.norm(omega)
            if theta > MIN_ANGLE_MAG: # Bad magic number
                axis = omega / theta
                # Rodrigues’ rotation formula for rotation matrix
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
                Fs_tmp = R @ Fs_tmp

            # Integrate increment
            Fs_tmp -= k_s * du_s
            # Apply Coulomb limit
            Fs_mag = np.linalg.norm(Fs_tmp)
            Fs_max = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)
            
            # Save elastic component for next time step
            Fs_old = Fs_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Fs_tmp -= eta_s * v_s
            # Apply Coulomb limit, again (this is a modelling choice!)
            Fs_mag = np.linalg.norm(Fs_tmp)
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Save total shear force
            Fs[i] = Fs_tmp.copy() # copy to avoid aliasing

    return Fs


def Fn_fail_test_6(contact_params, motions):
    """
    Failure mode for test 6:
    The total normal force is NOT clipped to repulsive — tensile Maxwell arm forces
    are allowed. When the arm spring force is negative (tensile, after a rapid
    separation) and its magnitude exceeds the base spring contribution, the contact
    pulls the particles back together, which is unphysical for a granular contact
    without cohesion.
    The arm still decays freely during out-of-contact periods (history preserved),
    so the only difference from the correct model is the missing max(0, ...) clip.
    """
    k_n     = contact_params['k_n']
    k_arm   = contact_params.get('armKn',   [0.0])[0]
    eta_arm = contact_params.get('armEtan', [0.0])[0]
    u_n     = motions['u_n'].reshape(-1)
    v_ijn   = motions['v_ijn']
    n_ij    = motions['n_ij']
    dt      = np.array(motions['dt'], dtype=float)

    has_arm = (k_arm > 0 and eta_arm > 0)
    tau     = eta_arm / k_arm if has_arm else 1.0

    active    = (u_n > 0)
    v_n       = -np.einsum('ij,ij->i', v_ijn, n_ij)
    N         = len(u_n)
    Fn_mag    = np.zeros(N)
    arm_force = 0.0

    for i in range(N):
        # Arm decays freely out of contact (v_approach = 0), same as correct model
        if has_arm:
            v_approach = v_n[i] if active[i] else 0.0
            decay      = np.exp(-dt[i] / tau)
            arm_force  = arm_force * decay + eta_arm * v_approach * (1.0 - decay)

        if active[i]:
            # Fail: do NOT clip to repulsive — tensile total force is allowed
            Fn_mag[i] = k_n * u_n[i] + arm_force

    Fn = -Fn_mag[:, None] * n_ij
    return Fn



def Fn_fail_test_7(contact_params, motions):
    """
    Failure mode for test 7:
    The Maxwell arm is always updated (it decays freely), but the velocity fed
    into it is capped at zero during separation (v_n < 0). This is asymmetric:
    on approach the arm is driven by the full approach velocity, on separation
    it only decays — as if the dashpot within the arm is a one-way valve that
    closes the moment the particles start moving apart. Energy is dissipated
    during approach but the rebound is stiffer than it should be.
    """
    k_n     = contact_params['k_n']
    k_arm   = contact_params.get('armKn',   [0.0])[0]
    eta_arm = contact_params.get('armEtan', [0.0])[0]
    u_n     = motions['u_n'].reshape(-1)
    v_ijn   = motions['v_ijn']
    n_ij    = motions['n_ij']
    dt      = np.array(motions['dt'], dtype=float)

    has_arm = (k_arm > 0 and eta_arm > 0)
    tau     = eta_arm / k_arm if has_arm else 1.0

    active    = (u_n > 0)
    v_n       = -np.einsum('ij,ij->i', v_ijn, n_ij)
    N         = len(u_n)
    Fn_mag    = np.zeros(N)
    arm_force = 0.0

    for i in range(N):
        # Arm always updates, but velocity is capped at zero during separation.
        # Out of contact counts as separation (v_approach = 0), same as correct model.
        if has_arm:
            v_approach = max(0.0, v_n[i])  # Fail: one-way valve — no negative drive
            decay      = np.exp(-dt[i] / tau)
            arm_force  = arm_force * decay + eta_arm * v_approach * (1.0 - decay)

        if active[i]:
            Fn_mag[i] = max(0.0, k_n * u_n[i] + arm_force)

    Fn = -Fn_mag[:, None] * n_ij
    return Fn


# Fn_fail_test_8, just use dashpot


def Fn_fail_test_9(contact_params, motions):
    """
    Loss of contact instantly erases the Maxwell arm history, instead of allowing it to decay freely.

    Normal mode: parallel spring k_n plus one Maxwell arm (armKn[0] in series
    with armEtan[0]), both repulsive-only.

    The arm history is NOT reset on loss of contact. Instead it decays freely
    with v_approach = 0 (pure exponential relaxation) so that a particle
    re-entering contact before full decay sees the correct residual arm force.
    The normal force applied to the bodies is still zero while out of contact.
    """
    k_n     = contact_params['k_n']
    k_arm   = contact_params.get('armKn',   [0.0])[0]
    eta_arm = contact_params.get('armEtan', [0.0])[0]
    u_n     = motions['u_n'].reshape(-1)
    v_ijn   = motions['v_ijn']
    n_ij    = motions['n_ij']
    dt      = np.array(motions['dt'], dtype=float)

    has_arm = (k_arm > 0 and eta_arm > 0)
    tau     = eta_arm / k_arm if has_arm else 1.0

    active = (u_n > 0)
    v_n    = -np.einsum('ij,ij->i', v_ijn, n_ij)

    N         = len(u_n)
    Fn_mag    = np.zeros(N)
    arm_force = 0.0

    for i in range(N):
        # Always update the arm, even out of contact: v_approach = 0 gives pure decay.
        # This preserves viscoelastic memory for re-contact before full relaxation.
        if has_arm:
            v_approach = v_n[i] if active[i] else 0.0
            decay      = np.exp(-dt[i] / tau)
            arm_force  = arm_force * decay + eta_arm * v_approach * (1.0 - decay)

        if active[i]:
            Fn_mag[i] = max(0.0, k_n * u_n[i] + arm_force)
        else:
            arm_force = 0.0  # Fail: loss of contact instantly erases arm history — no memory for re-contact
        # else: Fn_mag[i] remains 0 — no force applied across a gap

    Fn = -Fn_mag[:, None] * n_ij
    return Fn


def Fs_fail_test_10(contact_params, motions, Fn):
    """
    Failure mode for test 10:
    The viscous component of the shear force is incorrectly included in the accumulated
    elastic history (Fs_old). The rate-dependent dashpot force contaminates the elastic
    state carried to the next time step, causing the stored spring force to be
    velocity-dependent and introducing drift over time.
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu_s    = contact_params['mu_s']                        # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    du_s    = np.array(motions['du_s'], dtype=float)        # (N,3) - over the last time step
    omega_f = np.asarray(motions['omega_f'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    mask    = (u_n.ravel() == 0.0)
    Fn_mag  = np.linalg.norm(Fn, axis=1)
    N, dim  = du_s.shape

    Fs      = np.zeros((N, dim))
    Fs[0]   = 0
    Fs_tmp  = np.zeros(3)
    Fs_old  = np.zeros(3)
    for i in range(N):
        if mask[i]:
            Fs[i]   = 0.0
            Fs_old  = np.zeros(3)
        else:
            Fs_tmp  = Fs_old

            # Small-angle rotation update inside the loop
            omega   = omega_f[i] * dt[i]
            theta   = np.linalg.norm(omega)
            if theta > MIN_ANGLE_MAG:
                axis = omega / theta
                # Rodrigues' rotation formula for rotation matrix
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R       = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
                Fs_tmp  = R @ Fs_tmp

            # Integrate increment
            Fs_tmp -= k_s * du_s[i]

            # Apply Coulomb limit
            Fs_mag  = np.linalg.norm(Fs_tmp)
            Fs_max  = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Add viscous component
            Fs_tmp -= eta_s * v_s[i]

            # Apply Coulomb limit, again (this is a modelling choice!)
            Fs_mag  = np.linalg.norm(Fs_tmp)
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Fail: save total force (viscous included) as next step's elastic history
            Fs_old  = Fs_tmp.copy() # copy to avoid aliasing
            Fs[i]   = Fs_tmp.copy() # copy to avoid aliasing

    return Fs



def Fs_fail_test_11(contact_params, motions, Fn):
    """
    Failure mode for test 11:
    Delegates to Fs_spring_dashpot_Coulomb_ext, in which the viscous dashpot contribution
    is excluded from the Coulomb limit check. The dashpot force is added on top of the
    already-limited elastic force with no further cap, allowing the total shear force to
    exceed mu*|Fn| whenever there is a non-zero slip velocity.
    """
    return Fs_spring_dashpot_Coulomb_ext(contact_params, motions, Fn)





# End of file






#
# Below are some other contact models that we don't use right now.
#

def Fn_linear_elastic(contact_params, motions):
    """F_n = - k_n u_n"""
    kn = contact_params['k_n']
    u_n = motions['u_n'].reshape(-1)
    n_ij = motions['n_ij']
    Fn = kn * u_n[:, None] * n_ij
    return Fn



def Fn_hertzian(contact_params, motions):
    """
    Hertzian normal force: F_n = k_n * u_n^(3/2) * n_ij
    k_n = (4/3) * E* * sqrt(R*)
    """
    u_n = motions['u_n'].reshape(-1)   # (N,)
    n_ij = motions['n_ij']             # (N,3)

    E_star, _, R_star, _ = my_compute_effective_params(contact_params)

    k_n = (4.0 / 3.0) * E_star * np.sqrt(R_star)
    mag = k_n * u_n**1.5
    Fn = mag[:, None] * n_ij
    return Fn



def Fn_viscous_const(contact_params, motions):
    """
    Viscous normal damping (constant restitution model).

    eta_n = 2 * sqrt(m_star * k_n) * beta_n
    F_n_visc = - eta_n * v_n * n_ij
    """
    _, _, _, m_star = my_compute_effective_params(contact_params)
    k_n     = contact_params['k_n']
    beta_n  = contact_params['beta_n']
    v_ijn     = motions['v_ijn']             # (N,3)
    n_ij    = motions['n_ij']            # (N,3)

    # Normal velocity magnitude
    v_n = np.linalg.norm(v_ijn, axis=1)

    # viscous coefficient
    eta_n = 2.0 * np.sqrt(m_star * k_n) * beta_n

    # viscous normal force vector
    Fn_visc = - eta_n * v_n[:, None] * n_ij
    return Fn_visc

def Fn_viscous_veldep(contact_params, motions):
    """
    Viscous normal damping (velocity-dependent restitution model).

    beta = - ln(e) / sqrt(pi^2 + [ln(e)]^2)
    eta_n = 2 * sqrt(m_star * k_n) * beta
    F_n_visc = - eta_n * v_n * n_ij
    """
    _, _, _, m_star = my_compute_effective_params(contact_params)
    k_n  = contact_params['k_n']
    cor  = contact_params['restitution']
    v_ijn  = motions['v_ijn']
    n_ij = motions['n_ij']

    # Normal velocity magnitude
    v_n = np.linalg.norm(v_ijn, axis=1)

    # damping ratio
    beta = - np.log(cor) / np.sqrt(np.pi**2 + (np.log(cor))**2)

    # viscous coefficient
    eta_n = 2.0 * np.sqrt(m_star * k_n) * beta

    # viscous normal force
    Fn_visc = - eta_n * v_n[:, None] * n_ij
    return Fn_visc




def Fs_linear_shear(contact_params, motions, Fn):
    """
    Linear shear stiffness: F_t = -k_s0 * u_t, capped by Coulomb
    where k_s0 = 8 * G* * sqrt(R*)
    """
    u_t = motions['u_t']
    mu = contact_params['mu']

    _, G_star, R_star, _ = my_compute_effective_params(contact_params)
    k_s0 = 8.0 * G_star * np.sqrt(R_star)

    Fs = -k_s0 * u_t
    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Fs_mag = np.linalg.norm(Fs, axis=1)
    slip = Fs_mag > mu * Fn_mag
    Fs[slip] *= (mu * Fn_mag[slip] / Fs_mag[slip])[:, None]
    return Fs



def Fs_full_mindlin(contact_params, motions, Fn):
    """
    Full no-slip Mindlin: F_t = -k_s * u_t, capped by Coulomb
    where k_s = k_s0 * sqrt(u_n), k_s0 = 8 * G* * sqrt(R*)
    """
    u_t = motions['u_t']
    u_n = motions['u_n']
    mu = contact_params['mu']

    _, G_star, R_star, _ = my_compute_effective_params(contact_params)
    k_s0 = 8.0 * G_star * np.sqrt(R_star)

    k_s = k_s0 * np.sqrt(u_n)
    Fs = - (k_s[:, None] * u_t)

    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Fs_mag = np.linalg.norm(Fs, axis=1)
    slip = Fs_mag > mu * Fn_mag
    Fs[slip] *= (mu * Fn_mag[slip] / Fs_mag[slip])[:, None]
    return Fs



def Fs_partial_slip(contact_params, motions, Fn):
    """
    Mindlin–Deresiewitz partial-slip:
      F_t = -8 G* a [ u_t - ((a-c)/(3 a^2)) |u_t|^2 u_t ], Coulomb-limited
      where a = sqrt(R* u_n), c = a (1 - |u_t|/a)^(1/3)
    """
    u_t = motions['u_t']
    u_n = motions['u_n'].reshape(-1)
    mu = contact_params['mu']

    _, G_star, R_star, _ = my_compute_effective_params(contact_params)
    a = np.sqrt(R_star * u_n)

    u_t_mag = np.linalg.norm(u_t, axis=1)
    ratio = np.clip(1 - u_t_mag / a, 0.0, None)
    c = a * ratio**(1.0/3.0)

    term = (a - c) / (3.0 * a**2)
    diff = u_t - (term[:, None] * (u_t_mag**2)[:, None] * u_t / u_t_mag[:, None])
    Fs = -8.0 * G_star * a[:, None] * diff

    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Fs_mag = np.linalg.norm(Fs, axis=1)
    slip = Fs_mag > mu * Fn_mag
    Fs[slip] *= (mu * Fn_mag[slip] / Fs_mag[slip])[:, None]
    return Fs



def Fs_viscous_const(contact_params, motions):
    """
    Viscous tangential damping (constant restitution model).

    eta_t = 2 * sqrt(m_star * k_t) * beta_t
    F_t_visc = - eta_t * v_t
    """
    _, _, _, m_star = my_compute_effective_params(contact_params)
    k_t     = contact_params['k_t']
    beta_t  = contact_params['beta_t']
    v_t     = motions['v_t']               # (N,3)

    # viscous coefficient
    eta_t = 2.0 * np.sqrt(m_star * k_t) * beta_t

    # viscous tangential force
    Fs_visc = - eta_t * v_t
    return Fs_visc



def Fs_viscous_veldep(contact_params, motions):
    """
    Viscous tangential damping (velocity-dependent restitution model).

    beta = - ln(e) / sqrt(pi^2 + [ln(e)]^2)
    eta_t = 2 * sqrt(m_star * k_t) * beta
    F_t_visc = - eta_t * v_t
    """
    # unpack
    _, _, _, m_star = my_compute_effective_params(contact_params)
    k_t  = contact_params['k_t']
    cor  = contact_params['restitution']
    v_t  = motions['v_t']

    # damping ratio
    beta = - np.log(cor) / np.sqrt(np.pi**2 + (np.log(cor))**2)

    # viscous coefficient
    eta_t = 2.0 * np.sqrt(m_star * k_t) * beta

    # viscous tangential force
    Fs_visc = - eta_t * v_t
    return Fs_visc
