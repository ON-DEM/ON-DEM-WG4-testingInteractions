# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import numpy as np

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
    # because the internal viscous forces can indeed be tensile, they just can transmit over a gap.

    # Apply contact model
    Fn = - Fn_mag[:, None] * n_ij

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

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Accumulate shear force
    N, dim = du_s.shape
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

            # Small-angle rotation update inside the loop
            omega = omega_f[i]*dt[i]
            theta = np.linalg.norm(omega)
            if theta > 1e-12: # Bad magic number
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
            Fs_tmp -= k_s * du_s[i]
            # Apply Coulomb limit
            Fs_mag = np.linalg.norm(Fs_tmp)
            Fs_max = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)
            
            # Save elastic component for next time step
            Fs_old = Fs_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Fs_tmp -= eta_s * v_s[i]
            # Apply Coulomb limit, again (this is a modelling choice!)
            Fs_mag = np.linalg.norm(Fs_tmp)
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Save total shear force
            Fs[i] = Fs_tmp.copy() # copy to avoid aliasing

    return Fs

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

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Accumulate shear force
    N, dim = du_s.shape
    Fs = np.zeros((N,dim))
    Fs[0] = 0
    Fs_tmp = np.zeros(3)
    Fs_old = np.zeros(3)
    for i in range(N):
        # Shear force and displacement is lost if contact is lost
        if mask[i]:
            Fs[i] = 0.0
            Fs_old = np.zeros(3)
        else:
            # Retrieve elastic component previous shear force
            Fs_tmp = Fs_old

            # Small-angle rotation update inside the loop
            omega = omega_f[i]*dt[i]
            theta = np.linalg.norm(omega)
            if theta > 1e-12:
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
            Fs_tmp -= k_s * du_s[i]
            # Apply Coulomb limit
            Fs_mag = np.linalg.norm(Fs_tmp)
            Fs_max = mu_s * Fn_mag[i]
            if Fs_mag > Fs_max:
                Fs_tmp *= (Fs_max / Fs_mag)

            # Save elastic component for next time step
            Fs_old = Fs_tmp.copy()

            # Add viscous component
            Fs_tmp -= eta_s * v_s[i]
            # No limit on viscous contribution (this is a modelling choice!)

            # Save total shear force
            Fs[i] = Fs_tmp.copy() # copy to avoid aliasing

    return Fs

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

    # Accumulate rolling force
    N, dim = du_r.shape
    Tr = np.zeros((N,dim))
    Tr[0] = 0
    Tr_tmp = np.zeros(3)
    Tr_old = np.zeros(3)
    for i in range(N):
        # Rolling force is lost if contact is lost
        if mask[i]:
            Tr[i] = 0.0
            Tr_old = np.zeros(3)
        else:
            # Retrieve elastic component previous rolling force
            Tr_tmp = Tr_old

            # Small-angle rotation update inside the loop
            omega = omega_f[i]*dt[i]
            theta = np.linalg.norm(omega)
            if theta > 1e-12:
                axis = omega / theta
                # Rodrigues' rotation formula for rotation matrix
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
                Tr_tmp = R @ Tr_tmp

            # Integrate increment
            Tr_tmp -= np.cross(k_r * du_r[i], rollArm[i])
            # Apply Coulomb limit
            Tr_mag = np.linalg.norm(Tr_tmp)
            Tr_max = mu_r * R_eff * Fn_mag[i]
            if Tr_mag > Tr_max:
                Tr_tmp *= (Tr_max / Tr_mag)

            # Save elastic component for next time step
            Tr_old = Tr_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Tr_tmp -= np.cross(eta_r * v_r[i], rollArm[i])
            # Apply Coulomb limit, again (this is a modelling choice!)
            Tr_mag = np.linalg.norm(Tr_tmp)
            if Tr_mag > Tr_max:
                Tr_tmp *= (Tr_max / Tr_mag)

            # Save total rolling force
            Tr[i] = Tr_tmp.copy() # copy to avoid aliasing

    return Tr


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

    # Accumulate twisting torque
    N = len(u_n)
    Tt = np.zeros((N,3))
    Tt[0] = 0
    Tt_tmp = np.zeros(3)
    Tt_old = np.zeros(3)
    for i in range(N):
        # Twisting torque is lost if contact is lost
        if mask[i]:
            Tt[i] = 0.0
            Tt_old = np.zeros(3)
        else:
            # Retrieve elastic component previous twisting torque
            Tt_tmp = Tt_old
            
            # Integrate increment
            Tt_tmp -= k_t * dtheta_t[i]
            
            # Apply Coulomb limit
            Tt_mag = np.linalg.norm(Tt_tmp)
            Tt_max = mu_t * R_eff * Fn_mag[i]
            if Tt_mag > Tt_max:
                Tt_tmp *= (Tt_max / Tt_mag)

            # Save elastic component for next time step
            Tt_old = Tt_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Tt_tmp -= eta_t * omega_t[i] * dt[i]
            # Apply Coulomb limit, again (this is a modelling choice!)
            Tt_mag = np.linalg.norm(Tt_tmp)
            if Tt_mag > Tt_max:
                Tt_tmp *= (Tt_max / Tt_mag)

            # Save total twisting torque
            Tt[i] = Tt_tmp.copy() # copy to avoid aliasing

    return Tt

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

    # Accumulate twisting torque
    N = len(u_n)
    Tb = np.zeros((N,3))
    Tb[0] = 0
    Tb_tmp = np.zeros(3)
    Tb_old = np.zeros(3)
    for i in range(N):
        # Bending torque is lost if contact is lost
        if mask[i]:
            Tb[i] = 0.0
            Tb_old = np.zeros(3)
        else:
            # Retrieve elastic component previous bending torque
            Tb_tmp = Tb_old
            
            # Integrate increment
            Tb_tmp -= k_b * dtheta_b[i]
            
            # Apply Coulomb limit
            Tb_mag = np.linalg.norm(Tb_tmp)
            Tb_max = mu_b * R_eff * Fn_mag[i]
            if Tb_mag > Tb_max:
                Tb_tmp *= (Tb_max / Tb_mag)

            # Save elastic component for next time step
            Tb_old = Tb_tmp.copy() # copy to avoid aliasing

            # Add viscous component
            Tb_tmp -= eta_b * omega_b[i] * dt[i]
            # Apply Coulomb limit, again (this is a modelling choice!)
            Tb_mag = np.linalg.norm(Tb_tmp)
            if Tb_mag > Tb_max:
                Tb_tmp *= (Tb_max / Tb_mag)

            # Save total bending torque
            Tb[i] = Tb_tmp.copy() # copy to avoid aliasing

    return Tb

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

""" 
NOTES:

Fs_fail_test_1
    No elastic component, directly apply mu Fn

Fs_fail_test_2
    Keep accumulating elastic component or shear displacement, but still apply coulomb limit

Fs_fail_test_3_4
    Don't rotate the shear force vector at all.

Test 5: ratcheting - not sure yet how to fail.
    
Fn_fail_test_6
    Do not limit viscous force to be only repulse. Allow it to be tensile.

Fn_fail_test_7 # Not so sure about this one.
    Completely disable the viscous force after the force has touched zero.

Fn_fail_test_8: Continuity of viscous force - not sure yet how to fail, maybe Maxwell element.

Fs_fail_test_9
    Let the viscous component of the shear force contribute to the shear history.

Fs_fail_test_10; call Fs_spring_dashpot_Coulomb_ext

Test 11: shape dependence - not sure yet how to fail.

Test 12: call bending function instead of roll.

 """


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
            if theta > 1e-12:
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

    # Arm lengths for manual shear velocity calculation
    r_i = (R_i - u_n) * n_ij
    r_j = (R_j - u_n) * n_ij

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

            # Manually compute shear displacement and velocity
            v_rel = (v_j[i] + np.cross(omega_j[i], r_j[i]) - (v_i[i] + np.cross(omega_i[i], r_i[i])))
            v_s = v_rel - np.dot(n_ij[i], v_rel) * n_ij[i] # Relative velocity projected to tangent plane
            du_s = v_s * dt[i]

            # Small-angle rotation update inside the loop
            omega = omega_f[i]*dt[i]
            theta = np.linalg.norm(omega)
            if theta > 1e-12: # Bad magic number
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
    The total normal force is NOT clipped to zero — tensile viscous forces are allowed.
    During rapid separation the dashpot can pull the particles back together, which is
    unphysical for a granular contact without cohesion.
    """
    k_n     = contact_params['k_n']     # (1)
    eta_n   = contact_params['eta_n']   # (1)
    u_n     = motions['u_n'].reshape(-1)    # (N,1)
    v_ijn   = motions['v_ijn']              # (N,3)
    n_ij    = motions['n_ij']               # (N,3)

    active  = (u_n > 0)

    # Normal velocity magnitude (separation +, approach -), swapped so that approach is resistive
    v_n     = - np.einsum('ij,ij->i', v_ijn, n_ij)
    Fn_mag  = np.zeros_like(u_n)
    Fn_mag[active] = k_n * u_n[active] + eta_n * v_n[active]

    # Fail: do NOT apply np.maximum(Fn_mag, 0) — tensile total force is allowed
    Fn = - Fn_mag[:, None] * n_ij

    return Fn




def Fn_fail_test_7(contact_params, motions): # NOT EXACTLY WHAT I INTENDED

    """
    Failure mode for test 7:
    The viscous dashpot is completely disabled once the contact enters the separation phase
    (v_n < 0, i.e. particles moving apart). Only the elastic spring acts during separation,
    as if the dashpot is switched off the moment the force would first touch zero.
    This is asymmetric: energy is dissipated during approach but not during rebound.
    """
    k_n     = contact_params['k_n']     # (1)
    eta_n   = contact_params['eta_n']   # (1)
    u_n     = motions['u_n'].reshape(-1)    # (N,1)
    v_ijn   = motions['v_ijn']              # (N,3)
    n_ij    = motions['n_ij']               # (N,3)

    active  = (u_n > 0)

    # Normal velocity magnitude (separation +, approach -), swapped so that approach is resistive
    v_n     = - np.einsum('ij,ij->i', v_ijn, n_ij)

    # Fail: only add viscous damping during approach (v_n > 0), disabled during separation
    approaching = (v_n > 0)
    Fn_mag  = np.zeros_like(u_n)
    Fn_mag[active]              = k_n * u_n[active]
    Fn_mag[active & approaching] += eta_n * v_n[active & approaching]
    Fn_mag  = np.maximum(Fn_mag, 0) # No adhesion in this model

    Fn = - Fn_mag[:, None] * n_ij

    return Fn


# Fs_fail_test_8


def Fs_fail_test_9(contact_params, motions, Fn):
    """
    Failure mode for test 9:
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
            if theta > 1e-12:
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



def Fs_fail_test_10(contact_params, motions, Fn):
    """
    Failure mode for test 10:
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
