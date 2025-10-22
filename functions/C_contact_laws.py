# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

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
    eta_n   = contact_params['k_n']         # (1)
    u_n     = motions['u_n'].reshape(-1)    # (N,1)
    v_n     = motions['v_n']                # (N,1)
    n_ij    = motions['n_ij']               # (N,3)

    # Compute effective parameters?

    Fn = - ( k_n * u_n[:, None] + eta_n * v_n[:, None] ) * n_ij
    return Fn

#
#   SHEAR FORCE LAWS
#

def Fs_spring_dashpot_Coloumb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fs = - k_s * u_s * n_s - eta_s * v_s, with |Fs| <= mu*|Fn|
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu      = contact_params['mu']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    omega_b = np.asarray(motions['omega_b'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    # Shear displacement increment per time step
    du_s = v_s * dt[:,None]

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
            omega = omega_b[i]*dt[i]
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
            if Fs_mag > mu * Fn_mag[i]:
                Fs_tmp *= (mu * Fn_mag[i] / Fs_mag)
            
            # Save elastic component for next time step
            Fs_old = Fs_tmp.copy()

            # Add viscous component
            Fs_tmp -= eta_s * v_s
            # Apply Coulomb limit, again
            Fs_mag = np.linalg.norm(Fs_tmp)
            if Fs_mag > mu * Fn_mag[i]:
                Fs_tmp *= (mu * Fn_mag[i] / Fs_mag)

            # Save total shear force
            Fs[i] = Fs_tmp.copy() # copy to avoid aliasing

    return Fs

def Fs_spring_dashpot_Coloumb_ext(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fs = - k_s * u_s * n_s, with |Fs| <= mu*|Fn|
    Fs -= eta_s * v_s
    In other words, the dashpot is excluded from the Coulomb limit
    """
    k_s     = contact_params['k_s']                         # (1)
    eta_s   = contact_params['eta_s']                       # (1)
    mu      = contact_params['mu']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_s     = np.array(motions['v_s'], dtype=float)         # (N,3)
    omega_b = np.asarray(motions['omega_b'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    # Shear displacement increment per time step
    du_s = v_s * dt[:,None]

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
            omega = omega_b[i]*dt[i]
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
            if Fs_mag > mu * Fn_mag[i]:
                Fs_tmp *= (mu * Fn_mag[i] / Fs_mag)
            
            # Save elastic component for next time step
            Fs_old = Fs_tmp.copy()

            # Add viscous component
            Fs_tmp -= eta_s * v_s

            # Save total shear force
            Fs[i] = Fs_tmp.copy() # copy to avoid aliasing

    return Fs

#
#   ROLLING FORCE LAWS
#

def Fr_spring_dashpot_Coloumb(contact_params, motions, Fn):
    """
    Linear spring-dashpot in parallel capped by Coulomb limit
    Fr = - k_r * u_r * n_r - eta_r * v_r, with |Fr| <= mu*|Fr|
    """
    k_t     = contact_params['k_r']                         # (1)
    eta_s   = contact_params['eta_r']                       # (1)
    mu      = contact_params['mu']                          # (1)
    u_n     = np.array(motions['u_n'], dtype=float)         # (N,1)
    v_r     = np.array(motions['v_r'], dtype=float)         # (N,3)
    omega_b = np.asarray(motions['omega_b'], dtype=float)   # (N,3)
    dt      = np.array(motions['dt'], dtype=float)          # (1)

    # Shear displacement increment per time step
    du_r = v_r * dt[:,None]


    return Fr


#
#   TWISTING FORCE LAWS
#

def Tt_spring_dashpot_Coloumb(contact_params, motions, Fn):

    return Tt

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
    R_star = (R_i * R_j) / (R_i + R_j)

    # Effective mass
    if (m_i is not None) and (m_j is not None):
        m_star = (m_i * m_j) / (m_i + m_j)
    else:
        m_star = 1

    return E_star, G_star, R_star, m_star

# End of file





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
    v_n     = motions['v_n']             # (N,)
    n_ij    = motions['n_ij']            # (N,3)

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
    v_n  = motions['v_n']
    n_ij = motions['n_ij']

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
