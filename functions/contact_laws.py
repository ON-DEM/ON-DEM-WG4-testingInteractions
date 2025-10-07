# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import numpy as np

#
#   NORMAL FORCE LAWS
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



#
#   TANGENTIAL FORCE LAWS
#

def Ft_linear_Coloumb(contact_params, motions, Fn):
    """
    Linear shear stiffness: F_t -= k_t * du_t, capped by Coulomb limit mu*Fn
    """
    du_t    = np.array(motions['du_t'], dtype=float)
    k_t     = contact_params['k_t']
    mu      = contact_params['mu']
    u_n    = np.array(motions['u_n'], dtype=float)
    omega_b = np.asarray(motions['omega_b'], dtype=float)
    dt      = np.array(motions['dt'], dtype=float)

    # Test for contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly

    # Normal force magnitudes
    Fn_mag = np.linalg.norm(Fn, axis=1)

    # Accumulate shear force
    N, dim = du_t.shape
    Ft = np.zeros((N,dim))
    Ft[0] = 0
    Ft_tmp = np.zeros(3)
    for i in range(N):
        # Displacement is lost if contact is lost
        if mask[i]:
            Ft[i] = 0.0
        else:
            if i > 0:
                Ft_tmp = Ft[i-1]
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
                Ft_tmp = R @ Ft_tmp
            # Integrate increment
            Ft_tmp -= k_t * du_t[i]
            # Apply Coulomb limit
            Ft_mag = np.linalg.norm(Ft_tmp)
            if Ft_mag > mu * Fn_mag[i]:
                Ft_tmp *= (mu * Fn_mag[i] / Ft_mag)
            Ft[i] = Ft_tmp.copy() # copy to avoid aliasing

    return Ft



def Ft_linear_shear(contact_params, motions, Fn):
    """
    Linear shear stiffness: F_t = -k_s0 * u_t, capped by Coulomb
    where k_s0 = 8 * G* * sqrt(R*)
    """
    u_t = motions['u_t']
    mu = contact_params['mu']

    _, G_star, R_star, _ = my_compute_effective_params(contact_params)
    k_s0 = 8.0 * G_star * np.sqrt(R_star)

    Ft = -k_s0 * u_t
    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Ft_mag = np.linalg.norm(Ft, axis=1)
    slip = Ft_mag > mu * Fn_mag
    Ft[slip] *= (mu * Fn_mag[slip] / Ft_mag[slip])[:, None]
    return Ft



def Ft_full_mindlin(contact_params, motions, Fn):
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
    Ft = - (k_s[:, None] * u_t)

    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Ft_mag = np.linalg.norm(Ft, axis=1)
    slip = Ft_mag > mu * Fn_mag
    Ft[slip] *= (mu * Fn_mag[slip] / Ft_mag[slip])[:, None]
    return Ft



def Ft_partial_slip(contact_params, motions, Fn):
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
    Ft = -8.0 * G_star * a[:, None] * diff

    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Ft_mag = np.linalg.norm(Ft, axis=1)
    slip = Ft_mag > mu * Fn_mag
    Ft[slip] *= (mu * Fn_mag[slip] / Ft_mag[slip])[:, None]
    return Ft



def Ft_viscous_const(contact_params, motions):
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
    Ft_visc = - eta_t * v_t
    return Ft_visc



def Ft_viscous_veldep(contact_params, motions):
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
    Ft_visc = - eta_t * v_t
    return Ft_visc

# End of file