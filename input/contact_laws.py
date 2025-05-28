import numpy as np

#
#   NORMAL FORCE LAWS
#

def Fn_linear_elastic(contact_params, motions):
    """F_n = - k_n u_n"""
    kn = contact_params['k_n']
    u_n = motions['u_n'].reshape(-1)
    n_ij = motions['n_ij']
    return kn * u_n[:, None] * n_ij



def Fn_hertzian(contact_params, motions):
    """
    Hertzian normal force: F_n = k_n * u_n^(3/2) * n_ij
    k_n = (4/3) * E* * sqrt(R*)
    """
    u_n = motions['u_n'].reshape(-1)   # (N,)
    n_ij = motions['n_ij']             # (N,3)

    E_star, _, R_star = my_compute_effective_params(contact_params)

    k_n = (4.0 / 3.0) * E_star * np.sqrt(R_star)
    mag = k_n * u_n**1.5
    return mag[:, None] * n_ij



#
#   TANGENTIAL FORCE LAWS
#

def Ft_linear_shear(contact_params, motions, Fn):
    """
    Linear shear stiffness: F_t = -k_s0 * u_t, capped by Coulomb
    where k_s0 = 8 * G* * sqrt(R*)
    """
    u_t = motions['u_t']
    mu = contact_params['mu']

    _, G_star, R_star = my_compute_effective_params(contact_params)
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
    u_n = motions['u_n'].reshape(-1)
    mu = contact_params['mu']

    _, G_star, R_star = my_compute_effective_params(contact_params)
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

    _, G_star, R_star = my_compute_effective_params(contact_params)
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

# End of file