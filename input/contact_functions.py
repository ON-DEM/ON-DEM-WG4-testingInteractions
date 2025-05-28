import numpy as np



def my_simulate_contact_force(motions, contact_params, Fn_func, Ft_func):
    """
    Generalised contact force batch processor.

    Parameters
    ----------
    contact_params : dict
        Dictionary of contact parameters. Expected keys:
            'kn' : float - normal stiffness
            'kt' : float - tangential stiffness
            'mu' : float - friction coefficient
            'R'  : float - reference length (e.g., particle radius or sum of radii)
    motions : dict of ndarrays
        Input motion data with keys:
            't'      : (N,1) time array
            'x_i','x_j' : (N,3) position arrays
            'v_i','v_j' : (N,3) velocity arrays
            'omega_i','omega_j': (N,3) angular vel arrays
            'n_ij'   : (N,3) contact normals
            'v_ijn'  : (N,1) normal component of rel vel
            'l_ij'   : (N,1) center-center distance
            'u_n' : (N,1) normal displacement or penetration distance
            'u_t' : (N,3) tangential displacement
    Fn_func : callable
        Function to compute normal force:
            Fn = Fn_func(contact_params, motions)
        Returns ndarray shape (N,3).
    Ft_func : callable
        Function to compute tangential force:
            Ft = Ft_func(contact_params, motions, Fn)
        Returns ndarray shape (N,3).

    Returns
    -------
    result : dict
        Same as motions, with additional keys:
            'Fn' : (N,3) normal force
            'Ft' : (N,3) tangential force
            'F'  : (N,3) total force = Fn + Ft
            'T'  : (N,3) torque vector
    """
    # Extract contact parameters
    R  = contact_params['R']

    # Compute forces via provided functions
    Fn = Fn_func(contact_params, motions)          # shape (N,3)
    Ft = Ft_func(contact_params, motions, Fn)      # shape (N,3)

    # Total force
    F = Fn + Ft

    # Compute torque: T_i = r_i * (n_ij_i × F_i)
    n_ij = motions['n_ij']
    N = n_ij.shape[0]
    # Determine lever arm r per contact
    if 'u_n' in motions:
        u_n = motions['u_n'].reshape(-1)
        r = R - u_n
    else:
        r = np.full(N, R)
    cross_nF = np.cross(n_ij, F)
    T = (r.reshape(-1,1) * cross_nF)

    # Package results
    result = motions.copy()
    result.update({
        'Fn': Fn,
        'Ft': Ft,
        'F': F,
        'T': T
    })
    return result



def my_compute_effective_params(contact_params):
    """
    Compute effective contact parameters for two particles from contact_params dict:
      - E* effective normal modulus
      - G* effective shear modulus
      - R* effective radius

    Expects keys: 'E1','nu1','E2','nu2','R1','R2' (optional 'G1','G2').
    """
    E1, nu1 = contact_params['E1'], contact_params['nu1']
    E2, nu2 = contact_params['E2'], contact_params['nu2']
    R1, R2 = contact_params['R1'], contact_params['R2']
    G1 = contact_params.get('G1', None)
    G2 = contact_params.get('G2', None)

    # Effective normal modulus
    inv_E_star = (1 - nu1**2) / E1 + (1 - nu2**2) / E2
    E_star = 1.0 / inv_E_star

    # Determine shear moduli
    if G1 is None:
        G1 = E1 / (2.0 * (1.0 + nu1))
    if G2 is None:
        G2 = E2 / (2.0 * (1.0 + nu2))

    # Effective shear modulus
    inv_G_star = (2.0 - nu1) / G1 + (2.0 - nu2) / G2
    G_star = 1.0 / inv_G_star

    # Effective radius
    R_star = (R1 * R2) / (R1 + R2)

    return E_star, G_star, R_star



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
    ut = motions['u_t']
    mu = contact_params['mu']

    _, G_star, R_star = my_compute_effective_params(contact_params)
    k_s0 = 8.0 * G_star * np.sqrt(R_star)

    Ft = -k_s0 * ut
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
    ut = motions['u_t']
    u_n = motions['u_n'].reshape(-1)
    mu = contact_params['mu']

    _, G_star, R_star = my_compute_effective_params(contact_params)
    k_s0 = 8.0 * G_star * np.sqrt(R_star)

    k_s = k_s0 * np.sqrt(u_n)
    Ft = - (k_s[:, None] * ut)

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
    ut = motions['u_t']
    u_n = motions['u_n'].reshape(-1)
    mu = contact_params['mu']

    _, G_star, R_star = my_compute_effective_params(contact_params)
    a = np.sqrt(R_star * u_n)

    ut_mag = np.linalg.norm(ut, axis=1)
    ratio = np.clip(1 - ut_mag / a, 0.0, None)
    c = a * ratio**(1.0/3.0)

    term = (a - c) / (3.0 * a**2)
    diff = ut - (term[:, None] * (ut_mag**2)[:, None] * ut / ut_mag[:, None])
    Ft = -8.0 * G_star * a[:, None] * diff

    # Coulomb limit
    Fn_mag = np.linalg.norm(Fn, axis=1)
    Ft_mag = np.linalg.norm(Ft, axis=1)
    slip = Ft_mag > mu * Fn_mag
    Ft[slip] *= (mu * Fn_mag[slip] / Ft_mag[slip])[:, None]
    return Ft






# Example usage with dummy data
if __name__ == "__main__":
    N = 5
    kn, kt, mu, R = 1e5, 2e4, 0.5, 0.01
    un = np.random.rand(N, 1) * 1e-4
    ut = np.random.rand(N, 3) * 1e-4
    n  = np.random.rand(N, 3)
    n  = n / np.linalg.norm(n, axis=1)[:, None]

    # Needs to be adjusted
    results = my_simulate_contact_force(kn, kt, mu, R, un, ut, n)
    for key, val in results.items():
        print(f"{key}:\n{val}\n")

# Outdated
def my_cundall_strack(kn, kt, mu, R, un, ut, n):
    """
    Apply Cundall-Strack contact law over time.

    Parameters
    ----------
    kn : float
        Normal stiffness.
    kt : float
        Tangential stiffness.
    mu : float
        Friction coefficient.
    R : float
        Particle radius.
    un : ndarray of shape (N, 1) or (N,)
        Normal displacements over time (column vector).
    ut : ndarray of shape (N, 3)
        Tangential displacements over time.
    n : ndarray of shape (N, 3)
        Contact normals over time (unit vectors).

    Returns
    -------
    result : dict with keys
        "F"  : ndarray (N, 3) - total contact force over time.
        "T"  : ndarray (N, 3) - torque over time.
        "Fn" : ndarray (N, 3) - normal force over time.
        "Ft" : ndarray (N, 3) - tangential force over time.
    """
    # Ensure inputs are arrays
    un = np.asarray(un)
    ut = np.asarray(ut)
    n  = np.asarray(n)

    # Flatten un if it's Nx1 to a 1D array of length N
    if un.ndim == 2 and un.shape[1] == 1:
        un_flat = un[:, 0]
    else:
        un_flat = un

    N = un_flat.shape[0]

    # Preallocate output arrays
    F  = np.zeros((N, 3))
    T  = np.zeros((N, 3))
    Fn = np.zeros((N, 3))
    Ft = np.zeros((N, 3))

    # Loop over each time step (each contact state)
    for i in range(N):
        # Normal displacement at time i
        un_i = un_flat[i]

        # Tangential displacement and normal at time i
        ut_i = ut[i]    # shape (3,)
        n_i  = n[i]     # shape (3,)

        # Normal force: linear elastic model
        Fn_i = kn * un_i * n_i

        # Tangential force: spring model
        Ft_i = kt * ut_i

        # Magnitudes for friction check
        Fn_mag = np.linalg.norm(Fn_i)
        Ft_mag = np.linalg.norm(Ft_i)

        # Mohr-Coulomb friction limit
        if Ft_mag > mu * Fn_mag and Ft_mag > 0:
            Ft_i = mu * Fn_mag / Ft_mag * Ft_i

        # Total force
        F_i = Fn_i + Ft_i

        # Torque: lever arm r = R - un
        r = R - un_i
        T_i = r * np.cross(n_i, F_i)

        # Store results
        Fn[i] = Fn_i
        Ft[i] = Ft_i
        F[i]  = F_i
        T[i]  = T_i

    # Package into result dictionary
    result = {
        "F": F,
        "T": T,
        "Fn": Fn,
        "Ft": Ft
    }
    return result

