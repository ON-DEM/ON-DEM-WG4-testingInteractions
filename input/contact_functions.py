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
            'R_i','R_j'  : float - reference length to surface,
                particle radii in the case of spheres.
    motions : dict of ndarrays
        Input motion data with keys:
            't'      : (N,1) time array
            'x_i','x_j' : (N,3) position arrays
            'v_i','v_j' : (N,3) velocity arrays
            'quat_i','quat_j' : (N,4) orientation quaternions
            'omega_i','omega_j': (N,3) angular vel arrays
            'n_ij'   : (N,3) contact normals
            'v_ijn'  : (N,1) normal component of rel vel
            'l_ij'   : (N,3) center-center branch vector
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
    R_i  = contact_params['R_i']
    R_j  = contact_params['R_j']

    # Compute normal overlap
    l_ij = motions['l_ij']           # (N,3) center distance vectors
    l_mag = np.linalg.norm(l_ij, axis=1)
    motions['u_n'] = (R1 + R2 - l_mag).reshape(-1,1)

    # Compute shear kinematics
    v_slide, u_t = my_integrate_shear_displacement(contact_params, motions)
    motions['v_slide'] = v_slide
    motions['u_t']     = u_t

    # Compute forces via provided functions
    Fn = Fn_func(contact_params, motions)          # shape (N,3)
    Ft = Ft_func(contact_params, motions, Fn)      # shape (N,3)

    # Total force
    F_i = Fn + Ft
    F_j = - F_i

    # Compute torque: T_i = r_i * (n_ij_i × F_i)
    n_ij = motions['n_ij']
    # Determine lever arm r per contact
    u_n = motions['u_n'].reshape(-1)
    r_i = R_i - u_n
    r_j = R_j - u_n
    cross_nF_i = np.cross(n_ij, F_i)
    cross_nF_j = np.cross(-n_ij, F_j)
    T_i = (r_i.reshape(-1,1) * cross_nF_i)
    T_j = (r_j.reshape(-1,1) * cross_nF_j)

    # Package results
    result = motions.copy()
    result.update({
        'Fn': Fn,
        'Ft': Ft,
        'F_i': F_i,
        'F_j': F_j,
        'T_i': T_i,
        'T_j': T_j
    })
    return result



def my_integrate_shear_displacement(contact_params, motions):
    """
    Compute the instantaneous shear velocities and shear displacements
    for a batch of time steps, given a constant rigid-body angular velocity.

    Parameters
    ----------
    contact_params : dict
        Dictionary of contact parameters. Expected keys:
            'R_i' : float - radius of particle i
            'R_j' : float - radius of particle j
    motions : dict of ndarrays
        Input motion data with keys:
            'n_ij'      : (N,3) contact normal vectors at each time step
            'omega_i'   : (N,3) angular velocity vectors of particle i
            'omega_j'   : (N,3) angular velocity vectors of particle j
            'omega_b'   : (3,)  constant rigid-body angular velocity vector
            'dt'        : float  time step size

    Returns
    -------
    v_slide : ndarray, shape (N, 3)
        Instantaneous shear (sliding) velocities at each time step.
    u_t : ndarray, shape (N, 3)
        Shear displacement vectors over each time step dt.
    """
    n      = np.array(motions['n_ij'],    dtype=float)
    omega_i = np.array(motions['omega_i'], dtype=float)
    omega_j = np.array(motions['omega_j'], dtype=float)
    omega_b = np.asarray(motions['omega_b'], dtype=float)
    dt      = motions['dt']
    R_i      = contact_params['R_i']
    R_j      = contact_params['R_j']

    # Instantaneous shear velocity at each time
    v_slide = R_i * np.cross(omega_i - omega_b, n, axis=1) + \
              R_j * np.cross(omega_j - omega_b, n, axis=1)

    # Shear displacement per time step
    u_t = v_slide * dt

    return v_slide, u_t



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

# End of file