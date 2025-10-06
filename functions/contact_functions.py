# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import numpy as np

#
#   SIMULATE THE CONTACT INTERACTION
#

def my_simulate_contact(motions, contact_params, Fn_func, Ft_func):
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
            'q_i','q_j' : (N,4) orientation quaternions
            'omega_i','omega_j': (N,3) angular vel arrays
            'n_ij'   : (N,3) contact normals
            'v_ijn'  : (N,3) normal component of rel vel
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
    u_n = (R_i + R_j - l_mag).reshape(-1,1) # Surface-to-surface across entire contact
    u_n = np.maximum(u_n, 0.0)
    motions['u_n'] = u_n

    # Compute shear kinematics
    v_t, u_t = my_integrate_shear_displacement(contact_params, motions)
    motions['v_t'] = v_t
    motions['u_t'] = u_t

    # Compute forces via provided functions
    Fn = Fn_func(contact_params, motions)          # shape (N,3)
    Ft = Ft_func(contact_params, motions, Fn)      # shape (N,3)

    # Total force
    F_i = Fn + Ft
    F_j = -F_i

    # Compute torque: T_i = r_i * (n_ij_i × F_i)
    n_ij = motions['n_ij']
    # Determine lever arm r per contact
    r_i = R_i - u_n # should be 0.5*u_n to end up at contact point
    r_j = R_j - u_n
    cross_nF_i = np.cross(n_ij, F_i)
    cross_nF_j = np.cross(-n_ij, F_j)
    T_i = (r_i * cross_nF_i)
    T_j = (r_j * cross_nF_j)

    # Package results
    result = motions.copy()
    result.update({
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
            'omega_b'   : (3,) constant rigid-body angular velocity vector
            'dt'        : (1) float time step size
            'u_n'       : (N,1) the normal overlap distance

    Returns
    -------
    v_t : ndarray, shape (N, 3)
        Instantaneous shear velocities at each time step.
    u_t : ndarray, shape (N, 3)
        Shear displacement vectors over each time step dt.
    """
    n       = np.array(motions['n_ij'], dtype=float)
    omega_i = np.array(motions['omega_i'], dtype=float)
    omega_j = np.array(motions['omega_j'], dtype=float)
    omega_b = np.asarray(motions['omega_b'], dtype=float)
    dt      = np.array(motions['dt'], dtype=float)
    u_n     = np.asarray(motions['u_n'], dtype=float)
    R_i     = contact_params['R_i']
    R_j     = contact_params['R_j']

    # Instantaneous shear velocity at each time
    v_t = R_i * np.cross(omega_i - omega_b, n, axis=1) + \
          R_j * np.cross(omega_j - omega_b, n, axis=1)
    # Set to zero if no contact
    mask = (u_n.ravel() == 0.0) # This is ok because we set to 0.0 exactly
    v_t[mask, :] = 0.0
    
    # Shear displacement increment per time step
    du_t = v_t * dt[:,None]

    # Accumulated shear displacement
    N, dim = du_t.shape
    u_t = np.zeros((N,dim))
    acc = np.zeros(3)
    for i in range(N):
        # Displacement is lost if contact is lost
        if mask[i]:
            acc[:] = 0.0
            u_t[i] = 0.0
        else:
            acc += du_t[i]
            u_t[i] = acc.copy()   # copy to avoid aliasing

    return v_t, u_t



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