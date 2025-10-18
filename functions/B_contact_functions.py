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

    # Compute forces via provided functions
    Fn = Fn_func(contact_params, motions)          # shape (N,3)
    Ft = Ft_func(contact_params, motions, Fn)      # shape (N,3)
    
    # Total force
    F_i = Fn + Ft
    F_j = -F_i

    # Get normal penetration
    u_n = motions['u_n']
    # Compute torque: T_i = r_i * (n_ij_i × F_i)
    n_ij = motions['n_ij']
    # Determine lever arm r per contact
    r_i = R_i - 0.5*u_n
    r_j = R_j - 0.5*u_n
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

# End of file