import numpy as np

#
#   ANALYSE AND COMPARE DEM VS ANALYTICAL
#

def test_momentum_conservation(motions_list, masses, inertias, external_forces=None,
                                external_torques=None, r0=None, tol=1e-6):
    """
    Test linear and angular momentum conservation for a system of particles.

    Parameters
    ----------
    motions_list : list of dict
        Each dict must contain time series arrays (shape (N,3) or (N,1)):
            't'     : (N,1) time array (identical for all particles)
            'x'     : (N,3) positions
            'v'     : (N,3) linear velocities
            'omega' : (N,3) angular velocities
    masses : array-like of length M
        Mass of each of the M particles.
    inertias : array-like of length M or (M,3,3)
        Moment of inertia: scalar or 3x3 tensor per particle.
    external_forces : list of ndarray, optional
        External force time series for each particle, shape (M, N, 3).
    external_torques : list of ndarray, optional
        External torque time series for each particle, shape (M, N, 3).
    r0 : None or array-like of shape (3,) or (N,3)
        Reference point for angular momentum (origin or CoM). If None, use CoM at each time.
    tol : float
        Tolerance for conservation check.

    Returns
    -------
    result : dict
        'P'    : (N,3) total linear momentum
        'dPdt' : (N,3) time derivative of P
        'L'    : (N,3) total angular momentum
        'dLdt' : (N,3) time derivative of L
        'lin_conserved' : bool, |dPdt - F_ext_sum|<tol
        'ang_conserved' : bool, |dLdt - (torque_ext + r_rel×F_ext)|<tol
    """
    M = len(motions_list)
    # extract time
    t = motions_list[0]['t'].reshape(-1)
    N = t.size
    dt = t[1:] - t[:-1]
    
    # Initialize accumulators
    P = np.zeros((N, 3))
    L = np.zeros((N, 3))
    
    # Compute center of mass if needed
    xs = np.stack([m['x'] for m in motions_list], axis=0)  # (M,N,3)
    if r0 is None:
        # CoM at each time
        masses_arr = np.asarray(masses)[:, None, None]  # (M,1,1)
        com = (masses_arr * xs).sum(axis=0) / np.sum(masses)  # (N,3)
    else:
        com = np.broadcast_to(np.asarray(r0), (N, 3))
    
    # Loop over particles
    for i, m in enumerate(motions_list):
        mi = masses[i]
        xi = m['x']       # (N,3)
        vi = m['v']       # (N,3)
        wi = m['omega']   # (N,3)
        # linear momentum
        P += mi * vi
        # angular momentum: (r - r0) × m v + I ω
        r_rel = xi - com  # (N,3)
        L += np.cross(r_rel, mi * vi)
        # add rotational inertia term
        Ii = inertias[i]
        if np.ndim(Ii) == 0:
            L += Ii * wi
        else:
            # tensor product
            L += np.einsum('ij,nj->ni', Ii, wi)
    
    # time derivatives
    dPdt = np.vstack([ (P[j+1] - P[j])/(t[j+1]-t[j]) for j in range(N-1) ] + [ [0,0,0] ])
    dLdt = np.vstack([ (L[j+1] - L[j])/(t[j+1]-t[j]) for j in range(N-1) ] + [ [0,0,0] ])
    
    # external sums if given
    if external_forces is not None:
        Fext_sum = sum(external_forces)  # assume same shape (N,3)
    else:
        Fext_sum = np.zeros((N,3))
    if external_torques is not None:
        Text_sum = sum(external_torques)
    else:
        Text_sum = np.zeros((N,3))
    # also r_rel × Fext sum
    if external_forces is not None:
        torque_from_forces = np.zeros((N,3))
        for i, m in enumerate(motions_list):
            xi = m['x']
            r_rel = xi - com
            torque_from_forces += np.cross(r_rel, external_forces[i])
    else:
        torque_from_forces = np.zeros((N,3))
    
    # Check conservation
    lin_residual = dPdt - Fext_sum
    ang_residual = dLdt - (torque_from_forces + Text_sum)
    lin_conserved = np.allclose(lin_residual, 0, atol=tol)
    ang_conserved = np.allclose(ang_residual, 0, atol=tol)
    
    return {
        'P': P, 'dPdt': dPdt,
        'L': L, 'dLdt': dLdt,
        'lin_conserved': lin_conserved,
        'ang_conserved': ang_conserved
    }



def my_compare_results(sim_results, ana_results, tol=None):
    """
    Compare two 'results' dictionaries (simulation vs analytical) over time.

    Parameters
    ----------
    sim_results : dict
        Simulation output dict containing keys:
            't', 'x_i', 'x_j', 'v_i', 'v_j',
            'omega_i', 'omega_j', 'n_ij', 'v_ijn', 'l_ij',
            optionally 'u_n', 'u_t',
            'Fn', 'Ft', 'F', 'T',
            'q_i', 'q_j' (orientations as quaternions Nx4).
    ana_results : dict
        Analytical output dict with the same keys as sim_results.
    tol : dict, optional
        Per-key tolerances for comparison. Default:
            tol = {
                'position': 1e-6,
                'orientation': 1e-6,
                'force': 1e-6,
                'torque': 1e-6
            }

    Returns
    -------
    report : dict
        For each comparison, a boolean array of length N and an overall pass flag.
        Keys:
            'x_i', 'x_j', 'q_i', 'q_j', 'F', 'T'
            Each maps to dict with:
                'diff': ndarray (N,),
                'pass': ndarray (N,), 
                'all_pass': bool
    """
    if tol is None:
        tol = {'position':1e-6, 'orientation':1e-6, 'force':1e-6, 'torque':1e-6}

    def compare_array(key_sim, key_ana, tol_val):
        arr_sim = np.asarray(sim_results[key_sim])
        arr_ana = np.asarray(ana_results[key_ana])
        # compute norm of difference per time step
        diffs = np.linalg.norm(arr_sim - arr_ana, axis=1)
        passes = diffs <= tol_val
        return diffs, passes, bool(np.all(passes))

    report = {}

    # Positions
    report['x_i'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('x_i','x_i', tol['position'])
    ))
    report['x_j'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('x_j','x_j', tol['position'])
    ))

    # Orientations (quaternions)
    # Compute quaternion distance: norm of elementwise difference
    report['q_i'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('q_i','q_i', tol['orientation'])
    ))
    report['q_j'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('q_j','q_j', tol['orientation'])
    ))

    # Forces
    report['F'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('F','F', tol['force'])
    ))
    report['T'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('T','T', tol['torque'])
    ))

    return report

# End of file