# Danny van der Haven, dannyvdhaven@gmail.com
# Outdated.

import numpy as np
from scipy.spatial.transform import Rotation as R



def my_input_velocity_i(t, V_i, A_i, w_i, k_i):
    t = np.asarray(t)
    return V_i + A_i * np.sin(w_i * t) * np.exp(k_i * t)



def my_input_position_i(t, x_i0, V_i, A_i, w_i, k_i=None):
    t = np.asarray(t)
    if k_i is not None: # if not zero
        denom = w_i**2 + k_i**2
        exp_term = np.exp(k_i * t)
        trig_term = w_i * np.cos(w_i * t) - k_i * np.sin(w_i * t)
        full_term = A_i * exp_term * trig_term / denom
        return x_i0 + V_i * t + full_term - (A_i * w_i / denom)
    else:
        return x_i0 + V_i * t - (A_i / w_i) * (np.cos(w_i * t) - 1)



def my_get_contact_pen(pos_i, pos_j, vel_i, vel_j, R_i=None, R_j=None):
    """
    Computes relative displacement, relative velocity, contact normal, 
    and decomposes them into normal and tangential contributions for 
    arrays of particle positions and velocities. These definitions are
    independent of rigid body motions (when we apply the same translation
    to particles i and j), but a rigid body rotation will rotate the
    resulting vector quantities.

    pos_i, pos_j: 3xN arrays of positions
    vel_i, vel_j: 3xN arrays of velocities
    R_i, R_j: scalars or arrays of length N (optional particle radii)
    Assumes i->j for the relative quantities.

    Returns a dictionary with keys:
        delta_pos, delta_vel,
        n_ij, delta_pos_n, delta_pos_t,
        delta_vel_n, delta_vel_t,
        penetration (optional)
    """
    #pos_i, pos_j = np.asarray(pos_i), np.asarray(pos_j)
    #vel_i, vel_j = np.asarray(vel_i), np.asarray(vel_j)

    delta_pos = pos_j - pos_i  # From i to j
    delta_vel = vel_j - vel_i

    # Norm of displacement
    norm_delta_pos = np.linalg.norm(delta_pos, axis=1, keepdims=True)
    # Avoid division by zero
    norm_delta_pos = np.where(norm_delta_pos > 1e-12, norm_delta_pos, 1e-12)

    # Contact normal, unit vector i→j
    n_ij = delta_pos / norm_delta_pos

    # Project displacements and velocities onto the normal
    dot_pos = np.sum(delta_pos * n_ij, axis=1, keepdims=True)
    dot_vel = np.sum(delta_vel * n_ij, axis=1, keepdims=True)

    delta_pos_n = dot_pos * n_ij 
    delta_pos_t = delta_pos - delta_pos_n

    delta_vel_n = dot_vel * n_ij 
    delta_vel_t = delta_vel - delta_vel_n

    result = {
        "delta_pos": delta_pos,
        "delta_vel": delta_vel,
        "n_ij": n_ij,
        "delta_pos_n": delta_pos_n,
        "delta_pos_t": delta_pos_t,
        "delta_vel_n": delta_vel_n,
        "delta_vel_t": delta_vel_t
    }

    if R_i is not None and R_j is not None: # zeros
        penetration = (R_i + R_j) - norm_delta_pos
        result["penetration"] = penetration

    return result



def my_get_contact_rot(omega_i, omega_j, n):
    """Computes twist, roll, and pure shear displacements 
    of a contact interaction. The definitions are naturally 
    independent of the rigid body rotation of particles i 
    and j together.

    Parameters:
    omega_i : ndarray (3,) - angular velocity of particle i
    omega_j : ndarray (3,) - angular velocity of particle j
    n       : ndarray (3,) - contact normal (unit vector)

    Returns:
    twist, roll, shear: ndarrays (3,) each
    """
    # The relative angular velocity
    delta_omega = omega_i - omega_j

    # Compute the twisting or torsion angle.
    twist = np.sum(delta_omega * n, axis=1, keepdims=True) * n
    # Equal to dot(delta_omega, n) * n but per row. 
    # Alternatively, twist = (I - n ⊗ n) · delta_omega

    # Compute the rolling or bending angle.
    roll = delta_omega - twist

    # Rotation of the contact normal
    omega_mean = 0.5 * (omega_i + omega_j)
    omega_n = omega_mean - np.sum(omega_mean * n, axis=1, keepdims=True) * n
    # Same as (I - n⊗n) @ omega_mean

    # Compute the pure shear displacement.
    # Single vector equivalent for clarity:
    #   P = np.eye(3) - np.outer(n, n)  # Projection matrix
    #   shear = P @ (omega_i + omega_j - 2 * omega_n) # Matrix multiplication
    # Below is batched calculation.

    # Batch outer product: n ⊗ n → shape (N, 3, 3)
    n_outer = n[:, :, np.newaxis] * n[:, np.newaxis, :]

    # Identity matrix for each entry: shape (N, 3, 3)
    I = np.eye(3)[np.newaxis, :, :]  # shape (1, 3, 3), broadcast over N

    # Batch projection matrices: P_i = I - n ⊗ n
    P = I - n_outer  # shape (N, 3, 3)

    # Angular terms: shape (N, 3)
    omega_sum = omega_i + omega_j - 2 * omega_n

    # Apply projection: shear_i = P_i @ omega_sum_i
    shear = np.einsum('nij,nj->ni', P, omega_sum)

    return twist, roll, shear



def my_vector_rot(v, q):
    """
    Rotates vector v using unit quaternion q (as [x, y, z, w]).
    """
    r = R.from_quat(q)
    return r.apply(v)
# For angular velocity, we simply have \omega_i = \omega_i + d\omega



def my_vector_rot_inv(v_rot, q):
    """
    Applies the inverse rotation of q to the rotated vector v_rot to get the original vector.
    """
    r = R.from_quat(q)
    return r.inv().apply(v_rot)



# Unnecessary stuff, I think
def omega_n_from_delta_n(delta_n, n_old):
    # Solves ω_n × n_old = delta_n => ω_n = Pseudo-inverse 
    # or projected guess. Assuming n_old is normalised.
    cross_mat = lambda v: np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])
    return np.linalg.pinv(cross_mat(n_old)) @ delta_n

def omega_n_from_n_old_new(n_old, n_new, dt=1.0):
    """Computes the angular velocity of the contact normal from old and new normals.
       Assumes small-angle rotation and that n_old and n_new are normalised."""
    delta_n = n_new - n_old
    omega_n = np.cross(n_old, delta_n) / dt
    return omega_n

def omega_n_from_n_old_new_geometric(n_old, n_new, dt=1.0):
    """Computes the contact normal angular velocity ωₙ from old and new contact normals.
       Uses geometric interpretation for arbitrary angle."""
    axis = np.cross(n_old, n_new)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(n_old, n_new)
    theta = np.arctan2(sin_theta, cos_theta)
    if sin_theta != 0:
        axis /= sin_theta  # normalise rotation axis
        omega_n = (theta / dt) * axis
    else:
        omega_n = np.zeros(3)
    return omega_n



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