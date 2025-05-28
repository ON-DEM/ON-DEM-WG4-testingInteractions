import numpy as np
from scipy.spatial.transform import Rotation

#
#   SIMULATE MOTION OF TWO PARTICLES
#

def my_simulate_motion(
    x_b, v_b, omega_b,
    init_quat_i, init_quat_j,
    A, w, phi, k, l0,
    n_r, n_s,
    omega_t_func, omega_r_func, omega_s_func,
    t_end, dt
):
    """
    Simulate relative motions under combined twist, roll, and shear.

    Inputs
    ------
    x_b, v_b, omega_b : (3,) vectors
        Base position, velocity, and angular velocity.
    init_quat_i, init_quat_j : (4,) vectors
        Initial orientations of particles i and j given by quaternion
        with format [x, y, z, w]
    A, w, phi, k : scalars
        Loading amplitude, frequency, phase, and damping.
    l0 : (3,) vector
        Initial branch vector.
    n_r, n_s : (3,) vectors
        Roll and shear direction unit vectors (orthogonal to initial branch).
    omega_t_func, omega_r_func, omega_s_func : callables t->scalar
        Time-dependent twist, roll, and shear magnitudes.
    t_end, dt : scalars
        Simulation end time and time step.

    Returns
    -------
    dict of arrays:
      t: (N,1)
      x_i,x_j: (N,3)
      v_i,v_j: (N,3)
      quat_i,quat_j: (N,3)
      omega_i,omega_j: (N,3)
      n_ij,v_ij,l_ij: (N,3)
    """
    # Convert inputs to arrays
    x_b = np.asarray(x_b, float)
    v_b = np.asarray(v_b, float)
    omega_b = np.asarray(omega_b, float)
    l0 = np.asarray(l0, float)
    n_r = np.asarray(n_r, float)
    n_s = np.asarray(n_s, float)

    # Scalar validations
    if dt <= 0 or t_end <= 0:
        raise ValueError("Scalars dt and t_end must be positive.")
    if not np.isscalar(A) or not np.isscalar(w) or not np.isscalar(phi) or not np.isscalar(k):
        raise ValueError("Parameters A, w, phi, and k must be scalars.")
    # Vector validations
    if x_b.shape != (3,) or v_b.shape != (3,) or omega_b.shape != (3,):
        raise ValueError("Inputs x_b, v_b, omega_b must be 3-vectors.")
    if len(init_quat_i) != 4 or len(init_quat_j) != 4:
        raise ValueError("Inputs init_quat_i and init_quat_j must be 4-vectors.")
    if l0.shape != (3,):
        raise ValueError("Initial branch vector l0 must be a 3-vector.")
    if n_r.shape != (3,) or n_s.shape != (3,):
        raise ValueError("Vectors n_r and n_s must be 3-vectors.")

    # Initial branch direction
    norm_l0 = np.linalg.norm(l0)
    if np.isclose(norm_l0, 0):
        raise ValueError("Initial branch vector must be non-zero.")
    n0 = l0 / norm_l0

    # Orthogonality checks
    if not np.isclose(np.dot(n_r, n0), 0):
        raise ValueError("Vector n_r must be orthogonal to initial branch vector.")
    if not np.isclose(np.dot(n_s, n0), 0):
        raise ValueError("Vector n_s must be orthogonal to initial branch vector.")

    # Time array
    t = np.arange(0, t_end + dt/2, dt)
    N = t.size

    # Preallocate time-series arrays
    x_i = np.zeros((N,3)); x_j = np.zeros((N,3))
    v_i = np.zeros((N,3)); v_j = np.zeros((N,3))
    omega_i = np.zeros((N,3)); omega_j = np.zeros((N,3))

    # Precompute constants
    denom = w**2 + k**2
    zero_k = np.isclose(k, 0)

    for idx, ti in enumerate(t):
        # Body rotation
        Rb = my_rotation_matrix(omega_b, ti)

        # Compute relative normal velocity
        v_ijn = - A * np.sin(w * ti + phi) * np.exp(k * ti) * Rb.dot(n0)

        # Compute branch magnitude
        if zero_k:
            mag = norm_l0 + (A/w) * (np.cos(w * ti + phi) - 1)
        else:
            mag = (norm_l0
                   + (A * w) / denom
                   - (A / denom) 
                   * (w * np.cos(w * ti + phi) - k * np.sin(w * ti + phi))
                   * np.exp(k * ti)
                  )
        # Branch vector and normal
        n_ij = Rb.dot(n0)
        l_ij = mag * n_ij
        
        # Positions
        x_i[idx] = Rb.dot(x_b) + v_b * ti
        x_j[idx] = x_i[idx] + l_ij

        # Velocities
        v_i[idx] = v_b + np.cross(omega_b, x_i[idx])
        v_j[idx] = v_i[idx] + np.cross(omega_b, l_ij) + v_ijn

        # Angular velocities
        omegar_t = omega_t_func(ti)
        omegar_r = omega_r_func(ti)
        omegar_s = omega_s_func(ti)
        # Rotated direction vectors
        nr_r = Rb.dot(n_r)
        nr_s = Rb.dot(n_s)
        omega_i[idx] = (omega_b
                        + 0.5 * omegar_t * n_ij
                        + 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)
        omega_j[idx] = (omega_b
                        - 0.5 * omegar_t * n_ij
                        - 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)

    quat_i =  my_integrate_rotation(init_quat_i, omega_i, dt)
    quat_j =  my_integrate_rotation(init_quat_j, omega_j, dt)

    # Package results
    motions = {
        't': t.reshape(-1,1),
        'x_i': x_i, 'x_j': x_j,
        'v_i': v_i, 'v_j': v_j,
        'quat_i': quat_i, 'quat_j': quat_j,
        'omega_i': omega_i, 'omega_j': omega_j,
        'n_ij': n_ij, 'v_ijn': v_ijn, 'l_ij': l_ij
    }
    return motions



def my_rotation_matrix(omega, t):
    """
    Compute the rotation matrix exp(skew(omega) * t) via Rodrigues' formula.

    omega : array_like, shape (3,), constant angular velocity vector
    t     : float, time
    """
    omega = np.asarray(omega, float)
    theta = np.linalg.norm(omega) * t
    if np.isclose(theta, 0):
        return np.eye(3)
    axis = omega / np.linalg.norm(omega)
    # The skew-symmetric matrix of the axis
    K = np.array([[       0, -axis[2],  axis[1]],
                  [ axis[2],        0, -axis[0]],
                  [-axis[1],  axis[0],       0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)



def my_integrate_rotation(initial_quat, omega, dt):
    """
    Integrate quaternion orientation over time given angular velocities (batch mode).
    
    Parameters:
    - initial_quat: array-like, shape (4,)
        Initial orientation quaternion [x, y, z, w].
    - omega: ndarray, shape (N, 3)
        Time series of angular velocity vectors in rad/s for each timestep.
    - dt: float
        Timestep duration in seconds.
    - nsteps: int, optional
        Number of steps to simulate. If None, inferred from omega.shape[0].
    
    Returns:
    - quaternions: ndarray, shape (M, 4)
        Quaternion orientations [x, y, z, w] at each time including initial; 
        M = nsteps + 1.
    """
    # Ensure numpy arrays
    omega = np.asarray(omega)
    # Determine number of steps
    nsteps = omega.shape[0]

    # Initialize orientation and storage
    orientation = Rotation.from_quat(initial_quat)
    quats = np.empty((nsteps, 4))
    quats[0] = orientation.as_quat()
    
    # Loop over each timestep in batch
    for i in range(nsteps-1):
        # Compute incremental rotation from angular velocity at step i
        theta_vec = omega[i] * dt  # Rotation vector (axis * angle)
        delta_rot = Rotation.from_rotvec(theta_vec)
        # Update orientation by quaternion multiplication
        orientation = orientation * delta_rot
        # Store new quaternion
        quats[i + 1] = orientation.as_quat()
    
    return quats

# End of file