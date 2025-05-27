import numpy as np
from scipy.spatial.transform import Rotation



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



def my_simulate_contact(
    x_b, v_b, omega_b,
    A, w, phi, k, l0,
    n_r, n_s,
    omega_t_func, omega_r_func, omega_s_func,
    t_end, dt,
    R_i=None, R_j=None
):
    """
    Simulate spherical contact under combined twist, roll, and shear.

    Inputs
    ------
    x_b, v_b, omega_b : (3,) vectors
        Base position, velocity, and angular velocity.
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
    R_i, R_j : scalars, optional
        Radii for penetration depth u_n.

    Returns
    -------
    dict of arrays:
      t: (N,1)
      x_i,x_j, v_i,v_j, omega_i,omega_j : (N,3)
      u_n : (N,1), if R_i and R_j provided
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
    u_n = np.zeros((N,1)) if R_i is not None and R_j is not None else None
    u_t = np.zeros((N,3)) if R_i is not None and R_j is not None else None

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
        l_ij = mag * Rb.dot(n0)
        n_ij = Rb.dot(n0)

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
        nr_t = Rb.dot(n0)
        nr_r = Rb.dot(n_r)
        nr_s = Rb.dot(n_s)
        omega_i[idx] = (omega_b
                        + 0.5 * omegar_t * nr_t
                        + 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)
        omega_j[idx] = (omega_b
                        - 0.5 * omegar_t * nr_t
                        - 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)

        # Penetration depth
        if u_n is not None:
            u_n[idx] = R_i + R_j - abs(mag)

    # Package results
    results = {
        't': t.reshape(-1,1),
        'x_i': x_i, 'x_j': x_j,
        'v_i': v_i, 'v_j': v_j,
        'omega_i': omega_i, 'omega_j': omega_j,
        'n_ij': n_ij, 'v_ijn': v_ijn, 'l_ij': l_ij
    }
    if u_n is not None:
        results['u_n'] = u_n
        results['u_t'] = u_t
    return results



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



def my_integrate_shear_displacement(n, omega1, omega2, omega_b, R1, R2, dt):
    """
    Compute the instantaneous shear velocities and shear displacements
    for a batch of time steps, given a constant rigid-body angular velocity.

    Parameters
    ----------
    n : array_like, shape (N, 3)
        Sequence of already-normalised contact normal vectors at each time step.
    omega1 : array_like, shape (N, 3)
        Sequence of angular velocity vectors of particle 1.
    omega2 : array_like, shape (N, 3)
        Sequence of angular velocity vectors of particle 2.
    omega_b : array_like, shape (3,)
        Constant rigid-body angular velocity vector (projected along the normal).
    R1 : float
        Radius of particle 1 (constant).
    R2 : float
        Radius of particle 2 (constant).
    dt : float
        Time step size (constant).

    Returns
    -------
    v_slide : ndarray, shape (N, 3)
        Instantaneous shear (sliding) velocities at each time step.
    u_t : ndarray, shape (N, 3)
        Shear displacement vectors over each time step dt.
    """
    n = np.array(n, dtype=float)
    omega1 = np.array(omega1, dtype=float)
    omega2 = np.array(omega2, dtype=float)
    omega_b = np.asarray(omega_b, dtype=float)

    # Instantaneous shear velocity at each time
    v_slide = R1 * np.cross(omega1 - omega_b, n, axis=1) + \
              R2 * np.cross(omega2 - omega_b, n, axis=1)

    # Shear displacement per time step
    u_t = v_slide * dt

    return v_slide, u_t



# Example usage
if __name__ == "__main__":
     initial_quat = [0, 1, 0, 0]  # Identity quaternion
     angular_velocity = np.array([0.1, 0.2, 0.3])  # radians per second
     dt = 0.01  # timestep in seconds
     steps = 100  # simulate 1 second

     trajectory = my_integrate_rotation(initial_quat, angular_velocity, dt,
steps)

     # Print final orientation
     print("Final orientation (quaternion):", trajectory[-1])
