# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import numpy as np
from scipy.spatial.transform import Rotation

#
#   SIMULATE MOTION OF TWO PARTICLES
#

def my_simulate_motion(
    x_b, v_b, omega_b,
    init_q_i, init_q_j,
    A, B, w, phi, k, l0,
    A_t, B_t, w_t, phi_t, k_t,
    A_r, B_r, w_r, phi_r, k_r,
    A_s, B_s, w_s, phi_s, k_s,
    n_r, n_s,
    t_end, dt
):
    """
    Simulate relative motions under combined twist, roll, and shear.

    Inputs
    ------
    x_b, v_b, omega_b : (3,) vectors
        Base position, velocity, and angular velocity.
    init_q_i, init_q_j : (4,) vectors
        Initial orientations of particles i and j given by quaternion
        with format [x, y, z, w]
    A, B, w, phi, k : scalars
        Parameters of the loading velocity function, in order:
        Constant offset, amplitude, frequency, phase, and damping.
        The _t, _r, and _s indicate the twist, roll, and shear 
        angular velocity functions.
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
      q_i,q_j: (N,3)
      omega_i,omega_j: (N,3)
      n_ij,v_ij,l_ij: (N,3)
      omega_b: (3,)
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
    if not np.isscalar(A) or not np.isscalar(B) or not np.isscalar(w) or not np.isscalar(phi) or not np.isscalar(k):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_t) or not np.isscalar(B_t) or not np.isscalar(w_t) or not np.isscalar(phi_t) or not np.isscalar(k_t):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_r) or not np.isscalar(B_r) or not np.isscalar(w_r) or not np.isscalar(phi_r) or not np.isscalar(k_r):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_s) or not np.isscalar(B_s) or not np.isscalar(w_s) or not np.isscalar(phi_s) or not np.isscalar(k_s):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    # Vector validations
    if x_b.shape != (3,) or v_b.shape != (3,) or omega_b.shape != (3,):
        raise ValueError("Inputs x_b, v_b, omega_b must be 3-vectors.")
    if len(init_q_i) != 4 or len(init_q_j) != 4:
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
    n_ij = np.zeros((N,3)); v_ijn = np.zeros((N,3))
    l_ij = np.zeros((N,3))

    # Precompute constants
    denom = w**2 + k**2
    zero_k = np.isclose(k, 0)

    for idx, ti in enumerate(t):
        # Body rotation, this works because omega_b is constant.
        Rb = Rotation.from_rotvec(omega_b * ti)

        # Contact normal
        n_ij[idx] = Rb.apply(n0)

        # Compute relative normal velocity
        v_ijn[idx] = A - B * np.sin(w * ti + phi) * np.exp(k * ti) * n_ij[idx]

        # Compute branch magnitude
        if zero_k:
            mag = norm_l0 + A*ti + (B/w) * (np.cos(w * ti + phi) - 1)
        else:
            mag = (norm_l0
                   + A * ti
                   + (B * w) / denom
                   - (B / denom) 
                   * (w * np.cos(w * ti + phi) - k * np.sin(w * ti + phi))
                   * np.exp(k * ti)
                  )
        # Branch vector
        l_ij[idx] = mag * n_ij[idx]
        
        # Positions
        x_i[idx] = Rb.apply(x_b) + v_b * ti
        x_j[idx] = x_i[idx] + l_ij[idx]

        # Velocities
        v_i[idx] = v_b + np.cross(omega_b, x_i[idx])
        v_j[idx] = v_i[idx] + np.cross(omega_b, l_ij[idx]) + v_ijn[idx]

        # Angular velocities
        omegar_t = A_t - B_t * np.sin(w_t * ti + phi_t) * np.exp(k_t * ti)
        omegar_r = A_r - B_r * np.sin(w_r * ti + phi_r) * np.exp(k_r * ti)
        omegar_s = A_s - B_s * np.sin(w_s * ti + phi_s) * np.exp(k_s * ti)
        # Rotated direction vectors
        nr_r = Rb.apply(n_r)
        nr_s = Rb.apply(n_s)
        omega_i[idx] = (omega_b
                        + 0.5 * omegar_t * n_ij[idx]
                        + 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)
        omega_j[idx] = (omega_b
                        - 0.5 * omegar_t * n_ij[idx]
                        - 0.5 * omegar_r * nr_r
                        + 0.5 * omegar_s * nr_s)

    q_i =  my_integrate_rotation(init_q_i, omega_i, dt)
    q_j =  my_integrate_rotation(init_q_j, omega_j, dt)

    # Package results
    motions = {
        't': t.reshape(-1,1),'dt':dt,
        'x_i': x_i, 'x_j': x_j,
        'v_i': v_i, 'v_j': v_j,
        'quat_i': q_i, 'quat_j': q_j,
        'omega_i': omega_i, 'omega_j': omega_j,
        'n_ij': n_ij, 'v_ijn': v_ijn, 'l_ij': l_ij,
        'omega_b': omega_b
    }
    return motions



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

def writeDemInput(results,filename='dem_input.txt'):
    """
    Write the DEM inputs to a file. The input is a dictionnary produced by my_simulate_motion.
    The file will contain the time series of translational and angular velocities
    """
 
    demInputs = {k: results[k] for k in ['t', 'v_i', 'v_j', 'omega_i', 'omega_j']}

    def flatten_for_csv(arr):
        arr = np.asarray(arr)
        if arr.ndim > 1:
            return arr.reshape(arr.shape[0], -1)
        return arr

    # Prepare data for writing
    data = [flatten_for_csv(demInputs[k]) for k in demInputs]
    rows = np.hstack(data)


    file = open('test_results.txt', 'w')
    file.write("# initial position/orientation as X1,R1,X2,R2,Q1,Q2 (vector/quaternion)\n"
            "# init: 0 0 0 1 2 0 0 1 0 0 1 0 0 0 1 0\n"
            "# # Times series of translational and angular velocities)\n")
    import csv
    writer = csv.writer(file, delimiter=' ')
    file.write("# ")
    writer.writerow(demInputs.keys())
    writer.writerows(rows)
    file.close()

# End of file