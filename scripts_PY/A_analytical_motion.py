# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import numpy as np
from scipy.spatial.transform import Rotation

#
#   GENERATE MOTION OF TWO PARTICLES
#

def my_analytical_motion(
    x_f, v_f, omega_f,
    init_q_i, init_q_j,
    A, B, w, phi, k, l0,
    A_t, B_t, w_t, phi_t, k_t,
    A_r, B_r, w_r, phi_r, k_r,
    A_s, B_s, w_s, phi_s, k_s,
    n_r, n_s,
    t_end, dt,
    R_i, R_j
):
    """
    Simulate relative motions under combined twist, roll, and shear.

    Inputs
    ------
    x_f, v_f, omega_f : (3,) vectors
        Frame position, velocity, and angular velocity.
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
        Roll and shear direction unit vectors (must be orthogonal to initial branch).
    t_end, dt : scalars
        Simulation end time and time step.
    R_i, R_j : scalars
        Radii of the two particles (needed to proprely assing angular velocities).

    Returns
    -------
    dict of arrays:
      t: (N,1)
      x_i,x_j: (N,3)
      v_i,v_j: (N,3)
      v_i_half,v_j_half: (N,3)   exact half-step linear velocities v(t_k - dt/2)
      a_i,a_j: (N,3)
      q_i,q_j: (N,4)
      omega_i,omega_j: (N,3)
      omega_i_half,omega_j_half: (N,3)  exact half-step angular velocities omega(t_k - dt/2)
      n_ij,v_ijn,a_ijn,l_ij: (N,3)
      omega_f: (3,)
      v_s,v_r,omega_t,omega_b: (N,3)
      du_s,du_r,dtheta_t,dtheta_b: (N,3)
    """

    # Convert inputs to arrays
    x_f = np.asarray(x_f, float)
    v_f = np.asarray(v_f, float)
    omega_f = np.asarray(omega_f, float)
    l0 = np.asarray(l0, float)
    n_r = np.asarray(n_r, float)
    n_s = np.asarray(n_s, float)

    # Scalar validations
    if dt <= 0 or t_end <= 0:
        raise ValueError("Scalars dt and t_end must be positive.")
    if R_i <= 0 or R_j <= 0:
        raise ValueError("Scalars R_i and R_j must be positive.")
    if not np.isscalar(A) or not np.isscalar(B) or not np.isscalar(w) or not np.isscalar(phi) or not np.isscalar(k):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_t) or not np.isscalar(B_t) or not np.isscalar(w_t) or not np.isscalar(phi_t) or not np.isscalar(k_t):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_r) or not np.isscalar(B_r) or not np.isscalar(w_r) or not np.isscalar(phi_r) or not np.isscalar(k_r):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    if not np.isscalar(A_s) or not np.isscalar(B_s) or not np.isscalar(w_s) or not np.isscalar(phi_s) or not np.isscalar(k_s):
        raise ValueError("Parameters A, B, w, phi, and k must be scalars.")
    # Vector validations
    if x_f.shape != (3,) or v_f.shape != (3,) or omega_f.shape != (3,):
        raise ValueError("Inputs x_f, v_f, omega_f must be 3-vectors.")
    if len(init_q_i) != 4 or len(init_q_j) != 4:
        raise ValueError("Inputs init_q_i and init_q_j must be 4-vectors.")
    if l0.shape != (3,):
        raise ValueError("Initial branch vector l0 must be a 3-vector.")
    if n_r.shape != (3,) or n_s.shape != (3,):
        raise ValueError("Vectors n_r and n_s must be 3-vectors.")

    # Initial branch direction
    norm_l0 = np.linalg.norm(l0)
    if np.isclose(norm_l0, 0):
        raise ValueError("Initial branch vector must be non-zero.")
    # We assume spheres, so unit branch vector is contact normal
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
    a_i = np.zeros((N,3)); a_j = np.zeros((N,3))
    omega_i = np.zeros((N,3)); omega_j = np.zeros((N,3))
    n_ij = np.zeros((N,3)); v_ijn = np.zeros((N,3)); a_ijn = np.zeros((N,3))
    l_ij = np.zeros((N,3))
    u_n = np.zeros((N,1))
    v_r = np.zeros((N,3)); du_r = np.zeros((N,3))
    v_s = np.zeros((N,3)); du_s = np.zeros((N,3))
    omega_t = np.zeros((N,3)); dtheta_t = np.zeros((N,3))
    omega_b = np.zeros((N,3)); dtheta_b = np.zeros((N,3))
    dtheta_vec_i = np.zeros((N,3)); dtheta_vec_j = np.zeros((N,3))

    # Precompute constants
    denom = w**2 + k**2
    zero_k = np.isclose(k, 0)
    zero_w = np.isclose(w, 0)

    # Precompute constants for twist angular velocity
    denom_t = w_t**2 + k_t**2
    zero_k_t = np.isclose(k_t, 0)
    zero_w_t = np.isclose(w_t, 0)

    # Precompute constants for roll angular velocity
    denom_r = w_r**2 + k_r**2
    zero_k_r = np.isclose(k_r, 0)
    zero_w_r = np.isclose(w_r, 0)

    # Precompute constants for shear angular velocity
    denom_s = w_s**2 + k_s**2
    zero_k_s = np.isclose(k_s, 0)
    zero_w_s = np.isclose(w_s, 0)

    def _branch_mag(ti):
        """Analytical branch-vector magnitude |l_ij|(t) (Eq. 24)."""
        if zero_k and zero_w:
            return norm_l0 + A * ti - B * ti * np.sin(phi)
        elif zero_k:
            return norm_l0 + A * ti - (B / w) * (np.cos(phi) - np.cos(w * ti + phi))
        elif zero_w:
            return norm_l0 + A * ti - (B / k) * np.sin(phi) * (np.exp(k * ti) - 1.0)
        else:
            return (norm_l0
                    + A * ti
                    - (B / denom)
                    * (
                        ( k * np.sin(w * ti + phi) - w * np.cos(w * ti + phi) )
                        * np.exp(k * ti)
                        - ( k * np.sin(phi) - w * np.cos(phi) )
                    ))

    def _velocities_at(ti):
        """Exact analytical linear/angular velocities at an arbitrary time ti.

        Returns (v_i, v_j, omega_i, omega_j) as (3,) arrays. This evaluates the
        same closed-form expressions used on-step in the main loop, so it can be
        called at the half-step times t - dt/2 to obtain the exact half-step
        velocities (including the value at t = -dt/2 for idx 0)."""
        Rb = Rotation.from_rotvec(omega_f * ti)
        n = Rb.apply(n0)                                            # contact normal (Eq. 5)
        mag = _branch_mag(ti)                                       # |l_ij| (Eq. 24)
        l = mag * n                                                 # branch vector (Eq. 6)
        v_ijn_loc = (A - B * np.sin(w * ti + phi) * np.exp(k * ti)) * n  # (Eq. 23)
        xi = Rb.apply(x_f) + v_f * ti                              # x_i (Eq. 1)

        # Linear velocities (Eq. 3 and 4)
        vi = v_f + np.cross(omega_f, xi) - np.cross(omega_f, v_f) * ti
        vj = vi + np.cross(omega_f, l) + v_ijn_loc

        # Angular velocities (Eq. 23)
        omegar_t = A_t - B_t * np.sin(w_t * ti + phi_t) * np.exp(k_t * ti)
        omegar_r = A_r - B_r * np.sin(w_r * ti + phi_r) * np.exp(k_r * ti)
        omegar_s = A_s - B_s * np.sin(w_s * ti + phi_s) * np.exp(k_s * ti)
        nrr = Rb.apply(n_r)
        nrs = Rb.apply(n_s)
        wi = (omega_f
              + 0.5 * omegar_t * n
              + 0.5/R_i * omegar_r * nrr
              + 0.5/R_i * omegar_s * nrs)
        wj = (omega_f
              - 0.5 * omegar_t * n
              - 0.5/R_j * omegar_r * nrr
              + 0.5/R_j * omegar_s * nrs)
        return vi, vj, wi, wj

    for idx, ti in enumerate(t):
        # Body rotation, this works because omega_f is constant.
        Rb = Rotation.from_rotvec(omega_f * ti)

        # Contact normal (Eq. 5)
        n_ij[idx] = Rb.apply(n0)

        # Compute relative normal velocity (Eq. 23)
        v_ijn[idx] = (A - B * np.sin(w * ti + phi) * np.exp(k * ti)) * n_ij[idx]

        # Compute relative normal acceleration (Eq. 26)
        a_ijn[idx] = -B * np.exp(k * ti) * (w * np.cos(w * ti + phi) + k * np.sin(w * ti + phi)) * n_ij[idx]

        # Compute branch magnitude (Eq. 24)
        mag = _branch_mag(ti)
        # Branch vector (Eq. 6)
        l_ij[idx] = mag * n_ij[idx]
        
        # Positions (Eq. 1 and 2)
        x_i[idx] = Rb.apply(x_f) + v_f * ti
        x_j[idx] = x_i[idx] + l_ij[idx]

        # Velocities (Eq. 3 and 4)
        v_i[idx] = v_f + np.cross(omega_f, x_i[idx]) - np.cross(omega_f, v_f) * ti
        v_j[idx] = v_i[idx] + np.cross(omega_f, l_ij[idx]) + v_ijn[idx]
        # Equivalently: v_j[idx] = v_f + np.cross(omega_f, x_j[idx]) + v_ijn[idx]

        # Accelerations
        # a_i = omega_f × (omega_f × x_i)
        a_i[idx] = np.cross(omega_f, np.cross(omega_f, x_i[idx])) - np.cross(omega_f, np.cross(omega_f, v_f)) * ti
        # a_j = a_i + a_ijn + 2*v_ijn*(omega_f × n_ij) + |l_ij|*omega_f × (omega_f × n_ij)
        a_j[idx] = (a_i[idx] 
                    + a_ijn[idx] 
                    + 2 * np.linalg.norm(v_ijn[idx]) * np.cross(omega_f, n_ij[idx])
                    + mag * np.cross(omega_f, np.cross(omega_f, n_ij[idx])))

        # Angular velocities (Eq. 23)
        omegar_t = A_t - B_t * np.sin(w_t * ti + phi_t) * np.exp(k_t * ti)
        omegar_r = A_r - B_r * np.sin(w_r * ti + phi_r) * np.exp(k_r * ti)
        omegar_s = A_s - B_s * np.sin(w_s * ti + phi_s) * np.exp(k_s * ti)
        # Rotated direction vectors (Eq. 9 and 10)
        nr_r = Rb.apply(n_r)
        nr_s = Rb.apply(n_s)
        omega_i[idx] = (omega_f
                        + 0.5 * omegar_t * n_ij[idx]
                        + 0.5/R_i * omegar_r * nr_r
                        + 0.5/R_i * omegar_s * nr_s)
        omega_j[idx] = (omega_f
                        - 0.5 * omegar_t * n_ij[idx]
                        - 0.5/R_j * omegar_r * nr_r
                        + 0.5/R_j * omegar_s * nr_s)
        
        # Twist, roll, and shear velocities (Eq. 18 and 20)
        omega_t[idx] = omegar_t * n_ij[idx]
        v_r[idx] = omegar_r * np.cross(nr_r, n_ij[idx])
        v_s[idx] = omegar_s * np.cross(nr_s, n_ij[idx])
        omega_b[idx] = omega_i[idx] - omega_j[idx] - omega_t[idx]

        # Compute analytical displacement increments over last timestep
        if idx == 0:
            # First timestep: no previous timestep exists, so increments are zero
            dtheta_t[idx] = 0.0
            du_r[idx] = np.zeros(3)
            du_s[idx] = np.zeros(3)
            dtheta_b[idx] = 0.0
        else:
            # Compute integral of angular velocities over [t_{i-1}, t_i]
            t_prev = t[idx-1]
            
            # Twist displacement increment (scalar, integrated along n_ij direction)
            if zero_k_t and zero_w_t:
                du_theta_mag = A_t * dt - B_t * dt * np.sin(phi_t)
            elif zero_k_t:
                du_theta_mag = A_t * dt - (B_t / w_t) * (np.cos(w_t * t_prev + phi_t) - np.cos(w_t * ti + phi_t))
            elif zero_w_t:
                du_theta_mag = A_t * dt - (B_t / k_t) * np.sin(phi_t) * (np.exp(k_t * ti) - np.exp(k_t * t_prev))
            else:
                du_theta_mag = (A_t * dt
                               - (B_t / denom_t) 
                               * (
                                   ( k_t * np.sin(w_t * ti + phi_t) - w_t * np.cos(w_t * ti + phi_t) ) 
                                   * np.exp(k_t * ti)
                                   - ( k_t * np.sin(w_t * t_prev + phi_t) - w_t * np.cos(w_t * t_prev + phi_t) )
                                   * np.exp(k_t * t_prev)
                               ))
            dtheta_t[idx] = du_theta_mag * n_ij[idx]

            # Roll displacement increment (vector)
            if zero_k_r and zero_w_r:
                du_r_mag = A_r * dt - B_r * dt * np.sin(phi_r)
            elif zero_k_r:
                du_r_mag = A_r * dt - (B_r / w_r) * (np.cos(w_r * t_prev + phi_r) - np.cos(w_r * ti + phi_r))
            elif zero_w_r:
                du_r_mag = A_r * dt - (B_r / k_r) * np.sin(phi_r) * (np.exp(k_r * ti) - np.exp(k_r * t_prev))
            else:
                du_r_mag = (A_r * dt
                           - (B_r / denom_r) 
                           * (
                               ( k_r * np.sin(w_r * ti + phi_r) - w_r * np.cos(w_r * ti + phi_r) ) 
                               * np.exp(k_r * ti)
                               - ( k_r * np.sin(w_r * t_prev + phi_r) - w_r * np.cos(w_r * t_prev + phi_r) )
                               * np.exp(k_r * t_prev)
                           ))
            du_r[idx] = du_r_mag * np.cross(nr_r, n_ij[idx])

            # Shear displacement increment (vector)
            if zero_k_s and zero_w_s:
                du_s_mag = A_s * dt - B_s * dt * np.sin(phi_s)
            elif zero_k_s:
                du_s_mag = A_s * dt - (B_s / w_s) * (np.cos(w_s * t_prev + phi_s) - np.cos(w_s * ti + phi_s))
            elif zero_w_s:
                du_s_mag = A_s * dt - (B_s / k_s) * np.sin(phi_s) * (np.exp(k_s * ti) - np.exp(k_s * t_prev))
            else:
                du_s_mag = (A_s * dt
                           - (B_s / denom_s) 
                           * (
                               ( k_s * np.sin(w_s * ti + phi_s) - w_s * np.cos(w_s * ti + phi_s) ) 
                               * np.exp(k_s * ti)
                               - ( k_s * np.sin(w_s * t_prev + phi_s) - w_s * np.cos(w_s * t_prev + phi_s) )
                               * np.exp(k_s * t_prev)
                           ))
            du_s[idx] = du_s_mag * np.cross(nr_s, n_ij[idx])

            # Compute analytical rotation increments from t_{i-1} to t_i
            # For particle i: omega_i = omega_f + 0.5*omegar_t*n_ij + 0.5/R_i*omegar_r*nr_r + 0.5/R_i*omegar_s*nr_s
            # Integrating gives: dtheta_vec_i = omega_f*dt + 0.5*du_theta_mag*n_ij + 0.5/R_i*du_r_mag*nr_r + 0.5/R_i*du_s_mag*nr_s
            dtheta_vec_i[idx] = (omega_f * dt 
                                + 0.5 * du_theta_mag * n_ij[idx] 
                                + 0.5/R_i * du_r_mag * nr_r 
                                + 0.5/R_i * du_s_mag * nr_s)
            # For particle j: omega_j = omega_f - 0.5*omegar_t*n_ij - 0.5/R_j*omegar_r*nr_r + 0.5/R_j*omegar_s*nr_s
            dtheta_vec_j[idx] = (omega_f * dt 
                                - 0.5 * du_theta_mag * n_ij[idx] 
                                - 0.5/R_j * du_r_mag * nr_r 
                                + 0.5/R_j * du_s_mag * nr_s)

            # For bending
            dtheta_b[idx] = dtheta_vec_i[idx] - dtheta_vec_j[idx] - dtheta_t[idx]
    
    # Exact analytical half-step velocities.
    # Many codes' leapfrog integrator stores velocities at the half-steps v(t_k - dt/2)
    # (the value left over from the previous Newton update, used by the contact law
    # at force-evaluation time t_k). Rather than reconstructing these from the on-step
    # arrays by averaging (which costs O(dt^2) accuracy), we evaluate the closed-form
    # velocity expressions directly at t_k - dt/2. Entry idx therefore holds
    # v(t_k - dt/2); for idx 0 this is the exact value at t = -dt/2.
    v_i_half = np.zeros((N,3)); v_j_half = np.zeros((N,3))
    omega_i_half = np.zeros((N,3)); omega_j_half = np.zeros((N,3))
    for idx, ti in enumerate(t):
        t_half = ti - dt/2
        v_i_half[idx], v_j_half[idx], omega_i_half[idx], omega_j_half[idx] = _velocities_at(t_half)

    # Compute normal overlap (Eq. 12)
    l_mag = np.linalg.norm(l_ij, axis=1)
    u_n = (R_i + R_j - l_mag).reshape(-1,1) # Surface-to-surface across entire contact
    u_n = np.maximum(u_n, 0.0)

    # Compute orientation using the midpoint (half-step) angular velocity.
    # The rotation increment for the step [t_{k-1}, t_k] is omega(t_k - dt/2) * dt,
    # i.e. the midpoint-rule integral of the angular velocity. Because omega_*_half[k]
    # is evaluated exactly at the step midpoint t_k - dt/2, this is second-order
    # accurate in dt, whereas building the increment from the step-endpoint direction
    # vectors (dtheta_vec_i/j below) is only first-order whenever the contact frame
    # rotates (omega_f != 0). Index 0 is the initial orientation, so its increment is
    # zeroed (omega_*_half[0] = omega(-dt/2) would otherwise apply a spurious rotation).
    dtheta_mid_i = omega_i_half * dt
    dtheta_mid_j = omega_j_half * dt
    dtheta_mid_i[0] = 0.0
    dtheta_mid_j[0] = 0.0 # We don't use this now - check later.
    q_i = my_integrate_rotation(init_q_i, dtheta_vec_i)
    q_j = my_integrate_rotation(init_q_j, dtheta_vec_j)

    # Package results
    motions = {
        't': t.reshape(-1,1),'dt':[dt]*len(t),
        'x_i': x_i, 'x_j': x_j,
        'v_i': v_i, 'v_j': v_j,
        'v_i_half': v_i_half, 'v_j_half': v_j_half,
        'a_i': a_i, 'a_j': a_j,
        'q_i': q_i, 'q_j': q_j,
        'omega_i': omega_i, 'omega_j': omega_j, 'omega_f': [omega_f]*len(t),
        'omega_i_half': omega_i_half, 'omega_j_half': omega_j_half,
        'n_ij': n_ij, 'v_ijn': v_ijn, 'a_ijn': a_ijn, 'l_ij': l_ij,
        'u_n': u_n, 'v_s': v_s, 'v_r': v_r, 'omega_t': omega_t, 'omega_b': omega_b,
        'du_s': du_s, 'du_r': du_r, 'dtheta_t': dtheta_t, 'dtheta_b': dtheta_b
    }
    return motions

def my_integrate_rotation(initial_quat, theta_vecs):
    """
    Integrate quaternion orientation over time given rotation vectors (batch mode).
    
    Parameters:
    - initial_quat: array-like, shape (4,)
        Initial orientation quaternion [x, y, z, w].
    - theta_vecs: ndarray, shape (N, 3)
        Time series of rotation vectors (axis-angle representation) for each timestep.
        theta_vecs[i] represents the rotation increment from time i-1 to time i.
        theta_vecs[0] should be zero (initial orientation is given).
    
    Returns:
    - quaternions: ndarray, shape (N, 4)
        Quaternion orientations [x, y, z, w] at each time including initial.
    """
    # Ensure numpy arrays
    theta_vecs = np.asarray(theta_vecs)
    # Determine number of steps
    nsteps = theta_vecs.shape[0]

    # Initialize orientation and storage
    orientation = Rotation.from_quat(initial_quat)
    quats = np.empty((nsteps, 4))
    quats[0] = orientation.as_quat()
    
    # Loop over each timestep
    for i in range(nsteps-1):
        # Use the rotation vector for the next timestep (from t[i] to t[i+1])
        theta_vec = theta_vecs[i + 1]
        delta_rot = Rotation.from_rotvec(theta_vec)
        # Update orientation by quaternion multiplication
        orientation = delta_rot * orientation  # World frame rotation!
        # Store new quaternion
        quats[i + 1] = (orientation.as_quat()).copy()
    
    return quats

# End of file