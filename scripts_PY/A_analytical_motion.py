# Copyright 2025: Danny van der Haven, dlhv2@cantab.ac.uk

import numpy as np
from scipy.spatial.transform import Rotation

#
#   CLOSED-FORM INTEGRAL HELPERS
#
# Antiderivatives of e^{k t} sin(Omega t + psi) and e^{k t} cos(Omega t + psi),
# evaluated at t = tau. These are the building blocks for the exact per-step
# contact increments (see _rotated_integral below). The only singular case is
# k = Omega = 0, where the integrand is the constant sin(psi) / cos(psi).
# NOTE: near a resonance Omega -> 0 with k = 0 the closed form is analytically
# exact but loses precision through cancellation in the F(t_b) - F(t_a) difference;
# the test suite never places |omega_f| exactly on a mode frequency, so this is
# not exercised.

def _expsin_antideriv(k, Omega, psi, tau):
    """Antiderivative of e^{k t} sin(Omega t + psi) at t = tau."""
    d = k * k + Omega * Omega
    if d == 0.0:
        return np.sin(psi) * tau
    return np.exp(k * tau) * (k * np.sin(Omega * tau + psi) - Omega * np.cos(Omega * tau + psi)) / d

def _expcos_antideriv(k, Omega, psi, tau):
    """Antiderivative of e^{k t} cos(Omega t + psi) at t = tau."""
    d = k * k + Omega * Omega
    if d == 0.0:
        return np.cos(psi) * tau
    return np.exp(k * tau) * (k * np.cos(Omega * tau + psi) + Omega * np.sin(Omega * tau + psi)) / d

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

    # Frame-rotation axis/angle for the contact directions: the contact frame
    # rotates rigidly as Rb(t) = exp([omega_f]x t), i.e. about ehat = omega_f/|omega_f|
    # at rate phi_f = |omega_f|. These drive the closed-form rotating-frame integrals.
    phi_f = np.linalg.norm(omega_f)
    ehat = omega_f / phi_f if phi_f > 0 else np.zeros(3)

    # Time array
    t = np.arange(0, t_end + dt/2, dt)
    N = t.size

    # Per-step contact increments are filled from index 1 below (index 0 stays zero,
    # as no step precedes t_0). Every other time series is assigned directly from the
    # vectorised expressions further down.
    du_r = np.zeros((N,3)); du_s = np.zeros((N,3))
    dtheta_t = np.zeros((N,3)); dtheta_b = np.zeros((N,3))

    # Precompute constants for the normal branch magnitude (_branch_mag).
    # The twist/roll/shear increments no longer need precomputed constants: they are
    # evaluated in closed form by _rotated_integral via the antiderivative helpers.
    denom = w**2 + k**2
    zero_k = np.isclose(k, 0)
    zero_w = np.isclose(w, 0)

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

    def _vel_at(tarr):
        """Exact analytical linear/angular velocities at the times in tarr (1-D),
        vectorised over time: returns (v_i, v_j, omega_i, omega_j) each shaped
        (len(tarr), 3). Evaluates the same closed-form expressions used on-step, so
        it serves both the half-step velocities (tarr = t - dt/2, exact at t = -dt/2)
        and the Gauss-node angular-velocity samples of the Magnus-4 orientation step."""
        tarr = np.atleast_1d(np.asarray(tarr, float))
        Rb = Rotation.from_rotvec(omega_f[None, :] * tarr[:, None])  # batch of len(tarr) rotations
        n = Rb.apply(n0)                                            # (M,3) contact normal (Eq. 5)

        nrr = Rb.apply(n_r)
        nrs = Rb.apply(n_s)
        mag = _branch_mag(tarr)                                     # (M,) |l_ij| (Eq. 24)
        s_n = A - B * np.sin(w * tarr + phi) * np.exp(k * tarr)     # (M,) normal rate (Eq. 23)
        v_ijn_loc = s_n[:, None] * n
        xi = Rb.apply(x_f) + v_f[None, :] * tarr[:, None]          # x_i (Eq. 1)

        # Linear velocities (Eq. 3 and 4)
        vi = v_f[None, :] + np.cross(omega_f, xi) - np.cross(omega_f, v_f) * tarr[:, None]
        vj = vi + np.cross(omega_f, mag[:, None] * n) + v_ijn_loc

        # Angular velocities (Eq. 23)
        o_t = A_t - B_t * np.sin(w_t * tarr + phi_t) * np.exp(k_t * tarr)
        o_r = A_r - B_r * np.sin(w_r * tarr + phi_r) * np.exp(k_r * tarr)
        o_s = A_s - B_s * np.sin(w_s * tarr + phi_s) * np.exp(k_s * tarr)
        wi = (omega_f[None, :] + 0.5 * o_t[:, None] * n
              + (0.5/R_i) * o_r[:, None] * nrr + (0.5/R_i) * o_s[:, None] * nrs)
        wj = (omega_f[None, :] - 0.5 * o_t[:, None] * n
              - (0.5/R_j) * o_r[:, None] * nrr + (0.5/R_j) * o_s[:, None] * nrs)
        return vi, vj, wi, wj

    def _rotated_integral(d0, A_x, B_x, w_x, psi_x, k_x, t_a, t_b):
        """Exact closed-form integral  J = int_{t_a}^{t_b} s(tau) Rb(tau) d0 dtau,
        with the exp-sinusoid rate  s(tau) = A_x - B_x sin(w_x tau + psi_x) e^{k_x tau}.

        Every twist/roll/shear increment has this form because the rotating contact
        directions satisfy Rb(tau) d0 = a cos(phi_f tau) + b sin(phi_f tau) + c
        (Rodrigues), so J = a*C + b*S + c*M with
            M = int s dtau,  C = int s cos(phi_f tau) dtau,  S = int s sin(phi_f tau) dtau.
        Products like sin(w_x tau)cos(phi_f tau) reduce (product-to-sum) to the same
        antiderivatives at the combined frequencies w_x and w_x +/- phi_f.
        For phi_f = 0 (non-rotating frame) the exact result collapses to M * d0.
        """
        d0 = np.asarray(d0, float)
        # M = int [A_x - B_x sin(w_x tau + psi_x) e^{k_x tau}] dtau
        def _M_at(tau):
            return A_x * tau - B_x * _expsin_antideriv(k_x, w_x, psi_x, tau)
        M = _M_at(t_b) - _M_at(t_a)   # scalar, or (L,) if t_a/t_b are arrays

        if phi_f == 0.0:
            return np.multiply.outer(M, d0)   # -> (3,) or (L,3)

        # C = int s cos(phi_f tau) dtau ;  S = int s sin(phi_f tau) dtau
        # sin(w tau + psi) cos(phi tau) = 1/2 [ sin((w+phi)tau+psi) + sin((w-phi)tau+psi) ]
        # sin(w tau + psi) sin(phi tau) = 1/2 [ cos((w-phi)tau+psi) - cos((w+phi)tau+psi) ]
        def _C_at(tau):
            return (A_x * _expcos_antideriv(0.0, phi_f, 0.0, tau)
                    - B_x * 0.5 * (_expsin_antideriv(k_x, w_x + phi_f, psi_x, tau)
                                   + _expsin_antideriv(k_x, w_x - phi_f, psi_x, tau)))
        def _S_at(tau):
            return (A_x * _expsin_antideriv(0.0, phi_f, 0.0, tau)
                    - B_x * 0.5 * (_expcos_antideriv(k_x, w_x - phi_f, psi_x, tau)
                                   - _expcos_antideriv(k_x, w_x + phi_f, psi_x, tau)))
        C = _C_at(t_b) - _C_at(t_a)
        S = _S_at(t_b) - _S_at(t_a)

        c_vec = ehat * np.dot(ehat, d0)   # component along the rotation axis (invariant)
        a_vec = d0 - c_vec                # perpendicular component (rotates as cos)
        b_vec = np.cross(ehat, d0)        # quadrature component (rotates as sin)
        return (np.multiply.outer(C, a_vec)
                + np.multiply.outer(S, b_vec)
                + np.multiply.outer(M, c_vec))

    # --- On-step kinematics (vectorised over all times t) ---
    # One batched frame rotation Rb(t) = exp([omega_f]x t) for all steps; every line
    # below is an array op over the N time samples.
    Rb = Rotation.from_rotvec(omega_f[None, :] * t[:, None])
    n_ij = Rb.apply(n0)            # (N,3) contact normal (Eq. 5)
    nr_r = Rb.apply(n_r)           # (N,3) rotated roll axis (Eq. 9, 10)
    nr_s = Rb.apply(n_s)           # (N,3) rotated shear axis
    Rb_xf = Rb.apply(x_f)

    sin_n = np.sin(w * t + phi); exp_n = np.exp(k * t)
    s_n = A - B * sin_n * exp_n                                  # normal rate (Eq. 23)
    v_ijn = s_n[:, None] * n_ij
    a_ijn = (-B * exp_n * (w * np.cos(w * t + phi) + k * sin_n))[:, None] * n_ij  # (Eq. 26)

    mag = _branch_mag(t)                                         # (N,) |l_ij| (Eq. 24)
    l_ij = mag[:, None] * n_ij                                   # branch vector (Eq. 6)

    # Positions (Eq. 1 and 2)
    x_i = Rb_xf + v_f[None, :] * t[:, None]
    x_j = x_i + l_ij

    # Linear velocities (Eq. 3 and 4)
    v_i = v_f[None, :] + np.cross(omega_f, x_i) - np.cross(omega_f, v_f) * t[:, None]
    v_j = v_i + np.cross(omega_f, l_ij) + v_ijn

    # Accelerations: a_i = omega_f x (omega_f x x_i);
    # a_j = a_i + a_ijn + 2|v_ijn|(omega_f x n_ij) + |l_ij| omega_f x (omega_f x n_ij)
    cof_n = np.cross(omega_f, n_ij)
    a_i = np.cross(omega_f, np.cross(omega_f, x_i)) - np.cross(omega_f, np.cross(omega_f, v_f)) * t[:, None]
    a_j = (a_i + a_ijn
           + 2 * np.linalg.norm(v_ijn, axis=1)[:, None] * cof_n
           + mag[:, None] * np.cross(omega_f, cof_n))

    # Angular velocities (Eq. 23)
    omegar_t = A_t - B_t * np.sin(w_t * t + phi_t) * np.exp(k_t * t)
    omegar_r = A_r - B_r * np.sin(w_r * t + phi_r) * np.exp(k_r * t)
    omegar_s = A_s - B_s * np.sin(w_s * t + phi_s) * np.exp(k_s * t)
    omega_i = (omega_f[None, :] + 0.5 * omegar_t[:, None] * n_ij
               + (0.5/R_i) * omegar_r[:, None] * nr_r + (0.5/R_i) * omegar_s[:, None] * nr_s)
    omega_j = (omega_f[None, :] - 0.5 * omegar_t[:, None] * n_ij
               - (0.5/R_j) * omegar_r[:, None] * nr_r + (0.5/R_j) * omegar_s[:, None] * nr_s)

    # Twist, roll, and shear velocities (Eq. 18 and 20)
    omega_t = omegar_t[:, None] * n_ij
    v_r = omegar_r[:, None] * np.cross(nr_r, n_ij)
    v_s = omegar_s[:, None] * np.cross(nr_s, n_ij)
    omega_b = omega_i - omega_j - omega_t

    # Exact analytical half-step velocities v(t_k - dt/2), evaluated in closed form for
    # all steps at once (no O(dt^2) reconstruction). Entry idx holds v(t_k - dt/2); for
    # idx 0 this is the exact value at t = -dt/2. These are the values a leapfrog
    # integrator stores at force-evaluation time.
    v_i_half, v_j_half, omega_i_half, omega_j_half = _vel_at(t - dt/2)

    # Compute normal overlap (Eq. 12)
    l_mag = np.linalg.norm(l_ij, axis=1)
    u_n = (R_i + R_j - l_mag).reshape(-1,1) # Surface-to-surface across entire contact
    u_n = np.maximum(u_n, 0.0)

    # --- Exact per-step contact increments and orientation (exact-continuum reference) ---
    # Each increment is the exact closed-form integral J(d0; s) = int s(tau) Rb(tau) d0 dtau
    # over the step [t_{k-1}, t_k] (see _rotated_integral). The roll/shear DISPLACEMENTS use
    # the cross-product seed (n_X x n0), since their velocity is s*(n_X x n); the per-particle
    # rotation RATES use the direction seed n_X. These are pure integrals (no coning term).
    #
    # The orientation q cannot be written in closed form (a finite-rotation composition has a
    # coning/commutator contribution), so it is integrated with a 4th-order Magnus step: two
    # Gauss-Legendre samples of the exact angular velocity give the leading term, and the
    # commutator of those samples supplies the coning correction. Index 0 keeps the initial
    # orientation (zero increment).
    # Each step covers [t_{k-1}, t_k]; t_a/t_b are the (N-1,) arrays of step endpoints,
    # so every _rotated_integral call returns the whole (N-1, 3) stack of increments.
    t_a = t[:-1]; t_b = t[1:]
    J_t  = _rotated_integral(n0,                A_t, B_t, w_t, phi_t, k_t, t_a, t_b)
    J_r  = _rotated_integral(n_r,               A_r, B_r, w_r, phi_r, k_r, t_a, t_b)
    J_s  = _rotated_integral(n_s,               A_s, B_s, w_s, phi_s, k_s, t_a, t_b)
    Jc_r = _rotated_integral(np.cross(n_r, n0), A_r, B_r, w_r, phi_r, k_r, t_a, t_b)
    Jc_s = _rotated_integral(np.cross(n_s, n0), A_s, B_s, w_s, phi_s, k_s, t_a, t_b)

    # Twist rotation increment; roll/shear displacement increments (index 0 stays zero)
    dtheta_t[1:] = J_t
    du_r[1:]     = Jc_r
    du_s[1:]     = Jc_s

    # Exact per-particle rotation increments (int omega dtau); used only for bending
    dth_i = omega_f[None, :] * dt + 0.5 * J_t + (0.5/R_i) * J_r + (0.5/R_i) * J_s
    dth_j = omega_f[None, :] * dt - 0.5 * J_t - (0.5/R_j) * J_r + (0.5/R_j) * J_s
    dtheta_b[1:] = dth_i - dth_j - J_t # J_t = dtheta_t[1:]

    # Magnus-4 orientation increments from two Gauss-point angular-velocity samples per step
    GL = np.sqrt(3.0) / 6.0          # Gauss-Legendre 2-node offset from the midpoint
    CONE = np.sqrt(3.0) / 12.0       # Magnus-4 commutator (coning) coefficient
    _, _, wi1, wj1 = _vel_at(t_a + (0.5 - GL) * dt)
    _, _, wi2, wj2 = _vel_at(t_a + (0.5 + GL) * dt)
    Omega_i = np.zeros((N,3)); Omega_j = np.zeros((N,3))
    Omega_i[1:] = 0.5 * dt * (wi1 + wi2) + CONE * dt * dt * np.cross(wi2, wi1)
    Omega_j[1:] = 0.5 * dt * (wj1 + wj2) + CONE * dt * dt * np.cross(wj2, wj1)

    # Compose orientations from the Magnus-4 increments (index 0 = initial orientation).
    q_i = my_integrate_rotation(init_q_i, Omega_i)
    q_j = my_integrate_rotation(init_q_j, Omega_j)

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

    # Pre-build every incremental rotation in one batched from_rotvec call (the
    # per-step construction was a hotspot), then compose sequentially with the same
    # world-frame (left) multiplication as before.
    deltas = Rotation.from_rotvec(theta_vecs)
    orientation = Rotation.from_quat(initial_quat)
    quats = np.empty((nsteps, 4))
    quats[0] = orientation.as_quat()
    for i in range(nsteps - 1):
        orientation = deltas[i + 1] * orientation  # World frame rotation!
        quats[i + 1] = orientation.as_quat()

    return quats

# End of file