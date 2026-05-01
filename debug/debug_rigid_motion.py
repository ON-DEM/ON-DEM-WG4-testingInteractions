#!/usr/bin/env python3
"""
Debug script: compare force/torque magnitudes, separation distance, and
relative orientation angle for tests 13–16, across four conditions:
    ANA  no rigid-body motion
    ANA  with rigid-body motion
    DEM  no rigid-body motion
    DEM  with rigid-body motion

Expected folder layout (this script lives inside debug/):
    debug/
        no_rigid_motion/
            theoretical_output_test_13.json  ...
            dem_output_YADE_test_13.csv      ...
        rigid_motion/
            theoretical_output_test_13.json  ...
            dem_output_YADE_test_13.csv      ...
        figures/   ← output goes here

Three PNG files are written per test:
    debug_test_XX_forces.png       |F_i|(t) and |T_i|(t)
    debug_test_XX_positions.png    separation distance |x_j − x_i|(t)
    debug_test_XX_orientations.png relative rotation angle between q_i and q_j (t)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# File-loading helpers
# ---------------------------------------------------------------------------

def json_to_dict(filename):
    """Load a JSON file and convert nested lists to numpy arrays."""
    def to_ndarray(obj):
        if isinstance(obj, list):
            if obj and isinstance(obj[0], list):
                return np.array(obj)
            elif obj and isinstance(obj[0], (int, float)):
                return np.array(obj)
            else:
                return [to_ndarray(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_ndarray(v) for k, v in obj.items()}
        else:
            return obj

    with open(filename, 'r') as f:
        data = json.load(f)
    return to_ndarray(data)


def load_dem_csv(filepath):
    """
    Load DEM output CSV (written by G_generate_forces.py / saveForcesTorques).

    Column order (header written as a comment line):
        t
        x1 y1 z1   x2 y2 z2
        qx1 qy1 qz1 qw1   qx2 qy2 qz2 qw2
        v1x v1y v1z   v2x v2y v2z
        w1x w1y w1z   w2x w2y w2z
        f1x f1y f1z   f2x f2y f2z
        t1x t1y t1z   t2x t2y t2z

    Quaternion convention: [qx, qy, qz, qw] — scalar component last.
    """
    with open(filepath, 'r') as f:
        header_line = f.readline().strip()
        if header_line.startswith('#'):
            header_line = header_line[1:].strip()
        cols = header_line.split()
        data = np.array([list(map(float, row.split()))
                         for row in f
                         if row.strip() and not row.strip().startswith('#')])

    def idx(name):
        return cols.index(name)

    result = {}
    result['t']   = data[:, idx('t')]
    result['x_i'] = data[:, [idx('x1'),  idx('y1'),  idx('z1')]]
    result['x_j'] = data[:, [idx('x2'),  idx('y2'),  idx('z2')]]
    # Quaternions stored as [qx, qy, qz, qw]
    result['q_i'] = data[:, [idx('qx1'), idx('qy1'), idx('qz1'), idx('qw1')]]
    result['q_j'] = data[:, [idx('qx2'), idx('qy2'), idx('qz2'), idx('qw2')]]
    result['F_i'] = data[:, [idx('f1x'), idx('f1y'), idx('f1z')]]
    result['F_j'] = data[:, [idx('f2x'), idx('f2y'), idx('f2z')]]
    result['T_i'] = data[:, [idx('t1x'), idx('t1y'), idx('t1z')]]
    result['T_j'] = data[:, [idx('t2x'), idx('t2y'), idx('t2z')]]
    return result


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def safe_t(data):
    """Return 1-D time array regardless of shape (N,) or (N,1)."""
    return np.asarray(data['t']).ravel()


def force_mag(data):
    """Row-wise L2 norm of F_i."""
    return np.linalg.norm(np.asarray(data['F_i']), axis=1)


def torque_mag(data):
    """Row-wise L2 norm of T_i."""
    return np.linalg.norm(np.asarray(data['T_i']), axis=1)


def separation(data):
    """Centre-to-centre distance |x_j − x_i| at each time step."""
    xi = np.asarray(data['x_i'])
    xj = np.asarray(data['x_j'])
    return np.linalg.norm(xj - xi, axis=1)


def rel_rotation_angle(data):
    """
    Angle of the relative rotation between particle i and particle j.

    For two unit quaternions q_i and q_j the 4-D dot product satisfies
        |q_i · q_j| = cos(θ/2)
    so the relative rotation angle is
        θ = 2 · arccos( clip(|q_i · q_j|, 0, 1) )

    Returns values in [0, π] radians.
    Quaternion convention: [qx, qy, qz, qw] — scalar component last (column 3).
    """
    qi = np.asarray(data['q_i'])   # (N, 4)
    qj = np.asarray(data['q_j'])   # (N, 4)

    # Normalise rows to guard against any numerical drift
    qi = qi / np.linalg.norm(qi, axis=1, keepdims=True)
    qj = qj / np.linalg.norm(qj, axis=1, keepdims=True)

    cos_half = np.clip(np.abs(np.einsum('ij,ij->i', qi, qj)), 0.0, 1.0)
    return 2.0 * np.arccos(cos_half)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent        # script lives inside debug/
NO_RM_DIR  = SCRIPT_DIR / 'no_rigid_motion'
RM_DIR     = SCRIPT_DIR / 'rigid_motion'
FIGURE_DIR = SCRIPT_DIR / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [13, 14, 15, 16]

# Plotting style
plt.rcParams['font.family']    = 'serif'
plt.rcParams['font.size']      = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams['grid.alpha']     = 0.2

STYLES = {
    'ana_no_rm': dict(color='#4B0082', lw=1.8, ls='-',  label='ANA  no rigid motion'),
    'ana_rm':    dict(color='#B87000', lw=1.8, ls='-',  label='ANA  with rigid motion'),
    'dem_no_rm': dict(color='#1a7d1a', lw=1.2, ls='--', label='DEM  no rigid motion'),
    'dem_rm':    dict(color='#d63b3b', lw=1.2, ls='--', label='DEM  with rigid motion'),
}


# ---------------------------------------------------------------------------
# Loading helper
# ---------------------------------------------------------------------------

def load_condition(folder, test_id):
    """
    Load ANA (JSON) and DEM (CSV) for one test from the given folder.
    Returns (ana_data, dem_data); either may be None if the file is missing.
    """
    testname  = f'test_{test_id:02d}'
    ana_path  = folder / f'theoretical_output_{testname}.json'
    dem_path  = folder / f'dem_output_YADE_{testname}.csv'

    ana, dem = None, None

    if ana_path.exists():
        ana = json_to_dict(str(ana_path))
    else:
        print(f'  [WARN] ANA file not found: {ana_path}')

    if dem_path.exists():
        try:
            dem = load_dem_csv(str(dem_path))
        except Exception as e:
            print(f'  [WARN] Could not load DEM file {dem_path}: {e}')
    else:
        print(f'  [WARN] DEM file not found: {dem_path}')

    return ana, dem


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for test_id in TESTS:
    print(f'\n=== Test {test_id} ===')

    ana_no_rm, dem_no_rm = load_condition(NO_RM_DIR, test_id)
    ana_rm,    dem_rm    = load_condition(RM_DIR,    test_id)

    all_datasets = {
        'ana_no_rm': ana_no_rm,
        'ana_rm':    ana_rm,
        'dem_no_rm': dem_no_rm,
        'dem_rm':    dem_rm,
    }

    # ------------------------------------------------------------------
    # 1. Forces & torques  (two panels, all four conditions together)
    # ------------------------------------------------------------------
    fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    fig.suptitle(f'Test {test_id} — force & torque magnitudes', fontsize=10)

    for key, data in all_datasets.items():
        if data is None:
            continue
        ax_f.plot(safe_t(data), force_mag(data),  **STYLES[key])
        ax_t.plot(safe_t(data), torque_mag(data), **STYLES[key])

    ax_f.set_ylabel(r'$|\vec{F}_i|$ (N)')
    ax_t.set_ylabel(r'$|\vec{T}_i|$ (N$\cdot$m)')
    for ax in (ax_f, ax_t):
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', lw=0.4)
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f'debug_test_{test_id:02d}_forces.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: debug_test_{test_id:02d}_forces.png')

    # ------------------------------------------------------------------
    # 2. Separation distance (ANA panel / DEM panel to avoid clutter)
    # ------------------------------------------------------------------
    fig, (ax_ana, ax_dem) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    fig.suptitle(
        fr'Test {test_id} — separation distance $|\vec{{x}}_j - \vec{{x}}_i|$',
        fontsize=10
    )

    for key, data in all_datasets.items():
        if data is None:
            continue
        ax = ax_ana if key.startswith('ana') else ax_dem
        ax.plot(safe_t(data), separation(data), **STYLES[key])

    for ax, title in ((ax_ana, 'Analytical'), (ax_dem, 'DEM')):
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$|\vec{x}_j - \vec{x}_i|$ (m)')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f'debug_test_{test_id:02d}_positions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: debug_test_{test_id:02d}_positions.png')

    # ------------------------------------------------------------------
    # 3. Relative rotation angle between q_i and q_j
    # ------------------------------------------------------------------
    fig, (ax_ana, ax_dem) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    fig.suptitle(
        fr'Test {test_id} — relative rotation angle between $q_i$ and $q_j$',
        fontsize=10
    )

    for key, data in all_datasets.items():
        if data is None:
            continue
        ax = ax_ana if key.startswith('ana') else ax_dem
        ax.plot(safe_t(data), np.degrees(rel_rotation_angle(data)), **STYLES[key])

    for ax, title in ((ax_ana, 'Analytical'), (ax_dem, 'DEM')):
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Relative rotation angle $\theta$ (°)')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f'debug_test_{test_id:02d}_orientations.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: debug_test_{test_id:02d}_orientations.png')

print('\nDone.')