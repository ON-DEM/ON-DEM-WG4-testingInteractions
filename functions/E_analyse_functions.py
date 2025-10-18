# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import numpy as np

#
#   ANALYSE AND COMPARE DEM VS ANALYTICAL
#

def my_compare_results(sim_results, ana_results, tol=None):
    """
    Compare two 'results' dictionaries (simulation vs analytical) over time.

    Parameters
    ----------
    sim_results : dict
        Simulation output dict containing keys:
            't', (time as vector Nx1)
            'x_i','x_j', (positions as vectors Nx3)
            'q_i','q_j', (orientations as quaternions Nx4)
            'F_i','T_i', (forces and torques on particle i, both Nx3)
            'F_j','T_j'  (forces and torques on particle j, both Nx3)
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
            'x_i','x_j','q_i','q_j','F_i','F_j','T_i','T_j'
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

    # Orientations
    report['q_i'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('q_i','q_i', tol['orientation'])
    ))
    report['q_j'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('q_j','q_j', tol['orientation'])
    ))

    # Forces
    report['F_i'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('F_i','F_i', tol['force'])
    ))
    report['F_j'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('F_j','F_j', tol['force'])
    ))

    # Torques
    report['T_i'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('T_i','T_i', tol['torque'])
    ))
    report['T_j'] = dict(zip(
        ['diff','pass','all_pass'],
        compare_array('T_j','T_j', tol['torque'])
    ))

    return report



def my_check_all(report):
    for key in ['F_i', 'F_j', 'T_i', 'T_j']:
        all_ok = report[key]['all_pass']
        print(f"{key} check passed: {all_ok}")
        if not all_ok:
            failing_steps = np.where(~report[key]['pass'])[0]
            print(f"  Failed at steps: {failing_steps}")

# End of file