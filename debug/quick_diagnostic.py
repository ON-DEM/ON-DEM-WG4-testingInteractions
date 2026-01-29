#!/usr/bin/env python3
"""
Quick diagnostic script to identify major issues before full analysis.
Tests ALL components of vectors and provides detailed analysis.

Usage: python quick_diagnostic.py --testID 1
"""

import numpy as np
import argparse
from pathlib import Path


def load_theoretical_data_simple(filepath):
    """Load theoretical data with proper column parsing."""
    data = np.loadtxt(str(filepath), comments='#')
    
    # Parse columns according to header structure
    # t dt x_i(3) x_j(3) v_i(3) v_j(3) q_i(4) q_j(4) omega_i(3) omega_j(3) 
    # omega_b(3) n_ij(3) v_ijn(3) l_ij(3) u_n(1) v_s(3) v_r(3) v_theta(3) F_i(3) F_j(3) T_i(3) T_j(3)
    
    result = {}
    col = 0
    result['t'] = data[:, col]; col += 1           # col 0
    result['dt'] = data[:, col]; col += 1          # col 1
    result['x_i'] = data[:, col:col+3]; col += 3   # cols 2-4
    result['x_j'] = data[:, col:col+3]; col += 3   # cols 5-7
    result['v_i'] = data[:, col:col+3]; col += 3   # cols 8-10
    result['v_j'] = data[:, col:col+3]; col += 3   # cols 11-13
    result['q_i'] = data[:, col:col+4]; col += 4   # cols 14-17 (x,y,z,w)
    result['q_j'] = data[:, col:col+4]; col += 4   # cols 18-21 (x,y,z,w)
    col += 3  # skip omega_i
    col += 3  # skip omega_j
    col += 3  # skip omega_b
    col += 3  # skip n_ij
    col += 3  # skip v_ijn
    col += 3  # skip l_ij
    col += 1  # skip u_n
    col += 3  # skip v_s
    col += 3  # skip v_r
    col += 3  # skip v_theta
    result['F_i'] = data[:, col:col+3]; col += 3   # cols 50-52
    result['F_j'] = data[:, col:col+3]; col += 3   # cols 53-55
    result['T_i'] = data[:, col:col+3]; col += 3   # cols 56-58
    result['T_j'] = data[:, col:col+3]; col += 3   # cols 59-61
    
    return result


def load_dem_data_simple(filepath):
    """Load DEM data with proper column parsing."""
    data = np.loadtxt(str(filepath), comments='#')
    
    # t x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2 
    # f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
    
    result = {}
    result['t'] = data[:, 0]
    result['x_i'] = data[:, 1:4]      # cols 1-3
    result['x_j'] = data[:, 4:7]      # cols 4-6
    result['q_i'] = data[:, 7:11]     # cols 7-10 (x,y,z,w)
    result['q_j'] = data[:, 11:15]    # cols 11-14 (x,y,z,w)
    result['F_i'] = data[:, 15:18]    # cols 15-17
    result['F_j'] = data[:, 18:21]    # cols 18-20
    result['T_i'] = data[:, 21:24]    # cols 21-23
    result['T_j'] = data[:, 24:27]    # cols 24-26
    
    return result


def compare_vectors(name, ana_vec, dem_vec, tolerance=1e-6):
    """Compare vector quantities component by component."""
    components = ['x', 'y', 'z'] if ana_vec.shape[1] == 3 else ['x', 'y', 'z', 'w']
    
    print(f"\n  {name}:")
    all_match = True
    
    for i, comp in enumerate(components):
        ana_val = ana_vec[0, i]
        dem_val = dem_vec[0, i]
        diff = abs(ana_val - dem_val)
        
        if diff > tolerance:
            status = "✗ MISMATCH"
            all_match = False
        else:
            status = "✓"
        
        print(f"    {comp}: Ana={ana_val:12.6e}, DEM={dem_val:12.6e}, diff={diff:12.6e} {status}")
    
    return all_match


def quick_check(ana_file, dem_file, test_id):
    """
    Perform quick diagnostic checks with comprehensive component testing.
    """
    print("\n" + "="*80)
    print(f"COMPREHENSIVE DIAGNOSTIC FOR TEST {test_id:02d}")
    print("="*80)
    
    # Load data
    ana = load_theoretical_data_simple(ana_file)
    dem = load_dem_data_simple(dem_file)
    
    print(f"\n✓ Loaded files successfully")
    print(f"  Analytical: {ana_file.name}")
    print(f"  DEM:        {dem_file.name}")
    
    # Check dimensions
    print(f"\nDATA DIMENSIONS:")
    print(f"  Analytical: {len(ana['t'])} time steps")
    print(f"  DEM:        {len(dem['t'])} time steps")
    
    # ==========================================================================
    # TIME ANALYSIS
    # ==========================================================================
    print(f"\n" + "="*80)
    print("TIME ANALYSIS")
    print("="*80)
    
    ana_t = ana['t']
    dem_t = dem['t']
    
    print(f"\n  Analytical time:")
    print(f"    Start:    {ana_t[0]:.10f} s")
    print(f"    End:      {ana_t[-1]:.10f} s")
    print(f"    Duration: {ana_t[-1] - ana_t[0]:.10f} s")
    print(f"    Steps:    {len(ana_t)}")
    
    if len(ana_t) > 1:
        dt_ana = np.diff(ana_t)
        print(f"    Avg Δt:   {np.mean(dt_ana):.10f} s")
        print(f"    Min Δt:   {np.min(dt_ana):.10f} s")
        print(f"    Max Δt:   {np.max(dt_ana):.10f} s")
        if np.std(dt_ana) > 1e-10:
            print(f"    Std Δt:   {np.std(dt_ana):.10e} s (variable time step)")
        else:
            print(f"    Std Δt:   {np.std(dt_ana):.10e} s (constant time step)")
    
    print(f"\n  DEM time:")
    print(f"    Start:    {dem_t[0]:.10f} s")
    print(f"    End:      {dem_t[-1]:.10f} s")
    print(f"    Duration: {dem_t[-1] - dem_t[0]:.10f} s")
    print(f"    Steps:    {len(dem_t)}")
    
    if len(dem_t) > 1:
        dt_dem = np.diff(dem_t)
        print(f"    Avg Δt:   {np.mean(dt_dem):.10f} s")
        print(f"    Min Δt:   {np.min(dt_dem):.10f} s")
        print(f"    Max Δt:   {np.max(dt_dem):.10f} s")
        if np.std(dt_dem) > 1e-10:
            print(f"    Std Δt:   {np.std(dt_dem):.10e} s (variable time step)")
        else:
            print(f"    Std Δt:   {np.std(dt_dem):.10e} s (constant time step)")
    
    # Time synchronization check
    time_issues = []
    if abs(ana_t[0] - dem_t[0]) > 1e-10:
        time_issues.append("Start times differ")
        print(f"\n  ⚠ WARNING: Start times differ by {abs(ana_t[0] - dem_t[0]):.10e} s")
    
    if abs(ana_t[-1] - dem_t[-1]) > 1e-6:
        time_issues.append("End times differ")
        print(f"\n  ⚠ WARNING: End times differ by {abs(ana_t[-1] - dem_t[-1]):.10e} s")
    
    if len(ana_t) != len(dem_t):
        time_issues.append("Different number of steps")
        print(f"\n  ⚠ WARNING: Different number of time steps")
    elif len(ana_t) > 1:
        # Check if time arrays are identical
        max_time_diff = np.max(np.abs(ana_t - dem_t))
        if max_time_diff > 1e-10:
            time_issues.append("Time arrays not synchronized")
            print(f"\n  ⚠ WARNING: Time arrays differ (max diff: {max_time_diff:.10e} s)")
        else:
            print(f"\n  ✓ Time arrays are synchronized (max diff: {max_time_diff:.10e} s)")
    
    # ==========================================================================
    # INITIAL CONDITIONS CHECK
    # ==========================================================================
    print(f"\n" + "="*80)
    print("INITIAL CONDITIONS (t=0) - ALL COMPONENTS")
    print("="*80)
    
    ic_issues = []
    
    # Position checks
    print(f"\nPOSITIONS:")
    if not compare_vectors("Particle i position (x_i)", ana['x_i'], dem['x_i']):
        ic_issues.append("Initial position x_i mismatch")
    if not compare_vectors("Particle j position (x_j)", ana['x_j'], dem['x_j']):
        ic_issues.append("Initial position x_j mismatch")
    
    # Quaternion checks
    print(f"\nQUATERNIONS (format: x,y,z,w):")
    if not compare_vectors("Particle i orientation (q_i)", ana['q_i'], dem['q_i']):
        ic_issues.append("Initial quaternion q_i mismatch")
    if not compare_vectors("Particle j orientation (q_j)", ana['q_j'], dem['q_j']):
        ic_issues.append("Initial quaternion q_j mismatch")
    
    # Check quaternion normalization
    ana_q_i_norm = np.linalg.norm(ana['q_i'][0])
    dem_q_i_norm = np.linalg.norm(dem['q_i'][0])
    ana_q_j_norm = np.linalg.norm(ana['q_j'][0])
    dem_q_j_norm = np.linalg.norm(dem['q_j'][0])
    
    print(f"\n  Quaternion normalization check:")
    print(f"    q_i: Ana norm={ana_q_i_norm:.10f}, DEM norm={dem_q_i_norm:.10f}")
    print(f"    q_j: Ana norm={ana_q_j_norm:.10f}, DEM norm={dem_q_j_norm:.10f}")
    
    if abs(ana_q_i_norm - 1.0) > 1e-6 or abs(dem_q_i_norm - 1.0) > 1e-6:
        print(f"    ⚠ WARNING: q_i not normalized!")
    if abs(ana_q_j_norm - 1.0) > 1e-6 or abs(dem_q_j_norm - 1.0) > 1e-6:
        print(f"    ⚠ WARNING: q_j not normalized!")
    
    # Force checks
    print(f"\nINITIAL FORCES:")
    if not compare_vectors("Particle i force (F_i)", ana['F_i'], dem['F_i'], tolerance=1e-3):
        ic_issues.append("Initial force F_i mismatch")
    if not compare_vectors("Particle j force (F_j)", ana['F_j'], dem['F_j'], tolerance=1e-3):
        ic_issues.append("Initial force F_j mismatch")
    
    # Torque checks
    print(f"\nINITIAL TORQUES:")
    if not compare_vectors("Particle i torque (T_i)", ana['T_i'], dem['T_i'], tolerance=1e-3):
        ic_issues.append("Initial torque T_i mismatch")
    if not compare_vectors("Particle j torque (T_j)", ana['T_j'], dem['T_j'], tolerance=1e-3):
        ic_issues.append("Initial torque T_j mismatch")
    
    # ==========================================================================
    # FORCE AND TORQUE STATISTICS (ALL TIME)
    # ==========================================================================
    print(f"\n" + "="*80)
    print("FORCE AND TORQUE STATISTICS (ALL COMPONENTS, ALL TIME)")
    print("="*80)
    
    components = ['x', 'y', 'z']
    
    # Force statistics
    print(f"\nFORCE STATISTICS - PARTICLE i:")
    for i, comp in enumerate(components):
        ana_vals = ana['F_i'][:, i]
        dem_vals = dem['F_i'][:, i]
        print(f"\n  Component {comp}:")
        print(f"    Analytical: min={np.min(ana_vals):12.6e}, max={np.max(ana_vals):12.6e}, mean={np.mean(ana_vals):12.6e}")
        print(f"    DEM:        min={np.min(dem_vals):12.6e}, max={np.max(dem_vals):12.6e}, mean={np.mean(dem_vals):12.6e}")
        print(f"    Max diff:   {np.max(np.abs(ana_vals - dem_vals)):12.6e}")
        print(f"    RMS diff:   {np.sqrt(np.mean((ana_vals - dem_vals)**2)):12.6e}")
    
    print(f"\nFORCE STATISTICS - PARTICLE j:")
    for i, comp in enumerate(components):
        ana_vals = ana['F_j'][:, i]
        dem_vals = dem['F_j'][:, i]
        print(f"\n  Component {comp}:")
        print(f"    Analytical: min={np.min(ana_vals):12.6e}, max={np.max(ana_vals):12.6e}, mean={np.mean(ana_vals):12.6e}")
        print(f"    DEM:        min={np.min(dem_vals):12.6e}, max={np.max(dem_vals):12.6e}, mean={np.mean(dem_vals):12.6e}")
        print(f"    Max diff:   {np.max(np.abs(ana_vals - dem_vals)):12.6e}")
        print(f"    RMS diff:   {np.sqrt(np.mean((ana_vals - dem_vals)**2)):12.6e}")
    
    # Torque statistics
    print(f"\nTORQUE STATISTICS - PARTICLE i:")
    for i, comp in enumerate(components):
        ana_vals = ana['T_i'][:, i]
        dem_vals = dem['T_i'][:, i]
        print(f"\n  Component {comp}:")
        print(f"    Analytical: min={np.min(ana_vals):12.6e}, max={np.max(ana_vals):12.6e}, mean={np.mean(ana_vals):12.6e}")
        print(f"    DEM:        min={np.min(dem_vals):12.6e}, max={np.max(dem_vals):12.6e}, mean={np.mean(dem_vals):12.6e}")
        print(f"    Max diff:   {np.max(np.abs(ana_vals - dem_vals)):12.6e}")
        print(f"    RMS diff:   {np.sqrt(np.mean((ana_vals - dem_vals)**2)):12.6e}")
    
    print(f"\nTORQUE STATISTICS - PARTICLE j:")
    for i, comp in enumerate(components):
        ana_vals = ana['T_j'][:, i]
        dem_vals = dem['T_j'][:, i]
        print(f"\n  Component {comp}:")
        print(f"    Analytical: min={np.min(ana_vals):12.6e}, max={np.max(ana_vals):12.6e}, mean={np.mean(ana_vals):12.6e}")
        print(f"    DEM:        min={np.min(dem_vals):12.6e}, max={np.max(dem_vals):12.6e}, mean={np.mean(dem_vals):12.6e}")
        print(f"    Max diff:   {np.max(np.abs(ana_vals - dem_vals)):12.6e}")
        print(f"    RMS diff:   {np.sqrt(np.mean((ana_vals - dem_vals)**2)):12.6e}")
    
    # ==========================================================================
    # MAGNITUDE COMPARISONS
    # ==========================================================================
    print(f"\n" + "="*80)
    print("MAGNITUDE COMPARISONS")
    print("="*80)
    
    # Force magnitudes
    ana_F_i_mag = np.linalg.norm(ana['F_i'], axis=1)
    dem_F_i_mag = np.linalg.norm(dem['F_i'], axis=1)
    ana_F_j_mag = np.linalg.norm(ana['F_j'], axis=1)
    dem_F_j_mag = np.linalg.norm(dem['F_j'], axis=1)
    
    print(f"\nFORCE MAGNITUDES:")
    print(f"  Particle i:")
    print(f"    Analytical: min={np.min(ana_F_i_mag):12.6e}, max={np.max(ana_F_i_mag):12.6e}")
    print(f"    DEM:        min={np.min(dem_F_i_mag):12.6e}, max={np.max(dem_F_i_mag):12.6e}")
    print(f"    Max diff:   {np.max(np.abs(ana_F_i_mag - dem_F_i_mag)):12.6e}")
    
    print(f"  Particle j:")
    print(f"    Analytical: min={np.min(ana_F_j_mag):12.6e}, max={np.max(ana_F_j_mag):12.6e}")
    print(f"    DEM:        min={np.min(dem_F_j_mag):12.6e}, max={np.max(dem_F_j_mag):12.6e}")
    print(f"    Max diff:   {np.max(np.abs(ana_F_j_mag - dem_F_j_mag)):12.6e}")
    
    # Torque magnitudes
    ana_T_i_mag = np.linalg.norm(ana['T_i'], axis=1)
    dem_T_i_mag = np.linalg.norm(dem['T_i'], axis=1)
    ana_T_j_mag = np.linalg.norm(ana['T_j'], axis=1)
    dem_T_j_mag = np.linalg.norm(dem['T_j'], axis=1)
    
    print(f"\nTORQUE MAGNITUDES:")
    print(f"  Particle i:")
    print(f"    Analytical: min={np.min(ana_T_i_mag):12.6e}, max={np.max(ana_T_i_mag):12.6e}")
    print(f"    DEM:        min={np.min(dem_T_i_mag):12.6e}, max={np.max(dem_T_i_mag):12.6e}")
    print(f"    Max diff:   {np.max(np.abs(ana_T_i_mag - dem_T_i_mag)):12.6e}")
    
    print(f"  Particle j:")
    print(f"    Analytical: min={np.min(ana_T_j_mag):12.6e}, max={np.max(ana_T_j_mag):12.6e}")
    print(f"    DEM:        min={np.min(dem_T_j_mag):12.6e}, max={np.max(dem_T_j_mag):12.6e}")
    print(f"    Max diff:   {np.max(np.abs(ana_T_j_mag - dem_T_j_mag)):12.6e}")
    
    # ==========================================================================
    # SUMMARY AND RECOMMENDATIONS
    # ==========================================================================
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_issues = time_issues + ic_issues
    
    if not all_issues:
        print("\n✓ No critical issues detected")
        print("  Initial conditions match")
        print("  Time synchronization is good")
        print("\n  Ready for detailed comparison with compare_outputs.py")
    else:
        print(f"\n⚠ {len(all_issues)} issue(s) detected:\n")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n  Recommendations:")
        if time_issues:
            print("    - Review time integration setup")
            print("    - Verify time units are consistent")
        if ic_issues:
            print("    - Check initial condition setup")
            print("    - Verify coordinate system conventions")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Quick diagnostic check with full component testing')
    parser.add_argument('testID', type=int,
                       help='Test ID number (e.g., 1 for test_01)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    ana_file = script_dir / '..' / 'output_ANA' / f'theoretical_output_test_{args.testID:02d}.csv'
    dem_file = script_dir / '..' / 'output_DEM' / f'dem_output_YADE_test_{args.testID:02d}.csv'
    
    if not ana_file.exists():
        print(f"ERROR: File not found: {ana_file}")
        return 1
    if not dem_file.exists():
        print(f"ERROR: File not found: {dem_file}")
        return 1
    
    quick_check(ana_file, dem_file, args.testID)
    return 0


if __name__ == '__main__':
    exit(main())
