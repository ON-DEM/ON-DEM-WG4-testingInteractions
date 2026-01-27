#!/usr/bin/env python3
"""
Script to compare theoretical (analytical) and DEM simulation outputs.

This script loads data from both output folders and creates comprehensive
visualizations to identify differences between theoretical predictions and
DEM simulation results.

Usage:
    python compare_outputs.py --testID 1
    python compare_outputs.py --testID 2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
from pathlib import Path


def load_theoretical_data(filepath):
    """
    Load theoretical output data from CSV file.
    
    Returns a dictionary with parsed columns and metadata.
    """
    # Read header line to get column names
    with open(filepath, 'r') as f:
        header = f.readline().strip('# \n').split()
    
    # Load data
    data = np.loadtxt(filepath, comments='#')
    
    # Parse the data according to the header structure
    # Header: t dt x_i x_j v_i v_j q_i q_j omega_i omega_j omega_b n_ij v_ijn l_ij u_n v_s v_r v_theta F_i F_j T_i T_j
    # Each vector quantity (x, v, q, omega, etc.) has 3 components
    
    col_idx = 0
    result = {}
    
    # Time variables (scalars)
    result['t'] = data[:, col_idx]; col_idx += 1
    result['dt'] = data[:, col_idx]; col_idx += 1
    
    # Position vectors (3D each)
    result['x_i'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['x_j'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Velocity vectors (3D each)
    result['v_i'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['v_j'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Quaternions (4D each, format: x,y,z,w)
    result['q_i'] = data[:, col_idx:col_idx+4]; col_idx += 4
    result['q_j'] = data[:, col_idx:col_idx+4]; col_idx += 4
    
    # Angular velocity vectors (3D each)
    result['omega_i'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['omega_j'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Branch vector omega_b (3D)
    result['omega_b'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Contact normal (3D)
    result['n_ij'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Relative velocity at contact (3D)
    result['v_ijn'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Contact point (3D)
    result['l_ij'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Normal overlap (scalar)
    result['u_n'] = data[:, col_idx]; col_idx += 1
    
    # Sliding velocity, rolling velocity, twisting velocity (3D each)
    result['v_s'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['v_r'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['v_theta'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Forces (3D each)
    result['F_i'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['F_j'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    # Torques (3D each)
    result['T_i'] = data[:, col_idx:col_idx+3]; col_idx += 3
    result['T_j'] = data[:, col_idx:col_idx+3]; col_idx += 3
    
    return result


def load_dem_data(filepath):
    """
    Load DEM output data from CSV file.
    
    Returns a dictionary with parsed columns.
    """
    # Read header line to get column names
    with open(filepath, 'r') as f:
        header = f.readline().strip('# \n').split()
    
    # Load data
    data = np.loadtxt(filepath, comments='#')
    
    # Parse the DEM data
    # Header: t x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2 f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
    
    result = {}
    result['t'] = data[:, 0]
    result['x_i'] = data[:, 1:4]  # x1, y1, z1
    result['x_j'] = data[:, 4:7]  # x2, y2, z2
    result['q_i'] = data[:, 7:11]  # qx1, qy1, qz1, qw1
    result['q_j'] = data[:, 11:15]  # qx2, qy2, qz2, qw2
    result['F_i'] = data[:, 15:18]  # f1x, f1y, f1z
    result['F_j'] = data[:, 18:21]  # f2x, f2y, f2z
    result['T_i'] = data[:, 21:24]  # t1x, t1y, t1z
    result['T_j'] = data[:, 24:27]  # t2x, t2y, t2z
    
    return result


def compute_errors(ana_data, dem_data):
    """
    Compute differences between analytical and DEM data.
    
    Returns a dictionary of error metrics.
    """
    errors = {}
    
    # Interpolate DEM data to match analytical time steps if needed
    # (DEM might have different time steps)
    from scipy.interpolate import interp1d
    
    # For each vector quantity, interpolate DEM to analytical time
    dem_interp = {}
    
    for key in ['x_i', 'x_j', 'q_i', 'q_j', 'F_i', 'F_j', 'T_i', 'T_j']:
        if dem_data[key].shape[0] > 1:
            # Create interpolation functions for each component
            interp_funcs = []
            for i in range(dem_data[key].shape[1]):
                # Use linear interpolation, extrapolate at boundaries
                f = interp1d(dem_data['t'], dem_data[key][:, i], 
                            kind='linear', bounds_error=False, 
                            fill_value='extrapolate')
                interp_funcs.append(f)
            
            # Evaluate at analytical time points
            dem_interp[key] = np.column_stack([f(ana_data['t']) for f in interp_funcs])
        else:
            # Single point, just repeat
            dem_interp[key] = np.tile(dem_data[key], (len(ana_data['t']), 1))
    
    # Compute absolute and relative errors
    for key in ['x_i', 'x_j', 'F_i', 'F_j', 'T_i', 'T_j']:
        errors[f'{key}_abs'] = ana_data[key] - dem_interp[key]
        errors[f'{key}_mag_ana'] = np.linalg.norm(ana_data[key], axis=1)
        errors[f'{key}_mag_dem'] = np.linalg.norm(dem_interp[key], axis=1)
        errors[f'{key}_mag_diff'] = errors[f'{key}_mag_ana'] - errors[f'{key}_mag_dem']
    
    # Special handling for quaternions (need to consider quaternion distance)
    for key in ['q_i', 'q_j']:
        # Simple difference (note: proper quaternion distance would be better)
        errors[f'{key}_abs'] = ana_data[key] - dem_interp[key]
    
    errors['dem_interp'] = dem_interp
    
    return errors


def create_comparison_plots(ana_data, dem_data, errors, output_dir, test_id):
    """
    Create comprehensive comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up nice plotting style
    plt.style.use('fivethirtyeight')
    colors_ana = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    colors_dem = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown
    
    # ========================================================================
    # PLOT 1: Force components comparison (particle i)
    # ========================================================================
    fig1 = plt.figure(figsize=(16, 10))
    gs1 = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.3)
    
    components = ['x', 'y', 'z']
    
    for idx, comp in enumerate(components):
        # Analytical vs DEM
        ax = fig1.add_subplot(gs1[idx, 0])
        ax.plot(ana_data['t'], ana_data['F_i'][:, idx], 
               label=f'Analytical', color=colors_ana[idx], linewidth=2)
        ax.plot(dem_data['t'], dem_data['F_i'][:, idx], 
               label=f'DEM', color=colors_dem[idx], linewidth=2, linestyle='--', marker='o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Force {comp} (N)')
        ax.set_title(f'Particle i: Force {comp}-component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error plot
        ax = fig1.add_subplot(gs1[idx, 1])
        ax.plot(ana_data['t'], errors['F_i_abs'][:, idx], 
               color='red', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error (N)')
        ax.set_title(f'Particle i: Force {comp}-component Error (Ana - DEM)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig1.suptitle(f'Test {test_id:02d}: Force Comparison - Particle i', 
                  fontsize=16, fontweight='bold')
    fig1.savefig(os.path.join(output_dir, f'test_{test_id:02d}_force_i_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # ========================================================================
    # PLOT 2: Force components comparison (particle j)
    # ========================================================================
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = GridSpec(3, 2, figure=fig2, hspace=0.3, wspace=0.3)
    
    for idx, comp in enumerate(components):
        # Analytical vs DEM
        ax = fig2.add_subplot(gs2[idx, 0])
        ax.plot(ana_data['t'], ana_data['F_j'][:, idx], 
               label=f'Analytical', color=colors_ana[idx], linewidth=2)
        ax.plot(dem_data['t'], dem_data['F_j'][:, idx], 
               label=f'DEM', color=colors_dem[idx], linewidth=2, linestyle='--', marker='o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Force {comp} (N)')
        ax.set_title(f'Particle j: Force {comp}-component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error plot
        ax = fig2.add_subplot(gs2[idx, 1])
        ax.plot(ana_data['t'], errors['F_j_abs'][:, idx], 
               color='red', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error (N)')
        ax.set_title(f'Particle j: Force {comp}-component Error (Ana - DEM)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig2.suptitle(f'Test {test_id:02d}: Force Comparison - Particle j', 
                  fontsize=16, fontweight='bold')
    fig2.savefig(os.path.join(output_dir, f'test_{test_id:02d}_force_j_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # ========================================================================
    # PLOT 3: Force magnitude comparison
    # ========================================================================
    fig3 = plt.figure(figsize=(16, 8))
    gs3 = GridSpec(2, 2, figure=fig3, hspace=0.3, wspace=0.3)
    
    # Particle i - magnitude
    ax = fig3.add_subplot(gs3[0, 0])
    ax.plot(ana_data['t'], errors['F_i_mag_ana'], 
           label='Analytical', color='blue', linewidth=2)
    ax.plot(dem_data['t'], errors['F_i_mag_dem'][::len(dem_data['t'])//len(ana_data['t']) if len(dem_data['t']) > len(ana_data['t']) else 1], 
           label='DEM', color='red', linewidth=2, linestyle='--', marker='o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Magnitude (N)')
    ax.set_title('Particle i: Force Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Particle i - magnitude error
    ax = fig3.add_subplot(gs3[0, 1])
    ax.plot(ana_data['t'], errors['F_i_mag_diff'], 
           color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude Error (N)')
    ax.set_title('Particle i: Force Magnitude Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Particle j - magnitude
    ax = fig3.add_subplot(gs3[1, 0])
    ax.plot(ana_data['t'], errors['F_j_mag_ana'], 
           label='Analytical', color='blue', linewidth=2)
    ax.plot(dem_data['t'], errors['F_j_mag_dem'][::len(dem_data['t'])//len(ana_data['t']) if len(dem_data['t']) > len(ana_data['t']) else 1], 
           label='DEM', color='red', linewidth=2, linestyle='--', marker='o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Magnitude (N)')
    ax.set_title('Particle j: Force Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Particle j - magnitude error
    ax = fig3.add_subplot(gs3[1, 1])
    ax.plot(ana_data['t'], errors['F_j_mag_diff'], 
           color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Magnitude Error (N)')
    ax.set_title('Particle j: Force Magnitude Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig3.suptitle(f'Test {test_id:02d}: Force Magnitude Comparison', 
                  fontsize=16, fontweight='bold')
    fig3.savefig(os.path.join(output_dir, f'test_{test_id:02d}_force_magnitude.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # ========================================================================
    # PLOT 4: Torque components comparison (particle i)
    # ========================================================================
    fig4 = plt.figure(figsize=(16, 10))
    gs4 = GridSpec(3, 2, figure=fig4, hspace=0.3, wspace=0.3)
    
    for idx, comp in enumerate(components):
        # Analytical vs DEM
        ax = fig4.add_subplot(gs4[idx, 0])
        ax.plot(ana_data['t'], ana_data['T_i'][:, idx], 
               label=f'Analytical', color=colors_ana[idx], linewidth=2)
        ax.plot(dem_data['t'], dem_data['T_i'][:, idx], 
               label=f'DEM', color=colors_dem[idx], linewidth=2, linestyle='--', marker='o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Torque {comp} (N·m)')
        ax.set_title(f'Particle i: Torque {comp}-component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error plot
        ax = fig4.add_subplot(gs4[idx, 1])
        ax.plot(ana_data['t'], errors['T_i_abs'][:, idx], 
               color='red', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error (N·m)')
        ax.set_title(f'Particle i: Torque {comp}-component Error (Ana - DEM)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig4.suptitle(f'Test {test_id:02d}: Torque Comparison - Particle i', 
                  fontsize=16, fontweight='bold')
    fig4.savefig(os.path.join(output_dir, f'test_{test_id:02d}_torque_i_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # ========================================================================
    # PLOT 5: Torque components comparison (particle j)
    # ========================================================================
    fig5 = plt.figure(figsize=(16, 10))
    gs5 = GridSpec(3, 2, figure=fig5, hspace=0.3, wspace=0.3)
    
    for idx, comp in enumerate(components):
        # Analytical vs DEM
        ax = fig5.add_subplot(gs5[idx, 0])
        ax.plot(ana_data['t'], ana_data['T_j'][:, idx], 
               label=f'Analytical', color=colors_ana[idx], linewidth=2)
        ax.plot(dem_data['t'], dem_data['T_j'][:, idx], 
               label=f'DEM', color=colors_dem[idx], linewidth=2, linestyle='--', marker='o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Torque {comp} (N·m)')
        ax.set_title(f'Particle j: Torque {comp}-component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error plot
        ax = fig5.add_subplot(gs5[idx, 1])
        ax.plot(ana_data['t'], errors['T_j_abs'][:, idx], 
               color='red', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Error (N·m)')
        ax.set_title(f'Particle j: Torque {comp}-component Error (Ana - DEM)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig5.suptitle(f'Test {test_id:02d}: Torque Comparison - Particle j', 
                  fontsize=16, fontweight='bold')
    fig5.savefig(os.path.join(output_dir, f'test_{test_id:02d}_torque_j_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    # ========================================================================
    # PLOT 6: Position and Quaternion comparison
    # ========================================================================
    fig6 = plt.figure(figsize=(16, 12))
    gs6 = GridSpec(4, 2, figure=fig6, hspace=0.3, wspace=0.3)
    
    # Position particle i
    for idx, comp in enumerate(components):
        ax = fig6.add_subplot(gs6[idx, 0])
        ax.plot(ana_data['t'], ana_data['x_i'][:, idx], 
               label='Analytical', color=colors_ana[idx], linewidth=2)
        ax.plot(dem_data['t'], dem_data['x_i'][:, idx], 
               label='DEM', color=colors_dem[idx], linewidth=2, linestyle='--', marker='o', markersize=3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Position {comp} (m)')
        ax.set_title(f'Particle i: Position {comp}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Position particle j (z-component only, as x and y should be zero)
    ax = fig6.add_subplot(gs6[3, 0])
    ax.plot(ana_data['t'], ana_data['x_j'][:, 2], 
           label='Analytical', color=colors_ana[2], linewidth=2)
    ax.plot(dem_data['t'], dem_data['x_j'][:, 2], 
           label='DEM', color=colors_dem[2], linewidth=2, linestyle='--', marker='o', markersize=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position z (m)')
    ax.set_title('Particle j: Position z')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Quaternion particle i (show all 4 components)
    ax = fig6.add_subplot(gs6[0:2, 1])
    for i in range(4):
        ax.plot(ana_data['t'], ana_data['q_i'][:, i], 
               label=f'Ana q{i}', linewidth=2)
    for i in range(4):
        ax.plot(dem_data['t'], dem_data['q_i'][:, i], 
               label=f'DEM q{i}', linewidth=1.5, linestyle='--', marker='o', markersize=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion components')
    ax.set_title('Particle i: Quaternion')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Quaternion particle j
    ax = fig6.add_subplot(gs6[2:4, 1])
    for i in range(4):
        ax.plot(ana_data['t'], ana_data['q_j'][:, i], 
               label=f'Ana q{i}', linewidth=2)
    for i in range(4):
        ax.plot(dem_data['t'], dem_data['q_j'][:, i], 
               label=f'DEM q{i}', linewidth=1.5, linestyle='--', marker='o', markersize=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Quaternion components')
    ax.set_title('Particle j: Quaternion')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig6.suptitle(f'Test {test_id:02d}: Position and Orientation Comparison', 
                  fontsize=16, fontweight='bold')
    fig6.savefig(os.path.join(output_dir, f'test_{test_id:02d}_position_orientation.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig6)
    
    # ========================================================================
    # PLOT 7: Error summary plot
    # ========================================================================
    fig7 = plt.figure(figsize=(16, 10))
    gs7 = GridSpec(2, 2, figure=fig7, hspace=0.3, wspace=0.3)
    
    # Total force error magnitude (particle i)
    ax = fig7.add_subplot(gs7[0, 0])
    error_mag_i = np.linalg.norm(errors['F_i_abs'], axis=1)
    ax.plot(ana_data['t'], error_mag_i, color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Error Magnitude (N)')
    ax.set_title('Particle i: Total Force Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Total force error magnitude (particle j)
    ax = fig7.add_subplot(gs7[0, 1])
    error_mag_j = np.linalg.norm(errors['F_j_abs'], axis=1)
    ax.plot(ana_data['t'], error_mag_j, color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force Error Magnitude (N)')
    ax.set_title('Particle j: Total Force Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Total torque error magnitude (particle i)
    ax = fig7.add_subplot(gs7[1, 0])
    error_mag_ti = np.linalg.norm(errors['T_i_abs'], axis=1)
    ax.plot(ana_data['t'], error_mag_ti, color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque Error Magnitude (N·m)')
    ax.set_title('Particle i: Total Torque Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Total torque error magnitude (particle j)
    ax = fig7.add_subplot(gs7[1, 1])
    error_mag_tj = np.linalg.norm(errors['T_j_abs'], axis=1)
    ax.plot(ana_data['t'], error_mag_tj, color='red', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque Error Magnitude (N·m)')
    ax.set_title('Particle j: Total Torque Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    fig7.suptitle(f'Test {test_id:02d}: Error Summary', 
                  fontsize=16, fontweight='bold')
    fig7.savefig(os.path.join(output_dir, f'test_{test_id:02d}_error_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig7)
    
    print(f"✓ All plots saved to {output_dir}/")


def print_error_statistics(errors, test_id):
    """
    Print statistical summary of errors.
    """
    print("\n" + "="*70)
    print(f"ERROR STATISTICS FOR TEST {test_id:02d}")
    print("="*70)
    
    # Force errors
    print("\nFORCE ERRORS:")
    print("-" * 70)
    for particle in ['i', 'j']:
        key = f'F_{particle}_abs'
        error_mag = np.linalg.norm(errors[key], axis=1)
        print(f"\nParticle {particle}:")
        print(f"  Max error magnitude:  {np.max(error_mag):.6e} N")
        print(f"  Mean error magnitude: {np.mean(error_mag):.6e} N")
        print(f"  RMS error:            {np.sqrt(np.mean(error_mag**2)):.6e} N")
        print(f"  Max component errors: x={np.max(np.abs(errors[key][:, 0])):.6e} N, "
              f"y={np.max(np.abs(errors[key][:, 1])):.6e} N, "
              f"z={np.max(np.abs(errors[key][:, 2])):.6e} N")
    
    # Torque errors
    print("\nTORQUE ERRORS:")
    print("-" * 70)
    for particle in ['i', 'j']:
        key = f'T_{particle}_abs'
        error_mag = np.linalg.norm(errors[key], axis=1)
        print(f"\nParticle {particle}:")
        print(f"  Max error magnitude:  {np.max(error_mag):.6e} N·m")
        print(f"  Mean error magnitude: {np.mean(error_mag):.6e} N·m")
        print(f"  RMS error:            {np.sqrt(np.mean(error_mag**2)):.6e} N·m")
        print(f"  Max component errors: x={np.max(np.abs(errors[key][:, 0])):.6e} N·m, "
              f"y={np.max(np.abs(errors[key][:, 1])):.6e} N·m, "
              f"z={np.max(np.abs(errors[key][:, 2])):.6e} N·m")
    
    print("\n" + "="*70)


def main():
    """
    Main function to run the comparison.
    """
    parser = argparse.ArgumentParser(
        description='Compare theoretical and DEM simulation outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_outputs.py --testID 1
  python compare_outputs.py --testID 2
        """
    )
    parser.add_argument('--testID', type=int, required=True,
                       help='Test ID number (e.g., 1 for test_01)')
    
    args = parser.parse_args()
    test_id = args.testID
    
    # Construct file paths
    # Script is run from debug/ folder
    # Output folders are siblings: ../output_ANA and ../output_DEM
    script_dir = Path(__file__).parent
    ana_file = script_dir / '..' / 'output_ANA' / f'theoretical_output_test_{test_id:02d}.csv'
    dem_file = script_dir / '..' / 'output_DEM' / f'dem_output_YADE_test_{test_id:02d}.csv'
    output_dir = script_dir / 'comparison_plots'
    
    # Check if files exist
    if not ana_file.exists():
        print(f"ERROR: Analytical output file not found: {ana_file}")
        return 1
    if not dem_file.exists():
        print(f"ERROR: DEM output file not found: {dem_file}")
        return 1
    
    print(f"\nLoading test {test_id:02d} data...")
    print(f"  Analytical: {ana_file}")
    print(f"  DEM:        {dem_file}")
    
    # Load data
    ana_data = load_theoretical_data(str(ana_file))
    dem_data = load_dem_data(str(dem_file))
    
    print(f"✓ Loaded {len(ana_data['t'])} analytical time steps")
    print(f"✓ Loaded {len(dem_data['t'])} DEM time steps")
    
    # Compute errors
    print("\nComputing errors...")
    errors = compute_errors(ana_data, dem_data)
    print("✓ Error analysis complete")
    
    # Print statistics
    print_error_statistics(errors, test_id)
    
    # Create plots
    print(f"\nGenerating comparison plots...")
    create_comparison_plots(ana_data, dem_data, errors, str(output_dir), test_id)
    
    print(f"\n{'='*70}")
    print("DEBUGGING SUGGESTIONS:")
    print("="*70)
    print("\n1. TIME SYNCHRONIZATION:")
    print("   - Check if DEM and analytical use the same time steps")
    print("   - Verify time integration scheme matches between methods")
    
    print("\n2. COORDINATE SYSTEMS:")
    print("   - Verify both systems use the same coordinate conventions")
    print("   - Check sign conventions for forces and torques")
    
    print("\n3. QUATERNION CONVENTIONS:")
    print("   - Verify quaternion ordering (xyzw vs wxyz)")
    print("   - Check if quaternions are normalized")
    
    print("\n4. CONTACT DETECTION:")
    print("   - Verify overlap calculation methods match")
    print("   - Check contact point calculation")
    
    print("\n5. FORCE MODELS:")
    print("   - Compare stiffness parameters")
    print("   - Verify damping coefficients")
    print("   - Check friction coefficient values")
    
    print("\n6. LOOK FOR PATTERNS IN PLOTS:")
    print("   - Constant offsets suggest systematic errors")
    print("   - Growing errors suggest integration issues")
    print("   - Oscillating errors suggest time step mismatches")
    
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
