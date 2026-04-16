#!/usr/bin/env python3
"""
High-quality figure generation script for DEM vs analytical comparison.

Usage:
    python3 I_make_figure.py YADE 1
    python3 I_make_figure.py SOFTWARE_LABEL TEST_ID
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, math
from pathlib import Path

# Import helper functions
sys.path.append('../../functions')
from D_helpers import json_to_dict, load_grouped_csv

# Set high-quality plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.3
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 0.8

# Colors
COLOR_ANA = '#4B0082'  # Dark purple
COLOR_DEM = (0.1020, 0.8000, 0.1020)  # Light green
COLOR_ZERO = 'black'

def load_dem_data(filepath):
    """Load DEM output data from CSV file."""
    data = np.loadtxt(filepath, comments='#')
    
    # New header format:
    # t x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2 
    # v1x v1y v1z v2x v2y v2z w1x w1y w1z w2x w2y w2z 
    # f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
    
    result = {}
    result['t'] = data[:, 0]
    result['x_i'] = data[:, 1:4]      # x1, y1, z1
    result['x_j'] = data[:, 4:7]      # x2, y2, z2
    result['q_i'] = data[:, 7:11]     # qx1, qy1, qz1, qw1
    result['q_j'] = data[:, 11:15]    # qx2, qy2, qz2, qw2
    result['v_i'] = data[:, 15:18]    # v1x, v1y, v1z
    result['v_j'] = data[:, 18:21]    # v2x, v2y, v2z
    result['omega_i'] = data[:, 21:24]  # w1x, w1y, w1z
    result['omega_j'] = data[:, 24:27]  # w2x, w2y, w2z
    result['F_i'] = data[:, 27:30]    # f1x, f1y, f1z
    result['F_j'] = data[:, 30:33]    # f2x, f2y, f2z
    result['T_i'] = data[:, 33:36]    # t1x, t1y, t1z
    result['T_j'] = data[:, 36:39]    # t2x, t2y, t2z
    
    return result

def downsample_data(data, target_points=70):
    """Downsample data to approximately target_points."""
    n = len(data['t'])
    if n <= target_points:
        return data
    
    step = max(1, n // target_points)
    indices = np.arange(0, n, step)
    
    downsampled = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                downsampled[key] = value[indices]
            else:
                downsampled[key] = value[indices, :]
        else:
            downsampled[key] = value
    
    return downsampled


def compute_error_metrics(ana_data, dem_data):
    """
    Compute normalized error metrics using FULL data (no interpolation).
    
    Assumes both datasets have the same time points (or close enough).
    Uses ALL data points for error calculation.
    
    ER_i = sum(abs(x_dem_i - x_ana_i)) / t_max
    N = sum(abs(x_ana_i) + abs(x_ana_j)) / t_max
    MAPE = 100 * ER_i / N
    """
    
    # Ensure we're using the minimum length if they differ
    n_ana = len(ana_data['t'])
    n_dem = len(dem_data['t'])
    
    if n_ana != n_dem:
        print(f"WARNING: Different number of points (Ana: {n_ana}, DEM: {n_dem}). Using minimum.")
        n = min(n_ana, n_dem)
    else:
        n = n_ana
    
    # Handle both 1D and 2D time arrays
    t_ana = ana_data['t']
    if t_ana.ndim > 1:
        t_ana = t_ana.flatten()
    t_max = t_ana[n-1]
    
    metrics = {}
    
    # Compute for each quantity using FULL data
    quantities = {
        'Position': ('x_i', 'x_j'),
        'Velocity': ('v_i', 'v_j'),
        'Orientation': ('q_i', 'q_j'),
        'Angular velocity': ('omega_i', 'omega_j'),
        'Force': ('F_i', 'F_j'),
        'Torque': ('T_i', 'T_j')
    }
    
    for name, (key_i, key_j) in quantities.items():
        # Use full data (first n points)
        ana_i = ana_data[key_i][:n]
        ana_j = ana_data[key_j][:n]
        dem_i = dem_data[key_i][:n]
        dem_j = dem_data[key_j][:n]
        
        # Magnitude of data
        ana_mag_i = np.linalg.norm(ana_i, axis=1)
        ana_mag_j = np.linalg.norm(ana_j, axis=1)
        dem_mag_i = np.linalg.norm(dem_i, axis=1)
        dem_mag_j = np.linalg.norm(dem_j, axis=1)
        
        # Error for particle i and j
        ER_i = np.sum(np.abs(dem_mag_i - ana_mag_i)) / t_max
        ER_j = np.sum(np.abs(dem_mag_j - ana_mag_j)) / t_max
        
        # Normalization (use N=1 if N=0)
        N = np.sum(np.abs(ana_mag_i) + np.abs(ana_mag_j)) / t_max
        N_safe = N if N > 0 else 1.0
        
        # Normalized errors
        ER_NORM_i = ER_i / N_safe
        ER_NORM_j = ER_j / N_safe
        
        metrics[name] = {
            'ER_i': ER_i,
            'ER_j': ER_j,
            'N': N,
            'ER_NORM_i': ER_NORM_i,
            'ER_NORM_j': ER_NORM_j
        }
    
    # Force balance: F_i + F_j should be zero (Newton's 3rd law)
    F_i = dem_data['F_i'][:n]
    F_j = dem_data['F_j'][:n]
    F_balance = F_i + F_j
    F_balance_mag = np.linalg.norm(F_balance, axis=1)
    ER_F_balance = np.sum(F_balance_mag) / t_max
    N_F = metrics['Force']['N']
    N_F_safe = N_F if N_F > 0 else 1.0
    ER_NORM_F_balance = ER_F_balance / N_F_safe
    
    metrics['Force balance'] = {
        'ER': ER_F_balance,
        'ER_NORM': ER_NORM_F_balance,
        'N': N_F
    }
    
    # Torque balance: T_i - T_j should be zero (they use their own centers as reference)
    T_i = dem_data['T_i'][:n]
    T_j = dem_data['T_j'][:n]
    T_balance = T_i - T_j
    T_balance_mag = np.linalg.norm(T_balance, axis=1)
    ER_T_balance = np.sum(T_balance_mag) / t_max
    N_T = metrics['Torque']['N']
    N_T_safe = N_T if N_T > 0 else 1.0
    ER_NORM_T_balance = ER_T_balance / N_T_safe
    
    metrics['Torque balance'] = {
        'ER': ER_T_balance,
        'ER_NORM': ER_NORM_T_balance,
        'N': N_T
    }
    
    return metrics


def sci_str_latex(val, sig=3, thresh=1e-14):
    """
    Return a LaTeX-formatted scientific notation string, e.g. "$1.23\\times10^{-04}$".
    - sig: significant digits (>=1).
    - thresh: absolute-magnitude threshold below which we print "<1.0\\times10^{exp}".
    """
    if math.isnan(val):
        return r"\text{nan}"
    if math.isinf(val):
        return r"\infty" if val > 0 else r"-\infty"

    if abs(val) < thresh:
        # Preserve the machine-precision indicator exactly as requested
        exp_thresh = int(math.log10(thresh))
        return rf"$<1.0\times10^{{{exp_thresh}}}$"

    sign = "-" if val < 0 else ""
    s = f"{abs(val):.{sig}e}"          # e-format using sig significant digits, e.g. "1.234e-04"
    mantissa_str, exp_str = s.split("e")
    mantissa = float(mantissa_str)
    exp = int(exp_str)

    # decimal places = sig-1 because mantissa is in [1,10)
    dec_places = max(sig - 1, 0)
    mantissa_fmt = f"{mantissa:.{dec_places}f}"

    # Handle rare rounding case where mantissa becomes 10.0 after formatting
    if float(mantissa_fmt) >= 10.0:
        mantissa = mantissa / 10.0
        exp += 1
        mantissa_fmt = f"{mantissa:.{dec_places}f}"

    return rf"${sign}{mantissa_fmt}\times10^{{{exp}}}$"


def write_latex_table(metrics, test_id, software_label, output_dir):
    """Write error metrics to LaTeX table."""
    filename = output_dir / f'errors_{software_label}_test_{test_id:02d}.txt'
    
    # Mapping of quantity names to LaTeX vector notation
    quantity_latex = {
        'Position': r'Position $\vec{x}$',
        'Velocity': r'Velocity $\vec{v}$',
        'Orientation': r'Orientation $\vec{q}$',
        'Angular velocity': r'Ang. velocity $\vec{\omega}$',
        'Force': r'Force $\vec{F}$',
        'Torque': r'Torque $\vec{T}$'
    }
    
    with open(filename, 'w') as f:
        f.write(f"% Error metrics for Test {test_id} using {software_label}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Quantity & MAPE$_i$ (\\%) & MAPE$_j$ (\\%) \\\\\n")
        f.write("\\hline\n")
        
        for name in ['Position', 'Velocity', 'Orientation', 'Angular velocity', 'Force', 'Torque']:
            m = metrics[name]
            val_i = m['ER_NORM_i'] * 100
            val_j = m['ER_NORM_j'] * 100
            
            # Format values, using scientific notation for very small numbers
            #str_i = f"{val_i:.3g}" if val_i >= 1e-14 else r"$<10^{-14}$"
            #str_j = f"{val_j:.3g}" if val_j >= 1e-14 else r"$<10^{-14}$"
            str_i = sci_str_latex(val_i, sig=3, thresh=1e-14)
            str_j = sci_str_latex(val_j, sig=3, thresh=1e-14)
            
            f.write(f"{quantity_latex[name]} & {str_i} & {str_j} \\\\\n")
        
        f.write("\\hline\n")
        
        # Force and torque balance
        val_F = metrics['Force balance']['ER_NORM'] * 100
        val_T = metrics['Torque balance']['ER_NORM'] * 100
        #str_F = f"{val_F:.3g}" if val_F >= 1e-14 else r"$<10^{-14}$"
        #str_T = f"{val_T:.3g}" if val_T >= 1e-14 else r"$<10^{-14}$"
        str_F = sci_str_latex(val_F, sig=3, thresh=1e-14)
        str_T = sci_str_latex(val_T, sig=3, thresh=1e-14)
        
        f.write(f"Force imbalance & \\multicolumn{{2}}{{c}}{{{str_F}}} \\\\\n")
        f.write(f"Torque imbalance & \\multicolumn{{2}}{{c}}{{{str_T}}} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    
    print(f"✓ LaTeX table written to {filename}")


def plot_test_x(ana_data, dem_data_ds, test_id, output_dir, software_label):
    """Plot x force component vs time for Tests 1 and 2."""
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    
    # Handle both 1D and 2D time arrays
    t_ana = ana_data['t']
    if t_ana.ndim > 1:
        t_ana = t_ana.flatten()
    t_dem = dem_data_ds['t']
    if t_dem.ndim > 1:
        t_dem = t_dem.flatten()
    
    # Plot analytical data
    ax.plot(t_ana, ana_data['F_i'][:, 0], 
           color=COLOR_ANA, linewidth=1.5, zorder=1)
    
    # Plot DEM data (downsampled)
    ax.plot(t_dem, dem_data_ds['F_i'][:, 0], 
           'o', color=COLOR_DEM, markersize=3, markerfacecolor=COLOR_DEM, 
           markeredgewidth=0.3, markeredgecolor='black', zorder=2)
    
    # Zero line
    ax.axhline(0, color=COLOR_ZERO, linewidth=0.5, zorder=0)
    
    # Grid
    ax.grid(True, alpha=0.15, linewidth=0.3)
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$F_{i,x}$ (N)')
    ax.set_xlim(0, t_ana[-1])
    
    # Use scientific notation if values exceed 1000
    max_val = max(abs(ana_data['F_i'][:, 0].max()), abs(ana_data['F_i'][:, 0].min()))
    if max_val > 1000:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'figure_{software_label}_test_{test_id:02d}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'figure_{software_label}_test_{test_id:02d}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved for Test {test_id}")


def plot_test_3d(ana_data, dem_data_ds, test_id, comp1, comp2, output_dir, software_label):
    """Plot 3D trajectory of force components."""
    import matplotlib.ticker as ticker
    
    fig = plt.figure(figsize=(3.5, 3.0))
    ax = fig.add_subplot(111, projection='3d')
    
    # Handle both 1D and 2D time arrays
    t_ana = ana_data['t']
    if t_ana.ndim > 1:
        t_ana = t_ana.flatten()
    t_dem = dem_data_ds['t']
    if t_dem.ndim > 1:
        t_dem = t_dem.flatten()
    
    # Get component indices
    comp_map = {'x': 0, 'y': 1, 'z': 2}
    idx1 = comp_map[comp1]
    idx2 = comp_map[comp2]
    
    # Plot analytical data
    ax.plot(t_ana, ana_data['F_i'][:, idx1], ana_data['F_i'][:, idx2],
           color=COLOR_ANA, linewidth=1.5, zorder=1)
    
    # Plot DEM data (downsampled)
    ax.scatter(t_dem, dem_data_ds['F_i'][:, idx1], dem_data_ds['F_i'][:, idx2],
              c=[COLOR_DEM], s=20, edgecolors='black', linewidths=0.3, zorder=2)
    
    # Grid
    ax.grid(True, alpha=0.15, linewidth=0.3)
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'$F_{{i,{comp1}}}$ (N)')
    ax.set_zlabel(f'$F_{{i,{comp2}}}$ (N)')
    
    # Set time axis limits tightly
    ax.set_xlim(0, t_ana[-1])
    
    # Set force axis limits to round numbers that fit the data + 10% margin
    def get_nice_limits(data):
        """Get nice round limits for the axis with 10% margin."""
        import math
        data_min = data.min()
        data_max = data.max()
        
        # Get order of magnitude
        abs_max = max(abs(data_min), abs(data_max))
        if abs_max == 0:
            magnitude = 1
        else:
            magnitude = 10 ** math.floor(math.log10(abs_max))
        
        # Find nice step size
        nice_numbers = [1, 2, 5, 10]
        extended_range = data_max - data_min
        step = magnitude
        for nice in nice_numbers:
            test_step = nice * magnitude
            if extended_range / test_step < 10:
                step = test_step
                break
        
        # Round limits to step
        limit_min = math.floor(data_min / step) * step
        limit_max = math.ceil(data_max / step) * step
        
        return limit_min, limit_max
    
    # Component 1 limits (with 10% margin)
    data1 = np.concatenate([ana_data['F_i'][:, idx1], dem_data_ds['F_i'][:, idx1]])
    ylim_min, ylim_max = get_nice_limits(data1)
    ax.set_ylim(ylim_min, ylim_max)
    
    # Component 2 limits (with 10% margin)
    data2 = np.concatenate([ana_data['F_i'][:, idx2], dem_data_ds['F_i'][:, idx2]])
    zlim_min, zlim_max = get_nice_limits(data2)
    ax.set_zlim(zlim_min, zlim_max)
    
    # Use scientific notation on axes if values exceed 1000
    max_val1 = max(abs(ylim_min), abs(ylim_max))
    max_val2 = max(abs(zlim_min), abs(zlim_max))
    
    if max_val1 > 1000:
        # Scientific notation on the axis itself
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # Remove scientific notation from axis label
        ax.set_ylabel(f'$F_{{i,{comp1}}}$ (N)')
    
    if max_val2 > 1000:
        # Scientific notation on the axis itself
        ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
        # Remove scientific notation from axis label
        ax.set_zlabel(f'$F_{{i,{comp2}}}$ (N)')
    
    # Set box appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.savefig(output_dir / f'figure_{software_label}_test_{test_id:02d}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'figure_{software_label}_test_{test_id:02d}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Figure saved for Test {test_id}")


def main():
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python3 make_figure_pub.py SOFTWARE_LABEL TEST_ID")
        print("Example: python3 make_figure_pub.py YADE 5")
        sys.exit(1)
    
    software_label = sys.argv[1]
    test_id = int(sys.argv[2])
    
    # Set up paths
    script_dir = Path(__file__).parent
    ana_file = script_dir / '..' / 'output_ANA' / f'theoretical_output_test_{test_id:02d}.json'
    dem_file = script_dir / '..' / 'output_DEM' / f'dem_output_{software_label}_test_{test_id:02d}.csv'
    report_dir = script_dir / '..' / 'output_REPORT'
    output_dir = script_dir / '..' / 'figures' 
    
    # Check files exist
    if not ana_file.exists():
        print(f"ERROR: Analytical file not found: {ana_file}")
        sys.exit(1)
    if not dem_file.exists():
        print(f"ERROR: DEM file not found: {dem_file}")
        sys.exit(1)
    
    print(f"\nProcessing Test {test_id} with {software_label}")
    print(f"  Analytical: {ana_file.name}")
    print(f"  DEM:        {dem_file.name}")
    
    # Load data using helper functions
    ana_data = json_to_dict(str(ana_file))
    dem_data = load_dem_data(str(dem_file))
    
    print(f"✓ Loaded {len(ana_data['t'])} analytical points")
    print(f"✓ Loaded {len(dem_data['t'])} DEM points")
    
    # Downsample DEM data
    dem_data_ds = downsample_data(dem_data, target_points=70)
    print(f"✓ Downsampled DEM to {len(dem_data_ds['t'])} points")
    
    # Compute error metrics
    print("\nComputing error metrics...")
    metrics = compute_error_metrics(ana_data, dem_data)
    
    # Write LaTeX table
    write_latex_table(metrics, test_id, software_label, report_dir)
    
    # Create figure based on test ID
    print("\nGenerating high-quality figure...")
    if test_id in [1, 2]:
        plot_test_x(ana_data, dem_data_ds, test_id, output_dir, software_label)
    elif test_id == 3:
        plot_test_3d(ana_data, dem_data_ds, test_id, 'y', 'z', output_dir, software_label)
    elif test_id == 4:
        plot_test_3d(ana_data, dem_data_ds, test_id, 'x', 'y', output_dir, software_label)
    elif test_id == 5:
        plot_test_3d(ana_data, dem_data_ds, test_id, 'x', 'z', output_dir, software_label)
    elif test_id in [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        plot_test_x(ana_data, dem_data_ds, test_id, output_dir, software_label)
    else:
        print(f"WARNING: No figure specification for Test {test_id}")
    
    print(f"\n{'='*70}")
    print("SUMMARY OF ERROR METRICS")
    print('='*70)
    for name in ['Position', 'Velocity', 'Orientation', 'Angular velocity', 'Force', 'Torque']:
        m = metrics[name]
        print(f"\n{name}:")
        print(f"  Particle i: ER_NORM = {m['ER_NORM_i']:.4e}")
        print(f"  Particle j: ER_NORM = {m['ER_NORM_j']:.4e}")
    
    print(f"\nBalance checks:")
    print(f"  Force balance:  ER_NORM = {metrics['Force balance']['ER_NORM']:.4e}")
    print(f"  Torque balance: ER_NORM = {metrics['Torque balance']['ER_NORM']:.4e}")
    print('='*70 + '\n')


if __name__ == '__main__':
    main()