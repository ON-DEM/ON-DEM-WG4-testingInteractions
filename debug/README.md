# DEM vs Analytical Output Comparison Tool

## Overview

This tool compares theoretical (analytical) predictions with YADE DEM simulation results to identify discrepancies and help debug your simulation setup.

## Setup

### Directory Structure
```
project_root/
├── output_ANA/              # Analytical output files
│   ├── theoretical_output_test_01.csv
│   ├── theoretical_output_test_02.csv
│   └── ...
├── output_DEM/              # DEM simulation output files
│   ├── dem_output_YADE_test_01.csv
│   ├── dem_output_YADE_test_02.csv
│   └── ...
└── debug/                   # Working directory
    ├── compare_outputs.py   # This script
    └── comparison_plots/    # Generated plots (created automatically)
```

### Requirements
```bash
pip install numpy matplotlib scipy --break-system-packages
```

## Usage

Navigate to the `debug/` folder and run:

```bash
# For test 1
python compare_outputs.py --testID 1

# For test 2
python compare_outputs.py --testID 2
```

## Output

The script generates 7 comprehensive comparison plots for each test:

### 1. Force Comparison - Particle i (`test_XX_force_i_comparison.png`)
- Side-by-side comparison of force components (x, y, z)
- Left column: Analytical vs DEM overlaid
- Right column: Error (Analytical - DEM)

### 2. Force Comparison - Particle j (`test_XX_force_j_comparison.png`)
- Same as above but for particle j

### 3. Force Magnitude (`test_XX_force_magnitude.png`)
- Compares total force magnitudes
- Shows both particles
- Highlights magnitude differences

### 4. Torque Comparison - Particle i (`test_XX_torque_i_comparison.png`)
- Side-by-side comparison of torque components (x, y, z)
- Error plots for each component

### 5. Torque Comparison - Particle j (`test_XX_torque_j_comparison.png`)
- Same as above but for particle j

### 6. Position and Orientation (`test_XX_position_orientation.png`)
- Position trajectories for both particles
- Quaternion evolution (all 4 components)

### 7. Error Summary (`test_XX_error_summary.png`)
- Total error magnitudes over time
- Useful for identifying when/where errors grow

## Interpreting Results

### Error Statistics
The script prints detailed statistics including:
- Maximum error magnitude
- Mean error magnitude
- RMS (root-mean-square) error
- Component-wise maximum errors

### Common Issues and What to Look For

#### 1. **Constant Offset Errors**
- **Pattern**: Error is constant or slowly varying
- **Likely cause**: 
  - Coordinate system mismatch
  - Sign convention differences
  - Parameter value differences

#### 2. **Growing Errors**
- **Pattern**: Error increases over time
- **Likely cause**:
  - Time integration scheme mismatch
  - Accumulation of numerical errors
  - Quaternion drift (not normalized)

#### 3. **Oscillating Errors**
- **Pattern**: Error oscillates regularly
- **Likely cause**:
  - Time step mismatch
  - Sampling at different phases
  - Frequency mismatch in forcing functions

#### 4. **Sudden Jumps**
- **Pattern**: Error suddenly changes
- **Likely cause**:
  - Contact detection timing differences
  - State transition handling
  - Quaternion branch cuts

#### 5. **Force vs Torque Discrepancy**
- If forces match but torques don't (or vice versa):
  - Check contact point calculation
  - Verify moment arm calculations
  - Check frame transformations

## Key Differences Between Files

### Analytical Output Format
```
t dt x_i x_j v_i v_j q_i q_j omega_i omega_j omega_b n_ij v_ijn l_ij u_n v_s v_r v_theta F_i F_j T_i T_j
```
- Comprehensive kinematics and contact variables
- All vectors are 3D (xyz)
- Quaternions are 4D (xyzw)

### DEM Output Format
```
t x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2 f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
```
- Focused on state variables (positions, orientations, forces, torques)
- May have different time steps than analytical

## Debugging Checklist

### Before Running
- [ ] Verify both output folders exist
- [ ] Check file naming matches convention (test_01, test_02, etc.)
- [ ] Ensure both files have data for the same test

### After Running
- [ ] Check time step alignment (printed in console)
- [ ] Review error statistics for magnitude
- [ ] Examine plots for patterns
- [ ] Compare quaternion normalization
- [ ] Verify coordinate system consistency

### Specific Things to Check in Your Code

1. **Time Integration**
   - Do both methods use the same integration scheme?
   - Are time steps identical?
   - Check integration order (explicit vs implicit)

2. **Quaternions**
   - Same ordering convention? (xyzw vs wxyz)
   - Normalized after each step?
   - Branch cuts handled consistently?

3. **Coordinate Systems**
   - Same handedness (right-handed vs left-handed)?
   - Same axis definitions (z-up vs y-up)?
   - Force/torque directions consistent?

4. **Contact Model**
   - Same stiffness parameters?
   - Same damping coefficients?
   - Friction model identical?
   - Contact detection threshold?

5. **Initial Conditions**
   - Exact same initial positions?
   - Same initial orientations?
   - Same initial velocities?

## Advanced Usage

### Modifying the Script

The script is well-commented and modular. Key functions:

- `load_theoretical_data()`: Parse analytical output
- `load_dem_data()`: Parse DEM output
- `compute_errors()`: Calculate differences (includes interpolation)
- `create_comparison_plots()`: Generate all visualizations
- `print_error_statistics()`: Console output

### Adding Custom Plots

To add new comparisons, add a new figure in `create_comparison_plots()`:

```python
# Example: Add relative error plot
fig_new = plt.figure(figsize=(12, 6))
relative_error = errors['F_i_abs'] / (ana_data['F_i'] + 1e-10)
plt.plot(ana_data['t'], np.linalg.norm(relative_error, axis=1))
plt.xlabel('Time (s)')
plt.ylabel('Relative Error')
plt.title('Force Relative Error')
plt.savefig(os.path.join(output_dir, f'test_{test_id:02d}_custom.png'))
```

## Troubleshooting

### "File not found" error
- Check you're running from the `debug/` folder
- Verify folder structure matches expected layout
- Check test ID formatting (with leading zero)

### Import errors
```bash
pip install numpy matplotlib scipy --break-system-packages
```

### Empty or weird plots
- Check CSV file format hasn't changed
- Verify data is not all zeros
- Check for NaN values in data

### Memory issues
- For very long simulations, add downsampling
- Process data in chunks
- Reduce plot DPI

## Contact & Support

If you encounter issues with the script or need additional analyses, please provide:
1. Error messages (full traceback)
2. Sample of CSV files (first 10 lines)
3. Description of expected vs actual behavior

## Version History

- v1.0: Initial release with 7 comparison plots and comprehensive error analysis
