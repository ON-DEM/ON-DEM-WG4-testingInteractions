import sys
import subprocess
import os
import numpy as np

sys.path.append('../../scripts_PY')

from E_compare_functions import *
from D_helpers import *

# Get testID from command line argument
if len(sys.argv) < 2:
    print("Usage: python compare.py <testID>")
    print("Example: python compare.py 1")
    sys.exit(1)

testID = int(sys.argv[1])
testname = f'test_{testID:02d}'
#yadePath = 'YADE'  # Adjust this path if YADE is located elsewhere
#yadePath = '../../../../LS-DEM-dev/install-dev/bin/yade-2026-01-26.git-4c8f5a2'
yadePath = '../../../../LS-DEM-dev/install-dev/bin/yade-2026-04-16.git-28d91cd'

# Construct file paths
output_ana_file = f'../../output_ANA/theoretical_output_{testname}.json'
output_dem_file = f'../../output_DEM/dem_output_YADE_{testname}.csv'
output_report_file = f'../../output_REPORT/comparison_report_YADE_{testname}.txt'

# Create output directories if they don't exist
os.makedirs('../../output_DEM', exist_ok=True)
os.makedirs('../../output_REPORT', exist_ok=True)

# Run YADE simulation with input file
subprocess.run([yadePath,'-nx', 'G_generate_forces.py', output_ana_file, output_dem_file], check=True)

# Load reference (analytical) results
ref = json_to_dict(output_ana_file)

# Load DEM results
dem = load_grouped_csv(output_dem_file)

# Compare results
report = my_compare_results(dem, ref)

# Write report to file
with open(output_report_file, 'w') as f:
    f.write(f"Comparison Report for {testname}\n")
    f.write("="*60 + "\n\n")
    for key in ['x_i', 'x_j', 'q_i', 'q_j', 'F_i', 'F_j', 'T_i', 'T_j']:
        all_pass = report[key]['all_pass']
        f.write(f"{key:4s}: {'PASS' if all_pass else 'FAIL'}\n")
        if not all_pass:
            failing_steps = np.where(~report[key]['pass'])[0]
            f.write(f"       Failed at steps: {failing_steps.tolist()}\n")

print(report)