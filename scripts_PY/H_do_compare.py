import sys
import subprocess
import os
from pathlib import Path
import numpy as np

# H_do_compare.py lives in scripts_PY/ and is software-agnostic
# software-specific piece is scripts_DEM/<software>/G_generate_forces.py plus the
# executable path listed in scripts_DEM/<software>/executable.txt.
SCRIPT_DIR = Path(__file__).resolve().parent     # scripts_PY/
ROOT = SCRIPT_DIR.parent                          # project root
sys.path.append(str(SCRIPT_DIR))

from E_compare_functions import *
from D_helpers import *

# --- Command line arguments: <software> <testID> ---
if len(sys.argv) < 3:
    print("Usage: python H_do_compare.py <software> <testID>")
    print("Example: python H_do_compare.py YADE 1")
    sys.exit(1)

software = sys.argv[1]               # folder name under scripts_DEM/ (e.g. YADE)
testID = int(sys.argv[2])
testname = f'test_{testID:02d}'

# --- Locate the software-specific generator and its executable ---
software_dir = ROOT / 'scripts_DEM' / software
gen_script = 'G_generate_forces.py'          # run from software_dir
exec_file = software_dir / 'executable.txt'

if not software_dir.is_dir():
    print(f"ERROR: software folder not found: {software_dir}")
    sys.exit(1)
if not (software_dir / gen_script).exists():
    print(f"ERROR: {gen_script} not found in {software_dir}")
    sys.exit(1)
if not exec_file.exists():
    print(f"ERROR: executable list not found: {exec_file}")
    sys.exit(1)

def read_executable(path):
    """Return the executable path from *path*: the first non-comment, non-blank
    line. '#' starts a comment (both full-line and trailing), so inactive
    alternatives can be kept alongside the active one."""
    with open(path) as f:
        for raw in f:
            candidate = raw.split('#', 1)[0].strip()
            if candidate:
                return candidate
    raise RuntimeError(f"No active executable path found in {path}")

executable = read_executable(exec_file)

# --- Construct file paths (anchored to the project root, software in the name) ---
output_ana_file = ROOT / 'output_ANA' / f'theoretical_output_{testname}.json'
output_dem_file = ROOT / 'output_DEM' / f'dem_output_{software}_{testname}.csv'
output_report_file = ROOT / 'output_REPORT' / f'comparison_report_{software}_{testname}.txt'

# Create output directories if they don't exist
os.makedirs(ROOT / 'output_DEM', exist_ok=True)
os.makedirs(ROOT / 'output_REPORT', exist_ok=True)

# Run the DEM simulation. cwd is the software folder so that both the executable
# path and G_generate_forces.py's own relative imports resolve as before.
subprocess.run(
    [executable, '-nx', gen_script, str(output_ana_file), str(output_dem_file)],
    cwd=software_dir, check=True,
)

# Load reference (analytical) results
ref = json_to_dict(str(output_ana_file))

# Load DEM results
dem = load_grouped_csv(str(output_dem_file))

# Compare results
report = my_compare_results(dem, ref)

# Write report to file
with open(output_report_file, 'w') as f:
    f.write(f"Comparison Report for {software} {testname}\n")
    f.write("="*60 + "\n\n")
    for key in ['x_i', 'x_j', 'q_i', 'q_j', 'F_i', 'F_j', 'T_i', 'T_j']:
        all_pass = report[key]['all_pass']
        f.write(f"{key:4s}: {'PASS' if all_pass else 'FAIL'}\n")
        if not all_pass:
            failing_steps = np.where(~report[key]['pass'])[0]
            f.write(f"       Failed at steps: {failing_steps.tolist()}\n")

print(report)
