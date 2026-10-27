# WG4-testingInteractions

A framework for testing discrete-element-method (DEM) contact-interaction
implementations against **analytical reference solutions**. For each test case an
analytical trajectory and the corresponding contact forces/torques are generated,
a DEM code is driven through the same prescribed kinematics, and the two are
compared and plotted.

## Pipeline

The workflow is a four-stage pipeline. Scripts are named `A`–`J` in execution
order; the stages that matter for running the tests are `F`, `G`, `H` and `I`,
tied together by `J`:

| Stage | Script | Location | Role |
|-------|--------|----------|------|
| 1 | `F_generate_analytical.py` | `scripts_PY/` | Generate the analytical reference data for a test |
| 2 | `H_do_compare.py` | `scripts_PY/` | Run the DEM simulation and compare it to the reference |
| — | `G_generate_forces.py` | `scripts_DEM/<software>/` | **Software-specific** DEM driver, launched by `H` |
| 3 | `I_make_figure.py` | `scripts_PY/` | Produce publication figures and LaTeX error tables |

`G_generate_forces.py` is the **only** software-specific script. Everything else
is generic and lives in `scripts_PY/`.

### Orchestrator — `J_run_all.py`

`scripts_PY/J_run_all.py` runs the whole pipeline (stages 1→3) for any selection
of tests and DEM softwares:

```bash
cd scripts_PY

python J_run_all.py                   # all configured tests, all softwares
python J_run_all.py --tests 1 2 3     # only tests 1, 2, 3
python J_run_all.py --softwares YADE  # restrict to one software
python J_run_all.py --skipana         # reuse existing analytical data (skip stage 1)
python J_run_all.py --skipdem         # reuse existing DEM results (skip stage 2)
python J_run_all.py --err             # generate the faulty analytical cases (diagnostics)
```

The configured defaults (`ALL_TESTS`, `ALL_SOFTWARES`) are at the top of
`J_run_all.py`. A summary table of per-stage PASS/FAIL is printed at the end.

## Directory structure

```
WG4-testingInteractions/
├── README.md
├── scripts_PY/                     # generic, software-independent scripts
│   ├── F_generate_analytical.py
│   ├── H_do_compare.py
│   ├── I_make_figure.py
│   ├── J_run_all.py                # orchestrator
│   └── D_helpers.py, E_compare_functions.py, ...
├── scripts_DEM/
│   └── YADE/                       # one folder per DEM software
│       ├── G_generate_forces.py    # software-specific driver
│       └── executable.txt          # path to the solver executable
├── output_ANA/                     # analytical reference data (theoretical_output_test_XX.{json,csv})
├── output_ANA_ERR/                 # faulty analytical references (diagnostics)
├── output_DEM/                     # DEM results   (dem_output_<software>_test_XX.csv)
├── output_REPORT/                  # comparison reports + LaTeX error tables
├── output_figures/                 # generated figures (PDF + PNG)
└── docs/                           # analytical reference derivation
```

The DEM output folder is always `output_DEM`; only the filename encodes the
software label, e.g. `dem_output_YADE_test_01.csv`.

## Running a single stage

Stages can also be run on their own, from inside `scripts_PY/`:

```bash
python F_generate_analytical.py 1              # analytical reference for test 1
python H_do_compare.py YADE 1                  # DEM run + comparison, test 1, YADE
python I_make_figure.py YADE 1                 # figure + error table, test 1, YADE
```

`H_do_compare.py` and `I_make_figure.py` both take `<software> <testID>`, where
`<software>` is the folder name under `scripts_DEM/`.

## Adding a new DEM software

The generic scripts never need to change. To integrate a new code `FOO`:

1. Create `scripts_DEM/FOO/`.
2. Add `G_generate_forces.py` there — the driver that reads the analytical input
   JSON, runs `FOO` through the prescribed kinematics, and writes the DEM output
   CSV in the expected column layout (see below).
3. Add `executable.txt` with the path to the `FOO` solver (see next section).
4. Run with `python J_run_all.py --softwares FOO`, or add `FOO` to
   `ALL_SOFTWARES` in `J_run_all.py`.

### `executable.txt`

Each `scripts_DEM/<software>/` folder holds an `executable.txt` listing the
solver executable. `H_do_compare.py` uses the **first non-comment, non-blank
line**; lines beginning with `#` (full-line or trailing) are kept as inactive
alternatives, which is handy for swapping between solver builds. Paths are
resolved relative to that folder. Example (YADE):

```
#yade  # if YADE is on the PATH
../../../../LS-DEM-dev/install-dev/bin/yade-2026-09-28.git-f119cbc        # Clean
#../../../../LS-DEM-dev/install-noBCH/bin/yade-2026-06-23.git-3a543bc     # No BCH correction
#../../../../LS-DEM-dev/install-noInitRB/bin/yade-2026-06-23.git-7769868  # No rigid-body motion init
```

## DEM output format

`G_generate_forces.py` must write a space-delimited CSV with a `#`-commented
header. Column order is fixed (positions, orientations, velocities, angular
velocities, forces, torques):

```
t x1 y1 z1 x2 y2 z2 qx1 qy1 qz1 qw1 qx2 qy2 qz2 qw2
v1x v1y v1z v2x v2y v2z w1x w1y w1z w2x w2y w2z
f1x f1y f1z f2x f2y f2z t1x t1y t1z t2x t2y t2z
```

Velocity and angular-velocity columns may be written **on-step** (`v1x`, `w1x`,
…) or at the leapfrog **half-step** (`v1x_half`, `w1x_half`, …). The loaders
detect the convention from the header and compare against the matching analytical
series, so either is accepted.

## Outputs

### Figures (`output_figures/`)

Publication-quality `figure_<software>_test_XX.{pdf,png}`. The plot type depends
on the test (2D component-vs-time, 3D component trajectories, or 3D force/torque
space). Analytical data is drawn as a solid dark-purple line, DEM data as
downsampled light-green markers.

### Error tables (`output_REPORT/errors_<software>_test_XX.txt`)

LaTeX `tabular` (no outer wrapper, ready to `\input`) with the MAPE per particle
for position, velocity, orientation, angular velocity, force and torque, plus the
force and torque imbalance (Newton's-third-law checks). Values below `1e-14` %
are shown as `$<10^{-14}$`.

MAPE is computed from the per-step difference-vector magnitudes over the full
(non-interpolated) data:

$$ER_i = \frac{\sum \lVert x_{\text{DEM},i} - x_{\text{ANA},i}\rVert}{t_{max}}, \qquad
N = \frac{\sum (\lVert x_{\text{ANA},i}\rVert + \lVert x_{\text{ANA},j}\rVert)}{t_{max}}, \qquad
MAPE_i = 100\,\frac{ER_i}{N}.$$

### Comparison reports (`output_REPORT/comparison_report_<software>_test_XX.txt`)

Plain-text PASS/FAIL per quantity (position, orientation, force, torque), with
the failing step indices listed when a check fails.

## Requirements

```bash
pip install numpy scipy matplotlib
```

plus the DEM solver(s) referenced in each `scripts_DEM/<software>/executable.txt`.
