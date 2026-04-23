#!/usr/bin/env python3
"""
J_run_all.py  —  top-level orchestrator for the DEM contact-interaction test framework.

Sits at the project root and sequentially runs:
  1. F_generate_analytical.py  (scripts_PY/)   — analytical reference data
  2. H_do_compare.py           (scripts_DEM/<software>/)  — DEM simulation + comparison
  3. I_make_figure.py          (scripts_PY/)   — figures and error tables

Usage:
    python J_run_all.py [--tests 1 2 3 ...] [--softwares YADE ...]

Examples:
    python J_run_all.py                        # run all configured tests & softwares
    python J_run_all.py --tests 1 2 3          # run only tests 1, 2, 3
    python J_run_all.py --softwares YADE       # restrict to one software
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these lists to add/remove tests or DEM softwares
# ---------------------------------------------------------------------------
ALL_TESTS = list(range(1, 21))        # tests 1-20 are currently implemented
ALL_SOFTWARES = ['YADE']              # add other DEM codes here as they are integrated

# Path layout (all relative to this script, which lives at the project root)
ROOT = Path(__file__).parent.resolve()
SCRIPTS_PY  = ROOT / 'scripts_PY'
SCRIPTS_DEM = ROOT / 'scripts_DEM'

F_SCRIPT = 'F_generate_analytical.py'   # run from SCRIPTS_PY
I_SCRIPT = 'I_make_figure.py'           # run from SCRIPTS_PY
H_SCRIPT = 'H_do_compare.py'            # run from SCRIPTS_DEM/<software>/
# ---------------------------------------------------------------------------


def run(cmd, cwd, label):
    """Run *cmd* (list of strings) in directory *cwd*.
    Returns True on success, False on failure, printing a clear error summary."""
    cwd = Path(cwd)
    print(f"\n  $ {' '.join(cmd)}")
    print(f"    (cwd: {cwd.relative_to(ROOT)})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  ✗ FAILED  [{label}]  exit code {result.returncode}")
        return False
    print(f"  ✓ OK  [{label}]")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tests',     nargs='+', type=int, default=ALL_TESTS,
                        help='Test IDs to run (default: all implemented tests)')
    parser.add_argument('--softwares', nargs='+', default=ALL_SOFTWARES,
                        help='DEM software labels to use (default: YADE)')
    parser.add_argument('--skipana',   action='store_true',
                        help='Skip generating analytical reference data (Stage 1)')
    parser.add_argument('--skipdem', action='store_true',
                        help='Skip running DEM simulations and comparisons (Stage 2)')
    args = parser.parse_args()

    tests     = args.tests
    softwares = args.softwares
    skip_ana  = args.skipana
    skip_dem  = args.skipdem

    # Tracking: key = (stage, item) → bool
    results_F = {}  # test → bool
    results_H = {}  # (software, test) → bool
    results_I = {}  # (software, test) → bool

    # -----------------------------------------------------------------------
    # Stage 1 — generate analytical reference data for every test
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 1 — Analytical reference data  (F_generate_analytical.py)")
    print("="*70)

    if skip_ana:
        print("\n  SKIPPING Stage 1 as requested (--skipana). Assuming data exists.")
        for test in tests:
            results_F[test] = True
    else:
        for test in tests:
            label = f"F  test {test:02d}"
            ok = run(
                [sys.executable, F_SCRIPT, str(test)],
                cwd=SCRIPTS_PY,
                label=label,
            )
            results_F[test] = ok

    # -----------------------------------------------------------------------
    # Stage 2 — run DEM simulations and compare (H_do_compare.py)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 2 — DEM simulation + comparison  (H_do_compare.py)")
    print("="*70)

    if skip_dem:
        print("\n  SKIPPING Stage 2 as requested (--skipdem). Assuming DEM results exist.")
        for software in softwares:
            for test in tests:
                results_H[(software, test)] = True
    else:
        for software in softwares:
            h_dir = SCRIPTS_DEM / software
            if not h_dir.is_dir():
                print(f"\n  WARNING: directory not found for software '{software}': {h_dir}")
                for test in tests:
                    results_H[(software, test)] = False
                continue

            for test in tests:
                # Skip if the analytical reference failed — there is nothing to compare against
                if not results_F.get(test, False):
                    print(f"\n  SKIPPING {software} test {test:02d} — analytical reference unavailable")
                    results_H[(software, test)] = False
                    continue

                label = f"H  {software}  test {test:02d}"
                ok = run(
                    [sys.executable, H_SCRIPT, str(test)],
                    cwd=h_dir,
                    label=label,
                )
                results_H[(software, test)] = ok

    # -----------------------------------------------------------------------
    # Stage 3 — generate figures and error tables (I_make_figure.py)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STAGE 3 — Figures and error tables  (I_make_figure.py)")
    print("="*70)

    for software in softwares:
        for test in tests:
            # Skip if the DEM comparison failed — there is nothing to plot
            if not results_H.get((software, test), False):
                print(f"\n  SKIPPING {software} test {test:02d} — DEM results unavailable")
                results_I[(software, test)] = False
                continue

            label = f"I  {software}  test {test:02d}"
            ok = run(
                [sys.executable, I_SCRIPT, software, str(test)],
                cwd=SCRIPTS_PY,
                label=label,
            )
            results_I[(software, test)] = ok

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    any_failure = False

    print(f"\n{'Test':<6}  {'Ana':>5}  ", end='')
    for sw in softwares:
        print(f"{'DEM-'+sw:>10}  {'Fig-'+sw:>10}  ", end='')
    print()

    for test in tests:
        f_ok  = results_F.get(test, False)
        row   = f"  {test:02d}   {'OK' if f_ok else 'FAIL':>5}  "
        for sw in softwares:
            h_ok = results_H.get((sw, test), False)
            i_ok = results_I.get((sw, test), False)
            row += f"{'OK' if h_ok else 'FAIL':>10}  {'OK' if i_ok else 'FAIL':>10}  "
            if not (f_ok and h_ok and i_ok):
                any_failure = True
        print(row)

    print()
    if any_failure:
        print("Some steps FAILED — see output above for details.")
        sys.exit(1)
    else:
        print("All steps completed successfully.")

if __name__ == '__main__':
    main()
