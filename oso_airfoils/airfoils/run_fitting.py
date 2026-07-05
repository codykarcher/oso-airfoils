#!/usr/bin/env python
"""
run_fitting.py  –  Run all airfoil Kulfan-fitting scripts.

Each fitting script is executed as a separate Python subprocess, equivalent
to calling `python <script>` from the shell.  Comment out any entry in
SCRIPTS to skip it.

Usage:
    python run_fitting.py
"""

import pathlib
import subprocess
import sys

_here = pathlib.Path(__file__).parent.resolve()

# ── fitting scripts to run (comment out any you want to skip) ────────────────
SCRIPTS = [
    _here / 'du'     / 'datfiles' / 'originals' / 'du_fitting.py',
    _here / 'ffa'    / 'datfiles' / 'originals' / 'ffa_fitting.py',
    _here / 'mhkf1'  / 'datfiles' / 'originals' / 'mhkf1_fitting.py',
    _here / 'riso_a' / 'datfiles' / 'originals' / 'risoa_fitting.py',
    _here / 'riso_b' / 'datfiles' / 'originals' / 'risob_fitting.py',
    _here / 'riso_p' / 'datfiles' / 'originals' / 'risop_fitting.py',
]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for script in SCRIPTS:
        print(f"--- {script.relative_to(_here)} ---")
        if not script.exists():
            print(f"  [SKIP] file not found: {script}")
            print()
            continue
        result = subprocess.run([sys.executable, str(script)])
        if result.returncode != 0:
            print(f"  [ERROR] exited with code {result.returncode}")
        print()

    print("All fitting scripts complete.")
