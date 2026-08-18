#!/usr/bin/env python
"""
run_family_plots.py  –  Regenerate the family overview plot for each airfoil family.

Calls plot_family() directly from each family module; no subprocess overhead.
Comment out any entry in FAMILIES to skip that family.

Usage:
    python run_family_plots.py
"""

import pathlib
import sys

# ── families to plot (comment out any you want to skip) ──────────────────────
FAMILIES = [
    'du',
    'ffa',
    'mhkf1',
    'oso_2025_wt1',
    'oso_2025_wt2',
    'oso_2026_wt2s',
    'riso_a',
    'riso_b',
    'riso_p',
    's',
]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import importlib

    _here = pathlib.Path(__file__).parent.resolve()
    # Ensure the package root is on sys.path
    pkg_root = str(_here.parent.parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

    for family in FAMILIES:
        print(f"--- {family} ---")
        try:
            mod = importlib.import_module(f'oso_airfoils.airfoils.{family}.family_functions')
            mod.plot_family()
        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    print("All family plots complete.")
