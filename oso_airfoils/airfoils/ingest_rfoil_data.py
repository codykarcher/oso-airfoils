#!/usr/bin/env python
"""
ingest_rfoil_data.py  –  Ingest all rfoil polars for every airfoil family.

Scans each <family>/performance_data/rfoil/ for *.dat files and ingests
them using the shared ingest_data machinery directly (no dependency on
family_functions.py).  Duplicate runs are silently skipped via the existing
merge logic.

Usage:
    python ingest_rfoil_data.py
"""

import pathlib

# ── families to ingest (comment out any you want to skip) ─────────────────────
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
    from oso_airfoils.core.ingest_data import ingest

    _here = pathlib.Path(__file__).parent.resolve()
    _perf_root = _here.parent / 'data'

    print(f"Scanning for rfoil polar files under: {_perf_root}\n")

    for family in FAMILIES:
        rfoil_dir = _perf_root / family / 'performance_data' / 'rfoil'
        datfiles_dir = _here / family / 'datfiles'
        perf_dir = _perf_root / family / 'performance_data'

        if not rfoil_dir.is_dir():
            continue

        polar_files = sorted(rfoil_dir.glob('*.dat'))
        if not polar_files:
            print(f"[SKIP] {family}: rfoil dir is empty")
            continue

        print(f"[INGEST] {family} ({len(polar_files)} polar files)")
        try:
            ingest('rfoil_data', None, None, datfiles_dir,
                   perf_dir=perf_dir, polar_files=polar_files)
        except Exception as e:
            print(f"[ERROR] {family}: {e}")
        print()

    print("All done.")
