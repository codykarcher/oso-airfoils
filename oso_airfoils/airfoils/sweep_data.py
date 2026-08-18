#!/usr/bin/env python
"""
sweep_data.py  –  MPI aerodynamic parameter sweep across airfoil families.

Equivalent to sweep_data.sh but written in Python so it can be launched
directly under mpirun without a bash wrapper:

    python sweep_data.py                        # single process
    mpirun -n 32 python -m mpi4py sweep_data.py # MPI (recommended)

Family selection
----------------
Comment / uncomment entries in the FAMILIES list below to choose which
families are swept.

Mode selection
--------------
Set the flags at the top of the script:
    _RUN_ALPHA  – sweep alpha = [-30, 30] in 61 steps
    _RUN_CL     – sweep CL   = [0.0, 3.0] in 0.1 steps (XFoil only;
                               unachievable CLs are skipped automatically)
"""

import pathlib
import json
import numpy as np
from itertools import product

# ── families to sweep ─────────────────────────────────────────────────────────
# Comment out any line to skip that family.
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

# ── mode switches ─────────────────────────────────────────────────────────────
_RUN_ALPHA = False
_RUN_CL    = False
# ─────────────────────────────────────────────────────────────────────────────

# ── shared sweep parameters ───────────────────────────────────────────────────
Re_list    = [0.5e6, 1e6, 5e6, 10e6, 15e6, 20e6]
M_list     = [0.0]
conditions = [
    (9.0, 1.0,  1.0 ),   # (N_crit, xtp_u, xtp_l)  clean
    (3.0, 0.05, 0.05),   #                          rough
]
N_panels   = 160
nf_model   = 'xxxlarge'

# alpha sweep
alpha_list = list(np.linspace(-30, 30, 61))
alpha_solvers = ('xfoil', 'neuralfoil')

# CL sweep
cl_list    = list(np.round(np.arange(0.0, 3.0 + 0.05, 0.1), 4))
cl_solvers = ('xfoil','neuralfoil')
# ─────────────────────────────────────────────────────────────────────────────


def _run_family(family_name, datfiles_dir, perf_dir, run_cases, solvers,
                comm, rank, size):
    """Sweep all airfoils in one family for the given run_cases / solvers."""
    from oso_airfoils.core.airfoil_family import get_geometry_from_dir
    from oso_airfoils.core.sweep import run_sweep
    from oso_airfoils.core.ingest_data import find_json_path, _merge_runs, _NumpyEncoder

    airfoils = sorted(p.stem for p in datfiles_dir.glob('*.dat'))
    if not airfoils:
        if rank == 0:
            print(f"  [SKIP] {family_name}: no .dat files found")
        return

    family_label = family_name.upper()

    for afl_name in airfoils:
        afl = get_geometry_from_dir(afl_name, datfiles_dir, family_label)
        for solver in solvers:
            records = run_sweep(solver, afl, run_cases,
                                comm=comm, rank=rank, size=size,
                                nf_model=nf_model)
            if rank == 0 and records:
                jpath = find_json_path(afl_name, datfiles_dir, perf_dir)
                with open(jpath) as f:
                    jdata = json.load(f)
                merged, na, ns, nu, nc = _merge_runs(
                    jdata['runs'], records, source_hint=jpath.name)
                jdata['runs'] = merged
                with open(jpath, 'w') as f:
                    json.dump(jdata, f, indent=2, cls=_NumpyEncoder)
                parts = [f"{na} added"]
                if ns:
                    parts.append(f"{ns} skipped (duplicate)")
                if nu:
                    parts.append(f"{nu} null-populated")
                if nc:
                    parts.append(f"{nc} conflict(s)")
                print(f"    [{solver}] {afl_name}: {', '.join(parts)}")


if __name__ == '__main__':
    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _comm, _rank, _size = None, 0, 1

    _here = pathlib.Path(__file__).parent.resolve()
    _perf_root = _here.parent / 'data'
    # ── build run_cases for each active mode ──────────────────────────────────
    alpha_cases = []
    if _RUN_ALPHA:
        alpha_cases = [
            {'alpha': float(a), 'Re': Re, 'M': M, 'N_crit': Nc,
             'xtp_u': xu, 'xtp_l': xl, 'N_panels': N_panels}
            for Re, M, (Nc, xu, xl), a
            in product(Re_list, M_list, conditions, alpha_list)
        ]

    cl_cases = []
    if _RUN_CL:
        cl_cases = [
            {'mode': 'cl', 'cl': float(cl), 'Re': Re, 'M': M, 'N_crit': Nc,
             'xtp_u': xu, 'xtp_l': xl, 'N_panels': N_panels}
            for Re, M, (Nc, xu, xl), cl
            in product(Re_list, M_list, conditions, cl_list)
        ]
    # ─────────────────────────────────────────────────────────────────────────

    for family in FAMILIES:
        datfiles_dir = _here / family / 'datfiles'
        perf_dir     = _perf_root / family / 'performance_data'

        if not datfiles_dir.is_dir():
            if _rank == 0:
                print(f"[SKIP] {family}: datfiles/ not found")
            continue

        if _rank == 0:
            print(f"\n{'─' * 52}")
            print(f"  {family}")
            print(f"{'─' * 52}")

        if _RUN_ALPHA:
            if _rank == 0:
                print(f"  alpha sweep ({len(alpha_cases)} cases × {len(alpha_solvers)} solvers)")
            _run_family(family, datfiles_dir, perf_dir,
                        alpha_cases, alpha_solvers, _comm, _rank, _size)

        if _RUN_CL:
            if _rank == 0:
                print(f"  CL sweep ({len(cl_cases)} cases × {len(cl_solvers)} solver)")
            _run_family(family, datfiles_dir, perf_dir,
                        cl_cases, cl_solvers, _comm, _rank, _size)

    if _rank == 0:
        print("\nDone.")
