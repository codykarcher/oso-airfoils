#!/usr/bin/env python3
"""sweep_families.py — MPI-parallel aerodynamic sweep for selected airfoil families.

Run (parallel):
    mpirun -n 188 python -m mpi4py sweep_families.py

Run (serial fallback):
    python sweep_families.py

Families  : oso_2026_ht1, mhkf1, ffa, du, riso_b
Re        : 1.5e6
Alphas    : -5 to 25 by 1 deg (31 points)
Conditions: clean  (N_crit=9.0, xtp=1.0/1.0)
            rough  (N_crit=3.0, xtp=0.05/0.05)
Solvers   : xfoil  (BL data saved, one task per alpha — max parallelism)
            neuralfoil  (xxlarge, one batched task per airfoil/condition)
Output    : sweeps/<stem>.json  — one file per airfoil, same schema as data/
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

# ── MPI setup ──────────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    if _rank == 0:
        print(f"MPI active: {_size} ranks", flush=True)
except ImportError:
    _comm, _rank, _size = None, 0, 1
    print("mpi4py not available — running serially", flush=True)

# ── Configuration ──────────────────────────────────────────────────────────────
FAMILIES = ['oso_2026_ht1', 'mhkf1', 'ffa', 'du', 'riso_b']

RE     = 1.5e6
ALPHAS = list(range(-5, 26))  # -5 … 25 inclusive, step 1

TURB_CASES = [
    {'N_crit': 9.0, 'xtp_u': 1.0,   'xtp_l': 1.0},    # clean
    {'N_crit': 3.0, 'xtp_u': 0.05,  'xtp_l': 0.05},   # rough
]

NF_MODEL      = 'xxlarge'
XFOIL_TIMELIM = 20           # seconds per alpha call
KULFAN_ORDER  = 8

HERE      = pathlib.Path(__file__).resolve().parent
# Sweep results are family performance data, so they are written straight into the
# data tree (data/<family>/performance_data/<airfoil>.json) rather than into a
# separate runfiles/sweeps/ staging folder that then had to be merged by hand.
DATA_ROOT = HERE.parent / 'data'


def _perf_dir(family):
    d = DATA_ROOT / family / 'performance_data'
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── Imports from oso_airfoils ──────────────────────────────────────────────────
from oso_airfoils.core.data_utils     import _DEFAULT_AFL_ROOT
from oso_airfoils.core.ingest_data    import (
    _wrapper_result_to_records, _NumpyEncoder, _merge_runs, _get_xfoil_version,
)
from oso_airfoils.core.airfoil_family import get_geometry_from_dir

try:
    from oso_airfoils.airfoils.generate_empty_jsons import make_geometry_record
    _HAS_GEOM = True
except Exception:
    _HAS_GEOM = False

# Cache xfoil version once at startup (subprocess call)
_XFOIL_VERSION = _get_xfoil_version() if _rank == 0 else None
if _comm is not None and _size > 1:
    _XFOIL_VERSION = _comm.bcast(_XFOIL_VERSION, root=0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_kulfan(family: str, stem: str):
    """Load a Kulfan object from the installed oso_airfoils airfoil tree."""
    dat_dir = _DEFAULT_AFL_ROOT / family / 'datfiles'
    return get_geometry_from_dir(stem, dat_dir)


def _collect_airfoils() -> list[tuple[str, str]]:
    """Return sorted list of (family, stem) for all target families."""
    result = []
    for family in FAMILIES:
        dat_dir = _DEFAULT_AFL_ROOT / family / 'datfiles'
        if not dat_dir.is_dir():
            if _rank == 0:
                print(f"WARNING: datfiles dir not found: {dat_dir}", flush=True)
            continue
        for dat in sorted(dat_dir.glob('*.dat')):
            result.append((family, dat.stem))
    return result


def _build_tasks(airfoils: list[tuple[str, str]]) -> list[dict]:
    """
    Build the flat task list that will be round-robin distributed across ranks.

    XFoil   → one task per (airfoil, condition, alpha)   [finest granularity]
    NeuralFoil → one task per (airfoil, condition) with all alphas batched
    """
    tasks = []
    for family, stem in airfoils:
        for tc in TURB_CASES:
            # XFoil: individual alpha tasks for maximum parallelism
            for a in ALPHAS:
                tasks.append({
                    'family': family,
                    'stem':   stem,
                    'solver': 'xfoil',
                    'alpha':  float(a),
                    'Re':     RE,
                    **tc,
                })
            # NeuralFoil: batch all alphas in one task (vectorised internally)
            tasks.append({
                'family': family,
                'stem':   stem,
                'solver': 'neuralfoil',
                'alpha':  ALPHAS,   # full list — vectorised
                'Re':     RE,
                **tc,
            })
    return tasks


def _run_task(task: dict, afl) -> list[dict]:
    """Execute one task and return a list of run records (may be empty on failure)."""
    solver  = task['solver']
    alpha   = task['alpha']
    Re      = task['Re']
    N_crit  = task['N_crit']
    xtp_u   = task['xtp_u']
    xtp_l   = task['xtp_l']
    te_gap  = float(afl.constants.TE_gap)
    u_coef  = afl.upperCoefficients
    l_coef  = afl.lowerCoefficients

    if solver == 'xfoil':
        from oso_airfoils.core.xfoil_wrapper import run as _xfoil
        try:
            res = _xfoil(
                'alpha',
                u_coef, l_coef,
                val=alpha,
                Re=Re, M=0.0,
                N_crit=N_crit,
                xtp_u=xtp_u, xtp_l=xtp_l,
                TE_gap=te_gap,
                timelimit=XFOIL_TIMELIM,
                save_boundary_layer_data=True,
            )
            if res is None:
                return []
            return _wrapper_result_to_records('xfoil', res, version=_XFOIL_VERSION)
        except Exception:
            return []

    elif solver == 'neuralfoil':
        from oso_airfoils.core.neuralfoil_wrapper import run as _nf
        try:
            res = _nf(
                'alpha',
                u_coef, l_coef,
                val=alpha,
                Re=Re, M=0.0,
                N_crit=N_crit,
                xtp_u=xtp_u, xtp_l=xtp_l,
                TE_gap=te_gap,
                model=NF_MODEL,
                save_boundary_layer_data=True,
            )
            if res is None:
                return []
            return _wrapper_result_to_records('neuralfoil', res, model=NF_MODEL)
        except Exception:
            return []

    return []


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if _rank == 0:
        print(f"Output directory : {DATA_ROOT}/<family>/performance_data", flush=True)

    airfoils = _collect_airfoils()
    if _rank == 0:
        print(
            f"Airfoils found   : {len(airfoils)} "
            f"across {len(FAMILIES)} families",
            flush=True,
        )
        for fam, stem in airfoils:
            print(f"  {fam}/{stem}", flush=True)

    tasks = _build_tasks(airfoils)
    n_tasks = len(tasks)

    if _rank == 0:
        print(
            f"Total tasks      : {n_tasks}  "
            f"(~{math.ceil(n_tasks / _size)} per rank on {_size} ranks)",
            flush=True,
        )

    # Round-robin distribution — same order on every rank, deterministic
    my_tasks = tasks[_rank::_size]

    # Per-rank: load each Kulfan once, cache within this rank's work loop
    kulfan_cache: dict[tuple, object] = {}
    my_tagged_records: list[tuple[str, str, dict]] = []

    for i, task in enumerate(my_tasks):
        key = (task['family'], task['stem'])
        if key not in kulfan_cache:
            kulfan_cache[key] = _load_kulfan(*key)
        afl = kulfan_cache[key]

        recs = _run_task(task, afl)
        for rec in recs:
            my_tagged_records.append((task['family'], task['stem'], rec))

        if _rank == 0 and (i + 1) % max(1, len(my_tasks) // 10) == 0:
            print(
                f"  rank 0 progress: {i + 1}/{len(my_tasks)} tasks done",
                flush=True,
            )

    # MPI gather all tagged records to rank 0
    if _comm is not None and _size > 1:
        all_batches = _comm.gather(my_tagged_records, root=0)
    else:
        all_batches = [my_tagged_records]

    if _rank != 0:
        return

    # Flatten
    flat: list[tuple] = [item for batch in all_batches for item in batch]
    print(f"\nGathered {len(flat)} total records across all ranks", flush=True)

    # Group by (family, stem)
    grouped: dict[tuple, list] = defaultdict(list)
    for family, stem, rec in flat:
        grouped[(family, stem)].append(rec)

    # Write one JSON per airfoil
    written = 0
    for (family, stem), records in sorted(grouped.items()):
        out_path = _perf_dir(family) / f'{stem}.json'
        # Merge into the existing performance file if there is one, so a sweep adds
        # runs rather than replacing everything already recorded for this airfoil.
        existing = []
        if out_path.is_file():
            with open(out_path) as fh:
                existing = json.load(fh).get('runs', [])
        merged, n_added, _, _, _ = _merge_runs(existing, records)
        print(f"  {stem:40s}  {n_added} records", flush=True)

        doc: dict = {'airfoil_id': stem, 'geometry': None, 'runs': merged}

        if _HAS_GEOM:
            try:
                afl = _load_kulfan(family, stem)
                doc['geometry'] = make_geometry_record(afl, kulfan_order=KULFAN_ORDER)
            except Exception as e:
                print(f"  WARNING: geometry failed for {stem}: {e}", flush=True)

        with open(out_path, 'w') as fh:
            json.dump(doc, fh, indent=2, cls=_NumpyEncoder)
        written += 1

    print(f"\nDone — wrote {written} JSON files under {DATA_ROOT}", flush=True)


if __name__ == '__main__':
    main()
