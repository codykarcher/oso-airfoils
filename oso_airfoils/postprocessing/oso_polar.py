"""Rainbow polar plot for an optimizer run.

Usage (from any directory):
    python -m oso_airfoils.postprocessing.oso_polar [path]
                       [-t/--tool neuralfoil|xfoil]
                       [-c/--compare du ffa mhkf1 ...]
                       [-o/--output figure.pdf]

*path* may be:
  - a population JSON file directly, or
  - a run directory containing ``population_*.json`` files
    (the most recent / highest-generation file is used automatically), or
  - omitted → uses the current working directory.

Example (cd into the run folder first):
    cd data/cases_111_to_120/case_116/c116_t18_k16_n752_l13_e15__2026_06_15/
    oso-polar
    oso-polar -c mhkf1 ffa -t xfoil
    oso-polar population_c116_t18_k16_n752_l13_e15_g500.json -c du ffa
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import natsort
import numpy as np

from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
from metafoil.core.kulfan import Kulfan
from oso_airfoils.postprocessing.runners import run_and_plot_polars_rainbow

# ── Tunable parameters ────────────────────────────────────────────────────────
N_PARETO_AIRFOILS  = 21        # 0.00 → 1.00 in steps of 0.05
TAU_MATCH_TOL      = 0.015     # max |tau_ref - tau_run| for a family match

# Alpha sweep used when polars are not already cached
ALPHA_RANGE        = (-5, 30, 0.25)  # (start, stop, step) in degrees

# Turbulence / transition conditions   [N_crit, xtp_upper, xtp_lower]
TURB_CASES_CLEAN   = [[9.0, 1.0,  1.0 ]]
TURB_CASES_ROUGH   = [[3.0, 0.05, 0.05]]

SAVE_DATA          = True       # cache newly computed polars to JSON
# ─────────────────────────────────────────────────────────────────────────────

# Map CLI-friendly family aliases to the actual directory names.
# Sub-family aliases are tuples: (family_dir_name, [allowed_stems]).
_FAMILY_ALIASES: dict[str, str | tuple] = {
    'du'          : 'du',
    'ffa'         : 'ffa',
    'mhkf1'       : 'mhkf1',
    'risoa'       : 'riso_a',
    'riso_a'      : 'riso_a',
    'risob'       : 'riso_b',
    'riso_b'      : 'riso_b',
    'risop'       : 'riso_p',
    'riso_p'      : 'riso_p',
    # s20 = thin S-series (20-m class): s826, s825, s814, s815
    's20'         : ('s', ['s826', 's825', 's814', 's815']),
    # s40 = thick S-series (40-m class): s832, s831, s830
    's40'         : ('s', ['s832', 's831', 's830']),
    'osowt1'      : 'oso_2025_wt1',
    'oso_wt1'     : 'oso_2025_wt1',
    'oso_2025_wt1': 'oso_2025_wt1',
    'osowt2'      : 'oso_2025_wt2',
    'oso_wt2'     : 'oso_2025_wt2',
    'oso_2025_wt2': 'oso_2025_wt2',
    'osowt2s'     : 'oso_2026_wt2s',
    'oso_wt2s'    : 'oso_2026_wt2s',
    'oso_2026_wt2s': 'oso_2026_wt2s',
    # 'osowt3'   : 'oso_2026_wt3',
    # 'oso_2026_wt3': 'oso_2026_wt3',
}


def _resolve_json(path_arg: str) -> pathlib.Path:
    """Return the population JSON path from a file, run dir, or '.'."""
    p = pathlib.Path(path_arg).resolve()
    if p.is_file() and p.suffix == '.json':
        return p
    if p.is_dir():
        candidates = natsort.natsorted(
            list(p.glob('population_*.json')), alg=natsort.ns.IGNORECASE
        )
        if candidates:
            return pathlib.Path(candidates[-1])
        # try one level deeper (user may be in case_116/ rather than the run folder)
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            candidates = natsort.natsorted(
                list(sub.glob('population_*.json')), alg=natsort.ns.IGNORECASE
            )
            if candidates:
                return pathlib.Path(candidates[-1])
    raise FileNotFoundError(
        f"No population JSON found at or under: {path_arg}"
    )


def _tau_match(family_dir: pathlib.Path, tau: float, tol: float, allowed_stems=None):
    """Return (stem, actual_tau) of the closest airfoil in family_dir, or None.

    Parameters
    ----------
    allowed_stems : list of str, optional
        When given, only consider JSON files whose stem is in this list.
        Used for sub-family aliases such as ``s20`` and ``s40``.
    """
    pd = family_dir / 'performance_data'
    if not pd.is_dir():
        return None
    best, best_tau, best_dt = None, None, float('inf')
    for jf in sorted(pd.glob('*.json')):
        if allowed_stems is not None and jf.stem not in allowed_stems:
            continue
        try:
            d = json.loads(jf.read_text())
            t = d.get('geometry', {}).get('tau')
            if t is None:
                continue
            dt = abs(t - tau)
            if dt < best_dt:
                best, best_tau, best_dt = jf.stem, t, dt
        except Exception:
            continue
    if best is None or best_dt > tol:
        return None
    return best, best_tau


def _sample_pareto(pareto: list[dict], n: int) -> list[dict]:
    """Return up to *n* members evenly sampled along the Pareto front."""
    if len(pareto) <= n:
        return pareto
    indices = np.unique(
        np.round(np.linspace(0, len(pareto) - 1, n)).astype(int)
    )
    return [pareto[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(
        description='Rainbow polar plot for an optimizer run.'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Population JSON file or run directory (default: current directory).',
    )
    parser.add_argument(
        '-t', '--tool',
        default='neuralfoil',
        choices=['neuralfoil', 'xfoil'],
        help='Aerodynamic solver (default: neuralfoil).',
    )
    parser.add_argument(
        '-c', '--compare',
        nargs='+',
        metavar='FAMILY',
        default=[],
        help=(
            'Reference airfoil families to overlay (tau-matched). '
            'Choices: du ffa mhkf1 risoa risob risop s osowt1 osowt2 osowt2s osowt3'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        dest='output',
        help='Output figure path (default: next to the JSON file).',
    )
    args = parser.parse_args()

    # ── Locate JSON ───────────────────────────────────────────────────────────
    json_path = _resolve_json(args.path)
    print(f'Using: {json_path}')
    data = json.loads(json_path.read_text())
    params = data['input_parameters']

    tau     = float(params['tau'])
    Re      = float(params.get('Re', 1.5e6))
    CL      = params.get('CL')
    cl_design = float(CL) if CL is not None else None

    tools       = [args.tool]
    turb_cases  = TURB_CASES_CLEAN + TURB_CASES_ROUGH
    sweep_range = ALPHA_RANGE

    # ── Build Pareto-front airfoil list ───────────────────────────────────────
    pareto = sorted(
        [p for p in data['population'] if p['pareto_index'] == 1],
        key=lambda p: p['LoD_rough_at_design'],
    )
    pareto = _sample_pareto(pareto, N_PARETO_AIRFOILS)
    N_sampled = len(pareto)
    te_gap = float(params.get('TE_gap', 0.0))

    airfoils = []
    for i, p in enumerate(pareto):
        rf_norm = i / max(N_sampled - 1, 1)
        label = f'rf={rf_norm:.2f}'
        afl = Kulfan(TE_gap=te_gap)
        afl.upperCoefficients = p['K_upper']
        afl.lowerCoefficients = p['K_lower']
        airfoils.append([label, afl])

    # ── Build reference airfoils ──────────────────────────────────────────────
    reference_airfoils = []
    afl_root = _DEFAULT_AFL_ROOT
    for alias in args.compare:
        fam_key = alias.lower()
        if fam_key not in _FAMILY_ALIASES:
            print(f'  Warning: unknown family alias {alias!r} — skipping.')
            continue
        entry = _FAMILY_ALIASES[fam_key]
        if isinstance(entry, tuple):
            fam_dir_key, allowed_stems = entry
            fam_dir = _DEFAULT_PERF_ROOT / fam_dir_key
        else:
            fam_dir = _DEFAULT_PERF_ROOT / entry
            allowed_stems = None
        result = _tau_match(fam_dir, tau, TAU_MATCH_TOL, allowed_stems)
        if result is None:
            print(f'  No {alias} airfoil within tau±{TAU_MATCH_TOL} of {tau:.3f} — skipping.')
            continue
        stem, actual_tau = result
        print(f'  Comparing: {stem} (tau={actual_tau:.4f}) from {alias}')
        reference_airfoils.append((stem, 'k'))

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is not None:
        figure_path = args.output
    else:
        figure_path = str(json_path.parent / 'polar_plot.png')

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f'Running polars ({args.tool}, Re={Re:.2e}, tau={tau:.3f})...')
    run_and_plot_polars_rainbow(
        airfoils        = airfoils,
        reynolds_numbers= [Re],
        turb_cases      = turb_cases,
        tools           = tools,
        figure_path     = figure_path,
        sweep_param     = 'alpha',
        sweep_range     = sweep_range,
        load_geometry   = True,
        save_data       = SAVE_DATA,
        afl_root        = afl_root,
        reference_airfoils = reference_airfoils if reference_airfoils else None,
        cl_design       = cl_design,
    )
    print(f'Saved → {figure_path}')


if __name__ == '__main__':
    main()
