"""Rainbow boundary-layer plot for an optimizer run.

Usage (from any directory):
    python -m oso_airfoils.postprocessing.oso_bl [path]
                       [-t/--tool neuralfoil|xfoil]
                       [-c/--compare du ffa mhkf1 ...]
                       [--cl CL | --alpha ALPHA]
                       [-o/--output figure.pdf]

*path* may be:
  - a population JSON file directly, or
  - a run directory containing ``population_*.json`` files
    (the most recent / highest-generation file is used automatically), or
  - omitted → uses the current working directory.

Defaults to running at CL_design taken from the run's input_parameters.
Pass --alpha to override with a specific angle of attack instead.

Example:
    cd data/cases_111_to_120/case_116/c116_t18_k16_n752_l13_e15__2026_06_15/
    oso-bl
    oso-bl -c mhkf1 ffa
    oso-bl --alpha 5.0 -t xfoil
"""

from __future__ import annotations

import argparse
import json
import pathlib

import natsort
import numpy as np

from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
from metafoil.core.kulfan import Kulfan
from oso_airfoils.postprocessing.runners import run_and_plot_boundary_layer_rainbow
from oso_airfoils.postprocessing.oso_polar import _FAMILY_ALIASES

# ── Tunable parameters ────────────────────────────────────────────────────────
N_PARETO_AIRFOILS  = 21        # 0.00 → 1.00 in steps of 0.05
TAU_MATCH_TOL      = 0.015     # max |tau_ref - tau_run| for a family match

# Default turbulence condition (clean)
N_CRIT_CLEAN       = 9.0
XTP_TOP_CLEAN      = 1.0
XTP_BOT_CLEAN      = 1.0

# Default turbulence condition (rough)
N_CRIT_ROUGH       = 3.0
XTP_TOP_ROUGH      = 0.5
XTP_BOT_ROUGH      = 0.5

SAVE_DATA          = True       # cache newly computed BL data to JSON
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_json(path_arg: str) -> pathlib.Path:
    p = pathlib.Path(path_arg).resolve()
    if p.is_file() and p.suffix == '.json':
        return p
    if p.is_dir():
        candidates = natsort.natsorted(
            list(p.glob('population_*.json')), alg=natsort.ns.IGNORECASE
        )
        if candidates:
            return pathlib.Path(candidates[-1])
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            candidates = natsort.natsorted(
                list(sub.glob('population_*.json')), alg=natsort.ns.IGNORECASE
            )
            if candidates:
                return pathlib.Path(candidates[-1])
    raise FileNotFoundError(f"No population JSON found at or under: {path_arg}")


def _tau_match(family_dir: pathlib.Path, tau: float, tol: float, allowed_stems=None):
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
    if len(pareto) <= n:
        return pareto
    indices = np.unique(np.round(np.linspace(0, len(pareto) - 1, n)).astype(int))
    return [pareto[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(
        description='Rainbow boundary-layer plot for an optimizer run.'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Population JSON file or run directory (default: current directory).',
    )
    parser.add_argument(
        '-t', '--tool',
        default='xfoil',
        choices=['neuralfoil', 'xfoil', 'qfoil'],
        help='Aerodynamic solver (default: xfoil).',
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
    # Flight condition: default to CL_design, overrideable
    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument(
        '--cl',
        type=float,
        default=None,
        help='CL to evaluate boundary layer at (default: CL_design from JSON).',
    )
    cond_group.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Angle of attack [deg] to evaluate boundary layer at.',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        dest='output',
        help='Output figure path (default: next to the JSON file).',
    )
    rough_group = parser.add_mutually_exclusive_group()
    rough_group.add_argument(
        '--rough',
        action='store_true',
        default=False,
        help=f'Use rough transition conditions (N_crit={N_CRIT_ROUGH}, xtp={XTP_TOP_ROUGH}/{XTP_BOT_ROUGH}).',
    )
    rough_group.add_argument(
        '--clean',
        action='store_true',
        default=False,
        help=f'Use clean transition conditions (N_crit={N_CRIT_CLEAN}, xtp={XTP_TOP_CLEAN}/{XTP_BOT_CLEAN}) [default].',
    )
    args = parser.parse_args()

    # ── Locate JSON ───────────────────────────────────────────────────────────
    json_path = _resolve_json(args.path)
    print(f'Using: {json_path}')
    data   = json.loads(json_path.read_text())
    params = data['input_parameters']

    tau    = float(params['tau'])
    Re     = float(params.get('Re', 1.5e6))
    CL     = params.get('CL')
    te_gap = float(params.get('TE_gap', 0.0))

    # Turbulence condition
    if args.rough:
        N_crit  = N_CRIT_ROUGH
        xtp_top = XTP_TOP_ROUGH
        xtp_bot = XTP_BOT_ROUGH
        cond_tag = 'rough'
    else:
        N_crit  = N_CRIT_CLEAN
        xtp_top = XTP_TOP_CLEAN
        xtp_bot = XTP_BOT_CLEAN
        cond_tag = 'clean'

    # Determine flight condition
    if args.alpha is not None:
        mode_key, mode_val = 'alpha', args.alpha
    elif args.cl is not None:
        mode_key, mode_val = 'cl', args.cl
    elif CL is not None:
        mode_key, mode_val = 'cl', float(CL)
        print(f'Using CL_design = {mode_val:.3f} from run parameters.')
    else:
        print('Warning: No CL_design in run parameters; defaulting to alpha=0.')
        mode_key, mode_val = 'alpha', 0.0

    # ── Build Pareto-front airfoil list ───────────────────────────────────────
    pareto = sorted(
        [p for p in data['population'] if p['pareto_index'] == 1],
        key=lambda p: p['LoD_rough_at_design'],
    )
    pareto = _sample_pareto(pareto, N_PARETO_AIRFOILS)
    N_sampled = len(pareto)

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
        figure_path = str(json_path.parent / f'bl_{cond_tag}_plot.png')

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f'Running BL ({args.tool}, Re={Re:.2e}, tau={tau:.3f}, '
          f'{mode_key}={mode_val:.3f}, {cond_tag})...')

    run_and_plot_boundary_layer_rainbow(
        airfoils          = airfoils,
        **{mode_key: mode_val},
        Re                = Re,
        N_crit            = N_crit,
        xtp_top           = xtp_top,
        xtp_bot           = xtp_bot,
        source            = args.tool,
        figure_path       = figure_path,
        save_data         = SAVE_DATA,
        afl_root          = afl_root,
        reference_airfoils= reference_airfoils if reference_airfoils else None,
        show_airfoil      = True,
    )
    print(f'Saved → {figure_path}')


if __name__ == '__main__':
    main()
