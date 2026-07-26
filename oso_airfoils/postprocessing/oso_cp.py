"""Cp distribution plot for an optimizer run.

Produces a two-panel figure matching the style of ``standardPlot.py``:
  - Top panel: Cp vs x/c (inverted y-axis, upper surface blue, lower orange)
  - Bottom panel(s): airfoil geometry with displacement thickness overlay

Usage (from any directory):
    python -m oso_airfoils.postprocessing.oso_cp [path]
                       [-t/--tool neuralfoil|xfoil]
                       [-c/--compare du ffa mhkf1 ...]
                       [--cl CL | --alpha ALPHA]
                       [--rough | --clean]
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
    oso-cp
    oso-cp -c mhkf1 ffa
    oso-cp --alpha 5.0 -t xfoil
    oso-cp --rough -c du
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import natsort
import numpy as np

from oso_airfoils.core.colors import default_color_cycle as dcc
from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
from metafoil.core.kulfan import Kulfan
from oso_airfoils.postprocessing.runners import _get_bl_record, _resolve_entry, _load_kulfan
from oso_airfoils.postprocessing.polars import computeNormals
from oso_airfoils.postprocessing.oso_polar import _FAMILY_ALIASES

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

# ── Tunable parameters ────────────────────────────────────────────────────────
TAU_MATCH_TOL  = 0.015

N_CRIT_CLEAN   = 9.0
XTP_TOP_CLEAN  = 1.0
XTP_BOT_CLEAN  = 1.0

N_CRIT_ROUGH   = 3.0
XTP_TOP_ROUGH  = 0.5
XTP_BOT_ROUGH  = 0.5

SAVE_DATA      = True

# Upper/lower Cp colours for the optimized airfoil (dataset 1 = primary)
COLOR_UPPER_OPT = '#0065cc'
COLOR_LOWER_OPT = '#d55e00'
# Reference airfoil colours (lighter — dataset 2 palette from standardPlot)
COLOR_UPPER_REF = '#56b4ff'
COLOR_LOWER_REF = '#fca7c7'
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


def _best_pareto(data: dict) -> dict:
    """Return the single best Pareto individual (highest LoD_rough_at_design)."""
    pareto = [p for p in data['population'] if p.get('pareto_index') == 1]
    if not pareto:
        # fall back to best feasible
        pareto = [p for p in data['population'] if p.get('con_tag', 1) == 0]
    if not pareto:
        raise ValueError('No feasible individuals found in population JSON.')
    return max(pareto, key=lambda p: p.get('LoD_rough_at_design', -1e9))


def _plot_cp(ax, cp_data: dict, color_upper: str, color_lower: str,
             label_upper: str = '', label_lower: str = '',
             linewidth: float = 1.5, alpha: float = 1.0) -> None:
    """Plot Cp split into upper/lower surface."""
    cp_list = cp_data['cp']
    x_list  = cp_data['x']
    stag_idx = cp_list.index(max(cp_list))
    x_upper  = x_list[:stag_idx + 1]
    x_lower  = x_list[stag_idx:]
    cp_upper = cp_list[:stag_idx + 1]
    cp_lower = cp_list[stag_idx:]
    ax.plot(x_upper, cp_upper, color=color_upper, linewidth=linewidth,
            alpha=alpha, label=label_upper or None)
    ax.plot(x_lower, cp_lower, color=color_lower, linewidth=linewidth,
            alpha=alpha, label=label_lower or None)


def _plot_airfoil_with_dstar(ax, bl_record: dict,
                              color_upper: str, color_lower: str,
                              xlims: tuple[float, float],
                              linewidth: float = 1.5) -> None:
    """Plot airfoil geometry + displacement-thickness boundary layer envelope."""
    bl = bl_record.get('bl_data')
    if bl is None:
        return

    x_bl = bl['x']
    y_bl = bl['y']
    dstar = bl['Dstar']

    # Detect wake start
    wake_index = next((k for k, v in enumerate(x_bl) if v > 1), len(x_bl))

    lower_te_shift = 0.0
    if (wake_index < len(x_bl)
            and abs(x_bl[wake_index] - 1.0) < 1e-3
            and abs(y_bl[wake_index]) < 1e-3):
        lower_te_shift = y_bl[wake_index - 1]

    # Airfoil contour
    if lower_te_shift != 0:
        ax.plot([x_bl[0]] + x_bl, [lower_te_shift] + y_bl, 'k', linewidth=1.0)
    else:
        ax.plot(x_bl, y_bl, 'k', linewidth=1.0)

    # Displacement thickness envelope
    normals = computeNormals(x_bl, y_bl)
    split_frac = None
    bl_pts: list[list[float]] = []
    stag_off = 0

    for i, nm in enumerate(normals):
        x = x_bl[i]
        y = y_bl[i]
        if i > 0 and x_bl[i] >= 1 and x_bl[i - 1] <= 1:
            split_frac = dstar[0] / (dstar[i - 1] + dstar[0])
            continue
        if i == 0 or i == wake_index:
            continue
        sc = dstar[i]
        if split_frac is not None:
            xshift = x - nm[0] * sc * (1 - split_frac)
            yshift = y - nm[1] * sc * (1 - split_frac) + lower_te_shift
            bl_pts = [[xshift + nm[0] * sc, yshift + nm[1] * sc]] + bl_pts
            stag_off += 1
            bl_pts.append([xshift, yshift])
        else:
            bl_pts.append([x + nm[0] * sc, y + nm[1] * sc])

    if bl_pts:
        bl_arr = np.array(bl_pts)
        stag_i = stag_off
        ax.plot(bl_arr[:stag_i + 1, 0], bl_arr[:stag_i + 1, 1],
                color=color_upper, linewidth=linewidth)
        ax.plot(bl_arr[stag_i:, 0], bl_arr[stag_i:, 1],
                color=color_lower, linewidth=linewidth)

    ax.set_xlim(xlims)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)


def _make_info_str(rec: dict, multi: bool = False) -> str:
    """Build a parameter annotation string from a BL record."""
    Re    = float(rec.get('Re', 0))
    alpha = float(rec.get('alpha', 0))
    cl    = float(rec.get('cl', 0))
    cd    = float(rec.get('cd', 1e-9))
    cm    = float(rec.get('cm', 0))
    nc    = float(rec.get('N_crit', 0))
    xtu   = float(rec.get('xtr_top', rec.get('xtr_u', 1)))
    xtl   = float(rec.get('xtr_bot', rec.get('xtr_l', 1)))

    if not multi:
        s  = r'\begin{eqnarray*}'
        s += rf'Re &=& {Re:.2e} \\'
        s += rf'\alpha &=& {alpha:.4f}^\circ \\'
        s += rf'C_L &=& {cl:.4f} \\'
        s += rf'C_M &=& {cm:.4f} \\'
        s += rf'C_D &=& {cd:.4f} \\'
        s += rf'L/D &=& {cl/cd:.4f} \\'
        s += rf'N_{{cr}} &=& {nc:.4f} \\'
        s += rf'X_{{tr_u}} &=& {xtu:.4f} \\'
        s += rf'X_{{tr_l}} &=& {xtl:.4f} \\'
        s += r'\end{eqnarray*}'
    else:
        s  = rf'$Re = {Re:.2e} \quad \alpha = {alpha:.4f}^\circ \quad '
        s += rf'C_L = {cl:.4f} \quad C_M = {cm:.4f} \quad C_D = {cd:.4f}$'
    return s


def cp_compare(
    datasets: list[dict[str, Any]],
    figure_path: str | None = None,
    dpi: int = 300,
) -> 'matplotlib.figure.Figure':
    """Plot Cp for one or more datasets.

    Parameters
    ----------
    datasets : list of dict
        Each dict must have keys: ``label``, ``bl_record`` (a BL/Cp record),
        ``color_upper``, ``color_lower``.
    figure_path : str, optional
        Save path; skipped if None.
    """
    n = len(datasets)
    if n == 0:
        raise ValueError('datasets is empty')
    if n > 3:
        raise ValueError('cp_compare supports at most 3 datasets')

    fig_height = 8 if n == 1 else (10 if n == 2 else 12)
    fig = plt.figure(figsize=(10, fig_height), dpi=dpi)
    gs  = gridspec.GridSpec(n + 2, 1, figure=fig)
    ax_cp  = fig.add_subplot(gs[0:2, 0])
    afl_axes = [fig.add_subplot(gs[2 + i, 0]) for i in range(n)]

    xlims = (-0.1, 1.1)
    tick_h = 0.05

    # Cp axis decorations
    ax_cp.plot([-0.1, 1.1], [0, 0], color='k', linewidth=0.8)
    for xt in np.linspace(0, 1, 11):
        ax_cp.plot([xt, xt], [-tick_h, tick_h], color='k', linewidth=0.6)
    ax_cp.set_xlim(xlims)
    ax_cp.set_ylabel(r'$C_p$')
    ax_cp.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_cp.set_xticks([])

    all_cp_min = []

    for ii, ds in enumerate(datasets):
        rec = ds['bl_record']
        cp_data = rec.get('cp_data')
        if cp_data is None or not cp_data.get('cp'):
            print(f"  Warning: no cp_data for dataset {ii} ({ds.get('label', '?')}) — skipping.")
            continue

        lbl = ds.get('label', '')
        _plot_cp(ax_cp, cp_data,
                 color_upper=ds['color_upper'],
                 color_lower=ds['color_lower'],
                 label_upper=f'{lbl} upper' if lbl else '',
                 label_lower=f'{lbl} lower' if lbl else '',
                 linewidth=1.5 if ii == 0 else 1.2,
                 alpha=1.0 if ii == 0 else 0.75)
        all_cp_min.append(min(cp_data['cp']))

        # Airfoil + Dstar panel
        ax_afl = afl_axes[ii]
        _plot_airfoil_with_dstar(ax_afl, rec,
                                  color_upper=ds['color_upper'],
                                  color_lower=ds['color_lower'],
                                  xlims=xlims)

        # Annotation
        multi = (n > 1)
        info = _make_info_str(rec, multi=multi)
        if not multi:
            ax_cp.text(0.80, 0.95, info, transform=ax_cp.transAxes,
                       fontsize=8, verticalalignment='top')
        else:
            vshift = max(rec['bl_data']['y']) + 0.04 if rec.get('bl_data') else 0.12
            ax_afl.text(0.15, vshift, info, fontsize=7)

    # y-axis: Cp inverted, floor at min(cp) - 0.1 with at least -2
    cp_floor = min(all_cp_min) - 0.1 if all_cp_min else -2.0
    ax_cp.set_ylim([min(-2.0, cp_floor), 1.1])
    ax_cp.invert_yaxis()

    # Legend on Cp axis if multiple datasets
    if n > 1:
        from matplotlib.lines import Line2D
        handles = []
        for ds in datasets:
            handles.append(Line2D([0], [0], color=ds['color_upper'], linewidth=1.5,
                                   label=f"{ds.get('label', '')} upper"))
            handles.append(Line2D([0], [0], color=ds['color_lower'], linewidth=1.5,
                                   label=f"{ds.get('label', '')} lower"))
        ax_cp.legend(handles=handles, fontsize=7, loc='lower right',
                     ncols=2, framealpha=0.7)

    plt.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved → {figure_path}')

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='oso-cp',
        description='Cp distribution plot for an optimizer run.',
    )
    parser.add_argument('path', nargs='?', default='.',
        help='Population JSON file or run directory (default: current directory).')
    parser.add_argument('-t', '--tool', default='xfoil',
        choices=['neuralfoil', 'xfoil', 'qfoil'],
        help='Aerodynamic solver (default: xfoil).')
    parser.add_argument('-c', '--compare', nargs='+', metavar='FAMILY', default=[],
        help=(
            'Reference airfoil families to overlay (tau-matched). '
            'Choices: du ffa mhkf1 risoa risob risop s20 s40 osowt1 osowt2 osowt2s osowt3'
        ))
    cond_group = parser.add_mutually_exclusive_group()
    cond_group.add_argument('--cl', type=float, default=None,
        help='CL to evaluate Cp at (default: CL_design from JSON).')
    cond_group.add_argument('--alpha', type=float, default=None,
        help='Angle of attack [deg] to evaluate Cp at.')
    rough_group = parser.add_mutually_exclusive_group()
    rough_group.add_argument('--rough', action='store_true', default=False,
        help=f'Rough transition (N_crit={N_CRIT_ROUGH}, xtp={XTP_TOP_ROUGH}/{XTP_BOT_ROUGH}).')
    rough_group.add_argument('--clean', action='store_true', default=False,
        help=f'Clean transition (N_crit={N_CRIT_CLEAN}, xtp={XTP_TOP_CLEAN}/{XTP_BOT_CLEAN}) [default].')
    parser.add_argument('-o', '--output', default=None, dest='output',
        help='Output figure path (default: <run_dir>/cp_<cond>.png).')
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
        N_crit, xtp_top, xtp_bot, cond_tag = N_CRIT_ROUGH, XTP_TOP_ROUGH, XTP_BOT_ROUGH, 'rough'
    else:
        N_crit, xtp_top, xtp_bot, cond_tag = N_CRIT_CLEAN, XTP_TOP_CLEAN, XTP_BOT_CLEAN, 'clean'

    # Flight condition
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

    # ── Best Pareto airfoil ───────────────────────────────────────────────────
    best = _best_pareto(data)
    afl_opt = Kulfan(TE_gap=te_gap)
    afl_opt.upperCoefficients = best['K_upper']
    afl_opt.lowerCoefficients = best['K_lower']

    print(f'Running Cp for best Pareto ({args.tool}, Re={Re:.2e}, '
          f'tau={tau:.3f}, {mode_key}={mode_val:.3f}, {cond_tag})...')

    rec_opt = _get_bl_record(
        family=None, stem=None, afl_root=_DEFAULT_AFL_ROOT,
        mode=mode_key, val=mode_val, Re=Re,
        N_crit=N_crit, xtp_top=xtp_top, xtp_bot=xtp_bot,
        source=args.tool, save_data=False,
        kulfan=afl_opt,
    )

    datasets = [{
        'label':       'optimized',
        'bl_record':   rec_opt,
        'color_upper': COLOR_UPPER_OPT,
        'color_lower': COLOR_LOWER_OPT,
    }]

    # ── Reference airfoils ────────────────────────────────────────────────────
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
        print(f'  Reference: {stem} (tau={actual_tau:.4f}) from {alias}')

        fam_dir_key_str = fam_dir_key if isinstance(entry, tuple) else entry
        try:
            afl_ref = _load_kulfan(fam_dir_key_str, stem, _DEFAULT_AFL_ROOT)
        except Exception:
            print(f'  Warning: could not load geometry for {stem} — skipping.')
            continue

        rec_ref = _get_bl_record(
            family=fam_dir_key_str, stem=stem, afl_root=_DEFAULT_AFL_ROOT,
            mode=mode_key, val=mode_val, Re=Re,
            N_crit=N_crit, xtp_top=xtp_top, xtp_bot=xtp_bot,
            source=args.tool, save_data=SAVE_DATA,
            kulfan=None,
        )
        datasets.append({
            'label':       stem,
            'bl_record':   rec_ref,
            'color_upper': COLOR_UPPER_REF,
            'color_lower': COLOR_LOWER_REF,
        })

    # ── Output path ───────────────────────────────────────────────────────────
    figure_path = args.output or str(json_path.parent / f'cp_{cond_tag}.png')

    cp_compare(datasets, figure_path=figure_path)
    print('Done.')


if __name__ == '__main__':
    main()
