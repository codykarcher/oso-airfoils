"""Legacy postprocessing for optimizer runs (cases 64-69).

These runs saved population members as .txt files (numpy savetxt format) rather
than JSON.  Parameters (tau, CL, N_k, Re) are parsed from the run directory name.

By default ``oso-legacy`` produces all four outputs:
  1. objective_evolution.png  — best-per-generation objective + design variables
  2. variable_evolution.png   — Kulfan coefficient trajectories
  3. cp_comparison_clean.png  — Cp at design alpha, clean, vs reference families
  4. cp_comparison_rough.png  — Cp at design alpha, rough, vs reference families
  5. polar_compare_plot.png   — compare polar (best N feasible vs references)
  6. airfoil_evolution.gif    — shape evolution across all generations

Use ``--polar-only``, ``--gif-only``, ``--history-only``, or ``--cp-only``
to restrict to a single output group.

Usage examples:
    cd data/cases_61_to_70/case_67/c67_t21_l15_r122_k16_n376__2025_07_21_22-29/
    oso-legacy
    oso-legacy -c all
    oso-legacy -c du ffa -t xfoil -n 3
    oso-legacy -c all --dpi 80 --duration 150 --every 5
    oso-legacy --polar-only -c all
    oso-legacy --gif-only -c du ffa
    oso-legacy --history-only
    oso-legacy --cp-only -c du ffa

*path* may be:
  - a population .txt file directly (polar / cp only), or
  - a run directory containing ``population_*.txt`` files, or
  - omitted → uses the current working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import natsort
import numpy as np
from PIL import Image

try:
    from mpi4py import MPI as _MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False

from oso_airfoils.core.colors import default_color_cycle
from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
from oso_airfoils.core.xfoil_wrapper import run as xfoil_run
from metafoil.core.kulfan import Kulfan
from oso_airfoils.postprocessing.runners import run_and_plot_polars_compare

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})

# ── Tunable parameters ────────────────────────────────────────────────────────
N_BEST_AIRFOILS  = 1          # default number of best individuals for polar
TAU_MATCH_TOL    = 0.015      # max |tau_ref - tau_run| for a family match

# Alpha sweep range for polars
ALPHA_RANGE      = (-5, 30, 0.25)   # (start, stop, step) in degrees

# Turbulence / transition conditions   [N_crit, xtp_upper, xtp_lower]
TURB_CASES_CLEAN = [[9.0, 1.0,  1.0 ]]
TURB_CASES_ROUGH = [[3.0, 0.05, 0.05]]

SAVE_DATA        = True
# ─────────────────────────────────────────────────────────────────────────────

# Reynolds number lookup by tau, from the legacy design matrix
_RE_DESIGN: dict[float, float] = {
    0.15: 10.0e6,
    0.18: 10.0e6,
    0.21: 12.0e6,
    0.24: 13.0e6,
    0.27: 16.0e6,
    0.30: 18.0e6,
    0.33: 16.0e6,
    0.36: 13.0e6,
}

# TE gap lookup by tau_100 key (str), from legacy wt_objective.py
_TE_GAP: dict[str, float] = {
    '15': 0.00196,
    '18': 0.00230,
    '21': 0.00262,
    '24': 0.00751,
    '27': 0.01012,
    '30': 0.01828,
    '33': 0.02644,
    '36': 0.02896,
}

# CLI-friendly family aliases → actual directory names under _DEFAULT_PERF_ROOT.
# Sub-family aliases are tuples: (family_dir_name, [allowed_stems]).
_FAMILY_ALIASES: dict[str, str | tuple] = {
    'du'           : 'du',
    'ffa'          : 'ffa',
    'mhkf1'        : 'mhkf1',
    'risoa'        : 'riso_a',
    'riso_a'       : 'riso_a',
    'risob'        : 'riso_b',
    'riso_b'       : 'riso_b',
    'risop'        : 'riso_p',
    'riso_p'       : 'riso_p',
    # s20 = thin S-series (20-m class): s826, s825, s814, s815
    's20'          : ('s', ['s826', 's825', 's814', 's815']),
    # s40 = thick S-series (40-m class): s832, s831, s830
    's40'          : ('s', ['s832', 's831', 's830']),
    'osowt1'       : 'oso_2025_wt1',
    'oso_wt1'      : 'oso_2025_wt1',
    'oso_2025_wt1' : 'oso_2025_wt1',
    'osowt2'       : 'oso_2025_wt2',
    'oso_wt2'      : 'oso_2025_wt2',
    'oso_2025_wt2' : 'oso_2025_wt2',
    'osowt2s'      : 'oso_2026_wt2s',
    'oso_wt2s'     : 'oso_2026_wt2s',
    'oso_2026_wt2s': 'oso_2026_wt2s',
}

# Families included by -c all (non-OSO, non-mhkf1)
_ALL_FAMILIES = ['du', 'ffa', 'risoa', 'risob', 'risop', 's20', 's40']

# Color cycle for reference airfoils (black reserved for optimization airfoil)
_REF_COLORS = [c for c in list(default_color_cycle) if c != default_color_cycle.black]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_txt_last(path_arg: str) -> pathlib.Path:
    """Return the highest-generation population .txt path from a file or dir."""
    p = pathlib.Path(path_arg).resolve()
    if p.is_file() and p.suffix == '.txt':
        return p
    if p.is_dir():
        candidates = natsort.natsorted(
            list(p.glob('population_*.txt')), alg=natsort.ns.IGNORECASE
        )
        if candidates:
            return pathlib.Path(candidates[-1])
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            candidates = natsort.natsorted(
                list(sub.glob('population_*.txt')), alg=natsort.ns.IGNORECASE
            )
            if candidates:
                return pathlib.Path(candidates[-1])
    raise FileNotFoundError(f"No population .txt found at or under: {path_arg}")


def _resolve_run_dir(path_arg: str) -> pathlib.Path:
    """Return the run directory that contains population_*.txt files."""
    p = pathlib.Path(path_arg).resolve()
    if p.is_file() and p.suffix == '.txt':
        return p.parent
    if p.is_dir():
        if list(p.glob('population_*.txt')):
            return p
        for sub in sorted(p.iterdir()):
            if sub.is_dir() and list(sub.glob('population_*.txt')):
                return sub
    raise FileNotFoundError(f"No population .txt found at or under: {path_arg}")


def _parse_dirname(run_dir: pathlib.Path) -> dict:
    """Parse tau, CL, N_k, Re, and TE_gap from a legacy run directory name.

    Pattern: c{case}_t{tau100}_l{CL10}[_r{rLD}][_e{Re_MHz}]_k{Nk}_n{Npop}__{datetime}
    """
    name = run_dir.name

    m_t = re.search(r'(?:^|_)t(\d+)', name)
    if m_t is None:
        raise ValueError(f"Cannot parse tau from directory name: {name!r}")
    tau = int(m_t.group(1)) / 100.0

    m_l = re.search(r'(?:^|_)l(\d+)', name)
    if m_l is None:
        raise ValueError(f"Cannot parse CL from directory name: {name!r}")
    CL = int(m_l.group(1)) / 10.0

    m_k = re.search(r'(?:^|_)k(\d+)', name)
    if m_k is None:
        raise ValueError(f"Cannot parse N_k from directory name: {name!r}")
    N_k = int(m_k.group(1))

    m_e = re.search(r'(?:^|_)e(\d+(?:\.\d+)?)', name)
    if m_e is not None:
        Re = float(m_e.group(1)) * 1e6
    else:
        closest = min(_RE_DESIGN.keys(), key=lambda t: abs(t - tau))
        Re = _RE_DESIGN[closest]

    tau_key = str(int(round(tau * 100)))
    te_gap  = _TE_GAP.get(tau_key, 0.0)

    return {'tau': tau, 'CL': CL, 'N_k': N_k, 'Re': Re, 'te_gap': te_gap}


def _tau_match(family_dir: pathlib.Path, tau: float, tol: float, allowed_stems=None):
    """Return (stem, actual_tau) of the closest airfoil in family_dir, or None."""
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


def _expand_compare(aliases: list[str]) -> list[str]:
    """Expand the 'all' shorthand in a compare list."""
    result = []
    for alias in aliases:
        if alias.lower() == 'all':
            result.extend(_ALL_FAMILIES)
        else:
            result.append(alias)
    return result


def _resolve_references(compare_list: list[str], tau: float) -> list[tuple[str, str, str]]:
    """Resolve compare aliases to (stem, family_dir_key, display_alias) tuples."""
    resolved = []
    for alias in compare_list:
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
            fam_dir_key = entry
            allowed_stems = None
        result = _tau_match(fam_dir, tau, TAU_MATCH_TOL, allowed_stems)
        if result is None:
            print(f'  No {alias} airfoil within tau±{TAU_MATCH_TOL} of {tau:.3f} — skipping.')
            continue
        stem, actual_tau = result
        print(f'  Comparing: {stem} (tau={actual_tau:.4f}) from {alias}')
        resolved.append((stem, fam_dir_key, alias))
    return resolved


def _load_best_airfoils(txt_path: pathlib.Path, N_k: int) -> list:
    """Load and rank feasible airfoils from a legacy .txt population file.

    Returns list of dicts sorted ascending by ``obj`` (lower = better).
    """
    n_half        = N_k // 2
    idx_obj       = N_k
    idx_con_tag   = N_k + 1
    idx_lod_rough = N_k + 4

    data = np.loadtxt(str(txt_path))
    if data.ndim == 1:
        data = data[np.newaxis, :]

    feasible = data[data[:, idx_con_tag] == 0.0]
    if len(feasible) == 0:
        print('  Warning: no con_tag==0 individuals found; using all.')
        feasible = data

    feasible = feasible[np.argsort(feasible[:, idx_obj])]

    result = []
    for row in feasible:
        result.append({
            'K_upper'            : row[:n_half].tolist(),
            'K_lower'            : row[n_half:N_k].tolist(),
            'obj'                : float(row[idx_obj]),
            'LoD_rough_at_design': float(row[idx_lod_rough]),
        })
    return result


def _load_ref_kulfan(stem: str, fam_dir_key: str) -> Kulfan | None:
    """Load a reference airfoil Kulfan from its performance JSON geometry block."""
    jf = _DEFAULT_PERF_ROOT / fam_dir_key / 'performance_data' / (stem + '.json')
    if not jf.exists():
        return None
    try:
        d = json.loads(jf.read_text())
        geo    = d.get('geometry', {})
        K_up   = geo.get('upperCoefficients')
        K_lo   = geo.get('lowerCoefficients')
        te_gap = float(geo.get('TE_gap', 0.0))
        if K_up is None or K_lo is None:
            return None
        afl = Kulfan(TE_gap=te_gap)
        afl.upperCoefficients = K_up
        afl.lowerCoefficients = K_lo
        return afl
    except Exception:
        return None


# ── History plots ────────────────────────────────────────────────────────────


def _run_history(args: argparse.Namespace) -> None:
    """Plot objective-function evolution and design-variable evolution."""
    run_dir = _resolve_run_dir(args.path if hasattr(args, 'path') else '.')
    print(f'Run directory: {run_dir}')

    params = _parse_dirname(run_dir)
    tau, N_k = params['tau'], params['N_k']
    n_half   = N_k // 2
    idx_obj  = N_k

    all_txts = natsort.natsorted(
        list(run_dir.glob('population_*.txt')), alg=natsort.ns.IGNORECASE
    )
    if not all_txts:
        raise FileNotFoundError(f'No population_*.txt files found in {run_dir}')

    # Collect best-per-generation (row 0 = best by convention in legacy runs)
    best_objs   = []
    best_upper  = []   # list of K_upper arrays
    best_lower  = []   # list of K_lower arrays

    for txt_path in all_txts:
        data = np.loadtxt(str(txt_path))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        best_row = data[int(np.argmin(data[:, idx_obj]))]
        best_objs.append(float(best_row[idx_obj]))
        best_upper.append(best_row[:n_half])
        best_lower.append(best_row[n_half:N_k])

    gens = list(range(len(best_objs)))

    # 1. Objective evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best_objs, color=default_color_cycle.black, linewidth=1.2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Objective')
    ax.set_title(r'Objective evolution  ($\tau$' + f'$={tau:.2f}$)')
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    ext = getattr(args, 'ext', 'png')
    out = str(run_dir / f'objective_evolution.{ext}')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved → {out}')

    # 1b. Objective evolution — zoomed
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best_objs, color=default_color_cycle.black, linewidth=1.2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Objective')
    ax.set_title(r'Objective evolution (zoomed)  ($\tau$' + f'$={tau:.2f}$)')
    zoom_center = round(best_objs[-1] / 10, 0) * 10
    try:
        ax.set_ylim([zoom_center - 20, zoom_center + 20])
    except Exception:
        pass
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_zoom = str(run_dir / f'objective_evolution_zoomed.{ext}')
    fig.savefig(out_zoom, dpi=200)
    plt.close(fig)
    print(f'Saved → {out_zoom}')

    # 2. Design-variable evolution
    fig, ax = plt.subplots(figsize=(10, 5))
    upper_arr = np.array(best_upper)   # shape (N_gen, n_half)
    lower_arr = np.array(best_lower)
    for i in range(n_half):
        ax.plot(gens, upper_arr[:, i],
                color=default_color_cycle.blue, alpha=0.6, linewidth=0.9)
        ax.plot(gens, lower_arr[:, i],
                color=default_color_cycle.orange, alpha=0.6, linewidth=0.9)
    # phantom lines for legend
    ax.plot([], [], color=default_color_cycle.blue,   linewidth=1.5, label='Upper surface')
    ax.plot([], [], color=default_color_cycle.orange, linewidth=1.5, label='Lower surface')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Kulfan coefficient')
    ax.set_title(r'Design-variable evolution  ($\tau$' + f'$={tau:.2f}$)')
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    ext = getattr(args, 'ext', 'png')
    out = str(run_dir / f'variable_evolution.{ext}')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Cp comparison ─────────────────────────────────────────────────────────────


def _cp_xfoil(K_upper, K_lower, alpha, Re, N_crit, xtp_u, xtp_l, te_gap):
    """Run xfoil at a fixed alpha and return cp_data dict, or None on failure."""
    res = xfoil_run(
        'alfa', K_upper, K_lower, val=alpha,
        Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
        TE_gap=te_gap, timelimit=15,
        save_boundary_layer_data=True,
    )
    if res is None:
        return None
    cp = res.get('cp_data')
    if cp is None or not cp.get('x'):
        return None
    return cp


def _alpha_at_cl(K_upper, K_lower, CL, Re, te_gap,
                 N_crit=9.0, xtp_u=1.0, xtp_l=1.0,
                 alpha_range=(-5, 20, 0.5)):
    """Find the alpha that gives the target CL via a coarse alpha sweep."""
    a_start, a_end, a_step = alpha_range
    alphas = np.arange(a_start, a_end + a_step * 0.5, a_step)
    res = xfoil_run(
        'alfa', K_upper, K_lower, val=list(alphas),
        Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
        TE_gap=te_gap, timelimit=20,
    )
    if res is None:
        return None
    try:
        cl_arr = np.array(res['cl'])
        al_arr = np.array(res['alpha'])
        # only use the pre-stall monotonic portion
        peak = int(np.argmax(cl_arr))
        if peak < 2:
            return None
        return float(np.interp(CL, cl_arr[:peak], al_arr[:peak]))
    except Exception:
        return None


def _run_cp(args: argparse.Namespace) -> None:
    """Plot Cp distributions at design CL (clean and rough) vs reference airfoils."""
    txt_path = _resolve_txt_last(args.path if hasattr(args, 'path') else '.')
    run_dir  = txt_path.parent
    print(f'Using: {txt_path}')

    params = _parse_dirname(run_dir)
    tau, Re, CL, te_gap, N_k = (params[k] for k in ('tau', 'Re', 'CL', 'te_gap', 'N_k'))
    n_half  = N_k // 2
    idx_obj = N_k

    data     = np.loadtxt(str(txt_path))
    if data.ndim == 1:
        data = data[np.newaxis, :]
    best_row   = data[int(np.argmin(data[:, idx_obj]))]
    K_upper    = best_row[:n_half].tolist()
    K_lower    = best_row[n_half:N_k].tolist()
    print(f'  Best individual: obj={best_row[idx_obj]:.4f}')
    print(f'  Finding alpha at CL={CL:.3f} (clean sweep)...')
    alpha_design = _alpha_at_cl(K_upper, K_lower, CL, Re, te_gap)
    if alpha_design is None:
        print('  Warning: could not find alpha_design from clean sweep — falling back to stored col N_k+2.')
        alpha_design = float(best_row[N_k + 2]) if best_row.shape[0] > N_k + 2 else 5.0
    print(f'  alpha_design = {alpha_design:.3f} deg')

    # For reference airfoils, also find alpha at CL (clean) independently
    compare_list = _expand_compare(args.compare)
    refs         = _resolve_references(compare_list, tau)

    conditions = [
        ('clean', 9.0, 1.0,  1.0 ),
        ('rough', 3.0, 0.05, 0.05),
    ]

    for cond_tag, N_crit, xtp_u, xtp_l in conditions:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Reference airfoils
        for ref_idx, (stem, fam_dir_key, alias) in enumerate(refs):
            ref_kulf = _load_ref_kulfan(stem, fam_dir_key)
            if ref_kulf is None:
                print(f'  Warning: no geometry for {stem} — skipping from Cp.')
                continue
            ref_te = float(ref_kulf.TE_gap) if hasattr(ref_kulf, 'TE_gap') else te_gap
            # find alpha at CL for this reference airfoil (clean sweep)
            ref_alpha = _alpha_at_cl(
                ref_kulf.upperCoefficients, ref_kulf.lowerCoefficients,
                CL, Re, ref_te,
            )
            if ref_alpha is None:
                ref_alpha = alpha_design   # fallback
            cp = _cp_xfoil(
                ref_kulf.upperCoefficients, ref_kulf.lowerCoefficients,
                ref_alpha, Re, N_crit, xtp_u, xtp_l, ref_te,
            )
            if cp is None:
                print(f'  {stem} ({cond_tag}): xfoil did not converge — skipping.')
                continue
            color = _REF_COLORS[ref_idx % len(_REF_COLORS)]
            ax.plot(cp['x'], cp['cp'], color=color, alpha=0.75, linewidth=1.2, label=stem)

        # Optimized airfoil
        cp_opt = _cp_xfoil(K_upper, K_lower, alpha_design, Re, N_crit, xtp_u, xtp_l, te_gap)
        if cp_opt is not None:
            ax.plot(cp_opt['x'], cp_opt['cp'],
                    color=default_color_cycle.black, linewidth=2.0, label='optimized')
        else:
            print(f'  Optimized airfoil Cp ({cond_tag}): xfoil did not converge.')

        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(r'$x/c$')
        ax.set_ylabel(r'$C_p$')
        ax.set_title(
            r'$C_p$ comparison  '
            + r'($\tau$' + f'$={tau:.2f}$, '
            + r'$C_L$' + f'$={CL:.2f}$'
            + r' $\rightarrow$ $\alpha$' + f'$={alpha_design:.2f}' + r'^{\circ}$, '
            + f'{cond_tag})'
        )
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        ext = getattr(args, 'ext', 'png')
        out = str(run_dir / f'cp_comparison_{cond_tag}.{ext}')
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f'Saved → {out}')


# ── Polar ─────────────────────────────────────────────────────────────────────


def _run_polar(args: argparse.Namespace) -> None:
    txt_path = _resolve_txt_last(args.path if hasattr(args, 'path') else '.')
    run_dir  = txt_path.parent
    print(f'Using: {txt_path}')

    params = _parse_dirname(run_dir)
    tau, Re, CL, te_gap, N_k = (params[k] for k in ('tau', 'Re', 'CL', 'te_gap', 'N_k'))
    print(f'Parsed: tau={tau:.3f}, CL={CL:.2f}, Re={Re:.2e}, N_k={N_k}, TE_gap={te_gap:.5f}')

    compare_list = _expand_compare(args.compare)
    refs         = _resolve_references(compare_list, tau)

    individuals = _load_best_airfoils(txt_path, N_k)
    n_use       = min(args.n_airfoils, len(individuals))
    print(f'  Found {len(individuals)} feasible; using best {n_use}.')

    # Build airfoil list — opt airfoils first (black), then references (cycle)
    airfoils       = []
    color_override = {}

    for i, ind in enumerate(individuals[:n_use]):
        label = f'best_{i + 1}'
        afl   = Kulfan(TE_gap=te_gap)
        afl.upperCoefficients = ind['K_upper']
        afl.lowerCoefficients = ind['K_lower']
        airfoils.append([label, afl])
        color_override[label] = default_color_cycle.black

    for ref_idx, (stem, fam_dir_key, _alias) in enumerate(refs):
        airfoils.append(stem)
        color_override[stem] = _REF_COLORS[ref_idx % len(_REF_COLORS)]

    ext = getattr(args, 'ext', 'png')
    figure_path = args.polar_output if args.polar_output is not None else str(run_dir / f'polar_compare_plot.{ext}')

    print(f'Running polars (tool={args.tool}, Re={Re:.2e}, tau={tau:.3f})...')
    run_and_plot_polars_compare(
        airfoils         = airfoils,
        reynolds_numbers = [Re],
        turb_cases       = TURB_CASES_CLEAN + TURB_CASES_ROUGH,
        tools            = [args.tool],
        figure_path      = figure_path,
        sweep_param      = 'alpha',
        sweep_range      = ALPHA_RANGE,
        load_geometry    = True,
        save_data        = SAVE_DATA,
        afl_root         = _DEFAULT_AFL_ROOT,
        color_override   = color_override if color_override else None,
        show_cpmin       = False,
        cl_design        = CL,
    )
    print(f'Saved → {figure_path}')


# ── GIF ───────────────────────────────────────────────────────────────────────


def _gen_number_from_path(p: pathlib.Path) -> int:
    """Extract generation number from a population filename (e.g. _g376.txt → 376)."""
    m = re.search(r'_g(\d+)\.txt$', p.name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _run_gif(args: argparse.Namespace,
             mpi_comm=None, mpi_rank: int = 0, mpi_size: int = 1) -> None:
    run_dir = _resolve_run_dir(args.path if hasattr(args, 'path') else '.')
    print(f'Run directory: {run_dir}')

    params = _parse_dirname(run_dir)
    tau, te_gap, N_k = params['tau'], params['te_gap'], params['N_k']
    n_half = N_k // 2
    print(f'Parsed: tau={tau:.3f}, N_k={N_k}, TE_gap={te_gap:.5f}')

    # Collect all generation files
    all_txts = natsort.natsorted(
        list(run_dir.glob('population_*.txt')), alg=natsort.ns.IGNORECASE
    )
    all_txts = [pathlib.Path(p) for p in all_txts]
    if not all_txts:
        raise FileNotFoundError(f'No population_*.txt files found in {run_dir}')

    # Apply --every stride
    all_txts = all_txts[::args.every]
    print(f'  {len(all_txts)} generation files to render (stride={args.every}).')

    # Resolve reference airfoils
    compare_list = _expand_compare(args.compare)
    refs         = _resolve_references(compare_list, tau)

    # Pre-load reference Kulfan objects
    ref_afls: list[tuple[str, Kulfan, str]] = []  # (label, kulfan, color)
    for ref_idx, (stem, fam_dir_key, alias) in enumerate(refs):
        kulf = _load_ref_kulfan(stem, fam_dir_key)
        if kulf is None:
            print(f'  Warning: could not load geometry for {stem} — skipping from GIF.')
            continue
        ref_afls.append((stem, kulf, _REF_COLORS[ref_idx % len(_REF_COLORS)]))

    # Prepare frames directory
    frames_dir = run_dir / args.frames_dir
    frames_dir.mkdir(exist_ok=True)

    frame_paths = []
    idx_obj     = N_k
    idx_con_tag = N_k + 1

    # Each rank renders only its assigned frames (interleaved distribution)
    for frame_i, txt_path in enumerate(all_txts):
        if frame_i % mpi_size != mpi_rank:
            continue   # not this rank's frame

        gen_num = _gen_number_from_path(txt_path)

        data = np.loadtxt(str(txt_path))
        if data.ndim == 1:
            data = data[np.newaxis, :]

        # Find best individual (min obj)
        best_row_idx = int(np.argmin(data[:, idx_obj]))
        best_row     = data[best_row_idx]
        best_obj     = float(best_row[idx_obj])

        fig, ax = plt.subplots(figsize=(10, 4), dpi=args.dpi)

        # Plot all population airfoils faintly
        for row_i, row in enumerate(data):
            if row_i == best_row_idx:
                continue
            afl = Kulfan(TE_gap=te_gap)
            afl.upperCoefficients = row[:n_half].tolist()
            afl.lowerCoefficients = row[n_half:N_k].tolist()
            ax.plot(afl.xcoordinates, afl.ycoordinates,
                    color=default_color_cycle.black, alpha=0.015, linewidth=0.5)

        # Plot reference airfoils
        for label, kulf, color in ref_afls:
            ax.plot(kulf.xcoordinates, kulf.ycoordinates,
                    color=color, linewidth=1.5, label=label, alpha=0.85)

        # Highlight best individual
        best_afl = Kulfan(TE_gap=te_gap)
        best_afl.upperCoefficients = best_row[:n_half].tolist()
        best_afl.lowerCoefficients = best_row[n_half:N_k].tolist()
        ax.plot(best_afl.xcoordinates, best_afl.ycoordinates,
                color=default_color_cycle.black, linewidth=2.0,
                label=f'best (obj={best_obj:.4f})')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.25, 0.25)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$x/c$')
        ax.set_ylabel(r'$y/c$')
        ax.set_title(r'$\tau$' + f'$={tau:.2f}$, gen $={gen_num}$, '
                     + r'$\mathrm{obj}$' + f'$={best_obj:.4f}$')
        ax.grid(True, linewidth=0.4, alpha=0.5)
        if ref_afls:
            ax.legend(loc='upper right', fontsize=8)

        frame_path = frames_dir / f'frame_{frame_i:05d}.png'
        fig.tight_layout()
        fig.savefig(str(frame_path), dpi=args.dpi, bbox_inches='tight')
        frame_paths.append(frame_path)
        plt.close(fig)

        if (frame_i + 1) % 50 == 0 or frame_i == len(all_txts) - 1:
            print(f'  [rank {mpi_rank}] Rendered frame {frame_i + 1}/{len(all_txts)} (gen {gen_num})')

    # Wait for all ranks to finish rendering before assembling
    if mpi_comm is not None:
        mpi_comm.Barrier()

    # Only rank 0 assembles the GIF
    if mpi_rank != 0:
        return

    # Collect all frame paths in sorted order
    frame_paths = natsort.natsorted(
        list(frames_dir.glob('frame_*.png')), alg=natsort.ns.IGNORECASE
    )

    # Assemble GIF
    gif_path = args.gif_output if args.gif_output is not None else str(run_dir / 'airfoil_evolution.gif')
    print(f'Assembling GIF ({len(frame_paths)} frames) → {gif_path}')

    def _frame_gen(paths):
        for p in paths:
            with Image.open(str(p)) as im:
                yield im.copy()

    frames_iter = _frame_gen(frame_paths)
    try:
        first_frame = next(frames_iter)
    except StopIteration:
        print('No frames generated.')
        return

    frames_list = list(frames_iter)
    first_frame.save(
        gif_path,
        save_all=True,
        append_images=frames_list,
        duration=args.duration,
        loop=0,
    )
    print(f'Saved → {gif_path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def _relaunch_with_mpirun() -> None:
    """Re-exec the current process under mpirun if not already running under MPI."""
    import os, shutil, subprocess, sys
    # Already under MPI if PMI_RANK / OMPI_COMM_WORLD_RANK / SLURM_PROCID is set
    mpi_env_keys = ('PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'SLURM_PROCID',
                    'MV2_COMM_WORLD_RANK', 'MPI_LOCALRANKID')
    if any(k in os.environ for k in mpi_env_keys):
        return  # already inside an mpirun launch

    # Skip MPI relaunch if the user disabled the GIF (serial tasks only)
    _no_gif_flags = {'--no-gif', '--nogif', '--polar-only', '--history-only', '--cp-only'}
    if any(f in sys.argv for f in _no_gif_flags):
        return

    mpirun = shutil.which('mpirun') or shutil.which('mpiexec')
    if mpirun is None:
        return  # mpirun not on PATH, fall back to serial

    n_cpus = os.cpu_count() or 1
    cmd = [mpirun, '-n', str(n_cpus), sys.executable, '-m',
           'oso_airfoils.postprocessing.oso_legacy'] + sys.argv[1:]
    print(f'[oso-legacy] Relaunching under MPI with {n_cpus} ranks...')
    sys.exit(subprocess.call(cmd))


def main() -> None:
    _relaunch_with_mpirun()

    parser = argparse.ArgumentParser(
        prog='oso-legacy',
        description=(
            'Legacy postprocessing for optimizer runs (cases 64-69, .txt population files). '
            'Runs both the polar compare plot and evolution GIF by default.'
        ),
    )
    parser.add_argument('path', nargs='?', default='.',
        help='Population .txt file or run directory (default: current directory).')

    # Shared: comparison families
    parser.add_argument('-c', '--compare', nargs='+', metavar='FAMILY', default=[],
        help=('Reference families to overlay (tau-matched). '
              'Choices: du ffa mhkf1 risoa risob risop s20 s40 osowt1 osowt2 osowt2s '
              'or "all" (expands to: du ffa risoa risob risop s20 s40).'))

    # Polar-specific
    parser.add_argument('-t', '--tool', default='xfoil',
        choices=['neuralfoil', 'xfoil'],
        help='Aerodynamic solver for polar (default: xfoil).')
    parser.add_argument('-n', '--n-airfoils', type=int, default=N_BEST_AIRFOILS,
        dest='n_airfoils',
        help=f'Number of best feasible airfoils to plot in polar (default: {N_BEST_AIRFOILS}).')
    parser.add_argument('--polar-output', default=None, dest='polar_output',
        help='Polar figure path (default: <run_dir>/polar_compare_plot.png).')

    # GIF-specific
    parser.add_argument('--gif-output', default=None, dest='gif_output',
        help='GIF output path (default: <run_dir>/airfoil_evolution.gif).')
    parser.add_argument('--frames-dir', default='_gif_frames', dest='frames_dir',
        help='Subdirectory for temporary frame PNGs (default: _gif_frames).')
    parser.add_argument('--dpi', type=int, default=200,
        help='GIF frame DPI (default: 200).')
    parser.add_argument('--duration', type=int, default=100,
        help='Milliseconds per frame in the GIF (default: 100).')
    parser.add_argument('--every', type=int, default=1, dest='every',
        help='Use every Nth generation file for GIF (default: 1 = all).')
    parser.add_argument('-x', '--ext', default='png', dest='ext',
        help='Output file extension for plots (png, pdf, pgf, svg, …). Default: png.')

    # Mode selectors
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--polar-only', action='store_true', default=False,
        help='Run only the polar compare plot.')
    mode_group.add_argument('--gif-only', action='store_true', default=False,
        help='Run only the evolution GIF.')
    mode_group.add_argument('--history-only', action='store_true', default=False,
        help='Run only the objective / design-variable history plots.')
    mode_group.add_argument('--cp-only', action='store_true', default=False,
        help='Run only the Cp comparison plots.')

    # Skip flags
    parser.add_argument('--no-gif', '--nogif', action='store_true', default=False,
        dest='no_gif',
        help='Skip the evolution GIF (runs serial; implies no MPI relaunch).')

    args = parser.parse_args()

    run_polar   = not (args.gif_only or args.history_only or args.cp_only)
    run_gif     = not (args.polar_only or args.history_only or args.cp_only or args.no_gif)
    run_history = not (args.polar_only or args.gif_only or args.cp_only)
    run_cp      = not (args.polar_only or args.gif_only or args.history_only)

    # MPI setup
    if _HAS_MPI:
        _comm = _MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    else:
        _comm, _rank, _size = None, 0, 1

    if run_history and _rank == 0:
        print('\n── History ──────────────────────────────────────────────')
        _run_history(args)

    if run_cp and _rank == 0:
        print('\n── Cp comparison ────────────────────────────────────────')
        _run_cp(args)

    if run_polar and _rank == 0:
        print('\n── Polar ────────────────────────────────────────────────')
        _run_polar(args)

    if run_gif:
        if _rank == 0:
            print('\n── GIF ──────────────────────────────────────────────────')
            if _size > 1:
                print(f'  Rendering frames across {_size} MPI ranks.')
        _run_gif(args, mpi_comm=_comm, mpi_rank=_rank, mpi_size=_size)


if __name__ == '__main__':
    main()
