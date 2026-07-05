"""Cp alpha-sweep plot for a reference airfoil.

Runs xfoil at a range of angles of attack and overlays the Cp distributions
on a single inverted-axis plot, coloured by a sequential colormap so the
progression with alpha is visually clear.

Usage:
    python -m oso_airfoils.postprocessing.cp_alpha_sweep AIRFOIL_NAME
                       [--alpha-start ALPHA] [--alpha-stop ALPHA] [--alpha-step ALPHA]
                       [--re RE] [--n-crit N] [--xtp-u F] [--xtp-l F]
                       [--cmap CMAP] [-o OUTPUT]

Examples:
    python -m oso_airfoils.postprocessing.cp_alpha_sweep ffa-w3-270
    python -m oso_airfoils.postprocessing.cp_alpha_sweep du_91-w2-250 --alpha-start 0 --alpha-stop 12 --alpha-step 2
    python -m oso_airfoils.postprocessing.cp_alpha_sweep s830 --re 3e6 --rough
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT
from oso_airfoils.core.xfoil_wrapper import run as xfoil_run
from oso_airfoils.postprocessing.runners import _find_stem_in_tree, _load_kulfan
from oso_airfoils.postprocessing.polars import get_colors

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 13})

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_ALPHA_START = -2.0
DEFAULT_ALPHA_STOP  = 14.0
DEFAULT_ALPHA_STEP  =  2.0
DEFAULT_RE          = 3e6
DEFAULT_N_CRIT      = 9.0
DEFAULT_XTP         = 1.0
DEFAULT_CMAP        = 'coolwarm'
_CMAP_LOWER         = 0.30
_CMAP_UPPER         = 0.80
# ─────────────────────────────────────────────────────────────────────────────


def _plot_cp_split(ax, cp_data: dict, color, alpha_val: float,
                   linewidth: float = 1.4, line_alpha: float = 0.85) -> None:
    """Plot Cp split into upper/lower surface with a single color."""
    cp   = cp_data['cp']
    x    = cp_data['x']
    stag = cp.index(max(cp))
    ax.plot(x[:stag + 1], cp[:stag + 1], color=color,
            linewidth=linewidth, alpha=line_alpha)
    ax.plot(x[stag:],     cp[stag:],     color=color,
            linewidth=linewidth, alpha=line_alpha)


def cp_alpha_sweep(
    airfoil_name: str,
    alphas: list[float],
    Re: float = DEFAULT_RE,
    N_crit: float = DEFAULT_N_CRIT,
    xtp_u: float = DEFAULT_XTP,
    xtp_l: float = DEFAULT_XTP,
    cmap_name: str = DEFAULT_CMAP,
    figure_path: str | None = None,
    dpi: int = 300,
) -> 'matplotlib.figure.Figure':
    """Run xfoil at each alpha in *alphas* and plot the Cp distributions.

    Parameters
    ----------
    airfoil_name : str
        Airfoil name (resolved via ``_find_stem_in_tree``).
    alphas : list[float]
        Angles of attack [deg].
    Re, N_crit, xtp_u, xtp_l
        Xfoil run parameters.
    cmap_name : str
        Matplotlib colormap name for the alpha sweep colouring.
    figure_path : str, optional
        Save path; skipped if None.
    """
    # Load geometry
    family, stem = _find_stem_in_tree(airfoil_name, _DEFAULT_AFL_ROOT)
    afl = _load_kulfan(family, stem, _DEFAULT_AFL_ROOT)
    print(f'Loaded: {stem}  (family={family})')

    te_gap = float(afl.TE_gap) if hasattr(afl, 'TE_gap') else 0.0

    # Colormap: clipped exactly as BL rainbow plots (lower=0.30, upper=0.80)
    n      = len(alphas)
    colors = get_colors(n, cmap_name, lower=_CMAP_LOWER, upper=_CMAP_UPPER)

    # Figure: 2 rows — Cp top, airfoil bottom
    fig, (ax_cp, ax_afl) = plt.subplots(
        2, 1, figsize=(10, 8), dpi=dpi,
        gridspec_kw={'height_ratios': [3, 1]},
    )

    xlims = (-0.05, 1.05)
    tick_h = 0.04

    # Cp axis decorations
    ax_cp.plot(xlims, [0, 0], color='k', linewidth=0.7)
    for xt in np.linspace(0, 1, 11):
        ax_cp.plot([xt, xt], [-tick_h, tick_h], color='k', linewidth=0.5)
    ax_cp.set_xlim(xlims)
    ax_cp.set_ylabel(r'$C_p$')
    ax_cp.spines[['right', 'top', 'bottom']].set_visible(False)
    ax_cp.set_xticks([])

    all_cp_min: list[float] = []
    converged_alphas: list[float] = []
    converged_colors: list = []

    for i, alpha in enumerate(alphas):
        print(f'  Running alpha={alpha:.1f}°...')
        res = xfoil_run(
            'alfa',
            afl.upperCoefficients, afl.lowerCoefficients,
            val=alpha,
            Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
            TE_gap=te_gap, timelimit=20,
            save_boundary_layer_data=True,
        )
        if res is None:
            print(f'    → xfoil returned None, skipping.')
            continue
        cp_data = res.get('cp_data')
        if cp_data is None or not cp_data.get('cp'):
            print(f'    → no cp_data, skipping.')
            continue

        _plot_cp_split(ax_cp, cp_data, color=colors[i], alpha_val=alpha)
        all_cp_min.append(min(cp_data['cp']))
        converged_alphas.append(alpha)
        converged_colors.append(colors[i])

    if not converged_alphas:
        print('No converged solutions — nothing to plot.')
        plt.close(fig)
        return fig

    # y-axis: inverted, floor at min Cp - 0.1 (at least -2)
    cp_floor = min(all_cp_min) - 0.1
    ax_cp.set_ylim([min(-2.0, cp_floor), 1.1])
    ax_cp.invert_yaxis()

    # BL-style legend: explicit Line2D handles, upper right
    handles = [
        Line2D([0], [0], color=c, linewidth=1.4,
               label=rf'$\alpha={a:.1f}^\circ$')
        for c, a in zip(converged_colors, converged_alphas)
    ]
    ax_cp.legend(
        handles=handles,
        fontsize=8, loc='upper right', ncols=2,
        framealpha=0.7, handlelength=1.2,
    )

    # Title
    cond = (f'$N_{{cr}}={N_crit:.0f}$, '
            + (r'$x_{tp}=1$' if xtp_u >= 1.0 else rf'$x_{{tp}}={xtp_u:.2f}$')
            + f',  $Re={Re:.2e}$')
    ax_cp.set_title(
        rf'\textbf{{{stem.replace("_", "-")}}}  —  $C_p$ sweep  ({cond})',
        fontsize=12,
    )

    # Airfoil panel
    ax_afl.plot(afl.xcoordinates, afl.ycoordinates, 'k', linewidth=1.0)
    ax_afl.set_xlim(xlims)
    ax_afl.set_aspect('equal')
    ax_afl.set_xticks([])
    ax_afl.set_yticks([])
    ax_afl.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
    ax_afl.set_xlabel(r'$x/c$')

    plt.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved → {figure_path}')

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='cp-alpha-sweep',
        description='Cp alpha-sweep plot for a reference airfoil.',
    )
    parser.add_argument('airfoil', help='Airfoil name (e.g. ffa-w3-270, du_91-w2-250).')
    parser.add_argument('--alpha-start', type=float, default=DEFAULT_ALPHA_START,
        dest='alpha_start', help=f'Start angle [deg] (default: {DEFAULT_ALPHA_START}).')
    parser.add_argument('--alpha-stop', type=float, default=DEFAULT_ALPHA_STOP,
        dest='alpha_stop', help=f'Stop angle [deg] (default: {DEFAULT_ALPHA_STOP}).')
    parser.add_argument('--alpha-step', type=float, default=DEFAULT_ALPHA_STEP,
        dest='alpha_step', help=f'Step [deg] (default: {DEFAULT_ALPHA_STEP}).')
    parser.add_argument('--re', type=float, default=DEFAULT_RE,
        help=f'Reynolds number (default: {DEFAULT_RE:.0e}).')
    parser.add_argument('--n-crit', type=float, default=DEFAULT_N_CRIT,
        dest='n_crit', help=f'N_crit (default: {DEFAULT_N_CRIT}).')
    parser.add_argument('--xtp-u', type=float, default=DEFAULT_XTP,
        dest='xtp_u', help='Forced transition upper (default: 1.0 = free).')
    parser.add_argument('--xtp-l', type=float, default=DEFAULT_XTP,
        dest='xtp_l', help='Forced transition lower (default: 1.0 = free).')
    rough_group = parser.add_mutually_exclusive_group()
    rough_group.add_argument('--rough', action='store_true', default=False,
        help='Rough preset: N_crit=3, xtp=0.5/0.5.')
    rough_group.add_argument('--clean', action='store_true', default=False,
        help='Clean preset: N_crit=9, xtp=1.0/1.0 [default].')
    parser.add_argument('--cmap', default=DEFAULT_CMAP,
        help=f'Matplotlib colormap (default: {DEFAULT_CMAP}).')
    parser.add_argument('-o', '--output', default=None,
        help='Output file path (default: <airfoil>_cp_sweep.png in cwd).')
    parser.add_argument('--dpi', type=int, default=300,
        help='Output DPI (default: 300).')
    args = parser.parse_args()

    # Apply rough/clean presets
    if args.rough:
        n_crit, xtp_u, xtp_l = 3.0, 0.5, 0.5
    else:
        n_crit = args.n_crit
        xtp_u  = args.xtp_u
        xtp_l  = args.xtp_l

    alphas = list(np.arange(args.alpha_start,
                             args.alpha_stop + args.alpha_step * 0.5,
                             args.alpha_step))

    figure_path = args.output or f'{args.airfoil}_cp_sweep.png'

    cp_alpha_sweep(
        airfoil_name=args.airfoil,
        alphas=alphas,
        Re=args.re,
        N_crit=n_crit,
        xtp_u=xtp_u,
        xtp_l=xtp_l,
        cmap_name=args.cmap,
        figure_path=figure_path,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
