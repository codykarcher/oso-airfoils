"""oso_airfoils.postprocessing.boundary_layer
=============================================

Boundary-layer plots built from pre-computed aerodynamic run records.

Public API
----------
boundary_layer_compare(data_dict, ...)
    Fixed-colour plot for up to two BL datasets.  Upper surface: blue,
    lower: orange-red, wake: gold.  A second dataset gets lighter shades.

boundary_layer_rainbow(data_dict, ...)
    Rainbow plot for an arbitrary number of datasets.  Uses Blues, Oranges,
    and Greens colourmaps for upper, lower, and wake respectively.  The
    swept parameter is auto-detected from the records.

Input format
------------
Both functions accept::

    data_dict = {airfoil_name: [run_records]}

where each run record is a dict matching the JSON performance-file schema
(keys: ``alpha``, ``Re``, ``N_crit``, ``bl_data``, ``cp_data``, …).
Records without ``bl_data`` are silently skipped.

Style overrides
---------------
Pass a ``style`` dict.  Recognised keys and defaults::

    {
        # colours — compare mode
        'color_upper_1':  '#0065cc',
        'color_lower_1':  '#d55e00',
        'color_wake_1':   '#e69f00',
        'color_upper_2':  '#56b4ff',
        'color_lower_2':  '#fca7c7',
        'color_wake_2':   '#ede13f',
        # colormaps — rainbow mode
        'cmap_upper':     'Blues',
        'cmap_lower':     'Oranges',
        'cmap_wake':      'Greens',
        'cmap_lower_bound': 0.30,
        'cmap_upper_bound': 0.80,
        # layout / rendering
        'linewidth':      1.5,
        'fontsize':       15,
        'legend_fontsize': 7,
        'fig_width':      10,
        'fig_height':     14,
        'dpi':            300,
        'airfoil_color':  'k',
        'airfoil_lw':     1.0,
    }
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from oso_airfoils.postprocessing.polars import computeNormals, get_colors, handleZeroDivide
from oso_airfoils.core.colors import default_color_cycle as dcc

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
    color=dcc
)

# ── parameter aliases ─────────────────────────────────────────────────────────

alternativeNameDict: dict[str, str] = {
    'none': 'None', 'None': 'None',
    # CP
    'cp': 'CP', 'pressure_coefficient': 'CP',
    'coefficient_of_pressure': 'CP', 'pressure': 'CP',
    # Ue/Vinf
    'ue/vinf': 'Ue/Vinf', 'normalized_velocity': 'Ue/Vinf',
    'velocity': 'Ue/Vinf', 'edge_velocity': 'Ue/Vinf',
    'u_edge': 'Ue/Vinf', 'ue_over_vinf': 'Ue/Vinf',
    'ue_over_vinfinity': 'Ue/Vinf', 'freestream_normalized_velocity': 'Ue/Vinf',
    # Dstar
    'dstar': 'Dstar', 'deltastar': 'Dstar', 'delta_star': 'Dstar',
    'displacement_thickness': 'Dstar',
    # Theta
    'theta': 'Theta', 'momentum_thickness': 'Theta',
    # Tstar
    'tstar': 'Tstar', 'theta_star': 'Tstar',
    'kinetic_energy_thickness': 'Tstar', 'kinetic_thickness': 'Tstar',
    # H*
    'h*': 'H*', 'kinetic_energy_shape_parameter': 'H*',
    'kinetic_shape_parameter': 'H*', 'h_star': 'H*', 'hstar': 'H*',
    'thetastar_over_theta': 'H*',
    # H
    'h': 'H', 'shape_parameter': 'H', 'dstar_over_theta': 'H',
    'delta_over_theta': 'H', 'deltastar_over_theta': 'H',
    # m
    'm': 'm', 'mass_defect': 'm',
    # P
    'p': 'P', 'momentum_defect': 'P',
    # K
    'k': 'K', 'kinetic_defect': 'K', 'kinetic_energy_defect': 'K',
    # Cf
    'cf': 'Cf', 'skin_friction': 'Cf',
    'skin_friction_coefficient': 'Cf', 'friction_coefficient': 'Cf',
    # tau_w
    'tau_w': 'tau_w', 'tau': 'tau_w',
    'wall_shear_stress': 'tau_w', 'wall_stress': 'tau_w',
    'shear_stress': 'tau_w', 'wall_shear': 'tau_w',
}

# Ordered list of BL parameters shown when parameter=None
_BL_PARAMS: list[str] = [
    'CP',
    'H*', 'P', 'm', 'K', 'Tstar',
    'Ue/Vinf', 'H', 'Dstar', 'Theta', 'Cf', 'tau_w',
]

_LABEL_LOOKUP: dict[str, str] = {
    'CP':     r'$C_p$',
    'Ue/Vinf': r'$U_e / V_{\infty}$',
    'H*':     r'$H^*$',
    'H':      r'$H$',
    'P':      r'$P$',
    'm':      r'$m$',
    'Dstar':  r'$\delta^*$',
    'Theta':  r'$\theta$',
    'Cf':     r'$C_f$',
    'K':      r'$K$',
    'Tstar':  r'$\theta^*$',
    'tau_w':  r'$\tau_w$',
}

_DEFAULT_STYLE: dict[str, Any] = {
    # compare-mode fixed colours — BL data lines
    'color_upper_1':   '#0065cc',
    'color_lower_1':   '#d55e00',
    'color_wake_1':    '#e69f00',
    'color_upper_2':   '#56b4ff',
    'color_lower_2':   '#fca7c7',
    'color_wake_2':    '#ede13f',
    # compare-mode geometry contour colours
    'color_geom_1':    '#000000',
    'color_geom_2':    '#888888',
    # rainbow-mode colourmaps
    'cmap_upper':       'Blues',
    'cmap_lower':       'Oranges',
    'cmap_wake':        'Greens',
    'cmap_geom':        'Greys',
    'cmap_lower_bound': 0.30,
    'cmap_upper_bound': 0.80,
    'cmap_geom_lower':  0.20,
    'cmap_geom_upper':  0.65,
    # layout
    'linewidth':        1.5,
    'fontsize':         15,
    'legend_fontsize':  11,
    'fig_width':        10,
    'fig_height':       14,
    'dpi':              300,
    'airfoil_color':    'k',
    'airfoil_lw':       1.0,
    'info_fontsize':    8,
}


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_stagnation_index(record: dict) -> int:
    """Return the stagnation index for a record.

    Uses the stored ``stagnation_index`` field when present (rfoil data).
    Falls back to ``cp_data['cp'].index(max(cp_data['cp']))`` otherwise
    (xfoil data with boundary-layer output).

    Raises
    ------
    ValueError
        When neither source of stagnation information is available.
    """
    idx = record.get('stagnation_index')
    if idx is not None:
        return int(idx)
    cp_data = record.get('cp_data')
    if cp_data and cp_data.get('cp'):
        cp = cp_data['cp']
        return cp.index(max(cp))
    raise ValueError(
        "Cannot determine stagnation index: record has no 'stagnation_index' "
        "and no 'cp_data'."
    )


def _y_is_valid(y_vals: list) -> bool:
    """Return True if y coordinates contain real (non-NaN) values."""
    for v in y_vals:
        if v is None:
            return False
        try:
            if math.isnan(v):
                return False
        except TypeError:
            pass
    return True


def _compute_dstar_viz(bl: dict, normals: list, stag_idx: int, wake_idx: int):
    """Build displacement-thickness visualisation points along the airfoil.

    Returns a dict with keys ``x_upper``, ``ds_upper``, ``x_lower``,
    ``ds_lower`` (as lists), or ``None`` if the computation cannot be
    performed.
    """
    x_bl    = bl['x']
    y_bl    = bl['y']
    dstar   = bl.get('Dstar')
    if dstar is None:
        return None

    lower_te_shift = 0.0
    if (abs(x_bl[wake_idx] - 1.0) < 1e-3 and abs(y_bl[wake_idx]) < 1e-3):
        # blunt trailing edge
        lower_te_shift = y_bl[wake_idx - 1]

    viz_stag = stag_idx
    split_frac = None
    boundary_layer_pts: list = []

    for i, nm in enumerate(normals):
        x = x_bl[i]
        y = y_bl[i]
        dorun = True

        if i > 0 and x_bl[i] >= 1 and x_bl[i - 1] <= 1:
            dorun = False
            split_frac = (
                dstar[0] / (dstar[i - 1] + dstar[0])
                if (dstar[i - 1] + dstar[0]) != 0 else 0.5
            )

        if dorun and i > 0 and i != wake_idx:
            sc = dstar[i]
            if split_frac is not None:
                xshift = x - nm[0] * sc * (1 - split_frac)
                yshift = y - nm[1] * sc * (1 - split_frac) + lower_te_shift
                boundary_layer_pts = (
                    [[xshift + nm[0] * sc, yshift + nm[1] * sc]]
                    + boundary_layer_pts
                )
                viz_stag += 1
                boundary_layer_pts.append([xshift, yshift])
            else:
                boundary_layer_pts.append([x + nm[0] * sc, y + nm[1] * sc])

    viz_stag -= 1

    if not boundary_layer_pts:
        return None

    pts = np.array(boundary_layer_pts)
    return {
        'x_upper':  pts[:viz_stag + 1, 0].tolist(),
        'ds_upper': pts[:viz_stag + 1, 1].tolist(),
        'x_lower':  pts[viz_stag:, 0].tolist(),
        'ds_lower': pts[viz_stag:, 1].tolist(),
        'x_airfoil': x_bl[:wake_idx],
        'y_airfoil': y_bl[:wake_idx],
    }


def _prepare_record(record: dict) -> dict | None:
    """Pre-process a run record for BL plotting.

    Returns a dict with computed split data, or ``None`` if the record has
    no ``bl_data``.

    The returned dict contains the original record fields plus:

    ``plot``
        ``{param: {'upper': [x, y], 'lower': [x, y], 'wake': [x, y],
           'invert_y': bool, 'y_label': str}}``
    ``_viz``
        Displacement-thickness visualisation data (may be ``None``).
    ``_stag_idx``, ``_wake_idx``
        Integer indices used during splitting.
    """
    if record.get('bl_data') is None:
        return None

    # Work on copies so we do not mutate the caller's data
    bl = {k: list(v) if isinstance(v, (list, tuple)) else v
          for k, v in record['bl_data'].items()}
    cp_data = record.get('cp_data', {}) or {}

    # Derived quantities
    cf_arr  = np.array(bl.get('Cf',      [float('nan')] * len(bl['x'])))
    ue_arr  = np.array(bl.get('Ue/Vinf', [float('nan')] * len(bl['x'])))
    tau_raw = np.array(bl.get('tau',     [float('nan')] * len(bl['x'])))
    computed_tau = cf_arr * ue_arr ** 2
    tau_w = np.where(np.isnan(tau_raw), computed_tau, tau_raw)
    bl['tau_w'] = tau_w.tolist()

    hstar = np.array(bl.get('H*',    [float('nan')] * len(bl['x'])))
    theta = np.array(bl.get('Theta', [float('nan')] * len(bl['x'])))
    bl['Tstar'] = (hstar * theta).tolist()

    # Indices
    stag_idx = _get_stagnation_index(record)
    wake_idx = next(
        (k for k, v in enumerate(bl['x']) if v > 1.0),
        len(bl['x'])
    )

    # Displacement-thickness visualisation
    y_vals = bl.get('y', [])
    viz = None
    if y_vals and _y_is_valid(y_vals):
        try:
            normals = computeNormals(bl['x'], bl['y'])
            viz = _compute_dstar_viz(bl, normals, stag_idx, wake_idx)
        except Exception:
            viz = None

    # Split each parameter into upper / lower / wake segments
    plot: dict[str, dict] = {}

    # CP — uses cp_data x array for surface x-coordinates
    cp_list = cp_data.get('cp', [])
    if cp_list:
        cp_x = cp_data.get('x', bl['x'][:len(cp_list)])
        plot['CP'] = {
            'upper':    [list(reversed(cp_x[:stag_idx + 1])),
                         list(reversed(cp_list[:stag_idx + 1]))],
            'lower':    [bl['x'][stag_idx:wake_idx],
                         cp_list[stag_idx:]],
            'wake':     [],
            'invert_y': True,
            'y_label':  _LABEL_LOOKUP['CP'],
        }
    else:
        plot['CP'] = {'upper': [], 'lower': [], 'wake': [],
                      'invert_y': True, 'y_label': _LABEL_LOOKUP['CP']}

    for nm in _BL_PARAMS:
        if nm == 'CP':
            continue

        vals = bl.get(nm)
        if vals is None:
            plot[nm] = {'upper': [], 'lower': [], 'wake': [],
                        'invert_y': False, 'y_label': _LABEL_LOOKUP.get(nm, nm)}
            continue

        if nm == 'Ue/Vinf':
            upper_x = list(reversed(bl['x'][:stag_idx + 1]))
            upper_y = list(reversed(vals[:stag_idx + 1]))
            lower_x = bl['x'][stag_idx:wake_idx]
            lower_y = list(vals[stag_idx:wake_idx])

            # Ensure lower surface Ue/Vinf is non-negative (insert zero crossing)
            if lower_y and lower_y[0] > 0:
                if 0.0 not in lower_y:
                    for j, yv in enumerate(lower_y):
                        if yv < 0:
                            lower_y.insert(j, 0.0)
                            lower_x.insert(j, (lower_x[j - 1] + lower_x[j]) / 2.0)
                            break
                lower_y = np.abs(lower_y).tolist()

            if upper_y and upper_y[0] < 0:
                if 0.0 not in upper_y:
                    for j, yv in enumerate(upper_y):
                        if yv > 0:
                            upper_y.insert(j, 0.0)
                            upper_x.insert(j, (upper_x[j - 1] + upper_x[j]) / 2.0)
                            break
                upper_y = np.abs(upper_y).tolist()
                lower_y = np.abs(lower_y).tolist()

            plot[nm] = {
                'upper':    [upper_x, upper_y],
                'lower':    [lower_x, lower_y],
                'wake':     [],
                'invert_y': False,
                'y_label':  _LABEL_LOOKUP[nm],
            }
        else:
            wake: list = []
            if wake_idx < len(vals) and not math.isnan(vals[wake_idx]):
                wake = [bl['x'][wake_idx:], vals[wake_idx:]]

            plot[nm] = {
                'upper':    [list(reversed(bl['x'][:stag_idx + 1])),
                             list(reversed(vals[:stag_idx + 1]))],
                'lower':    [bl['x'][stag_idx:wake_idx],
                             vals[stag_idx:wake_idx]],
                'wake':     wake,
                'invert_y': False,
                'y_label':  _LABEL_LOOKUP.get(nm, nm),
            }

    out = dict(record)
    out['plot']      = plot
    out['_viz']      = viz
    out['_stag_idx'] = stag_idx
    out['_wake_idx'] = wake_idx
    return out


def _detect_sweep(records_with_names: list[tuple[str, dict]]):
    """Detect the swept parameter across a list of (name, record) pairs.

    Returns
    -------
    sweep_field : str
        Name of the varying field (or ``'_airfoil_name'``).
    sorted_pairs : list of (name, record)
        Input list sorted by the sweep value.
    sweep_values : list
        The unique sorted sweep values (for labelling).
    """
    candidates = ['alpha', 'Re', 'N_crit', 'xtp_top', 'xtp_bot', 'M']
    for field in candidates:
        vals = [r.get(field) for _, r in records_with_names]
        unique = set(v for v in vals if v is not None)
        if len(unique) > 1:
            sorted_pairs = sorted(
                records_with_names,
                key=lambda x: x[1].get(field, 0)
            )
            return field, sorted_pairs, sorted(unique)

    # All conditions identical — sweep over airfoil names
    sorted_pairs = sorted(records_with_names, key=lambda x: x[0])
    names = [n for n, _ in sorted_pairs]
    return '_airfoil_name', sorted_pairs, sorted(set(names))


def _sweep_label(field: str, record: dict, name: str) -> str:
    """Return a short label for the given sweep position."""
    labels = {
        'alpha':          lambda r: r'$\alpha = %.1f^\circ$' % r.get('alpha', 0),
        'Re':             lambda r: r'$Re = %.2e$' % r.get('Re', 0),
        'N_crit':         lambda r: r'$N_{crit} = %.1f$' % r.get('N_crit', 0),
        'xtp_top':        lambda r: r'$x_{tp,u} = %.2f$' % r.get('xtp_top', 0),
        'xtp_bot':        lambda r: r'$x_{tp,l} = %.2f$' % r.get('xtp_bot', 0),
        'M':              lambda r: r'$M = %.3f$' % r.get('M', 0),
        '_airfoil_name':  lambda r: name,
    }
    fn = labels.get(field)
    return fn(record) if fn else str(record.get(field, ''))


def _record_info_string(record: dict) -> tuple[str, str]:
    """Return two LaTeX info strings for a single-dataset annotation."""
    re    = record.get('Re',    0)
    alpha = record.get('alpha', 0)
    cl    = record.get('cl',    0)
    cm    = record.get('cm',    0)
    cd    = record.get('cd',    0)
    ncrit = record.get('N_crit', 0)
    xtr_u = record.get('xtr_top', 0)
    xtr_l = record.get('xtr_bot', 0)

    s1 = (r'$Re = %.2e \quad \alpha = %.4f^\circ \quad'
          r' C_L = %.4f \quad C_M = %.4f \quad C_D = %.4f$'
          % (re, alpha, cl, cm, cd))
    s2 = (r'$L/D = %.4f \quad N_{crit} = %.1f \quad'
          r' X_{tr,u} = %.4f \quad X_{tr,l} = %.4f$'
          % (cl / cd if cd else float('inf'), ncrit, xtr_u, xtr_l))
    return s1, s2


# ── axes drawing ──────────────────────────────────────────────────────────────

def _draw_param(ax, pdata: dict, color_upper, color_lower, color_wake,
                lw: float, first: bool, label: str | None = None):
    """Plot upper, lower, and wake lines for one parameter on one axes."""
    kw_upper = dict(color=color_upper, linewidth=lw)
    kw_lower = dict(color=color_lower, linewidth=lw)
    kw_wake  = dict(color=color_wake,  linewidth=lw)
    if label is not None:
        kw_upper['label'] = label

    upper = pdata['upper']
    lower = pdata['lower']
    wake  = pdata['wake']

    if upper and upper[0] and upper[1]:
        ax.plot(upper[0], upper[1], **kw_upper)
    if lower and lower[0] and lower[1]:
        ax.plot(lower[0], lower[1], **kw_lower)
    if wake and wake[0] and wake[1]:
        ax.plot(wake[0], wake[1], **kw_wake)

    if first:
        ax.set_ylabel(pdata['y_label'])
        if pdata.get('invert_y'):
            ax.invert_yaxis()
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(True, which='minor', linewidth=0.4, alpha=0.4)
        ax.set_xlabel(r'$x/c$')


def _make_bl_figure(n_params: int, st: dict):
    """Create figure with stacked full-width top rows and a parameter grid below.

    Layout
    ------
    Row 0 (narrow)  : legend    — full width
    Row 1 (medium)  : airfoil   — full width
    Rows 2 … n+1   : parameter panels — 2 columns

    Returns
    -------
    fig, ax_legend, ax_airfoil, axList
    """
    n_rows = (n_params + 1) // 2
    fig = plt.figure(
        figsize=(st['fig_width'], st['fig_height']),
        dpi=st['dpi'],
        layout='constrained',
    )
    gs = GridSpec(
        n_rows + 2, 2, figure=fig,
        height_ratios=[1.5, 1.4] + [1.0] * n_rows,
    )
    ax_legend  = fig.add_subplot(gs[0, :])
    ax_airfoil = fig.add_subplot(gs[1, :])

    axList = []
    for row in range(n_rows):
        for col in range(2):
            if row * 2 + col < n_params:
                axList.append(fig.add_subplot(gs[row + 2, col]))

    return fig, ax_legend, ax_airfoil, axList


def _fill_airfoil_panel(
    ax,
    prepared_list: list,
    colors_upper: list,
    colors_lower: list,
    st: dict,
    show_dstar: bool = True,
    colors_geom: list | None = None,
):
    """Draw airfoil geometries (and optionally displacement thickness) on *ax*.

    Contour lines use ``colors_geom`` when provided (e.g. black / dark-grey
    for compare mode), falling back to ``colors_upper``.
    Displacement-thickness envelopes always use the BL data colours.
    All geometry contours are drawn first, then all displacement-thickness
    lines are stacked on top.
    """
    any_drawn = False

    # Pass 1 — geometry contours
    for j, prep in enumerate(prepared_list):
        if prep is None:
            continue
        viz = prep.get('_viz')
        if viz is None:
            continue
        cg = colors_geom[j] if (colors_geom is not None and j < len(colors_geom)) else colors_upper[j]
        ax.plot(viz['x_airfoil'], viz['y_airfoil'],
                color=cg, linewidth=st['airfoil_lw'])
        any_drawn = True

    # Pass 2 — displacement-thickness envelopes
    if show_dstar:
        for j, prep in enumerate(prepared_list):
            if prep is None:
                continue
            viz = prep.get('_viz')
            if viz is None:
                continue
            cu = colors_upper[j]
            cl = colors_lower[j]
            ax.plot(viz['x_upper'], viz['ds_upper'],
                    color=cu, linewidth=st['linewidth'] * 0.8)
            ax.plot(viz['x_lower'], viz['ds_lower'],
                    color=cl, linewidth=st['linewidth'] * 0.8)

    if any_drawn:
        ax.set_aspect('equal')
        ax.set_xlim([-0.01, 1.1])
        ax.spines[['right', 'left', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.patch.set_alpha(0)
    else:
        ax.axis('off')


def _fill_compare_legend(
    ax,
    labels: list,
    colors_upper: list,
    colors_lower: list,
    colors_wake: list,
    fontsize: int,
    colors_geom: list | None = None,
    records: list | None = None,
):
    """Draw a two-column legend for compare mode.

    Each column corresponds to one dataset and contains three coloured line
    entries (Upper / Lower / Wake) followed by informational text entries
    (Re, alpha, Cl, Cm, Cd, L/D, Ncrit).
    """
    ax.axis('off')

    def _info_entries(j):
        rec = records[j] if (records and j < len(records)) else {}
        if rec is None:
            rec = {}
        Re    = rec.get('Re')
        alpha = rec.get('alpha')
        cl    = rec.get('cl')
        cm    = rec.get('cm')
        cd    = rec.get('cd')
        ncrit = rec.get('N_crit')
        src   = rec.get('source')
        ld    = (cl / cd) if (cl is not None and cd and cd != 0) else None

        _SRC_FMT = {'xfoil': 'XFoil', 'rfoil': 'RFoil', 'neuralfoil': 'NeuralFoil'}
        lines = []
        if src   is not None: lines.append(_SRC_FMT.get(src, src.capitalize()))
        if Re    is not None: lines.append(f'Re = {Re:.2e}')
        if alpha is not None: lines.append(f'$\\alpha$ = {alpha:.2f}$^\\circ$')
        if cl    is not None: lines.append(f'$C_L$ = {cl:.4f}')
        if cd    is not None: lines.append(f'$C_D$ = {cd:.5f}')
        if cm    is not None: lines.append(f'$C_M$ = {cm:.4f}')
        if ld    is not None: lines.append(f'L/D = {ld:.2f}')
        if ncrit is not None: lines.append(f'$N_{{crit}}$ = {ncrit:.1f}')
        return [Line2D([0], [0], color='none', linewidth=0, label=txt)
                for txt in lines]

    def _entries(j):
        name = labels[j] if j < len(labels) else ''
        colour_lines = [
            Line2D([0], [0], color=colors_upper[j], linewidth=1.5,
                   label=f'{name} \u2014 Upper'),
            Line2D([0], [0], color=colors_lower[j], linewidth=1.5,
                   label=f'{name} \u2014 Lower'),
            Line2D([0], [0], color=colors_wake[j],  linewidth=1.5,
                   label=f'{name} \u2014 Wake'),
        ]
        return colour_lines + _info_entries(j)

    n = len(labels)
    if n == 1:
        # Single dataset: coloured lines on the left, info entries on the right
        name = labels[0] if labels else ''
        colour_lines = [
            Line2D([0], [0], color=colors_upper[0], linewidth=1.5,
                   label=f'{name} \u2014 Upper'),
            Line2D([0], [0], color=colors_lower[0], linewidth=1.5,
                   label=f'{name} \u2014 Lower'),
            Line2D([0], [0], color=colors_wake[0],  linewidth=1.5,
                   label=f'{name} \u2014 Wake'),
        ]
        leg1 = ax.legend(handles=colour_lines, loc='upper right', frameon=False,
                         bbox_to_anchor=(0.47, 0.95), fontsize=fontsize)
        ax.add_artist(leg1)
        ax.legend(handles=_info_entries(0), loc='upper left', frameon=False,
                  bbox_to_anchor=(0.53, 0.95), fontsize=fontsize)
    else:
        col1 = _entries(0)
        leg1 = ax.legend(handles=col1, loc='upper center', frameon=False,
                         bbox_to_anchor=(0.25, 0.95), fontsize=fontsize)
        ax.add_artist(leg1)
        col2 = _entries(1)
        ax.legend(handles=col2, loc='upper center', frameon=False,
                  bbox_to_anchor=(0.75, 0.95), fontsize=fontsize)

# ── public API ────────────────────────────────────────────────────────────────

def boundary_layer_compare(
    data_dict,
    parameter=None,
    show_airfoil=True,
    figure_path=None,
    labels=None,
    style=None,
):
    """Boundary-layer plot with fixed colours for up to two datasets.

    Parameters
    ----------
    data_dict : dict
        ``{airfoil_name: [run_records]}``.  The total number of records with
        ``bl_data`` across all names must not exceed 2.
    parameter : str, optional
        A single parameter name (or alias) to show.  ``None`` shows all 12.
    show_airfoil : bool
        Include the airfoil-shape / displacement-thickness panel (only for
        single dataset with valid y coordinates).
    figure_path : str or path-like, or None
        Save path.  ``None`` skips saving.
    labels : list of str, optional
        Legend labels, one per dataset.  Auto-generated when ``None``.
    style : dict, optional
        Style overrides (see module docstring).

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    # Flatten and prepare records
    pairs: list[tuple[str, dict]] = []
    for name, records in data_dict.items():
        for r in records:
            if r.get('bl_data') is not None:
                pairs.append((name, r))

    if not pairs:
        raise ValueError('No records with bl_data found in data_dict.')
    if len(pairs) > 2:
        raise ValueError(
            'boundary_layer_compare supports at most 2 BL datasets. '
            'Use boundary_layer_rainbow for more.'
        )

    prepared = [_prepare_record(r) for _, r in pairs]
    names    = [n for n, _ in pairs]

    if labels is None:
        labels = list(names)

    # Colour assignment
    n = len(prepared)
    colors_upper = [st['color_upper_1'], st['color_upper_2']][:n]
    colors_lower = [st['color_lower_1'], st['color_lower_2']][:n]
    colors_wake  = [st['color_wake_1'],  st['color_wake_2']][:n]
    colors_geom  = [st['color_geom_1'],  st['color_geom_2']][:n]

    # Resolve parameter alias
    if parameter is not None:
        parameter = alternativeNameDict.get(parameter.lower(), parameter)

    params = _BL_PARAMS if parameter is None else [parameter]

    if parameter is None:
        fig, ax_legend, ax_airfoil, axList = _make_bl_figure(len(params), st)

        _fill_airfoil_panel(ax_airfoil, prepared, colors_upper, colors_lower, st,
                            show_dstar=show_airfoil, colors_geom=colors_geom)
        _fill_compare_legend(ax_legend, labels, colors_upper, colors_lower,
                             colors_wake, st['legend_fontsize'],
                             colors_geom=colors_geom,
                             records=prepared)

        for pi, nm in enumerate(params):
            ax = axList[pi]
            for j, prep in enumerate(prepared):
                pdata = prep['plot'].get(nm)
                if pdata is None:
                    continue
                _draw_param(ax, pdata,
                            colors_upper[j], colors_lower[j], colors_wake[j],
                            st['linewidth'], first=(j == 0))

    else:
        # Single-parameter view
        fig, ax = plt.subplots(
            figsize=(st['fig_width'], max(st['fig_height'] // 3, 5)),
            dpi=st['dpi'],
            layout='constrained',
        )
        for j, prep in enumerate(prepared):
            pdata = prep['plot'].get(parameter)
            if pdata is None:
                continue
            lbl = labels[j] if n > 1 else None
            _draw_param(ax, pdata,
                        colors_upper[j], colors_lower[j], colors_wake[j],
                        st['linewidth'], first=(j == 0), label=lbl)

        xlims = [0.0, 1.0]
        for prep in prepared:
            pd_ = prep['plot'].get(parameter)
            if pd_ and pd_.get('upper') and pd_['upper'][0]:
                xlims[1] = max(xlims[1], math.ceil(max(pd_['upper'][0])))
            if pd_ and pd_.get('wake') and pd_['wake'][0]:
                xlims[1] = max(xlims[1], math.ceil(max(pd_['wake'][0])))
        ax.set_xlim(xlims[0] - 0.003, xlims[1] + 0.003)
        if n > 1:
            ax.legend(fontsize=st['legend_fontsize'])

    if figure_path is not None:
        fig.savefig(figure_path, dpi=st['dpi'], bbox_inches='tight')
    return fig


def boundary_layer_rainbow(
    data_dict,
    parameter=None,
    show_airfoil=False,
    figure_path=None,
    labels=None,
    style=None,
    reference_data_dict=None,
    single_airfoil_name=None,
):
    """Boundary-layer rainbow plot.

    Colours track position in the auto-detected sweep (Blues for upper
    surface, Oranges for lower, Greens for wake).

    Parameters
    ----------
    data_dict : dict
        ``{airfoil_name: [run_records]}``.  All records with ``bl_data``
        are plotted.
    parameter : str, optional
        A single parameter name / alias.  ``None`` shows all 12 panels.
    show_airfoil : bool
        Attempt to draw the displacement-thickness airfoil visualisation
        for the first (lowest sweep value) record only.
    figure_path : str or path-like, or None
        Save path.
    labels : list of str, optional
        Override auto-generated legend labels (one per record, in sweep order).
    style : dict, optional
        Style overrides.

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    # Flatten records
    pairs: list[tuple[str, dict]] = []
    for name, records in data_dict.items():
        for r in records:
            if r.get('bl_data') is not None:
                pairs.append((name, r))

    if not pairs:
        raise ValueError('No records with bl_data found in data_dict.')

    # Detect sweep and sort
    sweep_field, sorted_pairs, sweep_vals = _detect_sweep(pairs)
    N = len(sorted_pairs)

    # When all names match 'rf=X.XX', override the sweep sort and order by rf
    # value so that color index 0 = rf=0.00 (lightest) and index N-1 = rf=1.00
    # (darkest), regardless of what alpha order _detect_sweep chose.
    import re as _re_bl
    _rf_pat_bl = _re_bl.compile(r'^rf=(\d+\.\d+)$', _re_bl.IGNORECASE)
    _all_rf = [_rf_pat_bl.match(n) for n, _ in sorted_pairs]
    if _all_rf and all(_all_rf):
        sorted_pairs = sorted(sorted_pairs,
                              key=lambda x: float(_rf_pat_bl.match(x[0]).group(1)))

    # Prepare records
    prepared = [_prepare_record(r) for _, r in sorted_pairs]

    # Auto-labels
    if labels is None:
        labels = [
            _sweep_label(sweep_field, r, n)
            for n, r in sorted_pairs
        ]

    # Colour arrays
    lo = st['cmap_lower_bound'];  hi = st['cmap_upper_bound']
    colors_upper = get_colors(N, st['cmap_upper'], lower=lo, upper=hi)
    colors_lower = get_colors(N, st['cmap_lower'], lower=lo, upper=hi)
    colors_wake  = get_colors(N, st['cmap_wake'],  lower=lo, upper=hi)
    colors_legend = get_colors(N, 'Greys', lower=lo, upper=hi)
    if single_airfoil_name is not None:
        colors_geom = ['k'] * N
    else:
        colors_geom   = get_colors(N, st['cmap_geom'],
                                   lower=st['cmap_geom_lower'],
                                   upper=st['cmap_geom_upper'])

    # Resolve parameter
    if parameter is not None:
        parameter = alternativeNameDict.get(parameter.lower(), parameter)

    params = _BL_PARAMS if parameter is None else [parameter]

    if parameter is None:
        fig, ax_legend, ax_airfoil, axList = _make_bl_figure(len(params), st)

        _fill_airfoil_panel(ax_airfoil, prepared, colors_upper, colors_lower, st,
                            show_dstar=show_airfoil, colors_geom=colors_geom)
        _ref_entries = (
            [(n, s.get('color', 'k')) for n, s in reference_data_dict.items()]
            if reference_data_dict else None
        )
        _add_rainbow_legend(ax_legend, labels, colors_legend, st['legend_fontsize'],
                            records=[r for _, r in sorted_pairs],
                            reference_entries=_ref_entries,
                            airfoil_name=single_airfoil_name)

        for pi, nm in enumerate(params):
            ax = axList[pi]
            for j, prep in enumerate(prepared):
                if prep is None:
                    continue
                pdata = prep['plot'].get(nm)
                if pdata is None:
                    continue
                _draw_param(
                    ax, pdata,
                    colors_upper[j], colors_lower[j], colors_wake[j],
                    st['linewidth'],
                    first=(j == 0),
                )

        # Reference airfoils — drawn on top of rainbow
        if reference_data_dict:
            for ref_name, ref_spec in reference_data_dict.items():
                ref_color = ref_spec.get('color', 'k')
                ref_prep  = _prepare_record(ref_spec['record'])
                if ref_prep is None:
                    continue
                _fill_airfoil_panel(
                    ax_airfoil, [ref_prep], [ref_color], [ref_color], st,
                    show_dstar=show_airfoil, colors_geom=None,
                )
                for pi, nm in enumerate(params):
                    pdata = ref_prep['plot'].get(nm)
                    if pdata is None:
                        continue
                    _draw_param(axList[pi], pdata,
                                ref_color, ref_color, ref_color,
                                st['linewidth'], first=False)

    else:
        # Single-parameter view
        fig, ax = plt.subplots(
            figsize=(st['fig_width'], max(st['fig_height'] // 3, 5)),
            dpi=st['dpi'],
            layout='constrained',
        )
        for j, prep in enumerate(prepared):
            if prep is None:
                continue
            pdata = prep['plot'].get(parameter)
            if pdata is None:
                continue
            _draw_param(
                ax, pdata,
                colors_upper[j], colors_lower[j], colors_wake[j],
                st['linewidth'],
                first=(j == 0),
                label=labels[j],
            )

        # Reference airfoils — single-panel view
        if reference_data_dict:
            for ref_name, ref_spec in reference_data_dict.items():
                ref_color = ref_spec.get('color', 'k')
                ref_prep  = _prepare_record(ref_spec['record'])
                if ref_prep is None:
                    continue
                pdata = ref_prep['plot'].get(parameter)
                if pdata is None:
                    continue
                _draw_param(ax, pdata, ref_color, ref_color, ref_color,
                            st['linewidth'], first=False, label=ref_name)

        xlims = [0.0, 1.0]
        for prep in prepared:
            if prep is None:
                continue
            pd_ = prep['plot'].get(parameter)
            if pd_ and pd_.get('upper') and pd_['upper'][0]:
                xlims[1] = max(xlims[1], math.ceil(max(pd_['upper'][0])))
            if pd_ and pd_.get('wake') and pd_['wake'][0]:
                xlims[1] = max(xlims[1], math.ceil(max(pd_['wake'][0])))
        ax.set_xlim(xlims[0] - 0.003, xlims[1] + 0.003)
        ax.legend(fontsize=st['legend_fontsize'])

    if figure_path is not None:
        fig.savefig(figure_path, dpi=st['dpi'], bbox_inches='tight')
    return fig


def _add_rainbow_legend(ax, labels, colors_upper, fontsize, records=None,
                        reference_entries=None, airfoil_name=None):
    """Draw rainbow sweep legend on a dedicated (axis-off) axes.

    Single 4-column legend, centred.  Cols 0–2 hold the coloured airfoil
    entries in row-major reading order; col 3 holds the run-case parameter
    text.  Matplotlib fills legends column-major, so the handles list is
    serialised column-major so the visual output is row-major.

    When *airfoil_name* is provided (single-airfoil sweep), an extra row is
    prepended at the top with a black line labelled with the airfoil name.
    """
    ax.axis('off')

    # If all labels look like 'rf=X.XX', sort by rf value so the legend
    # always reads 0.00 → 1.00 left-to-right, top-to-bottom regardless of
    # what order _detect_sweep chose (e.g. alpha-sorted).
    import re as _re_leg
    _rf_pat = _re_leg.compile(r'^rf=(\d+\.\d+)$', _re_leg.IGNORECASE)
    _rf_matches = [_rf_pat.match(lb) for lb in labels]
    if _rf_matches and all(_rf_matches):
        _pairs = sorted(zip(labels, colors_upper),
                        key=lambda x: float(_rf_pat.match(x[0]).group(1)))
        labels       = [lb for lb, _ in _pairs]
        colors_upper = [c  for _,  c in _pairs]

    _SWEPT = '[Swept Parameter]'

    # ── Build info lines (col 3) ──────────────────────────────────────────────
    info_lines: list[str] = []
    if records:
        rec = records[0]

        def _const_num(key, tol=1e-3, rel=False):
            vals = [r.get(key) for r in records if r.get(key) is not None]
            if len(vals) < 2:
                return True
            ref = abs(vals[0]) if (rel and vals[0] != 0) else 1.0
            return max(abs(v - vals[0]) / ref for v in vals[1:]) < tol

        def _const_str(key):
            vals = [r.get(key) for r in records if r.get(key) is not None]
            return len(set(vals)) <= 1

        Re    = rec.get('Re')
        alpha = rec.get('alpha')
        cl    = rec.get('cl')
        cm    = rec.get('cm')
        cd    = rec.get('cd')
        ncrit = rec.get('N_crit')
        src   = rec.get('source')
        ld    = (cl / cd) if (cl is not None and cd and cd != 0) else None

        _SRC_FMT = {'xfoil': 'XFoil', 'rfoil': 'RFoil', 'neuralfoil': 'NeuralFoil'}
        if src   is not None:
            info_lines.append(_SRC_FMT.get(src, src.capitalize()) if _const_str('source') else f'Source = {_SWEPT}')
        if Re    is not None:
            info_lines.append(f'Re = {Re:.2e}'                    if _const_num('Re', rel=True)    else f'Re = {_SWEPT}')
        if alpha is not None:
            info_lines.append(f'$\\alpha$ = {alpha:.2f}$^\\circ$' if _const_num('alpha', tol=1e-4) else f'$\\alpha$ = {_SWEPT}')
        if cl    is not None:
            info_lines.append(f'$C_L$ = {cl:.4f}'                 if _const_num('cl',  tol=1e-4)  else f'$C_L$ = {_SWEPT}')
        if cd    is not None:
            info_lines.append(f'$C_D$ = {cd:.5f}'                 if _const_num('cd',  tol=1e-6)  else f'$C_D$ = {_SWEPT}')
        if cm    is not None:
            info_lines.append(f'$C_M$ = {cm:.4f}'                 if _const_num('cm',  tol=1e-4)  else f'$C_M$ = {_SWEPT}')
        if ld    is not None:
            ld_const = _const_num('cl', tol=1e-4) and _const_num('cd', tol=1e-6)
            info_lines.append(f'L/D = {ld:.2f}'                   if ld_const                     else f'L/D = {_SWEPT}')
        if ncrit is not None:
            info_lines.append(f'$N_{{crit}}$ = {ncrit:.1f}'       if _const_num('N_crit', tol=1e-3) else f'$N_{{crit}}$ = {_SWEPT}')

    # ── Build 2D visual grid (row-major), serialise column-major ─────────────
    # Col 0-2: rainbow entries.  Col 3: info text (invisible handle).
    _rainbow_cols = 3
    _total_cols   = 4 if info_lines else _rainbow_cols

    n_header       = 1 if airfoil_name else 0
    n_rainbow      = len(labels)
    n_rainbow_rows = (n_rainbow + _rainbow_cols - 1) // _rainbow_cols if n_rainbow > 0 else 0
    n_ref          = len(reference_entries) if reference_entries else 0
    n_left_rows    = n_header + n_rainbow_rows + (1 if n_ref > 0 else 0)
    n_rows         = max(n_left_rows, len(info_lines))

    def _blank():
        return Line2D([0], [0], color='none', linewidth=0, label='')

    grid = [[_blank() for _ in range(_total_cols)] for _ in range(n_rows)]

    # Airfoil-name header row (single-airfoil mode) — col 0 of row 0
    if airfoil_name:
        grid[0][0] = Line2D([0], [0], color='k', linewidth=1.5, label=airfoil_name)

    # Rainbow entries — cols 0-2, row-major, offset by n_header rows
    for i, (lb, c) in enumerate(zip(labels, colors_upper)):
        row = n_header + i // _rainbow_cols
        col = i % _rainbow_cols
        grid[row][col] = Line2D([0], [0], color=c, linewidth=1.5, label=lb if '$' in lb else lb.upper())

    # Reference entry — col 0 of the row immediately after rainbow rows
    if reference_entries:
        for j, (ref_label, ref_color) in enumerate(reference_entries):
            if j < _rainbow_cols:
                grid[n_header + n_rainbow_rows][j] = Line2D([0], [0], color=ref_color, linewidth=1.5,
                                                             label=ref_label if '$' in ref_label else ref_label.upper())

    # Info text — col 3
    for i, txt in enumerate(info_lines):
        grid[i][3] = Line2D([0], [0], color='none', linewidth=0, label=txt)

    # Serialise column-major
    handles = [grid[row][col] for col in range(_total_cols) for row in range(n_rows)]

    ax.legend(handles=handles, loc='upper center', frameon=False,
              bbox_to_anchor=(0.5, 1.0), ncols=_total_cols, fontsize=fontsize,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)



if __name__ == '__main__':
    import pathlib
    from oso_airfoils.core import find_record

    _here = pathlib.Path(__file__).parent

    # ── Single airfoil: OSO-2026-WT2S-T24 alone ──────────────────────────────
    oso26_rec = find_record('oso_2026_wt2s', 'OSO-2026-WT2S-T24',
                            alpha=5.0, Re=10e6, N_crit=9.0,
                            xtp_top=1.0, xtp_bot=1.0,
                            source='xfoil', require_bl=True)
    _out_single = _here / 'boundary_layer_single_24pct.png'
    boundary_layer_compare(
        {'OSO-2026-WT2S-T24': [oso26_rec]},
        parameter=None,
        show_airfoil=True,
        figure_path=str(_out_single),
    )
    print(f'Saved: {_out_single}')

    # ── Compare: OSO-2025-WT2-T24 xfoil vs Ncrit sweep ───────────────────────
    # ffa_rec = find_record('ffa',          'FFA-W3-241',
    #                       alpha=5.0, Re=1e6, N_crit=9.0,
    #                       xtp_top=1.0, xtp_bot=1.0,
    #                       source='xfoil', require_bl=True)
    oso_rec = find_record('oso_2025_wt2', 'OSO-2025-WT2-T24',
                          alpha=5.0, Re=10e6, N_crit=9.0,
                          xtp_top=1.0, xtp_bot=1.0,
                          source='xfoil', require_bl=True)
    
    oso_nf_rec = find_record('oso_2025_wt2', 'OSO-2025-WT2-T24',
                          alpha=5.0, Re=10e6, N_crit=3.0,
                        #   xtp_top=1.0, xtp_bot=1.0,
                          source='xfoil', require_bl=True)


    data_dict = {
        # 'FFA-W3-241':       [ffa_rec],
        'OSO-2025-WT2-T24': [oso_rec, oso_nf_rec],
    }

    _out = _here / 'boundary_layer_compare_24pct.png'
    boundary_layer_compare(
        data_dict,
        parameter=None,
        show_airfoil=True,
        figure_path=str(_out),
    )
    print(f'Saved: {_out}')

    # ── Rainbow plot: all OSO-2025-WT2 airfoils at the same condition ─────────
    _WT2_STEMS = [
        'OSO-2025-WT2-T21',
        'OSO-2025-WT2-T24',
        'OSO-2025-WT2-T27',
        'OSO-2025-WT2-T30',
        'OSO-2025-WT2-T33',
        'OSO-2025-WT2-T36',
    ]
    rainbow_dict = {}
    for stem in _WT2_STEMS:
        try:
            rec = find_record('oso_2025_wt2', stem,
                              alpha=5.0, Re=10e6, N_crit=9.0,
                              xtp_top=1.0, xtp_bot=1.0,
                              source='xfoil', require_bl=True)
            rainbow_dict[stem] = [rec]
        except ValueError as e:
            print(f'Skipping {stem}: {e}')

    _out_rb = _here / 'boundary_layer_rainbow_wt2.png'
    boundary_layer_rainbow(
        rainbow_dict,
        parameter=None,
        show_airfoil=True,
        figure_path=str(_out_rb),
    )
    print(f'Saved: {_out_rb}')

