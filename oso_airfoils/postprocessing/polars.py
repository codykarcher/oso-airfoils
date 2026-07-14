"""oso_airfoils.postprocessing.polars
=====================================

Polar comparison plots built from pre-computed aerodynamic run records.

Public API
----------
polars_compare(data_dict, reynolds_numbers, turb_cases, tools, figure_path, ...)
    Fixed per-airfoil colour scheme.  Up to ~8 airfoils; multiple Re numbers
    get per-airfoil hue-family shading (same logic as the original
    compare_airfoils.compare_airfoils helper).

polars_rainbow(data_dict, reynolds_numbers, turb_cases, tools, figure_path, ...)
    Always turbo-colormap rainbow over airfoils.

Both functions accept pre-computed run-record dicts (as stored in the JSON
performance files) instead of running aerodynamic analyses live.  Pass
``geometry_dict`` to include the airfoil-shape panel.

Style overrides
---------------
Pass a ``style`` dict to either function.  Recognised keys and defaults::

    {
        'linewidth':      1.5,
        'fontsize':       15,
        'fig_width':      25,
        'bot_row_height': 7.0,
        'dpi':            300,
        'wspace':         0.12,
        'N_marks':        8,
        'palette':        [...],       # 9-colour fixed palette
        'cmap_rainbow':   'turbo',     # airfoil rainbow colourmap
        'cmap_re':        [...],       # per-airfoil Re-shading colourmaps
        'cmap_lower_re':  0.3,
        'cmap_upper_re':  0.8,
    }
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from oso_airfoils.core.colors import default_color_cycle as dcc

from metafoil.core.kulfan import Kulfan

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

_PALETTE = dcc
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=_PALETTE)

_DEFAULT_STYLE: dict[str, Any] = {
    'linewidth':           1.5,
    'fontsize':            15,   # kept for back-compat; sets both label and legend size
    'label_fontsize':      None, # axis labels / tick labels (falls back to 'fontsize')
    'legend_fontsize':     None, # top-panel legend text     (falls back to 'fontsize')
    'plot_legend_fontsize': None, # in-plot (bottom row) legends (falls back to 'fontsize')
    'fig_width':           25,
    'bot_row_height':      7.0,
    'dpi':                 300,
    'wspace':              0.12,
    'N_marks':             8,
    'palette':             _PALETTE,
    'cmap_rainbow':        'turbo',
    'cmap_lower_rainbow':  0.10,
    'cmap_upper_rainbow':  0.90,
    'cmap_re':             ['Blues', 'Oranges', 'Greens', 'Purples', 'Reds'],
    'cmap_lower_re':       0.3,
    'cmap_upper_re':       0.8,
}

# Tool priority order for linestyle assignment
_TOOL_ORDER = ['xfoil', 'qfoil', 'rfoil', 'neuralfoil']


# ── colour / geometry helpers ─────────────────────────────────────────────────

def handleZeroDivide(num, dem):
    return np.inf if dem == 0 else num / dem


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def get_colors(N, selected_map, lower=0.15, upper=0.75):
    cmap = truncate_colormap(plt.get_cmap(selected_map), lower, upper)
    norm = plt.Normalize(0, max(N - 1, 1))
    return cmap(norm(np.arange(N)))


def get_fractional_color(frac, selected_map, lower=0.15, upper=0.75):
    colors = get_colors(101, selected_map, lower, upper)
    return colors[int(math.floor(frac * 100.0))]


def computeNormals(xdata, ydata):
    xmin_index = xdata.index(min(xdata))
    normals = []
    for i, x in enumerate(xdata):
        y = ydata[i]

        if i == 0:
            dxdy = handleZeroDivide(ydata[i + 1] - y, xdata[i + 1] - x)
        elif i == len(xdata) - 1:
            dxdy = handleZeroDivide(y - ydata[i - 1], x - xdata[i - 1])
        else:
            dxdy_p = handleZeroDivide(ydata[i + 1] - y, xdata[i + 1] - x)
            dxdy_m = handleZeroDivide(y - ydata[i - 1], x - xdata[i - 1])
            if dxdy_p != dxdy_m and max(dxdy_p, dxdy_m) == np.inf:
                dxdy = min(dxdy_p, dxdy_m)
            else:
                dxdy = handleZeroDivide(
                    ydata[i + 1] - ydata[i - 1], xdata[i + 1] - xdata[i - 1]
                )

        if dxdy == 0.0:
            normal = [0, 1]
        elif dxdy == np.inf:
            normal = [1, 0]
        else:
            normal = [
                1 / (1 + 1 / dxdy ** 2) ** 0.5,
                -1 / dxdy / (1 + 1 / dxdy ** 2) ** 0.5,
            ]

        if i <= xmin_index:
            if normal[1] < 0:
                normal = [-normal[0], -normal[1]]
        elif x > 1:
            if normal[1] < 0:
                normal = [-normal[0], -normal[1]]
        else:
            if normal[1] > 0:
                normal = [-normal[0], -normal[1]]

        if xmin_index - 3 < i < xmin_index + 3 and normal[0] > 0:
            normal = [-normal[0], -normal[1]]

        normals.append(normal)
    return normals


def _arc_length_indices(x_vals, y_vals, n_marks):
    """Return indices evenly spaced by arc length along (x_vals, y_vals)."""
    xrange = max(x_vals.max() - x_vals.min(), 1e-9)
    yrange = max(y_vals.max() - y_vals.min(), 1e-9)
    dx_n = np.diff(x_vals) / xrange
    dy_n = np.diff(y_vals) / yrange
    arc = np.concatenate([[0], np.cumsum(np.sqrt(dx_n ** 2 + dy_n ** 2))])
    targets = np.linspace(0, arc[-1], n_marks)
    return [int(np.argmin(np.abs(arc - t))) for t in targets]


# ── record helpers ────────────────────────────────────────────────────────────

def _filter_records(records, Re, N_crit, xtp_u, xtp_l, source, tol=1e-3):
    """Return records matching the given condition and source.

    Parameters
    ----------
    records : list of dict
        Run records from the JSON performance file.
    Re, N_crit, xtp_u, xtp_l : float
        Condition to match.
    source : str
        Tool name, e.g. ``'xfoil'``, ``'neuralfoil'``, ``'rfoil'``.
    tol : float
        Relative tolerance for Re; absolute for N_crit/xtp.

    Returns
    -------
    list of dict
        Matching records, sorted by alpha.
    """
    result = []
    for r in records:
        if abs(r.get('Re', 0) - Re) / max(abs(Re), 1) > tol:
            continue
        if abs(r.get('N_crit', 0) - N_crit) > tol:
            continue
        if abs(r.get('xtp_top', 0) - xtp_u) > tol:
            continue
        if abs(r.get('xtp_bot', 0) - xtp_l) > tol:
            continue
        if r.get('source') != source:
            continue
        result.append(r)
    return sorted(result, key=lambda r: r.get('alpha', 0.0))


def _tool_linestyle(tool: str, turb_idx: int, active_tools: list[str]) -> Any:
    """Return the matplotlib linestyle for a tool at a given turbulence case index.

    Rank is determined by ``_TOOL_ORDER``.  Rank 0 (highest priority) gets
    solid/dashed, rank 1 gets dash-dot/dotted, rank 2 gets a custom dash pattern.

    Parameters
    ----------
    tool : str
        Tool name.
    turb_idx : int
        0 for clean, 1 for rough.
    active_tools : list[str]
        All tools active in this plot.
    """
    _styles_by_rank = [
        ['-',   '--'],                         # rank 0: primary
        ['-.', ':'],                           # rank 1: secondary
        [(0, (3, 1, 1, 1)), (0, (1, 1))],     # rank 2: tertiary
    ]
    sorted_tools = sorted(
        active_tools,
        key=lambda t: _TOOL_ORDER.index(t) if t in _TOOL_ORDER else 99
    )
    rank = sorted_tools.index(tool)
    return _styles_by_rank[min(rank, 2)][min(turb_idx, 1)]


def _tool_label(tool: str) -> str:
    return {'xfoil': 'XFOIL', 'qfoil': 'QFOIL', 'neuralfoil': 'NeuralFoil', 'rfoil': 'RFoil'}.get(tool, tool)


def _coords_from_geometry(geom) -> list[tuple]:
    """Convert various geometry representations to a list of (x, y) tuples."""
    if isinstance(geom, Kulfan):
        return list(zip(geom.xcoordinates, geom.ycoordinates))
    if isinstance(geom, np.ndarray):
        return [(float(geom[i, 0]), float(geom[i, 1])) for i in range(len(geom))]
    if isinstance(geom, (list, tuple)) and len(geom) == 2 and isinstance(geom[0], (list, tuple, np.ndarray)):
        return list(zip(geom[0], geom[1]))
    return list(geom)


# ── core polar plot ───────────────────────────────────────────────────────────

def polarPlot(
    dataList,
    airfoil_coords=None,
    legend_entries=None,
    linecolors=None,
    linestyles=None,
    coordinatecolors=None,
    coordinatestyles=None,
    width_ratios=None,
    show_cpmin=True,
    cl_design=None,
    legend_ncols=None,
    style=None,
):
    """Render a polar comparison figure.

    Parameters
    ----------
    dataList : list of lists of dicts
        Polar data.  One list per dataset; each dict is a run record.
    airfoil_coords : list of sequence, optional
        Airfoil coordinate arrays ``[(x, y), ...]``, one per airfoil.
    legend_entries : list of lists of dicts, optional
        Legend columns.  Each inner dict may have keys
        ``'text'``, ``'linestyle'``, ``'linecolor'``, ``'markersize'``.
    linecolors, linestyles : list, optional
        Per-dataset colour and linestyle.
    coordinatecolors, coordinatestyles : list, optional
        Per-airfoil shape colour and linestyle.
    width_ratios : list of float, optional
        Relative widths of the polar panels.
    show_cpmin : bool
        Include the Cp_min panel.
    cl_design : float, optional
        Horizontal design-CL reference line.
    legend_ncols : int, optional
        Force legend into 2 columns (merges cols 1+).
    style : dict, optional
        Style overrides (see module docstring).
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    _label_fs       = st['label_fontsize']       if st['label_fontsize']       is not None else st['fontsize']
    _legend_fs      = st['legend_fontsize']      if st['legend_fontsize']      is not None else st['fontsize']
    _plot_legend_fs = st['plot_legend_fontsize'] if st['plot_legend_fontsize'] is not None else st['fontsize']
    plt.rcParams.update({'font.size': _label_fs})

    palette = st['palette']

    for data in dataList:
        assert isinstance(data, list)
        for de in data:
            assert isinstance(de, dict)

    # Build per-dataset DataFrames
    dataframeList = []
    for data in dataList:
        dd = {k: [] for k in [
            'alpha', 'cl', 'cd', 'cm', 'cpmin',
            'xtp_u', 'xtp_l', 'xtr_u', 'xtr_l',
            're', 'm', 'n_crit', 'n_panels',
        ]}
        _key_map = {
            're':       'Re',
            'm':        'M',
            'n_crit':   'N_crit',
            'xtr_u':    'xtr_top',
            'xtr_l':    'xtr_bot',
            'xtp_u':    'xtp_top',
            'xtp_l':    'xtp_bot',
            'n_panels': 'N_panels',
        }
        for rdata in data:
            if rdata is None:
                continue
            for ky in dd:
                if ky in rdata:
                    dd[ky].append(rdata[ky])
                elif ky in _key_map and _key_map[ky] in rdata:
                    dd[ky].append(rdata[_key_map[ky]])
                else:
                    raise ValueError('Could not find key: %s' % ky)

        assert len(np.unique(dd['m']))     == 1
        assert len(np.unique(dd['xtp_u'])) == 1
        assert len(np.unique(dd['xtp_l'])) == 1
        assert len(np.unique(dd['n_crit'])) == 1

        dataframeList.append(pd.DataFrame.from_dict(dd))

    if width_ratios is None:
        if show_cpmin:
            width_ratios = [0.15, 0.2, 0.2, 0.25, 0.2]
        else:
            width_ratios = [0.2, 0.2, 0.25, 0.2]
    n_cols = len(width_ratios)
    gs_wr = [w / sum(width_ratios) * n_cols for w in width_ratios]

    show_top_row = (airfoil_coords is not None) or (legend_entries is not None)
    fig_width = st['fig_width']
    bot_h = st['bot_row_height']

    if show_top_row:
        if airfoil_coords is not None:
            all_c = np.vstack([np.array(c) for c in airfoil_coords])
            x_range = max(all_c[:, 0].max() - all_c[:, 0].min(), 1e-9)
            y_range = max(all_c[:, 1].max() - all_c[:, 1].min(), 1e-9)
            if show_cpmin:
                afl_frac = sum(width_ratios[2:]) / sum(width_ratios)
            else:
                afl_frac = 5.0 / 8.0
            afl_w_in = fig_width * afl_frac * 0.80
            top_h = max(afl_w_in * (y_range / x_range) + 1.2, 3.0)
        else:
            top_h = 3.5

        fig = plt.figure(
            figsize=(fig_width, top_h + bot_h), dpi=st['dpi'], layout='constrained'
        )
        outer = GridSpec(2, 1, figure=fig, height_ratios=[top_h, bot_h])
        if show_cpmin:
            top_gs = GridSpecFromSubplotSpec(
                1, 5, subplot_spec=outer[0], width_ratios=gs_wr, wspace=st['wspace']
            )
            ax_legend  = fig.add_subplot(top_gs[0, 0:2])
            ax_airfoil = fig.add_subplot(top_gs[0, 2:5])
        else:
            top_gs = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[0], width_ratios=[3, 5], wspace=st['wspace']
            )
            ax_legend  = fig.add_subplot(top_gs[0, 0])
            ax_airfoil = fig.add_subplot(top_gs[0, 1])
        bot_gs = GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[1], width_ratios=gs_wr, wspace=st['wspace']
        )
    else:
        fig = plt.figure(figsize=(fig_width, bot_h), dpi=st['dpi'], layout='constrained')
        bot_gs = GridSpec(1, n_cols, figure=fig, width_ratios=gs_wr, wspace=st['wspace'])

    if show_cpmin:
        ax0  = fig.add_subplot(bot_gs[0, 0])
        ax1  = fig.add_subplot(bot_gs[0, 1])
        ax2  = fig.add_subplot(bot_gs[0, 2])
        ax2r = ax2.twinx()
        ax3  = fig.add_subplot(bot_gs[0, 3])
        ax4  = fig.add_subplot(bot_gs[0, 4])
    else:
        ax0  = None
        ax1  = fig.add_subplot(bot_gs[0, 0])
        ax2  = fig.add_subplot(bot_gs[0, 1])
        ax2r = ax2.twinx()
        ax3  = fig.add_subplot(bot_gs[0, 2])
        ax4  = fig.add_subplot(bot_gs[0, 3])

    # ── Top row ───────────────────────────────────────────────────────────────
    if show_top_row:
        ax_legend.axis('off')
        if legend_entries is not None:
            if legend_ncols == 2 and len(legend_entries) > 2:
                _eff = [legend_entries[0],
                        [e for col in legend_entries[1:] for e in col]]
            else:
                _eff = legend_entries
            for ci, col_entries in enumerate(_eff):
                handles = []
                for entry in col_entries:
                    ms = entry.get('markersize', None)
                    marker = 'o' if ms else 'None'
                    h = Line2D(
                        [0], [0],
                        color=entry.get('linecolor', 'black'),
                        linestyle=entry.get('linestyle', '-'),
                        marker=marker,
                        markersize=ms or 0,
                        label=entry.get('text', ''),
                    )
                    handles.append(h)
                x_anchor = ci / max(len(_eff), 1) + 0.02
                leg = ax_legend.legend(
                    handles=handles, loc='upper left', frameon=False,
                    bbox_to_anchor=(x_anchor, 0.95),
                    fontsize=_legend_fs,
                )
                if ci < len(_eff) - 1:
                    ax_legend.add_artist(leg)

        if airfoil_coords is not None:
            for j, coords in enumerate(airfoil_coords):
                arr = np.array(coords)
                ac  = (coordinatecolors[j]  if coordinatecolors and j < len(coordinatecolors)
                       else palette[j % len(palette)])
                als = (coordinatestyles[j]  if coordinatestyles and j < len(coordinatestyles)
                       else '-')
                ax_airfoil.plot(arr[:, 0], arr[:, 1], color=ac, linestyle=als)
            ax_airfoil.set_aspect('equal')
            ax_airfoil.grid(True, linewidth=0.5, alpha=0.5)
            ax_airfoil.minorticks_on()
            ax_airfoil.grid(True, which='minor', linewidth=0.4, alpha=0.4)
            ax_airfoil.set_xlabel('$x/c$')
            ax_airfoil.set_ylabel('$y/c$')
        else:
            ax_airfoil.axis('off')

    # ── Bottom row: polar panels ───────────────────────────────────────────────
    clmin = np.inf;  clmax = -np.inf
    cmmin = np.inf;  cmmax = -np.inf
    N_marks = st['N_marks']

    for i, df in enumerate(dataframeList):
        plot_color = (linecolors[i] if linecolors and i < len(linecolors)
                      else palette[i % len(palette)])
        linestyle  = (linestyles[i] if linestyles and i < len(linestyles) else '-')
        lw = st['linewidth']

        entry = df.sort_values('alpha')
        clmin = min(clmin, entry['cl'].min())
        clmax = max(clmax, entry['cl'].max())
        cmmin = min(cmmin, entry['cm'].min())
        cmmax = max(cmmax, entry['cm'].max())

        kw = dict(color=plot_color, linestyle=linestyle, linewidth=lw)

        if ax0 is not None:
            ax0.plot(entry['cpmin'], entry['cl'], **kw)
            ax0.set_xlabel('$C_{p,min}$')
            ax0.set_ylabel('$C_L$', labelpad=4)
            ax0.xaxis.set_inverted(True)
            ax0.grid(True)

        ax1.plot(entry['cd'], entry['cl'], **kw)
        ax1.set_xlim([0, 0.05])
        ax1.set_ylabel('$C_L$', labelpad=4)
        ax1.set_xlabel('$C_D$')
        ax1.grid(True)

        ax2.plot(entry['alpha'], entry['cl'], **kw)
        ax2.set_ylabel('$C_L$', labelpad=4)
        ax2.set_xlabel(r'$\alpha$')
        ax2.grid(True)

        alpha_v = entry['alpha'].values
        cm_v    = entry['cm'].values
        idx_cm  = _arc_length_indices(alpha_v, cm_v, N_marks)
        ax2r.plot(alpha_v, cm_v, 'o', linestyle=linestyle,
                  color=plot_color, markersize=3, markevery=idx_cm, linewidth=lw)
        ax2r.set_ylabel('$C_M$', labelpad=+5)

        ax3.plot(entry['cl'] / entry['cd'], entry['cl'], **kw)
        ax3.set_xlabel('$L/D$')
        ax3.set_ylabel('$C_L$', labelpad=4)
        ax3.grid(True)

        ax4.plot(entry['xtr_u'], entry['cl'], **kw)
        xtr_l_v  = entry['xtr_l'].values
        cl_v     = entry['cl'].values
        idx_l    = _arc_length_indices(xtr_l_v, cl_v, N_marks)
        ax4.plot(xtr_l_v, cl_v, 'o', linestyle=linestyle,
                 color=plot_color, markersize=3, markevery=idx_l, linewidth=lw)
        ax4.set_xlabel('$x_{tr}/c$')
        ax4.set_ylabel('$C_L$', labelpad=4)
        ax4.set_xlim([0, 1.01])
        ax4.grid(True)

    if ax0 is not None:
        ax0.set_xlim(left=0.0)

    # Design CL reference line
    if cl_design is not None:
        _specs = [
            (ax1, 0.98, 'right', 'top'),
            (ax2, 0.02, 'left',  'bottom'),
            (ax3, 0.02, 'left',  'top'),
            (ax4, 0.98, 'right', 'bottom'),
        ]
        if ax0 is not None:
            _specs.append((ax0, 0.98, 'right', 'top'))
        for _ax, _tx, _ha, _va in _specs:
            _ax.axhline(cl_design, color='k', lw=0.8, ls='-', zorder=1.5)
            _yoff = 4 if _va == 'bottom' else -4
            _ax.annotate(
                r'$C_L$ Design',
                xy=(_tx, cl_design), xycoords=_ax.get_yaxis_transform(),
                xytext=(0, _yoff), textcoords='offset points',
                ha=_ha, va=_va, fontsize=7, color='k',
            )

    # Align CM right axis to CL left axis so gridlines coincide
    cl_lo, cl_hi = ax2.get_ylim()
    mid_cm = 0.5 * (cmmin + cmmax)
    mid_cm_r = round(mid_cm / 0.04) * 0.04
    zero_cp  = round((2 / 3 * clmin + 1 / 3 * clmax) / 0.2) * 0.2
    ax2r.set_ylim(
        (cl_lo  - zero_cp + 0.2 * mid_cm_r / 0.04) / 5,
        (cl_hi  - zero_cp + 0.2 * mid_cm_r / 0.04) / 5,
    )

    for _ax in [a for a in [ax0, ax1, ax2, ax3, ax4] if a is not None]:
        _ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2r.yaxis.set_major_locator(MultipleLocator(0.04))

    for _ax in [a for a in [ax0, ax1, ax2, ax2r, ax3, ax4] if a is not None]:
        _ax.set_axisbelow(True)
        _ax.minorticks_on()
        _ax.grid(True, which='minor', linewidth=0.4, alpha=0.4)

    ax2r.legend(handles=[
        Line2D([0], [0], color='black', linestyle='-', label='$C_L$'),
        Line2D([0], [0], color='black', linestyle='-', marker='o',
               markersize=3, label='$C_M$'),
    ], loc='upper left', framealpha=1.0, fontsize=_plot_legend_fs)

    ax4.legend(handles=[
        Line2D([0], [0], color='black', linestyle='-', label='Upper'),
        Line2D([0], [0], color='black', linestyle='-', marker='o',
               markersize=3, label='Lower'),
    ], loc='lower right', framealpha=1.0, fontsize=_plot_legend_fs)

    return fig


# ── legend builders ───────────────────────────────────────────────────────────

def _build_legend_cols(
    airfoil_names,
    reynolds_numbers,
    turb_cases,
    tools,
    linecolors_by_name,  # {name: color_or_list_per_re}
    is_rainbow: bool,
    multi_re: bool,
    st: dict,
    suppress_re: bool = False,
):
    """Build three legend column lists for polarPlot.

    Returns
    -------
    tuple of (legend_col_1, legend_col_2, legend_col_3)
    """
    palette = st['palette']
    cmap_re = st['cmap_re']

    legend_col_1 = []   # airfoil names + tool labels
    legend_col_2 = []   # Reynolds numbers
    legend_col_3 = []   # turbulence cases

    re_min = min(reynolds_numbers)
    re_max = max(reynolds_numbers)
    seen_re = []

    # Airfoil name entries
    for i, name in enumerate(airfoil_names):
        if multi_re:
            ccolor = get_fractional_color(1.0, cmap_re[i % len(cmap_re)], lower=0.6, upper=0.6)
            legend_col_1.append({'text': name, 'linestyle': '-', 'linecolor': ccolor, 'markersize': 0})
        else:
            color = linecolors_by_name.get(name, palette[i % len(palette)])
            legend_col_1.append({'text': name, 'linestyle': '-', 'linecolor': color, 'markersize': 0})

    # Re entries — when multi-Re, build a separate Re column; when single Re,
    # prepend it as a header in legend_col_3 so it renders on top and the
    # legend is reduced to two columns (avoids overlap / z-order issues).
    single_re = (not multi_re) and (len(set(reynolds_numbers)) == 1)
    for re in reynolds_numbers:
        if re in seen_re:
            continue
        seen_re.append(re)
        if multi_re:
            frac = (np.log10(re) - np.log10(re_min)) / max(np.log10(re_max) - np.log10(re_min), 1e-9)
            grey = get_fractional_color(frac, 'Greys', lower=0.3, upper=0.8)
            if not suppress_re:
                legend_col_2.append({'text': 'Re=%.2e' % re, 'linestyle': '-', 'linecolor': grey, 'markersize': 0})
        elif not single_re:
            if not suppress_re:
                legend_col_2.append({'text': 'Re=%.2e' % re, 'linestyle': '-', 'linecolor': 'k', 'markersize': 0})

    # Tool / turbulence entries
    sorted_tools = sorted(tools, key=lambda t: _TOOL_ORDER.index(t) if t in _TOOL_ORDER else 99)

    # When there is only one Re, prepend it as a header entry so it appears
    # in the same (last) legend column that renders on top.
    if single_re and not suppress_re:
        re_val = list(set(reynolds_numbers))[0]
        legend_col_3.append({'text': 'Re=%.2e' % re_val, 'linestyle': '-', 'linecolor': 'k', 'markersize': 0})

    for tool in sorted_tools:
        label = _tool_label(tool)
        for ii, tc in enumerate(turb_cases):
            ls = _tool_linestyle(tool, ii, tools)
            text = (
                r'%s -- $N_{crit}$: %.1f, $x_{tp,u}$: %.2f, $x_{tp,l}$: %.2f'
                % (label, tc[0], tc[1], tc[2])
            )
            legend_col_3.append({'text': text, 'linestyle': ls, 'linecolor': 'k', 'markersize': 0})

    return legend_col_1, legend_col_2, legend_col_3


# ── public API ────────────────────────────────────────────────────────────────

def polars_compare(
    data_dict,
    reynolds_numbers,
    turb_cases,
    tools,
    figure_path,
    geometry_dict=None,
    color_override=None,
    show_cpmin=True,
    cl_design=None,
    legend_ncols=None,
    style=None,
    force_compare_mode: bool = False,
):
    """Polar comparison plot from pre-computed run records.

    Parameters
    ----------
    data_dict : dict
        ``{airfoil_name: [run_records]}``.  Each run record is a dict matching
        the JSON performance-file schema.
    reynolds_numbers : list of float
        Reynolds numbers to include.  Each ``(Re, turb_case, tool)`` combination
        is looked up in the matching airfoil's record list; a
        :class:`ValueError` is raised if no matching record is found.
    turb_cases : list of [N_crit, xtp_u, xtp_l]
        One or two turbulence conditions.
    tools : list of str
        Solver names to plot.  Valid: ``'xfoil'``, ``'neuralfoil'``, ``'rfoil'``.
    figure_path : str or path-like, or None
        Where to save the figure.  Pass ``None`` to skip saving.
    geometry_dict : dict, optional
        ``{name: geometry}`` for the airfoil-shape panel.  Accepted types:
        ``Kulfan``, ``ndarray (N,2)``, ``([x], [y])``.
        Omit to skip the shape panel.
    color_override : dict, optional
        ``{name: colour}`` overrides for single-Re plots.
    cl_design : float, optional
        Design-CL reference line.
    legend_ncols : int, optional
        Force 2-column legend merging.
    style : dict, optional
        Style overrides (see module docstring).

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    # Resolve per-element font sizes (fall back to the unified 'fontsize' key)
    _label_fs       = st['label_fontsize']       if st['label_fontsize']       is not None else st['fontsize']
    _legend_fs      = st['legend_fontsize']      if st['legend_fontsize']      is not None else st['fontsize']
    _plot_legend_fs = st['plot_legend_fontsize'] if st['plot_legend_fontsize'] is not None else st['fontsize']
    plt.rcParams.update({'font.size': _label_fs})

    if color_override is None:
        color_override = {}
    if turb_cases is None:
        turb_cases = [[9.0, 1.0, 1.0]]

    if not force_compare_mode and len(data_dict) > 5 and len(reynolds_numbers) > 1:
        raise ValueError(
            'Too many airfoils and Reynolds numbers — the plot would be overcrowded. '
            'Reduce airfoils or Reynolds numbers.'
        )

    palette    = st['palette']
    cmap_re    = st['cmap_re']
    lo_re = st['cmap_lower_re'];  hi_re = st['cmap_upper_re']

    re_min = min(reynolds_numbers)
    re_max = max(reynolds_numbers)
    multi_re = (re_min != re_max) and not force_compare_mode

    # In force_compare_mode, detect the actual Re for each airfoil from its
    # records so we can embed it in the legend label.
    _per_airfoil_re: dict = {}
    if force_compare_mode:
        for _name, _recs in data_dict.items():
            for _r in _recs:
                _rv = _r.get('Re', 0)
                if _rv:
                    _per_airfoil_re[_name] = _rv
                    break

    data_list        = []
    coords_list      = []
    linecolors       = []
    linestyles       = []
    coordinatecolors = []
    linecolors_by_name: dict[str, Any] = {}
    _seen_names: set  = set()

    for i, (name, records) in enumerate(data_dict.items()):
        for re in reynolds_numbers:

            if multi_re:
                frac  = (np.log10(re) - np.log10(re_min)) / max(np.log10(re_max) - np.log10(re_min), 1e-9)
                color = get_fractional_color(frac, cmap_re[i % len(cmap_re)], lower=lo_re, upper=hi_re)
            else:
                color = palette[i % len(palette)]
                if name in color_override:
                    color = color_override[name]
            linecolors_by_name[name] = color

            # Airfoil shape panel — add once per airfoil
            if name not in _seen_names:
                _seen_names.add(name)
                if multi_re:
                    ccolor = get_fractional_color(1.0, cmap_re[i % len(cmap_re)], lower=0.6, upper=0.6)
                else:
                    ccolor = color
                coordinatecolors.append(ccolor)
                if geometry_dict is not None and name in geometry_dict:
                    coords_list.append(_coords_from_geometry(geometry_dict[name]))
                elif geometry_dict is not None:
                    coords_list.append(None)

            for ii, tc in enumerate(turb_cases):
                N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]

                for tool in tools:
                    filtered = _filter_records(records, re, N_crit, xtp_u, xtp_l, tool)
                    if not filtered:
                        if force_compare_mode:
                            continue
                        raise ValueError(
                            "No records for '%s', Re=%.2e, N_crit=%.1f, "
                            "xtp_u=%.2f, xtp_l=%.2f, source='%s'."
                            % (name, re, N_crit, xtp_u, xtp_l, tool)
                        )
                    data_list.append(filtered)
                    linecolors.append(color)
                    linestyles.append(_tool_linestyle(tool, ii, tools))

    # Remove None entries from coords_list (airfoils missing from geometry_dict)
    if geometry_dict is not None:
        valid_coords   = [(c, cc) for c, cc in zip(coords_list, coordinatecolors) if c is not None]
        coords_list       = [p[0] for p in valid_coords]
        coordinatecolors  = [p[1] for p in valid_coords]
    else:
        coords_list      = None
        coordinatecolors = None

    # Legend — in force_compare_mode, embed each airfoil's actual Re in its
    if force_compare_mode and _per_airfoil_re:
        _leg_names = [
            '%s  Re=%.2e' % (n, _per_airfoil_re[n]) if n in _per_airfoil_re else n
            for n in data_dict.keys()
        ]
        _leg_re = [next(iter(_per_airfoil_re.values()))]
    else:
        _leg_names = list(data_dict.keys())
        _leg_re = reynolds_numbers

    lc1, lc2, lc3 = _build_legend_cols(
        _leg_names,
        _leg_re,
        turb_cases, tools,
        linecolors_by_name, is_rainbow=False, multi_re=multi_re, st=st,
        suppress_re=force_compare_mode,
    )
    leg = [col for col in [lc1, lc2, lc3] if col]

    fig = polarPlot(
        data_list,
        airfoil_coords=coords_list,
        legend_entries=leg,
        linecolors=linecolors,
        linestyles=linestyles,
        coordinatecolors=coordinatecolors,
        show_cpmin=show_cpmin,
        cl_design=cl_design,
        legend_ncols=legend_ncols,
        style=st,
    )

    if figure_path is not None:
        fig.savefig(figure_path, dpi=st['dpi'])
    return fig


def tool_comparison_plot(
    records,
    reynolds_numbers,
    turb_cases,
    tools,
    airfoil_name,
    figure_path=None,
    geometry=None,
    show_cpmin=False,
    cl_design=None,
    style=None,
):
    """Per-tool comparison polar for a single airfoil.

    Each tool is assigned a distinct colour from the palette; the first
    turbulence case is plotted as a solid line (clean) and the second as a
    dashed line (rough).

    Parameters
    ----------
    records : list of dict
        Run records for the airfoil (e.g. from ``_get_polar_records``).
    reynolds_numbers : list of float
    turb_cases : list of [N_crit, xtp_u, xtp_l]
        One or two turbulence conditions.  Index 0 = clean (solid),
        index 1 = rough (dashed).
    tools : list of str
    airfoil_name : str
        Shown as the legend header.
    figure_path : str or path-like or None
        Save path; ``None`` to skip.
    geometry : optional
        ``Kulfan``, ``(N, 2)`` array, or ``([x], [y])`` for the shape panel.
    show_cpmin : bool
    cl_design : float, optional
    style : dict, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    palette = st['palette']
    _LS_BY_TURB = ['-', '--', ':', '-.']

    # Preserve the caller-supplied order so palette colours match the tool list.
    sorted_tools = list(tools)

    data_list    = []
    linecolors   = []
    linestyles   = []

    for ti, tool in enumerate(sorted_tools):
        color = palette[ti % len(palette)]
        for ii, tc in enumerate(turb_cases):
            N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]
            ls = _LS_BY_TURB[min(ii, len(_LS_BY_TURB) - 1)]
            for re in reynolds_numbers:
                filtered = _filter_records(records, re, N_crit, xtp_u, xtp_l, tool)
                if not filtered:
                    continue
                data_list.append(filtered)
                linecolors.append(color)
                linestyles.append(ls)

    # ── Legend ────────────────────────────────────────────────────────────────
    # Col 1: airfoil name + Re number(s)
    # Col 2: one coloured entry per tool
    # Col 3: one entry per turbulence case (clean / rough)
    legend_col_1 = [{'text': airfoil_name, 'linestyle': '-', 'linecolor': 'k', 'markersize': 0}]
    seen_re: list = []
    for re in reynolds_numbers:
        if re not in seen_re:
            seen_re.append(re)
            legend_col_1.append({'text': 'Re=%.2e' % re, 'linestyle': '-',
                                  'linecolor': 'k', 'markersize': 0})

    legend_col_2 = []
    for ti, tool in enumerate(sorted_tools):
        color = palette[ti % len(palette)]
        legend_col_2.append({'text': _tool_label(tool), 'linestyle': '-',
                              'linecolor': color, 'markersize': 0})

    legend_col_3 = []
    _turb_labels = ['Clean', 'Rough']
    for ii, tc in enumerate(turb_cases):
        ls    = _LS_BY_TURB[min(ii, len(_LS_BY_TURB) - 1)]
        label = _turb_labels[ii] if ii < len(_turb_labels) else f'Cond {ii}'
        text  = (r'%s -- $N_{crit}$: %.1f, $x_{tp}$: %.2f'
                 % (label, tc[0], tc[1]))
        legend_col_3.append({'text': text, 'linestyle': ls,
                              'linecolor': 'k', 'markersize': 0})

    legend_entries = [legend_col_1, legend_col_2, legend_col_3]

    airfoil_coords   = None
    coordinatecolors = None
    if geometry is not None:
        airfoil_coords   = [_coords_from_geometry(geometry)]
        # Use the current text color so the line is visible in both dark and
        # light matplotlib themes (white in dark mode, black in light mode).
        coordinatecolors = [plt.rcParams.get('text.color', 'k')]

    fig = polarPlot(
        data_list,
        airfoil_coords=airfoil_coords,
        legend_entries=legend_entries,
        linecolors=linecolors,
        linestyles=linestyles,
        coordinatecolors=coordinatecolors,
        show_cpmin=show_cpmin,
        cl_design=cl_design,
        style=st,
    )

    if figure_path is not None:
        fig.savefig(figure_path, dpi=st['dpi'])
    return fig


def polars_rainbow(
    data_dict,
    reynolds_numbers,
    turb_cases,
    tools,
    figure_path,
    geometry_dict=None,
    reference_data_dict=None,
    reverse_plot_order=False,
    show_cpmin=True,
    cl_design=None,
    legend_ncols=None,
    style=None,
):
    """Polar plot with turbo-colourmap rainbow over airfoils.

    All parameters have the same meaning as :func:`polars_compare` except:

    Parameters
    ----------
    reverse_plot_order : bool
        Draw airfoils in reversed order (useful when low-performance airfoils
        would otherwise overdraw high-performance ones).
    """
    st = dict(_DEFAULT_STYLE)
    if style:
        st.update(style)

    if turb_cases is None:
        turb_cases = [[9.0, 1.0, 1.0]]

    palette   = st['palette']
    cmap_re   = st['cmap_re']
    lo_re = st['cmap_lower_re'];  hi_re = st['cmap_upper_re']

    re_min = min(reynolds_numbers)
    re_max = max(reynolds_numbers)
    multi_re = re_min != re_max

    n_airfoils = len(data_dict)
    _rainbow_colors = get_colors(max(n_airfoils, 2), st['cmap_rainbow'],
                                 lower=st['cmap_lower_rainbow'], upper=st['cmap_upper_rainbow'])

    data_list        = []
    coords_list      = []
    linecolors       = []
    linestyles       = []
    coordinatecolors = []
    linecolors_by_name: dict[str, Any] = {}
    _seen_names: set = set()

    for i, (name, records) in enumerate(data_dict.items()):
        for re in reynolds_numbers:

            if multi_re:
                frac  = (np.log10(re) - np.log10(re_min)) / max(np.log10(re_max) - np.log10(re_min), 1e-9)
                color = get_fractional_color(frac, cmap_re[i % len(cmap_re)], lower=lo_re, upper=hi_re)
            else:
                color = _rainbow_colors[i]
            linecolors_by_name[name] = color

            if name not in _seen_names:
                _seen_names.add(name)
                coordinatecolors.append(color)
                if geometry_dict is not None and name in geometry_dict:
                    coords_list.append(_coords_from_geometry(geometry_dict[name]))
                elif geometry_dict is not None:
                    coords_list.append(None)

            for ii, tc in enumerate(turb_cases):
                N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]
                for tool in tools:
                    filtered = _filter_records(records, re, N_crit, xtp_u, xtp_l, tool)
                    if not filtered:
                        raise ValueError(
                            "No records for '%s', Re=%.2e, N_crit=%.1f, "
                            "xtp_u=%.2f, xtp_l=%.2f, source='%s'."
                            % (name, re, N_crit, xtp_u, xtp_l, tool)
                        )
                    data_list.append(filtered)
                    linecolors.append(color)
                    linestyles.append(_tool_linestyle(tool, ii, tools))

    if geometry_dict is not None:
        valid = [(c, cc) for c, cc in zip(coords_list, coordinatecolors) if c is not None]
        coords_list      = [p[0] for p in valid]
        coordinatecolors = [p[1] for p in valid]
    else:
        coords_list      = None
        coordinatecolors = None

    # Append reference airfoil geometry (drawn in reference colour on top)
    if reference_data_dict:
        for ref_name, ref_spec in reference_data_dict.items():
            ref_kulfan = ref_spec.get('kulfan')
            if ref_kulfan is not None:
                if coords_list is None:
                    coords_list = []
                    coordinatecolors = []
                coords_list.append(_coords_from_geometry(ref_kulfan))
                coordinatecolors.append(ref_spec.get('color', 'k'))

    # Reference airfoils — always plotted last (on top of rainbow)
    if reference_data_dict:
        for ref_name, ref_spec in reference_data_dict.items():
            ref_color   = ref_spec.get('color', 'k')
            ref_records = ref_spec['records']
            for re in reynolds_numbers:
                for ii, tc in enumerate(turb_cases):
                    N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]
                    for tool in tools:
                        filtered = _filter_records(ref_records, re, N_crit, xtp_u, xtp_l, tool)
                        if filtered:
                            data_list.append(filtered)
                            linecolors.append(ref_color)
                            linestyles.append(_tool_linestyle(tool, ii, tools))
                        else:
                            print(f'  Warning: no {tool} records for reference {ref_name!r}'
                                  f'  Re={re:.2e}  Ncrit={N_crit}  xtp={xtp_u}/{xtp_l}'
                                  f' — curve will be absent from plot.')

    # Reverse plotting order while keeping legend order
    if reverse_plot_order and n_airfoils > 0:
        _n_per = len(data_list) // n_airfoils

        def _rev_blocks(lst, k):
            return [e for blk in [lst[i * k:(i + 1) * k]
                                  for i in range(len(lst) // k)][::-1]
                    for e in blk]

        data_list   = _rev_blocks(data_list,  _n_per)
        linecolors  = _rev_blocks(linecolors, _n_per)
        linestyles  = _rev_blocks(linestyles, _n_per)
        if coords_list:
            coords_list      = coords_list[::-1]
            coordinatecolors = coordinatecolors[::-1]

    # Legend — rainbow layout
    lc1, lc2, lc3 = _build_legend_cols(
        list(data_dict.keys()), reynolds_numbers, turb_cases, tools,
        linecolors_by_name, is_rainbow=True, multi_re=multi_re, st=st,
    )
    # For rainbow: split airfoil entries across three legend columns
    n_tool_entries = len(tools) + (len(turb_cases) - 1) * len(tools)  # approx
    airfoil_entries = lc1[: n_airfoils]
    suffix_entries  = lc1[n_airfoils:]
    n_leg_cols = min(3, n_airfoils)
    split_leg = [[] for _ in range(n_leg_cols)]
    for j, entry in enumerate(airfoil_entries):
        split_leg[j % n_leg_cols].append(entry)
    for entry in suffix_entries:
        split_leg[0].append(entry)
    if lc2:
        split_leg[min(1, n_leg_cols - 1)].append(lc2[-1])
    for entry in lc3:
        split_leg[min(2, n_leg_cols - 1)].append(entry)

    # Reference airfoil legend entries — bottom of first column
    if reference_data_dict:
        for ref_name, ref_spec in reference_data_dict.items():
            split_leg[0].append({
                'text': ref_name, 'linestyle': '-',
                'linecolor': ref_spec.get('color', 'k'), 'markersize': 0,
            })

    fig = polarPlot(
        data_list,
        airfoil_coords=coords_list,
        legend_entries=split_leg,
        linecolors=linecolors,
        linestyles=linestyles,
        coordinatecolors=coordinatecolors,
        show_cpmin=show_cpmin,
        cl_design=cl_design,
        legend_ncols=legend_ncols,
        style=st,
    )

    if figure_path is not None:
        fig.savefig(figure_path, dpi=st['dpi'])
    return fig


if __name__ == '__main__':
    import pathlib
    from oso_airfoils.core import load_runs
    from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT as _afl_root

    _here = pathlib.Path(__file__).parent

    _WT2S_STEMS = ['OSO-2026-WT2S-T24']
    _WT2_STEMS  = ['OSO-2025-WT2-T21', 'OSO-2025-WT2-T24', 'OSO-2025-WT2-T27',
                   'OSO-2025-WT2-T30', 'OSO-2025-WT2-T33', 'OSO-2025-WT2-T36']

    # ── helper: load geometry for a list of (family, stem) pairs ─────────────
    def _load_geom(pairs):
        geom = {}
        for fam, stem in pairs:
            afl = Kulfan()
            afl.readFile(str(_afl_root / fam / 'datfiles' / f'{stem}.dat'))
            geom[stem] = afl
        return geom

    # ── 1. Single: OSO-2026-WT2S-T24, clean + rough, xfoil + neuralfoil ─────
    _out_single = _here / 'polars_single_wt2s_24pct.png'
    polars_compare(
        {'OSO-2026-WT2S-T24': load_runs('oso_2026_wt2s', 'OSO-2026-WT2S-T24')},
        reynolds_numbers=[10e6],
        turb_cases=[[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]],
        tools=['xfoil', 'neuralfoil'],
        figure_path=str(_out_single),
        geometry_dict=_load_geom([('oso_2026_wt2s', 'OSO-2026-WT2S-T24')]),
        show_cpmin=False,
    )
    print(f'Saved: {_out_single}')

    # ── 2. Compare: all OSO-2025-WT2 stems, clean + rough, xfoil + neuralfoil ─
    _out_compare = _here / 'polars_compare_wt2_roughness.png'
    polars_compare(
        {stem: load_runs('oso_2025_wt2', stem) for stem in _WT2_STEMS},
        reynolds_numbers=[10e6],
        turb_cases=[[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]],
        tools=['xfoil', 'neuralfoil'],
        figure_path=str(_out_compare),
        geometry_dict=_load_geom([('oso_2025_wt2', s) for s in _WT2_STEMS]),
        show_cpmin=False,
    )
    print(f'Saved: {_out_compare}')

    # ── 3. Rainbow: OSO-2025-WT2 thickness family, clean, xfoil, 10M Re ──────
    _out_rainbow = _here / 'polars_rainbow_wt2_thickness.png'
    polars_rainbow(
        {stem: load_runs('oso_2025_wt2', stem) for stem in _WT2_STEMS},
        reynolds_numbers=[10e6],
        turb_cases=[[9.0, 1.0, 1.0]],
        tools=['xfoil'],
        figure_path=str(_out_rainbow),
        geometry_dict=_load_geom([('oso_2025_wt2', s) for s in _WT2_STEMS]),
        show_cpmin=False,
    )
    print(f'Saved: {_out_rainbow}')

    # ── 4. Compare: FFA-W3-241 vs OSO-2025-WT2-T24 vs OSO-2026-WT2S-T24 ─────
    _out_24 = _here / 'polars_compare_24pct.png'
    polars_compare(
        {
            'FFA-W3-241':        load_runs('ffa',           'FFA-W3-241'),
            'OSO-2025-WT2-T24':  load_runs('oso_2025_wt2',  'OSO-2025-WT2-T24'),
            'OSO-2026-WT2S-T24': load_runs('oso_2026_wt2s', 'OSO-2026-WT2S-T24'),
        },
        reynolds_numbers=[1e6],
        turb_cases=[[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]],
        tools=['xfoil'],
        figure_path=str(_out_24),
        geometry_dict=_load_geom([
            ('ffa',           'FFA-W3-241'),
            ('oso_2025_wt2',  'OSO-2025-WT2-T24'),
            ('oso_2026_wt2s', 'OSO-2026-WT2S-T24'),
        ]),
        show_cpmin=False,
    )
    print(f'Saved: {_out_24}')

    # ── 5. Same as 4 but across all shared Re numbers (0.5M–10M) ─────────────
    _out_24_re = _here / 'polars_compare_24pct_allRe.png'
    polars_compare(
        {
            'FFA-W3-241':        load_runs('ffa',           'FFA-W3-241'),
            'OSO-2025-WT2-T24':  load_runs('oso_2025_wt2',  'OSO-2025-WT2-T24'),
            'OSO-2026-WT2S-T24': load_runs('oso_2026_wt2s', 'OSO-2026-WT2S-T24'),
        },
        reynolds_numbers=[0.5e6, 1e6, 5e6, 10e6],
        turb_cases=[[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]],
        tools=['xfoil'],
        figure_path=str(_out_24_re),
        geometry_dict=_load_geom([
            ('ffa',           'FFA-W3-241'),
            ('oso_2025_wt2',  'OSO-2025-WT2-T24'),
            ('oso_2026_wt2s', 'OSO-2026-WT2S-T24'),
        ]),
        show_cpmin=False,
    )
    print(f'Saved: {_out_24_re}')
