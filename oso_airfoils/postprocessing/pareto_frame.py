"""
pareto_frame.py  --  render ONE Pareto-evolution frame from a population snapshot.

This is the frame that ``generate_gif.py`` strings together into
``pareto_shapes_evolution.gif``: Pareto front on the left, Kulfan shape parameters
top-right, and the corresponding airfoil shapes bottom-right, all keyed by the same
turbo colour ramp along the front.

It lives here as a function so a single frame can be produced on demand -- for a live
dashboard during a run, or for a one-off figure -- instead of only as a by-product of
building the whole animation. ``generate_gif.py`` calls straight into it, so the GIF
and any single frame stay identical by construction.
"""

import json
import os

import matplotlib
import numpy as np

from metafoil.core.kulfan import Kulfan

DEFAULT_NAFL = 11

#: Turbo truncation used for the airfoil ramp. MUST match polars.py's rainbow
#: defaults (`cmap_lower_rainbow` / `cmap_upper_rainbow`), so an airfoil picked out
#: of the front here is the SAME colour in the polar rainbow. The two are read side
#: by side; a full-range turbo here against a truncated one there made matching an
#: airfoil between the plots guesswork.
CMAP_RAINBOW = 'turbo'
CMAP_LOWER = 0.10
CMAP_UPPER = 0.90
DEFAULT_FONT_SIZE = 20
DEFAULT_LEGEND_FONT_SIZE = 12


def auto_lim(vals, factor):
    """Symmetric-ish padded limits around ``vals`` (the GIF's own convention)."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    pad = factor * (hi - lo)
    return (lo - pad, hi + pad)


#: Extra room added to the RIGHT of the shape-parameter axes, as a fraction of the
#: data span, so the (16-entry) legend sits over blank space instead of over the
#: coefficient traces.
LEGEND_PAD_FRAC = 0.32

#: Margin around the airfoil geometry axes, in chord fractions, so the section isn't
#: drawn hard against the axes frame.
GEOMETRY_MARGIN = 0.03


def compute_limits(data, nafl=DEFAULT_NAFL,
                   xlim_top_factor=0.05, ylim_top_factor=0.05,
                   xlim_pareto_factor=0.05, ylim_pareto_factor=0.05,
                   ylim_bottom_factor=0.05,
                   xlim_bottom=(-GEOMETRY_MARGIN, 1.0 + GEOMETRY_MARGIN),
                   legend_pad_frac=LEGEND_PAD_FRAC):
    """Axis limits for a whole animation, derived from ONE reference snapshot.

    The GIF fixes its axes off the LAST generation so frames don't jitter; pass the
    same limits to every :func:`render_pareto_frame` call to get that behaviour, or
    pass ``None`` to let each frame autoscale.
    """
    pareto = [p for p in data['population'] if p['pareto_index'] == 1]
    if not pareto:
        return None
    rough = np.array([p['LoD_rough_at_design'] for p in pareto], float)
    clean = np.array([p['LoD_clean_at_design'] for p in pareto], float)
    K_all = np.concatenate([list(p['K_upper']) + list(p['K_lower']) for p in pareto])

    ys = []
    for p in pareto:
        afl = Kulfan(TE_gap=data['input_parameters']['TE_gap'])
        afl.upperCoefficients = p['K_upper']
        afl.lowerCoefficients = p['K_lower']
        ys.extend(afl.ycoordinates)

    # Widen the shape-parameter x-axis to the right only, leaving clear space under
    # the legend rather than shifting the traces off-centre.
    xt = auto_lim(rough, xlim_top_factor)
    if xt is not None and legend_pad_frac:
        xt = (xt[0], xt[1] + legend_pad_frac * (xt[1] - xt[0]))

    return dict(
        xlim_top=xt,
        ylim_top=auto_lim(K_all, ylim_top_factor),
        xlim_pareto=auto_lim(rough, xlim_pareto_factor),
        ylim_pareto=auto_lim(clean, ylim_pareto_factor),
        ylim_bottom=auto_lim(np.array(ys, float), ylim_bottom_factor),
        xlim_bottom=xlim_bottom,
    )


def render_pareto_frame(data, out_path, gen_label=None, limits=None,
                        nafl=DEFAULT_NAFL, dpi=40, fix_colormap_range=None,
                        font_size=DEFAULT_FONT_SIZE,
                        legend_font_size=DEFAULT_LEGEND_FONT_SIZE,
                        legend_loc='upper right'):
    """Render one frame to ``out_path``.

    ``data`` is a parsed ``population_*_g*.json`` snapshot (or a path to one).
    ``limits`` comes from :func:`compute_limits`; ``None`` autoscales this frame.
    Returns ``out_path``.
    """
    import matplotlib.pyplot as plt

    if isinstance(data, (str, os.PathLike)):
        with open(data) as f:
            data = json.load(f)
    if gen_label is None:
        gen_label = data['input_parameters'].get('current_generation', 0)

    lim = limits or {}
    xlim_top = lim.get('xlim_top'); ylim_top = lim.get('ylim_top')
    xlim_pareto = lim.get('xlim_pareto'); ylim_pareto = lim.get('ylim_pareto')
    ylim_bottom = lim.get('ylim_bottom')
    xlim_bottom = lim.get('xlim_bottom',
                          (-GEOMETRY_MARGIN, 1.0 + GEOMETRY_MARGIN))

    pop = data['population']
    N_k = int(data['input_parameters']['N_k'])
    te_gap = data['input_parameters']['TE_gap']

    pareto_points = sorted([p for p in pop if p['pareto_index'] == 1],
                           key=lambda p: p['LoD_rough_at_design'])
    if not pareto_points:
        return None

    from oso_airfoils.postprocessing.polars import get_colors
    _ramp = get_colors(max(nafl, 2), CMAP_RAINBOW, lower=CMAP_LOWER, upper=CMAP_UPPER)
    turbo_cmap = lambda idx: _ramp[int(np.clip(idx, 0, len(_ramp) - 1))]
    seismic_cmap = plt.get_cmap('seismic', N_k)
    rough_LD_vals = np.array([p['LoD_rough_at_design'] for p in pareto_points], float)
    clean_LD_vals = np.array([p['LoD_clean_at_design'] for p in pareto_points], float)

    # Sample evenly along the arc length of the front so the highlighted airfoils
    # are spread across it rather than bunched where points are dense.
    x_norm = float(xlim_pareto[1] - xlim_pareto[0]) if xlim_pareto else (float(np.ptp(rough_LD_vals)) or 1.0)
    y_norm = float(ylim_pareto[1] - ylim_pareto[0]) if ylim_pareto else (float(np.ptp(clean_LD_vals)) or 1.0)
    rn = rough_LD_vals / x_norm
    cn = clean_LD_vals / y_norm
    seg_lens = np.sqrt(np.diff(rn) ** 2 + np.diff(cn) ** 2)
    cum_arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_arc = cum_arc[-1]
    if total_arc <= 0 or len(rough_LD_vals) < 2:
        ixs = np.unique(np.linspace(0, max(len(rough_LD_vals) - 1, 0), nafl).astype(int))
    else:
        target_arcs = np.linspace(0.0, total_arc, nafl)
        ixs = np.array([int(np.argmin(np.abs(cum_arc - t))) for t in target_arcs])

    def color_for(i):
        if fix_colormap_range is not None:
            lo, hi = fix_colormap_range
            clamped = np.clip(rough_LD_vals[i], lo, hi)
            return turbo_cmap(int(np.clip((clamped - lo) / (hi - lo) * (nafl - 1), 0, nafl - 1)))
        return turbo_cmap(ixs.tolist().index(i))

    shape_curves = []
    for i in list(reversed(ixs)):
        afl = Kulfan(TE_gap=te_gap)
        afl.upperCoefficients = pareto_points[i]['K_upper']
        afl.lowerCoefficients = pareto_points[i]['K_lower']
        shape_curves.append((np.asarray(afl.xcoordinates), np.asarray(afl.ycoordinates),
                             color_for(i)))

    x_range_bottom = xlim_bottom[1] - xlim_bottom[0]
    if ylim_bottom is None:
        all_y = np.concatenate([yc for _, yc, _ in shape_curves])
        ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
        pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        ylim_b = (ymin - pad, ymax + pad)
    else:
        ylim_b = ylim_bottom
    y_range_bottom = ylim_b[1] - ylim_b[0]

    top_w_in, top_h_in = 20.0, 11.0
    bottom_h_in = top_w_in * (y_range_bottom / x_range_bottom)
    fig = plt.figure(figsize=(2 * top_w_in, top_h_in + bottom_h_in))
    gs = fig.add_gridspec(2, 2, width_ratios=[top_w_in, top_w_in],
                          height_ratios=[top_h_in, bottom_h_in], hspace=0.15, wspace=0.12)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    # Left: whole population greyed by front, Pareto front highlighted.
    for pix in list(reversed(np.sort(np.unique([p['pareto_index'] for p in pop])))):
        pts = sorted([p for p in pop if p['pareto_index'] == pix],
                     key=lambda p: p['LoD_rough_at_design'])
        r_ld = np.array([p['LoD_rough_at_design'] for p in pts])
        c_ld = np.array([p['LoD_clean_at_design'] for p in pts])
        clrval = min([1 - 1 / (pix + 1), 0.91])
        ax_left.plot(r_ld, c_ld, 'o', color=[clrval, clrval, clrval], ms=4, mew=0)
    for i in ixs:
        ax_left.plot(rough_LD_vals[i], clean_LD_vals[i], 'o', color=color_for(i), ms=9)
    ax_left.set_xlabel('Rough L/D', fontsize=font_size)
    ax_left.set_ylabel('Clean L/D', fontsize=font_size)
    ax_left.set_title('Pareto Front, Generation %d' % gen_label, fontsize=font_size)
    ax_left.tick_params(axis='both', labelsize=font_size)
    ax_left.grid(True)
    if xlim_pareto is not None:
        ax_left.set_xlim(xlim_pareto)
    if ylim_pareto is not None:
        ax_left.set_ylim(ylim_pareto)

    # Top right: Kulfan coefficients along the front.
    c = 0
    for i in range(0, N_k // 2):
        ax_top.plot(rough_LD_vals, np.array([p['K_upper'][i] for p in pareto_points]),
                    'o-', color=seismic_cmap(c), ms=3, label='Upper Surface %d' % (i + 1))
        c += 1
    c = N_k - 1
    for i in range(0, N_k // 2):
        ax_top.plot(rough_LD_vals, np.array([p['K_lower'][i] for p in pareto_points]),
                    'o-', color=seismic_cmap(c), ms=3, label='Lower Surface %d' % (i + 1))
        c -= 1
    for i in ixs:
        ax_top.plot(rough_LD_vals[i], 0, 'o', color=color_for(i), ms=5)
    ax_top.legend(loc=legend_loc, fontsize=legend_font_size)
    ax_top.set_xlabel('Rough L/D', fontsize=font_size)
    ax_top.set_ylabel('Airfoil Shape Parameters', fontsize=font_size)
    ax_top.set_title('Pareto Front Airfoil Shapes, Generation %d' % gen_label, fontsize=font_size)
    ax_top.tick_params(axis='both', labelsize=font_size)
    ax_top.grid(True)
    if xlim_top is not None:
        ax_top.set_xlim(xlim_top)
    if ylim_top is not None:
        ax_top.set_ylim(ylim_top)

    # Bottom right: the airfoils themselves.
    for x, y, col in shape_curves:
        ax_bot.plot(x, y, color=col, alpha=1)
    ax_bot.set_xlim(xlim_bottom)
    ax_bot.set_ylim(ylim_b)
    ax_bot.set_aspect('auto')
    ax_bot.grid(True)
    ax_bot.set_xlabel('x/c', fontsize=font_size)
    ax_bot.set_ylabel('y/c', fontsize=font_size)
    ax_bot.tick_params(axis='both', labelsize=font_size)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path
