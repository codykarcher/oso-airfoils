"""Active-constraint regime map for a gradient Pareto front.

Given a clean/rough L/D front (a list of airfoil designs), evaluate the hard
constraint set at every point, segment the front into regimes wherever the set
of *active* constraints changes, and render a full-width clean-vs-rough scatter
with the background shaded by regime and each band labelled with the constraints
active there beyond the always-on shape regularizers.

The key subtlety: constraints are stored raw (``R.Ixx - Ixx_con`` etc.), so their
natural scales differ by orders of magnitude. Activity is therefore judged on a
margin **relative to each constraint's own bound**, not a single absolute tol.

Typical use::

    from oso_airfoils.postprocessing.constraint_front import plot_constraint_front_for_thickness
    plot_constraint_front_for_thickness(24, oso_root, "pareto_t24.json", "cscatter_t24.png",
                                        ref={'clean_LD': 245, 'rough_LD': 117})
"""
import json
import numpy as np


# constraints tracked individually (everything else -- te_cone / curvature /
# aft-curvature / sign-flip / min-radius / non-intersection -- is a per-station
# shape regularizer that is active somewhere almost everywhere, so it is folded
# into the always-on baseline rather than used to split regimes).
def default_margins(R, p):
    """Return {label: (margin, tol)}; a constraint is ACTIVE where margin < tol.
    Margins are relative-to-bound where the bound has a natural scale."""
    v = lambda a: getattr(R, a).v
    m = {}
    m['Ixx'] = ((v('Ixx') - p['Ixx_con']) / p['Ixx_con'], 0.01)
    m['Iyy'] = ((v('Iyy') - p['Iyy_con']) / p['Iyy_con'], 0.01)
    if p['Izz_con'] > p['Ixx_con'] + p['Iyy_con']:
        m['Izz'] = ((v('Izz') - p['Izz_con']) / p['Izz_con'], 0.01)
    m['area'] = ((v('area') - p['A_con']) / p['A_con'], 0.01)
    m['LE-rad'] = (min((v('ler_u') - p['ler_con_upper']) / p['ler_con_upper'],
                       (v('ler_l') - p['ler_con_lower']) / p['ler_con_lower']), 0.01)
    if p.get('ler_skew_factor') is not None:
        m['radii-skew'] = ((p['ler_skew_factor'] * min(v('ler_u'), v('ler_l'))
                            - max(v('ler_u'), v('ler_l'))) / max(v('ler_u'), v('ler_l')), 0.02)
    mt = [v('taumax_psi') - p['max_thickness_loc']]
    if p.get('max_thickness_loc_upper') is not None:
        mt.append(v('taumax_psi_upper') - p['max_thickness_loc_upper'])
    if p.get('max_thickness_loc_lower') is not None:
        mt.append(v('taumax_psi_lower') - p['max_thickness_loc_lower'])
    m['maxThk-loc'] = (min(mt) / p['max_thickness_loc'], 0.01)
    m['stall-marg'] = (min(v('stall_margin_clean'), v('stall_margin_rough'))
                       - p['target_stall_margin'], 0.15)
    m['CLmax-reach'] = (v('lift_margin_clean'), 0.03)
    m['dCL-rough'] = (p['percent_delta_cl_from_roughness_threshold'] - abs(v('delta_cl_pct')), 0.005)
    return m


# map each tracked-margin label to its CONSTRAINT_GROUPS toggle key, so the regime
# map only shows constraints that were actually ENABLED during the optimization
# (e.g. radii_skew OFF => don't flag airfoils that merely sit near the skew bound).
_GROUP_OF = {
    'Ixx': 'moments_of_inertia', 'Iyy': 'moments_of_inertia', 'Izz': 'moments_of_inertia',
    'area': 'area', 'LE-rad': 'leading_edge_radius', 'radii-skew': 'radii_skew',
    'maxThk-loc': 'max_thickness_location', 'stall-marg': 'stall_margin',
    'CLmax-reach': 'reach_design_cl', 'dCL-rough': 'roughness_delta_cl',
}


def _nondominated(af):
    keep = []
    for i, p in enumerate(af):
        ri, ci = p['rough_LD'], p['clean_LD']
        if not any(q['rough_LD'] >= ri and q['clean_LD'] >= ci
                   and (q['rough_LD'] > ri or q['clean_LD'] > ci)
                   for j, q in enumerate(af) if j != i):
            keep.append(p)
    return keep


def active_constraint_regimes(airfoils, params, model, *, enabled=None,
                              margins_fn=default_margins, smooth=3):
    """Evaluate each front airfoil and segment into active-constraint regimes.

    Returns (rough, clean, chunks, perpetual) where chunks is a list of
    (start_idx, end_idx, active_labels) and perpetual is the set of tracked
    constraints active at every point.
    """
    from oso_airfoils.optimization import gradient_objective as og
    ctx = og.make_context(tool='nqfoil', model_size=model, params=params, enabled=enabled)
    ctx['n_aux'] = 1
    rg, cl, A, keys = [], [], [], None
    for a in airfoils:
        z = np.concatenate([np.asarray(a['upper_coefficients'], float),
                            np.asarray(a['lower_coefficients'], float),
                            [a.get('psi_star', 0.5)]])
        R = og.evaluate(z, ctx)
        m = margins_fn(R, params)
        if enabled is not None:               # only show constraints actually enforced
            m = {k: v for k, v in m.items() if enabled.get(_GROUP_OF.get(k, k), True)}
        if keys is None:
            keys = list(m.keys())
        rg.append(a['rough_LD']); cl.append(a['clean_LD'])
        A.append([m[k][0] < m[k][1] for k in keys])
    rg = np.asarray(rg, float); cl = np.asarray(cl, float)
    A = np.asarray(A, bool); n = len(rg)
    # median-smooth each constraint's active signal to drop single-point flicker
    S = A.copy()
    if smooth and n >= smooth:
        for j in range(len(keys)):
            for i in range(n):
                lo, hi = max(0, i - smooth // 2), min(n, i + smooth // 2 + 1)
                S[i, j] = np.sum(A[lo:hi, j]) >= (hi - lo) / 2.0
    perpetual = set(keys[j] for j in range(len(keys)) if S[:, j].all())
    varj = [j for j in range(len(keys)) if keys[j] not in perpetual]
    sig = [tuple(S[i, varj]) for i in range(n)]
    chunks, start = [], 0
    for i in range(1, n):
        if sig[i] != sig[start]:
            chunks.append((start, i - 1)); start = i
    chunks.append((start, n - 1))
    out = [(s, e, [keys[j] for j in varj if S[s, j]]) for (s, e) in chunks]
    return rg, cl, out, perpetual


def plot_active_constraint_front(airfoils, params, model, out_path, *, ref=None,
                                 title=None, enabled=None, nondominated=True,
                                 margins_fn=default_margins, figsize=(17, 6.4), overlay=None):
    """Render the active-constraint regime map for a front. `airfoils` is a list
    of design dicts (upper_coefficients, lower_coefficients, psi_star, clean_LD,
    rough_LD). `ref` optionally {'clean_LD','rough_LD'} draws the baseline star.
    `overlay` optionally draws a comparison front (path to a pareto json, or a list of
    airfoil dicts) as a light-grey line behind the scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    af = _nondominated(airfoils) if nondominated else list(airfoils)
    af = sorted(af, key=lambda p: p['rough_LD'])
    rg, cl, chunks, perpetual = active_constraint_regimes(
        af, params, model, enabled=enabled, margins_fn=margins_fn)
    n = len(rg)
    fig, ax = plt.subplots(figsize=figsize)
    cols = plt.cm.Pastel2(np.linspace(0, 1, 8))
    span = cl.max() - cl.min()
    ytop = cl.max() + span * 0.02
    for k, (s, e, act) in enumerate(chunks):
        x0 = rg[s] - (0.5 if s == 0 else (rg[s] - rg[s - 1]) / 2)
        x1 = rg[e] + (0.5 if e == n - 1 else (rg[e + 1] - rg[e]) / 2)
        ax.axvspan(x0, x1, color=cols[k % 8], alpha=0.6, zorder=0)
        lab = "\n".join(act) if act else "(baseline)"
        fs = float(np.clip((x1 - x0) * 2.0, 6.0, 10.5))
        ax.text((x0 + x1) / 2, ytop, lab, ha='center', va='top', fontsize=fs, color='#1a1a1a',
                zorder=5, linespacing=1.05,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.7))
    if overlay is not None:
        ov = json.load(open(overlay))['airfoils'] if isinstance(overlay, str) else list(overlay)
        ov = sorted(_nondominated(ov), key=lambda p: p['rough_LD'])
        ax.plot([p['rough_LD'] for p in ov], [p['clean_LD'] for p in ov],
                color='0.72', lw=2.6, zorder=1, label='v4 front', solid_capstyle='round')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.85)
    ax.scatter(rg, cl, c=np.arange(n), cmap='jet', s=62, zorder=4, edgecolors='k', linewidths=0.4)
    ax.plot(rg, cl, color='0.35', lw=0.8, zorder=3)
    if ref and ref.get('rough_LD') and ref.get('clean_LD'):
        ax.scatter([ref['rough_LD']], [ref['clean_LD']], marker='*', s=300, c='k', zorder=6)
    ax.set_xlabel("rough L/D", fontsize=12)
    ax.set_ylabel("clean L/D", fontsize=12)
    ax.set_ylim(cl.min() - span * 0.05, ytop + span * 0.02)
    ax.margins(x=0.005)
    base = ', '.join(sorted(perpetual)) if perpetual else '(none)'
    ax.set_title(title or f"active-constraint regimes along the front "
                 f"(always active: {base} + shape regularizers)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return chunks, perpetual


def plot_constraint_front_for_thickness(thickness, oso_root, front, out_path, *,
                                        ref=None, model=None, params_overrides=None, **kw):
    """Convenience wrapper: load the T-family params, read `front` (a path to a
    pareto json or a list of airfoil dicts), and render the regime map."""
    from oso_airfoils.optimization.pareto_gold import load_family_params, CONSTRAINTS
    if 'enabled' not in kw:                # default to the run's actual constraint toggles
        kw['enabled'] = CONSTRAINTS        # (so disabled constraints, e.g. radii_skew, aren't shown)
    p, _ = load_family_params(thickness, oso_root)
    p['alpha_min_clean'] = 0; p['alpha_min_rough'] = 0     # match the run's grid for a_des
    if params_overrides:
        p.update(params_overrides)
    if isinstance(front, str):
        d = json.load(open(front)); airfoils = d['airfoils']; model = model or d['meta'].get('model')
    else:
        airfoils = front
    title = f"OSO-WT2-T{thickness}  —  active-constraint regimes along the front"
    return plot_active_constraint_front(airfoils, p, model or 'xxlarge', out_path,
                                        ref=ref, title=title, **kw)
