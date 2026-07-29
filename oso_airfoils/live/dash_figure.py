"""
dash_figure.py -- one lightweight figure for the live dashboard.

Replaces the two project figures (pareto_frame + polar rainbow) for LIVE use. Those
are the right artefacts for a finished run, but they cost ~9 s a frame here: between
them they draw the whole population, all 16 coefficient traces, six polar panels and
a three-column legend, and the SVG write scales with that artist count.

This draws only what the dashboard reads, in one figure, top to bottom:

    1  airfoil geometries across the front
    2  the Pareto front
    3  Kulfan shape parameters (legend outside, right)
    4  C_L and C_M vs alpha
    5  L/D vs C_L
    6  C_p,min vs C_L   -- only when a cp_min constraint is actually active

The front's turbo ramp is kept, and matched to the project plots' truncation so
colours mean the same thing across all of them.

The figure is built ONCE and its artists updated in place on subsequent calls, which
is worth doing here (unlike in the project plots) because the artist count is fixed
and small.
"""

import numpy as np

CMAP_RAINBOW, CMAP_LOWER, CMAP_UPPER = 'turbo', 0.10, 0.90
MUTED_COLOR = '0.62'      # superseded phase-1 locators
CLEAN_LS, ROUGH_LS = '-', '--'
REF_COLOR = 'k'


def _ramp(n):
    from oso_airfoils.postprocessing.polars import get_colors
    return get_colors(max(n, 2), CMAP_RAINBOW, lower=CMAP_LOWER, upper=CMAP_UPPER)


def color_at(t):
    """Continuous truncated-turbo lookup at t in [0, 1].

    The ramp above is INDEXED BY POSITION, which is right for a GA population
    (every member exists at once, evenly spaced). It is wrong for a gradient run
    that produces points over time: with only the two endpoints solved, they take
    ramp slots 0 and 1 of n_front and both come out at the blue end, rather than
    sitting at the two extremes where they belong. Mapping the point's rough L/D
    onto [0, 1] instead puts the endpoints at the ends by construction and lets
    the interior fill in at its true colour as it arrives.
    """
    import matplotlib.pyplot as plt
    import numpy as _np
    t = float(_np.clip(t, 0.0, 1.0))
    return plt.get_cmap(CMAP_RAINBOW)(CMAP_LOWER + t * (CMAP_UPPER - CMAP_LOWER))


def cpmin_active(params) -> bool:
    """True when any C_p,min constraint is switched on for this case."""
    for k in ('cp_min_design', 'cp_min_at_alpha_offset', 'cp_min_prestall'):
        if params.get(k) is not None:
            for w in ('cp_min_design_weighting', 'cp_min_at_alpha_offset_weighting',
                      'cp_min_prestall_weighting'):
                if params.get(w):
                    return True
    return False


def nondominated(rough, clean):
    """Boolean mask of the Pareto-efficient members (maximise BOTH objectives).

    The gradient driver solves each epsilon level independently, so a point can
    easily be beaten on both axes by a neighbour -- a run of the T21 front had
    eps=132.0 reaching clean 168.3 / rough 132.6, which dominates the 128.1,
    129.4 and 130.7 points outright. Those are real solver output but they are
    NOT on the front, so they should not be joined by the front line, given a
    front colour, or have their geometry and polars drawn.
    """
    r = np.asarray(rough, float); c = np.asarray(clean, float)
    keep = np.ones(r.size, bool)
    for i in range(r.size):
        if not np.isfinite(r[i]) or not np.isfinite(c[i]):
            keep[i] = False
            continue
        # strictly beaten on one axis and not worse on the other
        keep[i] = not np.any(((r >= r[i]) & (c >= c[i])) &
                             ((r > r[i]) | (c > c[i])))
    return keep


def sample_front(snap, n):
    """Indices of front members sampled evenly along the front's arc length.

    Dominated members are excluded, so the sampled set -- which drives the
    geometry overlay and the polar rainbow -- contains only true front points.
    """
    rough = np.asarray(snap['front_rough'], float)
    clean = np.asarray(snap['front_clean'], float)
    if rough.size == 0:
        return []
    nd = nondominated(rough, clean)
    if nd.any():
        idx_nd = np.nonzero(nd)[0]
        sub = dict(snap, front_rough=rough[nd].tolist(),
                   front_clean=clean[nd].tolist())
        return [int(idx_nd[j]) for j in _arc_sample(sub, n)]
    return []


def _arc_sample(snap, n):
    rough = np.asarray(snap['front_rough'], float)
    clean = np.asarray(snap['front_clean'], float)
    if rough.size == 0:
        return []
    xs = float(np.ptp(rough)) or 1.0
    ys = float(np.ptp(clean)) or 1.0
    arc = np.concatenate(([0.0], np.cumsum(np.sqrt(
        np.diff(rough / xs) ** 2 + np.diff(clean / ys) ** 2))))
    if arc[-1] <= 0 or rough.size < 2:
        return list(np.unique(np.linspace(0, rough.size - 1, n).astype(int)))
    return list(np.unique([int(np.argmin(np.abs(arc - t)))
                           for t in np.linspace(0, arc[-1], n)]))


class DashFigure:
    """Persistent figure; :meth:`draw` refreshes it for a new generation."""

    def __init__(self, params, n_front=9, show_cpmin=None, figsize=None):
        """Layout: geometry across the top, then everything else side by side.

        The Pareto front and the shape parameters share an x-axis (both are against
        rough L/D), so they are stacked in one column with the axis drawn once. The
        polar panes sit beside them rather than below so they keep a sane aspect
        ratio instead of being stretched into ribbons.
        """
        import matplotlib.pyplot as plt
        self.params = params
        self.n_front = n_front
        self.show_cpmin = cpmin_active(params) if show_cpmin is None else show_cpmin
        self.colors = _ramp(n_front)

        # Left column gets 45% and the polar panes split the rest, so the stacked
        # Pareto/shape column is readable instead of squeezed.
        widths = [0.45, 0.275, 0.275] + ([0.275] if self.show_cpmin else [])
        n_cols = len(widths)
        if figsize is None:
            figsize = (16.5 if not self.show_cpmin else 20.5, 8.4)
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        outer = self.fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0])

        self.ax_geo = self.fig.add_subplot(outer[0])
        # Tight between the polar panes; the left column needs a little more room
        # because its y-labels sit outside.
        lower = outer[1].subgridspec(1, n_cols, width_ratios=widths, wspace=0.13)

        # column 0: Pareto over shape parameters, one shared x-axis
        col0 = lower[0].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.06)
        self.ax_par = self.fig.add_subplot(col0[0])
        self.ax_shp = self.fig.add_subplot(col0[1], sharex=self.ax_par)
        self.ax_par.tick_params(labelbottom=False)

        self.ax_cl = self.fig.add_subplot(lower[1])
        self.ax_ld = self.fig.add_subplot(lower[2])
        self.ax_cp = self.fig.add_subplot(lower[3]) if self.show_cpmin else None
        self.ax_cm = self.ax_cl.twinx()
        self.ax_cm.yaxis.set_label_position('right')
        self._restyle()

    # -- drawing ---------------------------------------------------------------

    def draw(self, snap, idx, polars, reference=None):
        """Refresh every pane from the compact payload. ``idx`` are the sampled
        front indices; ``polars`` maps position in ``idx`` -> per-condition dict."""
        te_gap = snap['ip']['TE_gap']
        N_k = int(snap['ip']['N_k'])
        fu, fl = snap['front_upper'], snap['front_lower']
        fr = np.asarray(snap['front_rough'], float)
        fc = np.asarray(snap['front_clean'], float)
        rough_s, clean_s = fr[idx], fc[idx]
        # One dominance mask for the whole figure: the Pareto line, the shape
        # traces and the sampled polars must all describe the SAME set of points.
        nd_all = nondominated(fr, fc) if fr.size else np.zeros(0, bool)

        # Value-based colours when the driver supplies the front's rough-L/D span
        # (gradient runs). Without it, fall back to the positional ramp so GA
        # behaviour is untouched.
        span = snap.get('front_span')
        if span and len(span) == 2 and span[1] > span[0]:
            lo, hi = float(span[0]), float(span[1])
            cols = [color_at((r - lo) / (hi - lo)) for r in rough_s]
        else:
            cols = [self.colors[i % len(self.colors)] for i in range(len(idx))]
        # Points the driver marks as superseded scaffolding drop to grey, so the
        # coloured set is exactly what the run will report.
        mut = snap.get('front_muted')
        if mut:
            cols = [MUTED_COLOR if mut[j] else c
                    for c, j in zip(cols, idx)]
        self._cols = cols

        for ax in (self.ax_geo, self.ax_par, self.ax_shp, self.ax_cl,
                   self.ax_cm, self.ax_ld):
            ax.clear()
        if self.ax_cp is not None:
            self.ax_cp.clear()
        self._restyle()

        # 1 ── geometries -------------------------------------------------------
        from metafoil.core.kulfan import Kulfan
        for i, j in enumerate(idx):
            a = Kulfan(TE_gap=te_gap)
            a.upperCoefficients = fu[j]; a.lowerCoefficients = fl[j]
            self.ax_geo.plot(a.xcoordinates, a.ycoordinates, color=self._cols[i], lw=1.1)
        if reference is not None and reference[2] is not None:
            rx, ry = reference[2]
            self.ax_geo.plot(rx, ry, color=REF_COLOR, lw=1.3, zorder=10,
                             label=reference[0])
            self.ax_geo.legend(loc='upper right', fontsize=8, frameon=False)

        # 2 ── Pareto front -----------------------------------------------------
        # Whole population as one scatter (not one call per front) -- the single
        # biggest artist saving versus the project frame.
        px = np.asarray(snap['pop_rough'], float)
        py = np.asarray(snap['pop_clean'], float)
        pf = np.asarray(snap['pop_front'], float)
        ok = np.isfinite(px) & np.isfinite(py) & (np.abs(px) < 1e4) & (np.abs(py) < 1e4)
        self.ax_par.scatter(px[ok], py[ok], s=7, c=np.clip(pf[ok], 1, 8),
                            cmap='Greys', vmin=0, vmax=10, linewidths=0, zorder=2)
        if fr.size:
            nd = nd_all
            # dominated points stay visible as plain grey dots -- they are real
            # results and their position is informative -- but carry no front
            # colour and are not joined by the line
            if (~nd).any():
                self.ax_par.plot(fr[~nd], fc[~nd], 'o', color='0.55', ms=4,
                                 mfc='none', mew=.9, zorder=3)
            if nd.sum() > 1:
                o = np.argsort(fr[nd])
                self.ax_par.plot(fr[nd][o], fc[nd][o], '-', color='0.55',
                                 lw=.8, zorder=3)
        for i in range(len(idx)):
            self.ax_par.plot(rough_s[i], clean_s[i], 'o', color=self._cols[i],
                             ms=8, zorder=4)

        # 3 ── shape parameters -------------------------------------------------
        import matplotlib.pyplot as plt
        seis = plt.get_cmap('seismic', N_k)
        if fr.size and nd_all.any():
            # Non-dominated members ONLY, sorted by rough L/D. Previously this
            # traced every member in list order: dominated points dragged each
            # coefficient curve off to an off-front design and back, and any
            # non-monotonic ordering drew the line backwards -- together that made
            # the pane look scrambled rather than showing how the coefficients
            # march along the front.
            o = np.argsort(fr[nd_all])
            xs = fr[nd_all][o]
            U = np.asarray(fu, float)[nd_all][o]
            L = np.asarray(fl, float)[nd_all][o]
            for j in range(N_k // 2):
                self.ax_shp.plot(xs, U[:, j], '-', color=seis(j), lw=1)
            for j in range(N_k // 2):
                self.ax_shp.plot(xs, L[:, j], '-', color=seis(N_k - 1 - j), lw=1)
            # legend outside, to the right, so it never covers the traces
            # No legend here -- 16 traces can't be labelled usefully in this column,
            # and the seismic ramp (blue upper / red lower) already reads at a glance.

        # 4/5/6 ── polars -------------------------------------------------------
        for i in range(len(idx)):
            pol = polars.get(i)
            if not pol:
                continue
            for cond, ls in (('clean', CLEAN_LS), ('rough', ROUGH_LS)):
                d = pol.get(cond)
                if not d:
                    continue
                z = 3 if cond == 'clean' else 2      # clean layered on top of rough
                self.ax_cl.plot(d['alpha'], d['cl'], ls, color=self._cols[i], lw=1.2, zorder=z)
                self.ax_cm.plot(d['alpha'], d['cm'], ls, color=self._cols[i], lw=.7,
                                alpha=.55, zorder=z)
                self.ax_ld.plot(d['lod'], d['cl'], ls, color=self._cols[i], lw=1.2, zorder=z)
                if self.ax_cp is not None:
                    self.ax_cp.plot(d['cpmin'], d['cl'], ls, color=self._cols[i],
                                    lw=1.2, zorder=z)
        if reference is not None and reference[1]:
            for cond, ls in (('clean', CLEAN_LS), ('rough', ROUGH_LS)):
                d = reference[1].get(cond)
                if not d:
                    continue
                self.ax_cl.plot(d['alpha'], d['cl'], ls, color=REF_COLOR, lw=1.4, zorder=20)
                self.ax_cm.plot(d['alpha'], d['cm'], ls, color=REF_COLOR, lw=.8,
                                alpha=.6, zorder=20)
                self.ax_ld.plot(d['lod'], d['cl'], ls, color=REF_COLOR, lw=1.4, zorder=20)
                if self.ax_cp is not None:
                    self.ax_cp.plot(d['cpmin'], d['cl'], ls, color=REF_COLOR,
                                    lw=1.4, zorder=20)

        # Keep C_M in the BOTTOM HALF of the pane. Sharing the full height with C_L
        # makes the two sets of curves cross and become unreadable; squashing C_M's
        # axis into the lower half separates them while keeping the shared alpha axis.
        cm_lo, cm_hi = self.ax_cm.get_ylim()
        if np.isfinite(cm_lo) and np.isfinite(cm_hi) and cm_hi > cm_lo:
            self.ax_cm.set_ylim(cm_lo, cm_lo + 2.0 * (cm_hi - cm_lo))

        cl_design = self.params.get('CL')
        for ax in (self.ax_cl, self.ax_ld, self.ax_cp):
            if ax is not None and cl_design is not None:
                ax.axhline(cl_design, color='0.35', lw=.8, zorder=1.2)
        self.ax_ld.set_xlim(left=0)
        return self.fig

    def _restyle(self):
        self.ax_geo.set_xlabel('x/c'); self.ax_geo.set_ylabel('y/c')
        # 'box' not 'datalim': with a wide, short axis datalim-adjustment pads the
        # x-range out to +/-0.5 instead of shrinking the axes box to fit the chord.
        self.ax_geo.set_aspect('equal', adjustable='box')
        self.ax_geo.set_xlim(-0.02, 1.02)
        self.ax_par.set_ylabel('Clean L/D')
        self.ax_shp.set_xlabel('Rough L/D'); self.ax_shp.set_ylabel('Kulfan coeff.')
        self.ax_par.tick_params(labelbottom=False)
        self.ax_cl.set_xlabel(r'$\alpha$  [deg]'); self.ax_cl.set_ylabel('$C_L$')
        self.ax_cm.set_ylabel('$C_M$', labelpad=1)
        self.ax_cm.yaxis.set_label_position('right')
        self.ax_ld.set_xlabel('$L/D$'); self.ax_ld.set_ylabel('$C_L$')
        axes = [self.ax_geo, self.ax_par, self.ax_shp, self.ax_cl, self.ax_ld]
        if self.ax_cp is not None:
            self.ax_cp.set_xlabel('$C_{p,min}$'); self.ax_cp.set_ylabel('$C_L$')
            self.ax_cp.xaxis.set_inverted(True)
            axes.append(self.ax_cp)
        for ax in axes:
            ax.grid(True, lw=.4, alpha=.5)

    def save(self, path):
        self.fig.savefig(path)
        return path


# ── polar data straight off the batched surrogate ────────────────────────────

def batched_polars(kulfans, Re, turb_cases, alphas, backend, model, device='cpu'):
    """Clean+rough polars for MANY airfoils in ONE forward.

    Returns ``[{cond: {alpha, cl, cd, cm, cpmin, lod}}]`` in input order. This is the
    cheap half of a frame (~0.02 s for a dozen airfoils); the figure is the rest.
    """
    from oso_airfoils.optimization.batch_surrogate import BatchSurrogate, _key
    key = ('_dash', backend, model, device)
    bs = _CACHE.get(key)
    if bs is None:
        bs = _CACHE[key] = BatchSurrogate(backend=backend, model_size=model, device=device)
    names = ('clean', 'rough')
    sweeps = [dict(name=names[i], Re=Re, ncrit=tc[0], xtr_u=tc[1], xtr_l=tc[2],
                   alphas=np.asarray(alphas, float))
              for i, tc in enumerate(turb_cases[:2])]
    ups = np.array([np.asarray(k.upperCoefficients, float) for k in kulfans])
    los = np.array([np.asarray(k.lowerCoefficients, float) for k in kulfans])
    tes = np.array([float(k.constants.TE_gap) for k in kulfans], float)
    bs.build_population_cache(ups, los, tes, sweeps)
    run = bs.make_cached_run()
    out = []
    for i, k in enumerate(kulfans):
        rec = {}
        for sw in sweeps:
            r = run('alpha', ups[i], los[i], val=list(alphas), Re=Re,
                    N_crit=sw['ncrit'], xtp_u=sw['xtr_u'], xtp_l=sw['xtr_l'],
                    TE_gap=float(tes[i]), model=model)
            cl = np.asarray(r['cl'], float); cd = np.asarray(r['cd'], float)
            with np.errstate(divide='ignore', invalid='ignore'):
                lod = np.where(cd > 0, cl / cd, np.nan)
            rec[sw['name']] = dict(alpha=np.asarray(r['alpha'], float), cl=cl, cd=cd,
                                   cm=np.asarray(r['cm'], float),
                                   cpmin=np.asarray(r['cpmin'], float), lod=lod)
        out.append(rec)
    return out


_CACHE: dict = {}
