"""
multicase_surrogate.py  —  ONE batched GPU pulse spanning MANY GA cases.

NEW FILE — builds on batch_surrogate.BatchSurrogate without modifying it. Where BatchSurrogate
does one forward for one case's population, MultiCaseSurrogate concatenates the children (or
initial populations) of EVERY active case into a SINGLE forward, then slices the result back
into each case's own BatchSurrogate cache so the existing make_cached_run/_pack_res serving path
is reused unchanged. The heavy transonic-idle GPU is thus fed one big saturating matmul per
generation instead of N small underutilizing ones (see the sizing analysis: this is the ~3x
utilization win that lifts the concurrent-case ceiling on a big GPU).

Geometry (the constraint path's batched Kulfan) is genome-keyed, so it is precomputed per
geo-config group (n_pts, spacing, toothpick) and handed back per case for installation right
before that case's Phase-B evaluation.

Aero cache keys are per-case (each case owns a BatchSurrogate whose _cache/_sweep_lookup are set
from the shared forward), so two cases sharing (ncrit, xtr) but differing in Re never collide.
"""

import numpy as np

from oso_airfoils.optimization.batch_surrogate import BatchSurrogate, _key
from oso_airfoils.optimization import batch_geometry as _bg


class MultiCaseSurrogate:
    def __init__(self, backend="nxfoil", model_size="xxxlarge", device="cuda",
                 use_cuda_graph=False):
        # the shared net lives here; per-case BatchSurrogate instances reuse it via nxfoil's
        # get_net / nqfoil's load_model model cache, so there is only ONE copy of weights on GPU.
        self._shared = BatchSurrogate(backend=backend, model_size=model_size, device=device,
                                      use_cuda_graph=use_cuda_graph)
        self.backend = backend
        self.model_size = model_size
        self.device = self._shared.device

    def new_case_cache(self):
        """A BatchSurrogate used purely as a per-case cache container + serving front-end.
        It shares the GPU net with self._shared; we never call its build_population_cache —
        pulse_aero() fills its _cache directly from the shared forward."""
        return BatchSurrogate(backend=self.backend, model_size=self.model_size,
                              device=self.device)

    # -------------------------------------------------------------------------------------
    # ONE forward over every case's (airfoil x sweep x alpha) rows, sliced back per case.
    # -------------------------------------------------------------------------------------
    def pulse_aero(self, items):
        """items: list of dicts, each {'surr': <per-case BatchSurrogate>, 'uppers': (P,8),
        'lowers': (P,8), 'tes': (P,) or scalar, 'sweeps': [sweep-dict, ...]}.
        Runs ONE batched forward across all items and populates each item['surr']._cache /
        _sweep_lookup so make_cached_run(...) serves that case."""
        if not items:
            return 0

        ru, rl, rt, ra, rRe, rnc, rxu, rxl = ([] for _ in range(8))
        # per-item bookkeeping so we can slice the single forward output back afterwards
        plans = []          # list of (item, uniq_keys, uidx, row_span{(kpos,sweepname):(start,n)})
        cur = 0
        for it in items:
            U = np.asarray(it['uppers'], float); L = np.asarray(it['lowers'], float)
            P = len(U)
            tes = it['tes']
            tes = (np.broadcast_to(np.asarray(tes, float).ravel(), (P,))
                   if np.ndim(tes) else np.full(P, float(tes)))
            sweeps = it['sweeps']

            keys = [_key(U[i], L[i], tes[i]) for i in range(P)]
            uniq = {}
            for i, k in enumerate(keys):
                uniq.setdefault(k, i)
            uidx = list(uniq.values())

            row_span = {}
            for pos, i in enumerate(uidx):
                for sw in sweeps:
                    al = np.asarray(sw["alphas"], float).ravel(); n = len(al)
                    ru.append(np.tile(U[i], (n, 1))); rl.append(np.tile(L[i], (n, 1)))
                    rt.append(np.full(n, tes[i])); ra.append(al)
                    rRe.append(np.full(n, float(sw["Re"]))); rnc.append(np.full(n, float(sw["ncrit"])))
                    rxu.append(np.full(n, float(sw["xtr_u"]))); rxl.append(np.full(n, float(sw["xtr_l"])))
                    row_span[(pos, sw["name"])] = (cur, n); cur += n
            plans.append((it, list(uniq.keys()), uidx, row_span))

        # ---- the single forward for the whole fleet ----
        U8u = np.concatenate(ru, 0); U8l = np.concatenate(rl, 0)
        A = np.concatenate(ra); TE = np.concatenate(rt)
        RE = np.concatenate(rRe); NC = np.concatenate(rnc); XU = np.concatenate(rxu); XL = np.concatenate(rxl)
        out = self._shared._forward(U8u, U8l, TE, A, RE, NC, XU, XL)
        self.n_rows_last = cur

        # ---- slice back into each case's BatchSurrogate cache ----
        for (it, keyorder, uidx, row_span) in plans:
            surr = it['surr']; sweeps = it['sweeps']
            cache = {}
            for kpos, k in enumerate(keyorder):
                for sw in sweeps:
                    s, n = row_span[(kpos, sw["name"])]
                    rec = {kk: vv[s:s + n] for kk, vv in out.items()}
                    rec["_alphas"] = np.asarray(sw["alphas"], float).ravel()
                    cache[(k, sw["name"])] = rec
            surr._cache = cache
            surr._sweep_lookup = {(round(float(sw["ncrit"]), 6), round(float(sw["xtr_u"]), 6),
                                   round(float(sw["xtr_l"]), 6)): sw["name"] for sw in sweeps}
            surr.n_rows = sum(n for (_, n) in row_span.values())
        return cur

    # -------------------------------------------------------------------------------------
    # Batched constraint geometry, grouped by geo-config (genome-keyed => shareable per group).
    # -------------------------------------------------------------------------------------
    def pulse_geometry(self, items):
        """items: list of dicts, each {'id': hashable, 'uppers', 'lowers', 'tes',
        'n_pts', 'spacing', 'tooth'}. Returns {id: (registry, psi, tooth)} — a genome-keyed
        TorchKulfan registry for each case, batched across cases that share (n_pts, spacing,
        tooth). Install with TorchKulfan.install_registry(...) before that case's eval."""
        groups = {}
        for it in items:
            gk = (int(it['n_pts']), str(it['spacing']), _tooth_key(it['tooth']))
            groups.setdefault(gk, []).append(it)

        result = {}
        for (n_pts, spacing, _tk), gitems in groups.items():
            tooth = gitems[0]['tooth']
            Us = [np.asarray(it['uppers'], float) for it in gitems]
            Ls = [np.asarray(it['lowers'], float) for it in gitems]
            tes = []
            for it in gitems:
                P = len(np.asarray(it['uppers'], float))
                te = it['tes']
                tes.append(np.broadcast_to(np.asarray(te, float).ravel(), (P,))
                           if np.ndim(te) else np.full(P, float(te)))
            U = np.concatenate(Us, 0); L = np.concatenate(Ls, 0); TE = np.concatenate(tes)
            registry, psi = _bg.precompute_population_geometry(
                U, L, TE, n_pts=n_pts, spacing=spacing, toothpick_location=tooth,
                device=self.device)
            for it in gitems:
                result[it['id']] = (registry, psi, tooth)
        return result


def _tooth_key(t):
    if t is None:
        return None
    try:
        return round(float(t), 10)
    except (TypeError, ValueError):
        return tuple(np.round(np.asarray(t, float).ravel(), 10).tolist())
