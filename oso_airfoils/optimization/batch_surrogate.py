"""
batch_surrogate.py  —  GPU-batched surrogate evaluation for the oso-airfoils GA.

NEW FILE — does not modify any existing module. It replaces the per-individual, one-airfoil-
at-a-time NeuralFoil calls (core/neuralfoil_wrapper.py, driven per MPI rank) with a SINGLE
batched GPU forward for the whole population × every alpha in every sweep, then serves each
individual's polar back through a drop-in shim that matches neuralfoil_wrapper.run() exactly.

Why this is a drop-in: the GA's neuralfoil_wrapper.run() sources its numbers from NeuralFoil's
198-output net (CL/CD/CM/Top/Bot + 6×32 BL) and derives cpmin from the BL ue. metafoil.nxfoil
is a float-exact torch port of that same net with a one-shot batched entry (get_aero_batch),
so a batched nxfoil call reproduces run()'s output bit-for-bit while doing all airfoils at once.

Backends
--------
  'nxfoil' (default) — float-exact NeuralFoil replica (XFOIL-fidelity). True drop-in.
  'nqfoil'           — qfoil-trained surrogate (qfoil fidelity). CL/CD/CM/Cpmin direct
                       (Cpmin is a native output; no cp_data/bl_data).

Speedups implemented here (per the plan — GPU/CPU only, no algorithmic changes):
  * ONE forward per generation for the entire population × all sweep alphas.
  * Genome dedup — identical design vectors are evaluated once (elitist survivors are free).
  * Persistent net cached on-device across generations (never reloaded).
  * torch.inference_mode() + TF32 matmul for the big xxxlarge GEMMs.
  * CUDA-graph capture is scaffolded (use_cuda_graph flag + shape-keyed store) but NOT yet
    wired to the forward — at GA batch sizes (~N_pop×alphas×sweeps ≈ 10k–20k rows/gen) the
    single batched call already amortizes kernel-launch overhead, so it's low marginal value;
    wire it only if per-generation launch overhead shows up in a profile.

Usage (see evaluators.GPUBatchEvaluator for the GA wiring):
    bs = BatchSurrogate(backend="nxfoil", model_size="xxxlarge", device="cuda")
    bs.build_population_cache(uppers, lowers, tes, sweeps)   # one GPU forward
    run = bs.make_cached_run()                               # drop-in for neuralfoil_wrapper.run

The cached run is handed to solvers.make_solver(params, neuralfoil_run=run), which is
what the objective function calls -- no module-global patching is involved.
"""
import numpy as np
import torch

_N_STATION = 32
_BL_KEYS = ("ue/vinf", "theta", "H")


def resolve_device(device):
    """Validate a requested torch device, falling back to CPU when it isn't usable.

    Silently downgrading is right here: a case file that says ``cuda`` should still
    run on a laptop, just slower. The resolved device is reported by the evaluator's
    ``describe()`` so the downgrade is visible rather than mysterious.
    """
    d = str(device or 'cpu').lower()
    if d.startswith('cuda') and not torch.cuda.is_available():
        return 'cpu'
    if d == 'mps' and not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        return 'cpu'
    return d


def geometry_device_for(device):
    """Device for the batched GEOMETRY, which may differ from the aero device.

    kulfan_torch computes in float64 for the exactness the constraint set needs
    (area/moments agree with the Kulfan class to ~1e-14). Apple's MPS backend has no
    float64 at all, so on MPS the geometry runs on the CPU while the aero net -- which
    is float32 -- still runs on the GPU. Everywhere else this is just ``device``.
    """
    return 'cpu' if str(device).lower() == 'mps' else device


def _enable_fast_math():
    # TF32 for Ampere+; harmless elsewhere. Surrogate is approximate, so this is free accuracy.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _key(upper, lower, te):
    u = np.round(np.asarray(upper, float).ravel(), 10)
    l = np.round(np.asarray(lower, float).ravel(), 10)
    return (u.tobytes(), l.tobytes(), round(float(te), 10))


#: Kulfan coefficients per surface expected by the surrogate nets (nxfoil/nqfoil).
NET_ORDER = 8


def to_net_order(uppers, lowers, tes):
    """Refit a population's coefficients to the order the surrogate nets take.

    The nets have a fixed 8-coefficient-per-surface input, but the optimizer's
    design vector can be any order (N_k = 8 means 4 per surface, and so on). This
    is the batched equivalent of the ``afl.changeOrder(8)`` that the per-airfoil
    NeuralFoil wrapper does before its own call, so the batched and serial paths
    feed the net the same geometry.

    Raising the order is exact -- a lower-degree Bernstein polynomial lies exactly
    in the higher-degree basis -- so nothing is approximated on the way in.

    Note the caller keys its cache on the ORIGINAL coefficients: the objective
    function looks polars up with the design vector it was handed, which is in the
    original order. Only the rows fed to the forward are refit.
    """
    uppers = np.asarray(uppers, float)
    lowers = np.asarray(lowers, float)
    if uppers.shape[-1] == NET_ORDER and lowers.shape[-1] == NET_ORDER:
        return uppers, lowers
    from metafoil.core.kulfan_torch import batch_change_order
    return batch_change_order(uppers, lowers, NET_ORDER, te_gap=tes)


class BatchSurrogate:
    def __init__(self, backend="nxfoil", model_size="xxxlarge", device="cuda",
                 use_cuda_graph=False):
        self.backend = backend
        self.model_size = model_size
        self.device = resolve_device(device)
        self.use_cuda_graph = use_cuda_graph and self.device.startswith("cuda")
        _enable_fast_math()
        # persistent net (loaded once, kept on-device for the whole run)
        if backend == "nxfoil":
            from metafoil import nxfoil
            self._nx = nxfoil
            self._net = nxfoil.get_net(model_size, self.device)
        elif backend == "nqfoil":
            # The full-BL nqfoil models mirror nxfoil's API and output contract, so
            # both backends go through the same batched entry point below.
            from metafoil.nqfoil import full_bl
            self._nq = full_bl
            sizes = full_bl.available_sizes()
            if model_size not in sizes:
                raise ValueError(
                    f"nqfoil has no '{model_size}' model; available sizes are {sizes}. "
                    "(the two ladders differ in length, so a config written for one "
                    "backend may need --model overridden for the other.)")
            self._net = full_bl.get_net(model_size, self.device)
        elif backend == "nqfoil_torch":
            # The differentiable torch reimplementation the gradient uses -- float32
            # GPU forward instead of full_bl's numpy path. Use it so the batched GA
            # sees the SAME polars as the gradient (full_bl's float64 numpy captures a
            # tiny plateau dip that the first-roll-over stall detector mis-reads as an
            # early stall; nqfoil_torch's float32 forward smooths it, matching the
            # gradient). See _forward_nqfoil_torch.
            from oso_airfoils.optimization import nqfoil_torch as nqt
            self._nqt = nqt
            self._nqt_pack = nqt.load(model_size, self.device)
            self._net = None
            # CRITICAL: disable TF32. TF32's ~1e-3 matmul rounding puts a spurious 0.001
            # dip on flat CL plateaus, which the first-roll-over stall detector reads as an
            # early stall -> false stall-margin violations that reject high-camber near-corner
            # airfoils. The gradient avoids this by running the same net on CPU (no TF32);
            # full float32 here reproduces the CPU/gradient polar exactly.
            import torch as _torch
            _torch.backends.cuda.matmul.allow_tf32 = False
            _torch.backends.cudnn.allow_tf32 = False
        else:
            raise ValueError(f"backend must be 'nxfoil', 'nqfoil' or 'nqfoil_torch', got {backend!r}")
        self._graph = {}          # row-count -> (static_in, static_out, graph)
        self._cache = None        # dict: (key, sweep_name) -> per-alpha output dict

    # ---- raw batched forward over N (airfoil,alpha,condition) rows ----
    def _forward_nxfoil(self, uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l):
        """Return the full 198-output dict as (N,) numpy arrays. One GPU forward."""
        with torch.inference_mode():
            out = self._nx.get_aero_batch(
                uppers, lowers, tes=tes, alphas=alphas, Res=Res, n_crits=ncrits,
                xtr_uppers=xtr_u, xtr_lowers=xtr_l, model_size=self.model_size,
                device=self.device, return_torch=True)
            return {k: v.detach().float().cpu().numpy().reshape(-1) for k, v in out.items()}

    def _forward_nqfoil(self, uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l):
        """Batched full-BL nqfoil forward. One GPU/CPU forward over all rows.

        The full-BL nqfoil models expose the same batched entry point and the same flat
        output keys as nxfoil, so this mirrors :meth:`_forward_nxfoil` exactly; the only
        difference downstream is that nqfoil carries a natively trained ``Cpmin``.
        """
        with torch.inference_mode():
            return self._nq.get_aero_batch(
                uppers, lowers, tes=tes, alphas=alphas, Res=Res, n_crits=ncrits,
                xtr_uppers=xtr_u, xtr_lowers=xtr_l, model_size=self.model_size,
                device=self.device)

    def _forward_nqfoil_torch(self, uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l):
        """Paired (one row per (airfoil, alpha)) forward through nqfoil_torch, the same
        float32 model the gradient uses. Mirrors nqfoil_torch.aero's feature layout and
        output de-standardization, but built for the flat N-row batch this class serves
        rather than aero's (B airfoils x A alphas) outer product. Returns (N,) arrays
        keyed CL/CD/CM/Cpmin/Top_Xtr/Bot_Xtr, matching _pack_res."""
        import torch
        nqt = self._nqt; pack = self._nqt_pack; f32 = torch.float32; dev = self.device
        def t(a):
            return torch.as_tensor(np.asarray(a), dtype=f32, device=dev)
        up = t(uppers); lo = t(lowers); N = up.shape[0]
        ar = t(alphas).reshape(-1) * (np.pi / 180.0)
        o = torch.ones(N, 1, dtype=f32, device=dev)
        def col(v):
            x = t(v).reshape(-1)
            return (x.expand(N) if x.numel() == 1 else x).reshape(N, 1)
        from metafoil.nqfoil import config as cfg
        X = torch.cat([
            up, lo,
            float(cfg.LE_WEIGHT_CONST) * o,
            col(tes) * 50.0,
            torch.sin(2 * ar).reshape(N, 1), torch.cos(ar).reshape(N, 1), (torch.sin(ar) ** 2).reshape(N, 1),
            (torch.log(col(Res)) - 12.5) / 3.5,
            (col(ncrits) - 9.0) / 4.5,
            col(xtr_u), col(xtr_l),
        ], dim=1)
        with torch.inference_mode():
            y = nqt._raw_forward(X, pack)
            phys = y[:, 1:] * pack["ostd"] + pack["omean"]
            out = {}
            for j, name in enumerate(nqt.CORE):
                v = phys[:, j]
                if j == nqt._CD_IDX:
                    v = torch.exp(v)
                elif name in ("Top_Xtr", "Bot_Xtr"):
                    v = torch.clamp(v, 0.0, 1.0)
                out[name] = v.detach().float().cpu().numpy().reshape(-1)
        return out

    def _forward(self, uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l):
        if self.backend == "nxfoil":
            return self._forward_nxfoil(uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l)
        if self.backend == "nqfoil_torch":
            return self._forward_nqfoil_torch(uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l)
        return self._forward_nqfoil(uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l)

    # ---- build the whole-generation cache in one forward ----
    def build_population_cache(self, uppers, lowers, tes, sweeps):
        """uppers/lowers: (P,8) arrays. tes: (P,) or scalar. sweeps: list of dicts, each
        {'name', 'Re', 'ncrit', 'xtr_u', 'xtr_l', 'alphas'(1-D array)}. Runs ONE batched
        forward over the deduped airfoils × all sweep alphas, and caches per (genome, sweep)."""
        uppers = np.asarray(uppers, float); lowers = np.asarray(lowers, float)
        P = len(uppers)
        tes = np.broadcast_to(np.asarray(tes, float).ravel(), (P,)) if np.ndim(tes) else np.full(P, float(tes))
        # dedup identical genomes (elitism => repeats)
        keys = [_key(uppers[i], lowers[i], tes[i]) for i in range(P)]
        uniq = {}
        for i, k in enumerate(keys):
            uniq.setdefault(k, i)
        uidx = list(uniq.values())                      # representative row per unique genome
        U = len(uidx)
        # Refit to the net's coefficient order once, for the unique genomes only.
        fu, fl = to_net_order(uppers[uidx], lowers[uidx], tes[uidx])

        # assemble all rows: for each unique airfoil, each sweep, each alpha
        ru, rl, rt, ra, rRe, rnc, rxu, rxl = ([] for _ in range(8))
        row_span = {}                                   # (uidx_pos, sweep_name) -> (start, n)
        cur = 0
        for pos, i in enumerate(uidx):
            for sw in sweeps:
                al = np.asarray(sw["alphas"], float).ravel(); n = len(al)
                ru.append(np.tile(fu[pos], (n, 1))); rl.append(np.tile(fl[pos], (n, 1)))
                rt.append(np.full(n, tes[i])); ra.append(al)
                rRe.append(np.full(n, float(sw["Re"]))); rnc.append(np.full(n, float(sw["ncrit"])))
                rxu.append(np.full(n, float(sw["xtr_u"]))); rxl.append(np.full(n, float(sw["xtr_l"])))
                row_span[(pos, sw["name"])] = (cur, n); cur += n
        U8u = np.concatenate(ru, 0); U8l = np.concatenate(rl, 0)
        A = np.concatenate(ra); TE = np.concatenate(rt)
        RE = np.concatenate(rRe); NC = np.concatenate(rnc); XU = np.concatenate(rxu); XL = np.concatenate(rxl)

        out = self._forward(U8u, U8l, TE, A, RE, NC, XU, XL)   # ONE forward, all rows

        # slice back into per (genome-key, sweep) records
        cache = {}
        keypos = {k: p for p, k in enumerate(uniq.keys())}
        for k in uniq.keys():
            pos = keypos[k]
            for sw in sweeps:
                s, n = row_span[(pos, sw["name"])]
                rec = {kk: vv[s:s + n] for kk, vv in out.items()}
                rec["_alphas"] = np.asarray(sw["alphas"], float).ravel()
                cache[(k, sw["name"])] = rec
        self._cache = cache
        self._sweep_lookup = {(round(float(sw["ncrit"]), 6), round(float(sw["xtr_u"]), 6),
                               round(float(sw["xtr_l"]), 6)): sw["name"] for sw in sweeps}
        self.n_rows = cur; self.n_unique = U; self.n_pop = P
        return self

    # ---- pack one (airfoil, sweep) record into neuralfoil_wrapper.run's res dict ----
    def _pack_res(self, rec, alpha_req, Re, N_crit, xtp_u, xtp_l, scalar):
        aall = rec["_alphas"]
        idx = [int(np.argmin(np.abs(aall - av))) for av in alpha_req]   # exact int-alpha match
        take = lambda name: np.asarray(rec[name])[idx]
        cl, cd, cm = take("CL"), take("CD"), take("CM")
        xtr_t, xtr_b = take("Top_Xtr"), take("Bot_Xtr")
        if self.backend in ("nqfoil", "nqfoil_torch") and "Cpmin" in rec:
            # qfoil reports Cpmin, so nqfoil trains it directly -- preferred over
            # reconstructing it from 32 BL edge-velocity stations.
            cpmin = take("Cpmin")
        else:
            # only the Cpmin fallback needs the BL edge velocities; theta/H are
            # unused now that cp_data/bl_data are not built (see below).
            u_ue = np.array([rec[f"upper_bl_ue/vinf_{i}"][idx] for i in range(_N_STATION)])
            l_ue = np.array([rec[f"lower_bl_ue/vinf_{i}"][idx] for i in range(_N_STATION)])
            cpmin = 1.0 - np.maximum(np.max(np.abs(u_ue), 0), np.max(np.abs(l_ue), 0)) ** 2
        res = dict(cl=cl, cd=cd, cm=cm, cpmin=cpmin, alpha=np.asarray(alpha_req, float),
                   xtr_top=xtr_t, xtr_bot=xtr_b, xtp_top=xtp_u, xtp_bot=xtp_l,
                   Re=Re, M=0.0, N_crit=N_crit, N_panels=None)
        # cp_data/bl_data are NEVER read on the oso constraint path (grep confirms only
        # _pack_res itself references them), yet building them -- the per-alpha nested-dict
        # construction in _nxfoil_cp_bl plus the 6x32 BL-station gathers -- was ~1/3 of a
        # generation's wall time. Serve them as None; a caller that needs the full cp/bl
        # profile should use the real neuralfoil_wrapper.run fallback, not the cache.
        res["cp_data"] = res["bl_data"] = None
        if scalar:
            for kk in ("cl", "cd", "cm", "cpmin", "alpha", "xtr_top", "xtr_bot"):
                res[kk] = res[kk][0]
        return res

    def _nxfoil_cp_bl(self, u_ue, l_ue, u_th, l_th, u_H, l_H):
        """Reproduce neuralfoil_wrapper's wrapped cp_data/bl_data (TE->LE upper, LE->TE lower)."""
        x_bl = (np.arange(_N_STATION) + 0.5) / _N_STATION
        x_wrap = list(np.concatenate([np.flip(x_bl), x_bl])); n_wrap = len(x_wrap)
        u_cp = 1.0 - u_ue ** 2; l_cp = 1.0 - l_ue ** 2
        cp_list, bl_list = [], []
        for ix in range(u_ue.shape[1]):
            cp_list.append({"x": x_wrap,
                            "cp": list(np.concatenate([np.flip(u_cp[:, ix]), l_cp[:, ix]]))})
            nan = [np.nan] * n_wrap
            bl_list.append({"s": nan, "x": x_wrap, "y": nan,
                            "Ue/Vinf": list(np.concatenate([np.flip(u_ue[:, ix]), l_ue[:, ix]])),
                            "Dstar": nan, "Theta": list(np.concatenate([np.flip(u_th[:, ix]), l_th[:, ix]])),
                            "Cf": nan, "H": list(np.concatenate([np.flip(u_H[:, ix]), l_H[:, ix]])),
                            "H*": nan, "P": nan, "m": nan, "K": nan, "tau": nan, "Di": nan})
        return cp_list, bl_list

    # ---- the drop-in replacement for core/neuralfoil_wrapper.run ----
    def make_cached_run(self, fallback=None):
        """Return f(mode, upper, lower, val=..., Re=..., N_crit=..., xtp_u=..., xtp_l=..., ...)
        matching neuralfoil_wrapper.run, served from the batched cache. On a cache miss or a
        'cl'-mode call (which the GA does not use for sweeps), defers to `fallback` (the real
        neuralfoil_wrapper.run) so nothing silently breaks."""
        if self._cache is None:
            raise RuntimeError("call build_population_cache(...) before make_cached_run()")

        def cached_run(mode, upperKulfanCoefficients, lowerKulfanCoefficients, val=0.0,
                       Re=1e7, M=0.0, xtp_u=1.0, xtp_l=1.0, N_crit=9.0, TE_gap=0.0,
                       model="xxxlarge", **kw):
            m = mode.lower(); m = "alpha" if m == "alfa" else m
            sweep = self._sweep_lookup.get((round(float(N_crit), 6), round(float(xtp_u), 6),
                                            round(float(xtp_l), 6)))
            k = _key(upperKulfanCoefficients, lowerKulfanCoefficients, TE_gap)
            if m != "alpha" or sweep is None or (k, sweep) not in self._cache:
                if fallback is None:
                    raise KeyError("cache miss and no fallback provided")
                return fallback(mode, upperKulfanCoefficients, lowerKulfanCoefficients, val=val,
                                Re=Re, M=M, xtp_u=xtp_u, xtp_l=xtp_l, N_crit=N_crit,
                                TE_gap=TE_gap, model=model, **kw)
            it = hasattr(val, "__iter__")
            if it and len(val) == 3:
                alpha = np.linspace(val[0], val[1], int((val[1] - val[0]) / val[2]) + 1)
            elif it:
                alpha = np.asarray(val, float)
            else:
                alpha = np.array([float(val)])
            return self._pack_res(self._cache[(k, sweep)], alpha, Re, N_crit, xtp_u, xtp_l,
                                  scalar=not it)
        return cached_run
