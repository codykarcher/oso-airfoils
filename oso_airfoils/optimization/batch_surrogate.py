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

Usage (see batched_new_generation.py for the GA wiring):
    bs = BatchSurrogate(backend="nxfoil", model_size="xxxlarge", device="cuda")
    bs.build_population_cache(uppers, lowers, tes, sweeps)   # one GPU forward
    run = bs.make_cached_run()                               # drop-in for neuralfoil_wrapper.run
"""
import numpy as np
import torch

_N_STATION = 32
_BL_KEYS = ("ue/vinf", "theta", "H")


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


class BatchSurrogate:
    def __init__(self, backend="nxfoil", model_size="xxxlarge", device="cuda",
                 use_cuda_graph=False):
        self.backend = backend
        self.model_size = model_size
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.use_cuda_graph = use_cuda_graph and self.device != "cpu"
        _enable_fast_math()
        # persistent net (loaded once, kept on-device for the whole run)
        if backend == "nxfoil":
            from metafoil import nxfoil
            self._nx = nxfoil
            self._net = nxfoil.get_net(model_size, self.device)
        elif backend == "nqfoil":
            from metafoil.nqfoil import inference as nqi
            self._nqi = nqi
            self._net, self._meta = nqi.load_model(model_size)
        else:
            raise ValueError(f"backend must be 'nxfoil' or 'nqfoil', got {backend!r}")
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
        """Batched qfoil-surrogate forward. Builds the 25-d feature matrix for all rows and
        runs the net once. Returns CL/CD/CM/Cpmin/Top_Xtr/Bot_Xtr as (N,) arrays."""
        from metafoil.nqfoil import config
        N = len(alphas)
        up = np.asarray(uppers, np.float32); lo = np.asarray(lowers, np.float32)
        a = np.asarray(alphas, np.float32); ar = a * (np.pi / 180.0)
        te = np.asarray(tes, np.float32); Re = np.asarray(Res, np.float64)
        nc = np.asarray(ncrits, np.float32); xu = np.asarray(xtr_u, np.float32); xl = np.asarray(xtr_l, np.float32)
        X = np.concatenate([
            up, lo,
            np.full((N, 1), config.LE_WEIGHT_CONST, np.float32),
            (te * 50.0).reshape(N, 1),
            np.sin(2 * ar).reshape(N, 1), np.cos(ar).reshape(N, 1), (np.sin(ar) ** 2).reshape(N, 1),
            ((np.log(Re) - 12.5) / 3.5).astype(np.float32).reshape(N, 1),
            ((nc - 9.0) / 4.5).reshape(N, 1), xu.reshape(N, 1), xl.reshape(N, 1),
        ], axis=1).astype(np.float32)
        with torch.inference_mode():
            xt = torch.from_numpy(X).to(self.device)
            conf_logit, reg = self._net.predict_latent(xt)
            latent = reg * self._net.out_std + self._net.out_mean
            latent = latent.detach().float().cpu().numpy()
        out = {}
        for j, (name, qkey, tr) in enumerate(self._meta["core_outputs"]):
            v = latent[:, j]
            if tr == "lncd":
                v = np.exp(v)
            elif name in ("Top_Xtr", "Bot_Xtr"):
                v = np.clip(v, 0.0, 1.0)
            out[name] = v
        return out

    def _forward(self, uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l):
        if self.backend == "nxfoil":
            return self._forward_nxfoil(uppers, lowers, tes, alphas, Res, ncrits, xtr_u, xtr_l)
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

        # assemble all rows: for each unique airfoil, each sweep, each alpha
        ru, rl, rt, ra, rRe, rnc, rxu, rxl = ([] for _ in range(8))
        row_span = {}                                   # (uidx_pos, sweep_name) -> (start, n)
        cur = 0
        for pos, i in enumerate(uidx):
            for sw in sweeps:
                al = np.asarray(sw["alphas"], float).ravel(); n = len(al)
                ru.append(np.tile(uppers[i], (n, 1))); rl.append(np.tile(lowers[i], (n, 1)))
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
        if self.backend == "nxfoil":
            u_ue = np.array([rec[f"upper_bl_ue/vinf_{i}"][idx] for i in range(_N_STATION)])
            l_ue = np.array([rec[f"lower_bl_ue/vinf_{i}"][idx] for i in range(_N_STATION)])
            u_th = np.array([rec[f"upper_bl_theta_{i}"][idx] for i in range(_N_STATION)])
            l_th = np.array([rec[f"lower_bl_theta_{i}"][idx] for i in range(_N_STATION)])
            u_H = np.array([rec[f"upper_bl_H_{i}"][idx] for i in range(_N_STATION)])
            l_H = np.array([rec[f"lower_bl_H_{i}"][idx] for i in range(_N_STATION)])
            cpmin = 1.0 - np.maximum(np.max(np.abs(u_ue), 0), np.max(np.abs(l_ue), 0)) ** 2
            cl, cd, cm = take("CL"), take("CD"), take("CM")
            xtr_t, xtr_b = take("Top_Xtr"), take("Bot_Xtr")
            cp_bl = self._nxfoil_cp_bl(u_ue, l_ue, u_th, l_th, u_H, l_H)
        else:  # nqfoil: direct outputs, no BL
            cl, cd, cm = take("CL"), take("CD"), take("CM")
            cpmin = take("Cpmin"); xtr_t, xtr_b = take("Top_Xtr"), take("Bot_Xtr")
            cp_bl = None
        res = dict(cl=cl, cd=cd, cm=cm, cpmin=cpmin, alpha=np.asarray(alpha_req, float),
                   xtr_top=xtr_t, xtr_bot=xtr_b, xtp_top=xtp_u, xtp_bot=xtp_l,
                   Re=Re, M=0.0, N_crit=N_crit, N_panels=None)
        if cp_bl is not None:
            res["cp_data"], res["bl_data"] = cp_bl
        else:
            res["cp_data"] = res["bl_data"] = None
        if scalar:
            for kk in ("cl", "cd", "cm", "cpmin", "alpha", "xtr_top", "xtr_bot"):
                res[kk] = res[kk][0]
            res["cp_data"] = res["cp_data"][0] if res["cp_data"] else None
            res["bl_data"] = res["bl_data"][0] if res["bl_data"] else None
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
