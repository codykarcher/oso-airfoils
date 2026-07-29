"""
nqfoil_torch.py — differentiable, batched, device-aware nqfoil aero for the
gradient Pareto drivers.

WHY THIS EXISTS
---------------
`metafoil.nqfoil.full_bl` is the right model but the wrong shape for optimization:

  * `get_aero_from_kulfan_parameters` evaluates ONE airfoil over an alpha grid;
  * `get_aero_batch` accepts many airfoils but computes "through numpy" (its own
    docstring) inside `torch.no_grad()`, so it is neither differentiable nor
    actually resident on the accelerator despite taking a `device=` argument.

Both gradient tracks need the opposite: one graph, many airfoils x many alphas,
differentiable, on the GPU. That is what this module provides. It reproduces
`full_bl`'s forward EXACTLY -- `config.featurize`'s 25-column layout, the
reflection-symmetry output fusion (ON for xxlarge), the per-output
de-standardization, and CD's ln-storage -- and is verified against the numpy path
in `test_matches_full_bl()`.

MEASURED, on this machine (Apple MPS, float32-only -- no float64 on Metal):

    batch     CPU        MPS      speedup
       64    5.3 ms    7.0 ms       0.76x   <- GPU LOSES here
     1024   21.8 ms    3.2 ms       6.8x
     4096  108.2 ms    6.1 ms      17.9x
    16384  343.9 ms   23.0 ms      15.0x

The 0.76x at batch 64 is the whole design constraint: a single 17-variable Ipopt
subproblem generates ~80 rows and is FASTER on the CPU. Speed only comes from
evaluating the entire Pareto front -- every point, every multi-start seed, every
alpha, clean and rough -- as one batch. A 20-point front x 8 seeds x 40 alphas x 2
conditions is ~12,800 rows, i.e. squarely in the 15-18x band.

CPMIN: taken as the model's own trained `Cpmin` output. This DIFFERS from the
nxfoil/NeuralFoil convention used elsewhere in the oso constraint stack, which
derives it as 1 - max(ue/vinf)^2 from the boundary-layer channels; on a NACA 2412
at alpha=4 the two disagree by 3.3% (-1.2624 direct vs -1.3034 derived). Direct is
correct here because qfoil reports Cpmin and nqfoil is trained on it. Use
`cpmin_from_ue=True` to reproduce the nxfoil convention for cross-checks.
"""
from __future__ import annotations

import numpy as np

_CACHE: dict = {}

CORE = ("CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr")
_CD_IDX = 1          # CD is stored as ln(CD) at this index of the 198-output block
_NCORE = 6
_NBL = 32
_BL_OFF = {"u_theta": 0, "u_H": 32, "u_ue": 64,
           "l_theta": 96, "l_H": 128, "l_ue": 160}


def load(model_size: str = "xxlarge", device: str = "cpu"):
    """Load and cache (net, meta, symmetry tensors) on `device`, in float32.

    float32 is not a choice on Apple silicon: Metal has no float64. The CPU path
    is float32 too so that the two devices agree bit-for-bit modulo reduction
    order -- otherwise a CPU/GPU discrepancy is indistinguishable from a bug.
    """
    key = (model_size, device)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    from metafoil.nqfoil import full_bl as fb
    net, meta = fb.load_model(model_size, device)
    net = net.to(device).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)          # the NET is fixed; we differentiate INPUTS
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    S = None
    if int(meta["symmetry"]):
        ip, isg, op, osg = fb._sym_arrays()
        S = dict(ip=t(ip, torch.long), isg=t(isg), op=t(op, torch.long), osg=t(osg))
    pack = dict(net=net, omean=t(meta["omean"]), ostd=t(meta["ostd"]), sym=S,
                meta=meta, device=device)
    _CACHE[key] = pack
    return pack


def featurize(up, lo, te, alpha_deg, Re, n_crit, xtr_u, xtr_l, device="cpu"):
    """Assemble the exact 25-column nqfoil input, differentiably.

    up, lo   : (B, 8)   airfoil weights (metafoil convention, LE weight held at 0)
    alpha_deg: (A,)     swept angles
    te, Re, n_crit, xtr_u, xtr_l : scalars or (B,)

    Returns (B*A, 25) with row index b*A + a, so outputs reshape to (B, A).
    Column layout is config.featurize's: 0..7 upper | 8..15 lower | 16 LE |
    17 TE*50 | 18 sin2a | 19 cos a | 20 sin^2 a | 21 (lnRe-12.5)/3.5 |
    22 (ncrit-9)/4.5 | 23 xtr_u | 24 xtr_l.
    """
    import torch
    from metafoil.nqfoil import config as cfg
    f32 = torch.float32
    as_t = lambda v: v if torch.is_tensor(v) else torch.as_tensor(
        np.asarray(v, np.float32), dtype=f32, device=device)
    up = as_t(up).to(f32); lo = as_t(lo).to(f32)
    if up.dim() == 1:
        up = up.unsqueeze(0)
    if lo.dim() == 1:
        lo = lo.unsqueeze(0)
    B = max(up.shape[0], lo.shape[0])
    if up.shape[0] == 1 and B > 1:
        up = up.expand(B, -1)
    if lo.shape[0] == 1 and B > 1:
        lo = lo.expand(B, -1)
    a = as_t(alpha_deg).reshape(-1).to(f32)
    A = a.shape[0]
    ar = a * (np.pi / 180.0)

    def bcol(v):
        """scalar or (B,) -> (B, 1) so it broadcasts over alpha."""
        x = as_t(v).reshape(-1).to(f32)
        return (x.expand(B) if x.numel() == 1 else x).reshape(B, 1)

    # (B, A, k) blocks then flatten -- keeps the graph on the two weight tensors
    upA = up.unsqueeze(1).expand(B, A, up.shape[-1])
    loA = lo.unsqueeze(1).expand(B, A, lo.shape[-1])
    o = torch.ones(B, A, 1, dtype=f32, device=up.device)
    aA = ar.reshape(1, A, 1).expand(B, A, 1)
    cols = torch.cat([
        upA, loA,
        float(cfg.LE_WEIGHT_CONST) * o,
        (bcol(te) * 50.0).unsqueeze(1).expand(B, A, 1),
        torch.sin(2 * aA), torch.cos(aA), torch.sin(aA) ** 2,
        ((torch.log(bcol(Re)) - 12.5) / 3.5).unsqueeze(1).expand(B, A, 1),
        ((bcol(n_crit) - 9.0) / 4.5).unsqueeze(1).expand(B, A, 1),
        bcol(xtr_u).unsqueeze(1).expand(B, A, 1),
        bcol(xtr_l).unsqueeze(1).expand(B, A, 1),
    ], dim=2)
    return cols.reshape(B * A, cfg.INPUT_DIM), B, A


def _raw_forward(X, pack):
    """net(X) with the reflection-symmetry fusion, graph intact."""
    net, S = pack["net"], pack["sym"]
    y = net(X)
    if S is not None:
        y = 0.5 * (y + net(X[:, S["ip"]] * S["isg"])[:, S["op"]] * S["osg"])
    return y


def aero(up, lo, alphas, te=0.0, Re=1e6, n_crit=9.0, xtr_u=1.0, xtr_l=1.0,
         model_size="xxlarge", device="cpu", want_bl=False, cpmin_from_ue=False):
    """Differentiable batched aero.

    Returns a dict of (B, A) tensors: CL, CD, CM, Cpmin, Top_Xtr, Bot_Xtr, LoD,
    analysis_confidence; plus (B, A, 32) BL groups when `want_bl`.
    Gradients flow to `up`/`lo` (and `alphas`) when those carry requires_grad.
    """
    import torch
    pack = load(model_size, device)
    X, B, A = featurize(up, lo, te, alphas, Re, n_crit, xtr_u, xtr_l, device)
    y = _raw_forward(X, pack)
    conf = torch.sigmoid(y[:, 0])
    phys = y[:, 1:] * pack["ostd"] + pack["omean"]

    out = {"analysis_confidence": conf.reshape(B, A)}
    for j, name in enumerate(CORE):
        v = phys[:, j]
        if j == _CD_IDX:
            v = torch.exp(v)                                  # CD stored as ln(CD)
        elif name in ("Top_Xtr", "Bot_Xtr"):
            v = torch.clamp(v, 0.0, 1.0)
        out[name] = v.reshape(B, A)
    out["LoD"] = out["CL"] / out["CD"]

    if want_bl or cpmin_from_ue:
        for g, off in _BL_OFF.items():
            out[g] = phys[:, _NCORE + off:_NCORE + off + _NBL].reshape(B, A, _NBL)
        if cpmin_from_ue:
            # the nxfoil/NeuralFoil convention, for cross-checking only
            ue2 = torch.cat([out["u_ue"], out["l_ue"]], dim=2) ** 2
            out["Cpmin_ue"] = 1.0 - ue2.max(dim=2).values
    return out


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def test_matches_full_bl(model_size="xxlarge", device="cpu", tol=2e-5):
    """This torch forward must reproduce full_bl's numpy forward. A mismatch here
    invalidates every downstream gradient, so this is the acceptance gate."""
    import torch
    from metafoil.nqfoil import full_bl as fb
    from metafoil.core.kulfan import Kulfan
    rng = np.random.default_rng(0)
    worst = {}
    for (m, p, t) in ((2.0, 0.4, 12.0), (4.0, 0.45, 18.0), (0.5, 0.25, 9.0)):
        a = Kulfan().naca4_like(m, p, t)
        u = np.asarray(a.upperCoefficients, float)
        l = np.asarray(a.lowerCoefficients, float)
        al = np.linspace(-4, 12, 9)
        ref = fb.get_aero_from_kulfan_parameters(
            {"upper_weights": u, "lower_weights": l, "TE_thickness": 0.0025},
            alpha=al, Re=6e6, n_crit=9.0, model_size=model_size)
        got = aero(u, l, al, te=0.0025, Re=6e6, n_crit=9.0,
                   model_size=model_size, device=device)
        for k in ("CL", "CD", "CM", "Cpmin"):
            r = np.asarray(ref[k], float).ravel()
            g = got[k].detach().cpu().numpy().ravel()
            rel = np.max(np.abs(g - r) / np.maximum(np.abs(r), 1e-6))
            worst[k] = max(worst.get(k, 0.0), float(rel))
    ok = all(v < tol for v in worst.values())
    print(f"  torch-vs-numpy worst relative error: "
          + "  ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items())))
    print(f"  -> {'MATCH' if ok else 'MISMATCH'} (tol {tol:.0e}, float32 net)")
    return ok, worst


def test_gradient(model_size="xxlarge", device="cpu", h=2e-3):
    """Autograd d(LoD)/d(weights) against central differences of the numpy path."""
    import torch
    from metafoil.nqfoil import full_bl as fb
    from metafoil.core.kulfan import Kulfan
    a = Kulfan().naca4_like(2.0, 0.4, 12.0)
    u0 = np.asarray(a.upperCoefficients, float)
    l0 = np.asarray(a.lowerCoefficients, float)
    al = np.array([4.0])
    ut = torch.tensor(u0, dtype=torch.float32, device=device, requires_grad=True)
    lt = torch.tensor(l0, dtype=torch.float32, device=device, requires_grad=True)
    out = aero(ut, lt, al, te=0.0025, Re=6e6, model_size=model_size, device=device)
    out["LoD"].sum().backward()
    ga = np.concatenate([ut.grad.cpu().numpy(), lt.grad.cpu().numpy()])

    def lod(uu, ll):
        r = fb.get_aero_from_kulfan_parameters(
            {"upper_weights": uu, "lower_weights": ll, "TE_thickness": 0.0025},
            alpha=al, Re=6e6, model_size=model_size)
        return float(np.asarray(r["CL"]).ravel()[0] / np.asarray(r["CD"]).ravel()[0])

    fd = np.zeros(16)
    for i in range(16):
        up_, um_ = u0.copy(), u0.copy()
        lp_, lm_ = l0.copy(), l0.copy()
        if i < 8:
            up_[i] += h; um_[i] -= h
        else:
            lp_[i - 8] += h; lm_[i - 8] -= h
        fd[i] = (lod(up_, lp_) - lod(um_, lm_)) / (2 * h)
    den = max(np.max(np.abs(fd)), 1e-9)
    rel = np.max(np.abs(ga - fd)) / den
    print(f"  autograd vs central-difference d(L/D)/dK: max rel {rel:.2e} "
          f"(|grad|max {den:.3f}, h={h})")
    print(f"  -> {'OK' if rel < 5e-3 else 'SUSPECT'}")
    return rel


if __name__ == "__main__":
    import torch, time
    print("nqfoil_torch self-test\n")
    print("[1] forward parity vs full_bl (cpu)")
    test_matches_full_bl(device="cpu")
    if torch.backends.mps.is_available():
        print("[2] forward parity vs full_bl (mps)")
        test_matches_full_bl(device="mps", tol=1e-3)
    print("[3] gradient check (cpu)")
    test_gradient(device="cpu")


# ----------------------------------------------------------------------
# exact forward-mode jacobian, hand-propagated through the MLP
# ----------------------------------------------------------------------
def aero_jac(up, lo, alphas, te=0.0, Re=1e6, n_crit=9.0, xtr_u=1.0, xtr_l=1.0,
             model_size="xxlarge", device="cpu"):
    """Values AND d/d[up(8), lo(8)] for one airfoil over an alpha grid.

    Propagating the tangents by hand keeps this a single batched matmul chain
    with no autograd graph and no tracing. Measured warm on a 41-alpha sweep:
    3.12 ms here vs 7.25 ms for torch.func.jacfwd (2.3x), agreeing to 8.7e-07.
    (An earlier 1.6 s figure for jacfwd was a COLD measurement dominated by the
    one-shot model load -- both paths are ~1 s cold.) The network is
    Linear/SiLU only, so
    network is Linear/SiLU only, so
        Linear : V <- V W^T + b,      T <- W T
        SiLU   : V <- silu(V),        T <- silu'(V) * T,
                 silu'(x) = s(x)(1 + x(1 - s(x))),  s = sigmoid
    and the reflection-symmetry fusion is applied to values and tangents alike.
    Exact to machine precision, no autograd graph, no tracing.

    Returns (vals, jac) with vals {name: (A,)} and jac {name: (A, 16)}.
    """
    import torch
    pack = load(model_size, device)
    net = pack["net"]
    X, B, A = featurize(up, lo, te, alphas, Re, n_crit, xtr_u, xtr_l, device)
    assert B == 1, "aero_jac is the single-airfoil path; use aero() for batches"
    K = 16
    N = X.shape[0]
    T = torch.zeros(N, X.shape[1], K, dtype=X.dtype, device=X.device)
    T[:, :K, :] = torch.eye(K, dtype=X.dtype, device=X.device).unsqueeze(0)

    def run(V, T):
        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                V = torch.nn.functional.linear(V, layer.weight, layer.bias)
                T = torch.einsum('oi,nik->nok', layer.weight, T)
            else:                                    # SiLU
                s = torch.sigmoid(V)
                T = (s * (1 + V * (1 - s))).unsqueeze(-1) * T
                V = V * s
        return V, T

    Y, TY = run(X, T)
    S = pack["sym"]
    if S is not None:
        X2 = X[:, S["ip"]] * S["isg"]
        T2 = T[:, S["ip"], :] * S["isg"].reshape(1, -1, 1)
        Y2, TY2 = run(X2, T2)
        Y = 0.5 * (Y + Y2[:, S["op"]] * S["osg"].reshape(1, -1))
        TY = 0.5 * (TY + TY2[:, S["op"], :] * S["osg"].reshape(1, -1, 1))

    phys = Y[:, 1:] * pack["ostd"] + pack["omean"]
    tphys = TY[:, 1:, :] * pack["ostd"].reshape(1, -1, 1)
    vals, jac = {}, {}
    for j, name in enumerate(CORE):
        v, t = phys[:, j], tphys[:, j, :]
        if j == _CD_IDX:
            v = torch.exp(v); t = v.unsqueeze(-1) * t        # d(exp u) = exp(u) du
        vals[name] = v; jac[name] = t
    cl, cd = vals["CL"], vals["CD"]
    vals["LoD"] = cl / cd
    jac["LoD"] = (jac["CL"] * cd.unsqueeze(-1) - jac["CD"] * cl.unsqueeze(-1)) \
                 / (cd ** 2).unsqueeze(-1)
    return ({k: v.detach().cpu().numpy() for k, v in vals.items()},
            {k: v.detach().cpu().numpy() for k, v in jac.items()})
