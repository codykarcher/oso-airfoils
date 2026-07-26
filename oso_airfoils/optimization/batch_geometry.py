"""
batch_geometry.py  —  GPU/CPU-batched airfoil GEOMETRY for a whole population in one shot.

NEW FILE — does not modify anything. Wraps metafoil.kulfan_torch.kulfan_geometry (the batched,
differentiable torch reimplementation of the Kulfan geometry) so you can evaluate a "run of
geometry parameters" for many airfoils at once instead of building one metafoil.Kulfan object
per airfoil. kulfan_torch is validated against the Kulfan class to ~1e-10 (area/moments) and
exact for LE radius, so the batched values match the per-object class.

    from oso_airfoils.optimization.batch_geometry import run_geometry_batch
    g = run_geometry_batch(uppers, lowers, tes=0.0, device="cuda")   # one call, all airfoils
    g["area"], g["Ixx"], g["Iyy"], g["Izz"], g["max_thickness"], g["leading_edge_radius_upper"], ...
    #   -> each a (P,) numpy array

Returned keys (all (P,) arrays, chord = 1):
    area, xcentroid, ycentroid, Ixx, Iyy, Izz, perimeter,
    max_thickness, max_thickness_x, leading_edge_radius_upper, leading_edge_radius_lower, TE_gap

Notes
-----
* Batched: one Gauss-Legendre quadrature over all airfoils; ~2450x the per-object Kulfan class.
* Differentiable: pass return_torch=True to keep autograd tensors (e.g. for constraint grads).
* The GA's *constraint* set also needs curvature (d2zeta_dpsi2), the TE-cone (psi/zetaUpper/
  zetaLower) and self-intersection height — those are pure CST-polynomial ops not yet in
  kulfan_torch; until they're ported+validated, the GA driver keeps the real Kulfan for those
  constraints (see batched_new_generation.py). This module covers the structural/thickness/LE
  quantities, which is what a standalone geometry-parameter sweep needs.
"""
import numpy as np


def run_geometry_batch(uppers, lowers, tes=0.0, te_shift=0.0, N1=0.5, N2=1.0,
                       device="cpu", return_torch=False):
    """uppers/lowers: (P,8) arrays (or (8,)). tes/te_shift: scalar or (P,). One batched call.
    Returns a dict of (P,) numpy arrays (or torch tensors if return_torch)."""
    import torch
    from metafoil.kulfan_torch import kulfan_geometry
    up = np.atleast_2d(np.asarray(uppers, float)); lo = np.atleast_2d(np.asarray(lowers, float))
    P = max(len(up), len(lo))
    g = kulfan_geometry(up, lo, te_gap=tes, te_shift=te_shift, N1=N1, N2=N2, device=device)
    if return_torch:
        return g
    out = {}
    for k, v in g.items():
        a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        out[k] = a.reshape(-1)
    return out


def _key(upper, lower, te):
    u = np.round(np.asarray(upper, float).ravel(), 10)
    l = np.round(np.asarray(lower, float).ravel(), 10)
    return (u.tobytes(), l.tobytes(), round(float(te), 10))


def precompute_population_geometry(uppers, lowers, tes=0.0, n_pts=140, spacing="cosinele",
                                   toothpick_location=None, N1=0.5, N2=1.0, device="cpu"):
    """Batched geometry for the whole population in a handful of kulfan_torch calls. Returns
    (registry, internal_psi): registry maps genome-key -> a record dict with every quantity the
    oso constraint set reads (Ixx/Iyy/Izz/area/tau/taumax_*/LE-radii + zeta/d2zeta over the
    internal grid + thickness at the grid and the toothpick location). One record per UNIQUE
    genome (elitist repeats are computed once)."""
    import torch
    from metafoil.kulfan_torch import kulfan_geometry, kulfan_surfaces, kulfan_surface_extrema
    from metafoil.core.kulfan import _psi_grid
    U = np.atleast_2d(np.asarray(uppers, float)); L = np.atleast_2d(np.asarray(lowers, float))
    P = max(len(U), len(L))
    tes = np.broadcast_to(np.asarray(tes, float).ravel(), (P,)) if np.ndim(tes) else np.full(P, float(tes))
    keys = [_key(U[i], L[i], tes[i]) for i in range(P)]
    uniq = {}
    for i, k in enumerate(keys):
        uniq.setdefault(k, i)
    idx = list(uniq.values()); Uu = U[idx]; Lu = L[idx]
    Tu = torch.as_tensor(tes[idx], dtype=torch.float64)     # (U,) tensor: bypasses float(v) on arrays
    psi = _psi_grid(n_pts, spacing); psi_int = psi[1:-1]

    g = kulfan_geometry(Uu, Lu, te_gap=Tu, N1=N1, N2=N2, device=device)
    surf = kulfan_surfaces(Uu, Lu, psi, te_gap=Tu, N1=N1, N2=N2, device=device, with_d2=False)
    d2 = kulfan_surfaces(Uu, Lu, psi_int, te_gap=Tu, N1=N1, N2=N2, device=device, with_d2=True)
    ext = kulfan_surface_extrema(Uu, Lu, te_gap=Tu, N1=N1, N2=N2, device=device)
    tooth = None
    if toothpick_location is not None:
        ts = kulfan_surfaces(Uu, Lu, [float(toothpick_location)], te_gap=Tu, N1=N1, N2=N2,
                             device=device, with_d2=False)
        tooth = (ts["zeta_upper"] - ts["zeta_lower"]).detach().cpu().numpy().reshape(-1)
    npx = lambda t: t.detach().cpu().numpy()
    zu = npx(surf["zeta_upper"]); zl = npx(surf["zeta_lower"])
    d2u = npx(d2["d2zeta_upper"]); d2l = npx(d2["d2zeta_lower"])
    G = {kk: npx(vv) for kk, vv in g.items()}
    tmu = npx(ext["taumax_psi_upper"]); tml = npx(ext["taumax_psi_lower"]); tmt = npx(ext["taumax_psi"])
    registry = {}
    for p, k in enumerate(uniq.keys()):
        registry[k] = dict(
            Ixx=float(G["Ixx"][p]), Iyy=float(G["Iyy"][p]), Izz=float(G["Izz"][p]),
            area=float(G["area"][p]), tau=float(G["max_thickness"][p]),
            ler_u=float(G["leading_edge_radius_upper"][p]), ler_l=float(G["leading_edge_radius_lower"][p]),
            taumax_psi=float(tmt[p]), taumax_psi_upper=float(tmu[p]), taumax_psi_lower=float(tml[p]),
            zeta_upper=zu[p], zeta_lower=zl[p], thickness_grid=zu[p] - zl[p],
            d2_upper_int=d2u[p], d2_lower_int=d2l[p],
            tooth=(float(tooth[p]) if tooth is not None else None))
    return registry, psi


class TorchKulfan:
    """Read-only drop-in for metafoil.core.kulfan.Kulfan on the oso constraint path, backed by a
    precomputed batched-geometry record (set via install_registry). Constructed as Kulfan(TE_gap=)
    then .upper/lowerCoefficients assigned — exactly how core_fitness_function uses it."""
    _REGISTRY = {}; _PSI = None; _TOOTH = None; _N1 = 0.5; _N2 = 1.0

    @classmethod
    def install_registry(cls, registry, psi, toothpick_location=None):
        cls._REGISTRY = registry; cls._PSI = np.asarray(psi, float); cls._TOOTH = toothpick_location

    def __init__(self, TE_gap=0.0, **kw):
        self._te = float(TE_gap); self._u = None; self._l = None; self._rec = None; self._chord = 1.0

    def _bind(self):
        if self._u is not None and self._l is not None and self._rec is None:
            self._rec = self._REGISTRY.get(_key(self._u, self._l, self._te))
            if self._rec is None:                       # off-population genome: compute one-off
                self._rec = self._compute_one()
        return self._rec

    def _compute_one(self):
        reg, _ = precompute_population_geometry(self._u[None, :], self._l[None, :], self._te,
                                                n_pts=len(self._PSI), toothpick_location=self._TOOTH,
                                                N1=self._N1, N2=self._N2)
        return next(iter(reg.values()))

    # coefficient setters/getters
    @property
    def upperCoefficients(self): return self._u
    @upperCoefficients.setter
    def upperCoefficients(self, v): self._u = np.asarray(v, float); self._rec = None; self._bind()
    @property
    def lowerCoefficients(self): return self._l
    @lowerCoefficients.setter
    def lowerCoefficients(self, v): self._l = np.asarray(v, float); self._rec = None; self._bind()
    @property
    def chord(self): return self._chord
    @chord.setter
    def chord(self, v): self._chord = getattr(v, "magnitude", v)

    # read-only geometry the constraints use
    @property
    def psi(self): return self._PSI
    @property
    def zetaUpper(self): return self._bind()["zeta_upper"]
    @property
    def zetaLower(self): return self._bind()["zeta_lower"]
    @property
    def Ixx(self): return self._bind()["Ixx"]
    @property
    def Iyy(self): return self._bind()["Iyy"]
    @property
    def Izz(self): return self._bind()["Izz"]
    @property
    def area(self): return self._bind()["area"]
    @property
    def tau(self): return self._bind()["tau"]
    @property
    def taumax_psi(self): return self._bind()["taumax_psi"]
    @property
    def taumax_psi_upper(self): return self._bind()["taumax_psi_upper"]
    @property
    def taumax_psi_lower(self): return self._bind()["taumax_psi_lower"]
    def leadingEdgeRadius(self): r = self._bind(); return [r["ler_u"], r["ler_l"]]
    def getNormalizedHeight(self, psi=None):
        r = self._bind()
        if psi is None:
            return r["thickness_grid"]
        if self._TOOTH is not None and abs(float(psi) - float(self._TOOTH)) < 1e-12 and r["tooth"] is not None:
            return r["tooth"]
        from metafoil.kulfan_torch import kulfan_surfaces
        s = kulfan_surfaces(self._u, self._l, [float(psi)], te_gap=self._te, with_d2=False)
        return float((s["zeta_upper"] - s["zeta_lower"]).reshape(-1)[0])
    def d2zeta_dpsi2(self, psi, side="upper"):
        r = self._bind(); pin = np.asarray(psi, float)
        if pin.shape == self._PSI[1:-1].shape and np.allclose(pin, self._PSI[1:-1]):
            return r["d2_upper_int"] if side == "upper" else r["d2_lower_int"]
        from metafoil.kulfan_torch import kulfan_surfaces
        s = kulfan_surfaces(self._u, self._l, pin, te_gap=self._te, N1=self._N1, N2=self._N2)
        return (s["d2zeta_upper"] if side == "upper" else s["d2zeta_lower"]).reshape(-1).detach().cpu().numpy()


def _selftest(n=6, seed=0):
    """Validate the batched geometry against per-object metafoil.Kulfan for a random pop."""
    import numpy as np
    from metafoil.core.kulfan import Kulfan
    mag = lambda q: float(q.magnitude) if hasattr(q, "magnitude") else float(q)
    rng = np.random.default_rng(seed)
    U = rng.uniform(0.05, 0.25, (n, 8)); L = rng.uniform(-0.20, -0.01, (n, 8))
    te = 0.0
    g = run_geometry_batch(U, L, tes=te, device="cpu")
    worst = {}
    for i in range(n):
        k = Kulfan(TE_gap=te); k.upperCoefficients = U[i]; k.lowerCoefficients = L[i]
        ref = dict(area=mag(k.area), Ixx=mag(k.Ixx), Iyy=mag(k.Iyy), Izz=mag(k.Izz),
                   ler_u=mag(k.leadingEdgeRadius()[0]))
        got = dict(area=g["area"][i], Ixx=g["Ixx"][i], Iyy=g["Iyy"][i], Izz=g["Izz"][i],
                   ler_u=g["leading_edge_radius_upper"][i])
        for key in ref:
            e = abs(ref[key] - got[key]) / (abs(ref[key]) + 1e-12)
            worst[key] = max(worst.get(key, 0.0), e)
    print(f"[batch_geometry selftest] {n} airfoils, max relative error vs metafoil.Kulfan:")
    for key, e in worst.items():
        print(f"  {key:6s}: {e:.2e}")
    ok = all(e < 1e-6 for e in worst.values())
    print("PASS" if ok else "FAIL (>1e-6)")
    return ok


if __name__ == "__main__":
    _selftest()
