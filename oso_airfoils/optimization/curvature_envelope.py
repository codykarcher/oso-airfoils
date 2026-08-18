"""Frozen d2kappa/ds2 ENVELOPE + a fixed-grid, DIFFERENTIABLE |d2kappa/ds2|
computation shared by BOTH optimizers (GA objective_function + gradient
gradient_objective) so their curvature-acceleration constraint matches exactly.

The envelope E(x/c) per thickness/surface is frozen in curvature_envelope.json
(built from curvature_plots.E_smooth on the GA V13 fronts). The constraint is
    |d2kappa/ds2(x_i)| <= E_frozen(x_i)   for every station x_i >= XSTART
enforced on both surfaces.

d2kappa/ds2 is EXACT-ANALYTIC at each fixed station x_i (=psi_i on the CST, N1=0.5,
N2=1). We tried finite-differencing kappa on the cosine XG / a uniform-psi grid as
the spec suggested, but the FD is ill-conditioned at the LE/TE (XG spacing ~1e-4
there), which spuriously pushed GA airfoils outside the tight-margin envelope. The
closed form is grid-independent, exact, and still Dual-differentiable (everything is
a Dual arithmetic combo of zeta',zeta'',zeta''',zeta'''', each LINEAR in the coeffs):

    kappa   = z2 * P^-1.5 ,                         P = 1 + z1^2
    kappa'  = z3 P^-1.5 - 3 z1 z2^2 P^-2.5
    kappa'' = z4 P^-1.5 - (9 z1 z2 z3 + 3 z2^3) P^-2.5 + 15 z1^2 z2^3 P^-3.5
    d2k/ds2 = kappa''/P - kappa' * z1 z2 / P^2
with z1..z4 = zeta'..zeta'''' = Jk @ A (+te_off for z1); Jk constant matrices on the
fixed grid (same class/shape closed form as gradient_objective.geometry_duals,
extended to 4th order). It tracks curvature_plots' resampled values to within the
envelope's own margin (GA population stays inside; blocky gradient foils poke out).
"""
import os, json
import numpy as np
from math import comb
from numpy.polynomial import polynomial as _Pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOC = json.load(open(os.path.join(_HERE, "curvature_envelope.json")))

XSTART = float(_DOC["XSTART"])
XG = np.asarray(_DOC["XG"], float)                    # full 140-pt envelope grid
M_COEFFS = int(_DOC.get("M_coeffs", 8))
# stations the curvature is evaluated at; clamp the exact-0/1 nodes off the CST singularity
PSI_EVAL = np.clip(XG, 1e-4, 1.0 - 1e-6)
STATION_IDX = np.where(XG >= XSTART)[0]               # stations the constraint acts on
E_TAB = {T: {s: np.asarray(_DOC["E"][T][s], float) for s in ("upper", "lower")}
         for T in _DOC["E"]}


def tau_to_T(tau):
    return f"{int(round(float(tau) * 100)):02d}"


def _zeta_deriv_basis(psi, M=M_COEFFS):
    """J1..J4 on `psi` s.t. zeta^(k)(psi) = Jk @ A (+te_off for k=1), N1=0.5,N2=1.
    zeta = C*S + psi*te, C=psi^0.5(1-psi); Jk = sum_i C(k,i) C^(i) B^(k-i)."""
    psi = np.asarray(psi, float); n = M - 1; npsi = len(psi)
    # Bernstein basis and derivatives 0..4 (exact polynomials -> no LE/TE singularity)
    Bd = [np.zeros((npsi, M)) for _ in range(5)]
    for k in range(M):
        poly = comb(n, k) * _Pl.polymul([0.0] * k + [1.0], _Pl.polypow([1.0, -1.0], n - k))
        for j in range(5):
            Bd[j][:, k] = _Pl.polyval(psi, _Pl.polyder(poly, j) if j else poly)
    # class function C = psi^0.5 - psi^1.5 and derivatives 0..4
    with np.errstate(divide="ignore", invalid="ignore"):
        C = [psi ** 0.5 - psi ** 1.5,
             0.5 * psi ** -0.5 - 1.5 * psi ** 0.5,
             -0.25 * psi ** -1.5 - 0.75 * psi ** -0.5,
             0.375 * psi ** -2.5 + 0.375 * psi ** -1.5,
             -0.9375 * psi ** -3.5 - 0.5625 * psi ** -2.5]
    J = []
    for kk in range(1, 5):
        Jk = np.zeros((npsi, M))
        for i in range(kk + 1):
            Jk += comb(kk, i) * C[i][:, None] * Bd[kk - i]
        J.append(Jk)
    return J   # [J1, J2, J3, J4]


J1G, J2G, J3G, J4G = _zeta_deriv_basis(PSI_EVAL, M_COEFFS)   # constant matrices on the frozen grid


def d2k_ds2_float(coeffs, te_off):
    """Exact d2kappa/ds2 on the full frozen grid (numpy float), for the GA."""
    A = np.asarray(coeffs, float)
    z1 = J1G @ A + te_off; z2 = J2G @ A; z3 = J3G @ A; z4 = J4G @ A
    P = 1.0 + z1 * z1
    kp = z3 * P ** -1.5 - 3.0 * z1 * z2 * z2 * P ** -2.5
    kpp = z4 * P ** -1.5 - (9.0 * z1 * z2 * z3 + 3.0 * z2 ** 3) * P ** -2.5 + 15.0 * z1 * z1 * z2 ** 3 * P ** -3.5
    return kpp / P - kp * z1 * z2 / (P * P)


def curvature_accel_violation(up, lo, te_gap, tau, return_detail=False):
    """max over stations (x>=XSTART) and both surfaces of (|d2kappa/ds2| - E_frozen)_+.
    0 => inside the envelope (feasible). Uses each surface's TE offset +/- te_gap/2."""
    T = tau_to_T(tau)
    Etab = E_TAB.get(T)
    if Etab is None:
        return (0.0, {}) if return_detail else 0.0
    d2u = d2k_ds2_float(up, +te_gap / 2.0)
    d2l = d2k_ds2_float(lo, -te_gap / 2.0)
    s = STATION_IDX
    vu = np.maximum(0.0, np.abs(d2u[s]) - Etab["upper"][s])
    vl = np.maximum(0.0, np.abs(d2l[s]) - Etab["lower"][s])
    viol = float(max(vu.max() if vu.size else 0.0, vl.max() if vl.size else 0.0))
    if return_detail:
        return viol, dict(d2u=d2u, d2l=d2l, T=T)
    return viol


# ======================================================================
# Lower-surface secondary-bulge guard  (2026-08-01)
# ----------------------------------------------------------------------
# A "bulge" is a rising secondary curvature peak on the lower surface a few
# percent aft of the LE shoulder (visible as a pooch at ~10% chord). It is NOT
# a d2kappa/ds2 spike (the hump is smooth, ~30-100x under the accel envelope) and
# NOT an inflection (kappa stays one-signed), so neither existing guard catches it.
# Detect it RELATIVELY: kappa in the mid-band may not exceed the 2%-chord shoulder
# reference by more than a small margin -- i.e. curvature may keep falling or stay
# flat aft of the shoulder, but may not climb back into a second peak. Relative =>
# thickness-agnostic (the reference scales with the section) and SLACK (delta < 0,
# never binding) unless kappa is actually humping. On the WT2 fronts it fires only
# on the t21 rough-end bulges; t24..t36 sit well clear. kappa = |zeta''| / (1+zeta'^2)^1.5.
BULGE_REF_X = 0.02                       # shoulder reference station (chord fraction)
BULGE_BAND = (0.065, 0.115)              # mid-chord hump search band
BULGE_N_BAND = 11                        # stations across the band
BULGE_MARGIN = 0.5                       # allowed kappa rise above the shoulder
_BULGE_PSI = np.concatenate(([BULGE_REF_X],
                             np.linspace(BULGE_BAND[0], BULGE_BAND[1], BULGE_N_BAND)))
_BJ1, _BJ2 = _zeta_deriv_basis(_BULGE_PSI, M_COEFFS)[:2]   # constant J1/J2 on the bulge stations


def kappa_at(coeffs, te_off, J1=_BJ1, J2=_BJ2):
    """|kappa| = |zeta''| / (1+zeta'^2)^1.5 at the fixed bulge stations (numpy float)."""
    A = np.asarray(coeffs, float)
    z1 = J1 @ A + te_off
    z2 = J2 @ A
    return np.abs(z2) / (1.0 + z1 * z1) ** 1.5


def bulge_violation(lower, te_gap=0.0, margin=BULGE_MARGIN, return_detail=False):
    """max(0, max_band |kappa_lower| - |kappa_lower(2%)| - margin). 0 => no rising
    interior curvature hump (feasible). Lower surface only; te offset -te_gap/2."""
    kap = kappa_at(lower, -te_gap / 2.0)
    delta = float(kap[1:].max() - kap[0] - margin)
    viol = max(0.0, delta)
    if return_detail:
        return viol, dict(kref=float(kap[0]), kband_max=float(kap[1:].max()), delta=delta)
    return viol
