"""
Trailing-edge boundary-layer inputs for the acoustic models.

Every noise model in this package is driven by a small set of boundary-layer
quantities evaluated at the trailing edge of each surface:

- ``delta_star`` : displacement thickness            [m]
- ``theta``      : momentum thickness                [m]
- ``delta99``    : boundary-layer thickness          [m]
- ``cf``         : skin-friction coefficient at the TE, normalized by the
                   FREESTREAM dynamic pressure (tau / (0.5 rho Uinf^2)),
                   matching NAFNoise's Cf = TAU(TE)/(0.5 QINF^2)
- ``ue``         : boundary-layer edge velocity at the TE  [m/s]

Three sources are provided:

``bl_from_nqfoil``  (preferred) — metafoil's nqfoil full-BL neural surrogate,
    which returns theta, H, and ue/Uinf at 32 stations per surface for any
    metafoil-convention Kulfan airfoil. TE values are extrapolated from the
    last stations; delta99 uses Drela's correlation
    delta99 = theta*(3.15 + 1.72/(Hk-1)) + delta*  (the same one NAFNoise
    applies to XFOIL output), and cf comes from the XFOIL/Coles turbulent
    skin-friction closure cf(Hk, Re_theta) scaled by (ue/Uinf)^2.

``bl_from_qfoil`` — metafoil's qfoil (RFOIL-modified XFOIL) solver with
    save_boundary_layer_data=True; slower but a real viscous solve, with Cf
    taken directly from the solver.

``bl_thickness_bpm`` (in bpm.py) — the BPM NACA-0012 correlations; only
    thicknesses, so TNO cannot run from it.

Conventions: for alpha >= 0 the suction side is the upper surface. Kulfan
weights are metafoil-convention (fit_order-8, N1=0.5/N2=1.0, no
leading-edge-modulation term); ``te_gap`` is the Kulfan TE thickness
normalized by chord.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SurfaceBL:
    """Boundary-layer state at the trailing edge of one surface (SI units)."""
    delta_star: float
    theta: float
    delta99: float
    cf: float           # tau_TE / (0.5 rho Uinf^2); np.nan if unavailable
    ue: float           # edge velocity [m/s]


@dataclass
class BLInputs:
    """Trailing-edge BL state for both surfaces plus provenance."""
    suction: SurfaceBL
    pressure: SurfaceBL
    source: str
    analysis_confidence: float = 1.0

    def has_cf(self):
        return np.isfinite(self.suction.cf) and np.isfinite(self.pressure.cf)


def cf_coles(hk, re_theta):
    """XFOIL's incompressible turbulent skin-friction closure (Coles fit).

    Returns cf based on the LOCAL edge dynamic pressure. From XFOIL's CFT
    (xblsys.f), with the compressibility terms dropped (M -> 0).
    """
    grt = max(np.log(max(re_theta, 1.0)), 3.0)
    cf0 = 0.3 * np.exp(max(-1.33 * hk, -20.0)) * (grt / 2.3026) ** (-1.74 - 0.31 * hk)
    return cf0 + 1.1e-4 * (np.tanh(4.0 - hk / 0.875) - 1.0)


def delta99_drela(theta, hk, delta_star):
    """BL thickness from integral quantities (Drela's XFOIL correlation)."""
    return theta * (3.15 + 1.72 / (hk - 1.0)) + delta_star


def _te_surface_from_stations(x, theta_c, hk, ue_ratio, chord, u, re):
    """Extrapolate station data to x/c = 1 and assemble a SurfaceBL.

    ``theta_c`` is theta/c, ``ue_ratio`` is ue/Uinf, at stations ``x`` (x/c).
    Linear extrapolation from the last two stations, which for nqfoil sit at
    x/c = 0.953 and 0.984.
    """
    def extrap(v):
        return v[-1] + (v[-1] - v[-2]) / (x[-1] - x[-2]) * (1.0 - x[-1])

    th_c = max(float(extrap(theta_c)), 1e-8)
    h = float(extrap(hk))
    h = max(h, 1.05)                       # keep the closures finite
    ue_r = float(extrap(ue_ratio))

    dstar_c = h * th_c
    d99_c = delta99_drela(th_c, h, dstar_c)
    re_theta = re * abs(ue_r) * th_c
    cf_local = cf_coles(h, re_theta)
    cf_freestream = cf_local * ue_r ** 2

    return SurfaceBL(delta_star=dstar_c * chord,
                     theta=th_c * chord,
                     delta99=d99_c * chord,
                     cf=cf_freestream,
                     ue=ue_r * u)


def bl_from_nqfoil(upper_weights, lower_weights, te_gap, alpha, re, chord, u,
                   n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0,
                   model_size="xxlarge"):
    """TE boundary-layer state from metafoil's nqfoil full-BL surrogate.

    ``alpha`` in degrees, ``re`` the chord Reynolds number, ``chord`` [m],
    ``u`` [m/s]. Weights are metafoil-convention Kulfan; ``te_gap`` is the
    TE thickness normalized by chord.
    """
    from metafoil.nqfoil import full_bl

    out = full_bl.get_aero_from_kulfan_parameters(
        {"upper_weights": np.asarray(upper_weights, float).ravel(),
         "lower_weights": np.asarray(lower_weights, float).ravel(),
         "TE_thickness": float(te_gap)},
        alpha=float(alpha), Re=float(re), n_crit=n_crit,
        xtr_upper=xtr_upper, xtr_lower=xtr_lower, model_size=model_size)

    x = np.asarray(out["bl_x"], float)
    upper = _te_surface_from_stations(
        x, out["upper_theta"], out["upper_H"], out["upper_ue"], chord, u, re)
    lower = _te_surface_from_stations(
        x, out["lower_theta"], out["lower_H"], out["lower_ue"], chord, u, re)

    suction, pressure = (upper, lower) if alpha >= 0.0 else (lower, upper)
    return BLInputs(suction=suction, pressure=pressure, source="nqfoil",
                    analysis_confidence=float(out["analysis_confidence"]))


def _te_surface_from_qfoil(side, chord, u, re):
    """SurfaceBL from a qfoil per-surface BL dict (last on-airfoil station)."""
    x = np.asarray(side["x"], float)
    on_foil = np.nonzero(x <= 1.0 + 1e-6)[0]
    i = int(on_foil[np.argmax(x[on_foil])])

    dstar_c = float(side["Dstar"][i])
    th_c = max(float(side["Theta"][i]), 1e-8)
    ue_r = float(side["Ue/Vinf"][i])
    cf = float(side["Cf"][i])              # already tau/(0.5 qinf^2)
    h = float(side["H"][i])
    d99_c = delta99_drela(th_c, h, dstar_c)

    return SurfaceBL(delta_star=dstar_c * chord, theta=th_c * chord,
                     delta99=d99_c * chord, cf=cf, ue=ue_r * u)


def bl_from_qfoil(upper_weights, lower_weights, te_gap, alpha, re, chord, u,
                  n_crit=9.0, xtr_upper=1.0, xtr_lower=1.0, **qfoil_kwargs):
    """TE boundary-layer state from a qfoil viscous solve (slower, exact)."""
    import metafoil.qfoil as qfoil

    result = qfoil.run_from_kulfan(
        'alpha', np.asarray(upper_weights, float).ravel(),
        np.asarray(lower_weights, float).ravel(),
        val=float(alpha), Re=float(re), N_crit=n_crit,
        xtp_u=xtr_upper, xtp_l=xtr_lower, TE_gap=float(te_gap),
        save_boundary_layer_data=True, **qfoil_kwargs)

    if isinstance(result, list):
        result = result[0]
    if not result.get('converged', True):
        raise RuntimeError(
            f"qfoil did not converge at alpha={alpha}, Re={re:g}")
    bl = result.get('bl_data')
    if bl is None:
        raise RuntimeError("qfoil result contains no boundary-layer data; "
                           "expected save_boundary_layer_data=True output")

    upper = _te_surface_from_qfoil(bl['upper'], chord, u, re)
    lower = _te_surface_from_qfoil(bl['lower'], chord, u, re)
    suction, pressure = (upper, lower) if alpha >= 0.0 else (lower, upper)
    return BLInputs(suction=suction, pressure=pressure, source="qfoil")
