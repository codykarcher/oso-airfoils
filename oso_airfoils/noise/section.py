"""
Section-level airfoil noise prediction: Kulfan coefficients in, 1/3-octave
noise spectra out.

This is the main entry point of the package:

    from oso_airfoils.noise import predict_section_noise, Observer

    res = predict_section_noise(
        upper_weights, lower_weights, te_gap=0.002,
        alpha=3.0, u=63.92, chord=0.2286, span=0.509,
        observer=Observer(r=1.22),
        inflow_turbulence={"intensity": 0.05, "length_scale": 0.06})

    res.freq, res.spl["tbl_te_suction"], res.total, res.oaspl_a

Noise sources (each an SPL spectrum in ``res.spl``):
    tbl_te_pressure, tbl_te_suction : turbulent-BL trailing-edge noise
    separation                      : angle-of-attack / stall contribution
    lbl_vs                          : laminar-BL vortex shedding (untripped)
    bluntness                       : TE bluntness vortex shedding
    inflow                          : turbulent inflow noise (if requested)

Boundary-layer inputs come from metafoil by default (``bl_source='nqfoil'``,
~30 ms per case) with 'qfoil' (real viscous solve) and 'bpm' (NACA-0012
correlations) as alternatives. TBL-TE noise can use the BPM scaling laws
(default) or the TNO-Blake model (``tbl_te_method='tno'``).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from . import bpm, inflow as inflow_mod, tno
from .atmosphere import Atmosphere, Observer
from .boundary_layer import BLInputs, SurfaceBL, bl_from_nqfoil, bl_from_qfoil
from .spectrum import (SPL_FLOOR, a_weighting, oaspl, spl_sum,
                       third_octave_bands)


@dataclass
class SectionNoiseResult:
    """Per-source 1/3-octave spectra for one airfoil section / observer."""
    freq: np.ndarray
    spl: dict
    bl: BLInputs = None
    meta: dict = field(default_factory=dict)

    @property
    def total(self):
        return spl_sum(*self.spl.values())

    @property
    def oaspl(self):
        return oaspl(self.total)

    @property
    def oaspl_a(self):
        return oaspl(self.total, a_weighted=True, freq=self.freq)

    def as_dict(self):
        out = {"freq": self.freq, **self.spl}
        out["total"] = self.total
        return out


def _thickness_params(upper_weights, lower_weights, te_gap):
    """t/c at 1% and 10% chord for the simplified Guidati correction."""
    from metafoil.core.kulfan import Kulfan
    k = Kulfan(upper_coefficients=np.asarray(upper_weights, float).ravel(),
               lower_coefficients=np.asarray(lower_weights, float).ravel(),
               te_gap=float(te_gap))
    t = k.thickness(np.array([0.01, 0.10]))
    return float(t[0]), float(t[1])


def predict_section_noise(upper_weights, lower_weights, te_gap=0.0, *,
                          alpha, u, chord, span,
                          observer=None, atmosphere=None,
                          bl_source="nqfoil", tbl_te_method="bpm",
                          include_lbl=None, include_bluntness=None,
                          h_te=None, te_psi_deg=14.0,
                          trip="none", xtr_upper=1.0, xtr_lower=1.0,
                          n_crit=9.0, inflow_turbulence=None,
                          freq=None, model_size="xxlarge",
                          bl_inputs=None):
    """Predict airfoil section noise from Kulfan coefficients.

    Parameters
    ----------
    upper_weights, lower_weights : metafoil-convention Kulfan weights.
    te_gap : Kulfan TE thickness normalized by chord (drives both the
        geometry and, by default, the bluntness noise source).
    alpha : angle of attack [deg]. u : flow speed [m/s]. chord, span : [m].
    observer : `Observer` (default r = 1 m, theta = phi = 90 deg).
    atmosphere : `Atmosphere` (default ISA sea level).
    bl_source : 'nqfoil' (default), 'qfoil', or 'bpm'.
    tbl_te_method : 'bpm' (default) or 'tno' ('tno' needs a solver BL
        source; falls back to BPM with a warning if cf <= 0 at the TE).
    include_lbl : laminar vortex-shedding noise. Default: only when the
        boundary layer is untripped (trip == 'none' and both xtr >= 1).
    include_bluntness : default True when the TE has finite thickness.
    h_te : bluntness thickness [m]; default te_gap * chord.
    te_psi_deg : TE solid angle [deg] for the bluntness model.
    trip : 'none' | 'heavy' | 'light' — used by the BPM BL correlations
        (bl_source='bpm'); solver sources express tripping via xtr_*.
    inflow_turbulence : None, or dict with keys ``intensity`` (u'/U as a
        fraction), ``length_scale`` [m], and optionally ``method``
        ('guidati_simplified' default, or 'amiet') and ``nlr_offset``.
    freq : band centers [Hz]; default the 34 standard bands 10 Hz - 20 kHz.
    bl_inputs : optionally pass a precomputed `BLInputs` to skip the solver.

    Returns
    -------
    SectionNoiseResult
    """
    observer = observer or Observer(r=1.0)
    atm = atmosphere or Atmosphere()
    freq = np.asarray(freq if freq is not None else third_octave_bands(),
                      dtype=float)
    re = u * chord / atm.nu
    alpha_star = abs(float(alpha))
    theta_deg, phi_deg, r = observer.theta_deg, observer.phi_deg, observer.r

    untripped = trip == "none" and xtr_upper >= 1.0 and xtr_lower >= 1.0
    if include_lbl is None:
        include_lbl = untripped
    if h_te is None:
        h_te = float(te_gap) * chord
    if include_bluntness is None:
        include_bluntness = h_te > 0.0

    # ---- boundary-layer inputs at the trailing edge -----------------------
    if bl_inputs is not None:
        bl = bl_inputs
    elif bl_source == "nqfoil":
        bl = bl_from_nqfoil(upper_weights, lower_weights, te_gap,
                            alpha, re, chord, u, n_crit=n_crit,
                            xtr_upper=xtr_upper, xtr_lower=xtr_lower,
                            model_size=model_size)
    elif bl_source == "qfoil":
        bl = bl_from_qfoil(upper_weights, lower_weights, te_gap,
                           alpha, re, chord, u, n_crit=n_crit,
                           xtr_upper=xtr_upper, xtr_lower=xtr_lower)
    elif bl_source == "bpm":
        th = bpm.bl_thickness_bpm(chord, u, alpha_star, trip, atm.c0, atm.nu)
        nan = float("nan")
        bl = BLInputs(
            suction=SurfaceBL(th["dstar_s"], nan, nan, nan, u),
            pressure=SurfaceBL(th["dstar_p"], nan, th["delta_p"], nan, u),
            source="bpm")
    else:
        raise ValueError(f"unknown bl_source '{bl_source}'")

    spl = {}

    # ---- TBL-TE + separation ---------------------------------------------
    spl_p, spl_s, spl_a = bpm.tbl_te(
        freq, alpha_star, chord, u, span, r, theta_deg, phi_deg,
        atm.c0, atm.nu, bl.suction.delta_star, bl.pressure.delta_star)
    if tbl_te_method == "tno":
        try:
            if bl.source == "bpm" or not bl.has_cf():
                raise ValueError("TNO needs solver-derived cf/delta99 "
                                 "(bl_source 'nqfoil' or 'qfoil')")
            spl_p, spl_s = tno.tno_te(freq, span, r, theta_deg, phi_deg,
                                      bl, u, atm.c0, atm.rho, atm.nu)
        except ValueError as err:
            warnings.warn(f"TNO unavailable ({err}); using BPM TBL-TE")
    elif tbl_te_method != "bpm":
        raise ValueError(f"unknown tbl_te_method '{tbl_te_method}'")
    spl["tbl_te_pressure"] = spl_p
    spl["tbl_te_suction"] = spl_s
    spl["separation"] = spl_a

    # ---- LBL vortex shedding ---------------------------------------------
    if include_lbl:
        if np.isfinite(bl.pressure.delta99):
            delta_p = bl.pressure.delta99
        else:
            delta_p = bpm.bl_thickness_bpm(chord, u, alpha_star, "none",
                                           atm.c0, atm.nu)["delta_p"]
        spl["lbl_vs"] = bpm.lbl_vs(freq, alpha_star, chord, u, span, r,
                                   theta_deg, phi_deg, atm.c0, atm.nu,
                                   delta_p)

    # ---- TE bluntness -----------------------------------------------------
    if include_bluntness:
        if h_te <= 0.0:
            raise ValueError("bluntness noise requires h_te > 0 "
                             "(or a nonzero te_gap)")
        spl["bluntness"] = bpm.bluntness(
            freq, u, span, r, theta_deg, phi_deg, atm.c0, h_te, te_psi_deg,
            bl.suction.delta_star, bl.pressure.delta_star)

    # ---- turbulent inflow -------------------------------------------------
    if inflow_turbulence is not None:
        ti = dict(inflow_turbulence)
        method = ti.pop("method", "guidati_simplified")
        nlr = ti.pop("nlr_offset", True)
        t1 = ti.pop("thickness_1pc", None)
        t10 = ti.pop("thickness_10pc", None)
        if method == "guidati_simplified" and (t1 is None or t10 is None):
            t1, t10 = _thickness_params(upper_weights, lower_weights, te_gap)
        spl["inflow"] = inflow_mod.turbulent_inflow(
            freq, alpha, u, chord, span, r, theta_deg, phi_deg,
            atm.c0, atm.rho, ti.pop("intensity"), ti.pop("length_scale"),
            thickness_1pc=t1, thickness_10pc=t10,
            method=method, nlr_offset=nlr)
        if ti:
            raise TypeError(f"unknown inflow_turbulence keys: {list(ti)}")

    meta = {"alpha": alpha, "u": u, "chord": chord, "span": span, "re": re,
            "mach": u / atm.c0, "bl_source": bl.source,
            "tbl_te_method": tbl_te_method, "h_te": h_te,
            "observer": observer, "atmosphere": atm}
    return SectionNoiseResult(freq=freq, spl=spl, bl=bl, meta=meta)
