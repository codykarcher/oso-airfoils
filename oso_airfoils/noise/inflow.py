"""
Turbulent inflow noise: Amiet flat-plate model with the low-frequency
correction, plus the simplified Guidati airfoil-shape correction.

References
----------
- Amiet, R., "Acoustic Radiation from an Airfoil in a Turbulent Stream,"
  J. Sound and Vibration 41(4), 1975.
- Moriarty, P., Guidati, G., Migliore, P., "Prediction of Turbulent Inflow
  and Trailing-Edge Noise for Wind Turbines," AIAA 2005-2881 (the simplified
  Guidati thickness correction and the +10 dB NLR calibration offset).
- Implementation matches NAFNoise's InflowNoise / Simple_TI subroutines.

The turbulence intensity here is a FRACTION of the mean velocity (0.05 means
5 %). Note that NAFNoise's example input file lists "0.050 (%)", which its
reader divides by 100 — i.e. that historic case actually runs at 0.05 %
intensity.
"""

import numpy as np

from .bpm import directivity_high, directivity_low
from .spectrum import SPL_FLOOR


def amiet_inflow(freq, alpha, u, chord, span, r, theta_deg, phi_deg,
                 c0, rho, turbulence_intensity, length_scale):
    """Amiet flat-plate turbulent-inflow noise with LFC blending.

    ``turbulence_intensity`` is u'_rms / U (fraction); ``length_scale`` the
    streamwise integral scale [m]. Returns per-band SPL [dB].
    """
    freq = np.asarray(freq, dtype=float)
    mach = u / c0
    beta2 = 1.0 - mach ** 2
    ti = float(turbulence_intensity)
    if ti <= 0.0:
        return np.full_like(freq, SPL_FLOOR)

    d_low = directivity_low(mach, theta_deg, phi_deg)
    d_high = directivity_high(mach, theta_deg, phi_deg)
    f_cutoff = 10.0 * u / (np.pi * chord)
    directivity = np.where(freq <= f_cutoff, d_low, d_high)

    ke = 3.0 / (4.0 * length_scale)
    k = 2.0 * np.pi * freq / u
    k_bar = k * chord / 2.0
    k_hat = k / ke

    spl_high = 10.0 * np.log10(
        rho ** 2 * c0 ** 4 * length_scale * (span / 2.0) / r ** 2
        * mach ** 5 * ti ** 2 * k_hat ** 3
        * (1.0 + k_hat ** 2) ** (-7.0 / 3.0) * directivity) + 78.4
    spl_high = spl_high + 10.0 * np.log10(
        1.0 + 9.0 * np.radians(alpha) ** 2)

    sears = 1.0 / (2.0 * np.pi * k_bar / beta2
                   + 1.0 / (1.0 + 2.4 * k_bar / beta2))
    lfc = 10.0 * sears * mach * k_bar ** 2 / beta2

    return np.maximum(spl_high + 10.0 * np.log10(lfc / (1.0 + lfc)),
                      SPL_FLOOR)


def guidati_thickness_correction(freq, u, chord, thickness_1pc,
                                 thickness_10pc):
    """Simplified Guidati airfoil-shape correction, dSPL [dB] (<= 0).

    ``thickness_1pc`` / ``thickness_10pc`` are the airfoil thicknesses at 1 %
    and 10 % chord, normalized by chord. Accurate for Strouhal numbers
    (f c / U) below roughly 75 (AIAA 2005-2881).
    """
    freq = np.asarray(freq, dtype=float)
    ti_param = thickness_1pc + thickness_10pc
    slope = 1.123 * ti_param + 5.317 * ti_param ** 2
    return -slope * (2.0 * np.pi * freq * chord / u + 5.0)


def turbulent_inflow(freq, alpha, u, chord, span, r, theta_deg, phi_deg,
                     c0, rho, turbulence_intensity, length_scale,
                     thickness_1pc=None, thickness_10pc=None,
                     method="guidati_simplified", nlr_offset=True):
    """Turbulent inflow noise at the observer.

    method : 'amiet' (flat plate) or 'guidati_simplified' (Amiet plus the
        thickness correction; requires thickness_1pc/thickness_10pc).
    nlr_offset : apply the +10 dB calibration NAFNoise adds to Guidati-
        corrected predictions to match NLR wind-tunnel data.
    """
    spl = amiet_inflow(freq, alpha, u, chord, span, r, theta_deg, phi_deg,
                       c0, rho, turbulence_intensity, length_scale)
    if method == "amiet":
        return spl
    if method == "guidati_simplified":
        if thickness_1pc is None or thickness_10pc is None:
            raise ValueError("guidati_simplified needs thickness_1pc and "
                             "thickness_10pc (t/c at 1% and 10% chord)")
        spl = spl + guidati_thickness_correction(freq, u, chord,
                                                 thickness_1pc,
                                                 thickness_10pc)
        if nlr_offset:
            spl = spl + 10.0
        return np.maximum(spl, SPL_FLOOR)
    raise ValueError(f"unknown inflow method '{method}'")
