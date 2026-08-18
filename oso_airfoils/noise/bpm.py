"""
Brooks-Pope-Marcolini (BPM) semi-empirical airfoil self-noise model.

Implements the scaling laws of Brooks, T., Pope, D., and Marcolini, M.,
"Airfoil Self-Noise and Prediction," NASA Reference Publication 1218, 1989:

- turbulent-boundary-layer trailing-edge (TBL-TE) noise, pressure side,
  suction side, and the separation (angle-of-attack) contribution,
- laminar-boundary-layer vortex-shedding (LBL-VS) noise,
- trailing-edge bluntness vortex-shedding noise,
- tip vortex formation noise,
- the BPM flat-plate/NACA-0012 boundary-layer thickness correlations.

The acoustic kernels take boundary-layer thicknesses as explicit inputs so
they can be driven either by the BPM correlations (`bl_thickness_bpm`) or by
solver-derived values (see `boundary_layer.py`, which feeds them from
metafoil's nqfoil/qfoil).

All kernels are vectorized over the frequency array and return 1/3-octave
band SPL in dB re 20 uPa at the observer. Angles follow `atmosphere.py`
conventions; alpha_star is the angle of attack in degrees (BPM measures it
from the zero-lift line for cambered sections; in practice the geometric
angle is commonly used, as in NAFNoise).

Numerical behavior is verified against the BPM reference implementation
distributed with NREL NAFNoise (bpm/PREDICT.FOR); see tests/. One knowing
deviation: the reference code leaves the G5 bluntness helper's `mu` and `m`
coefficients undefined exactly at segment boundaries (uninitialized-variable
behavior in Fortran); here the segments are closed intervals.
"""

import numpy as np

from .spectrum import SPL_FLOOR, spl_sum


# ----------------------------------------------------------------------------
# Directivity (RP-1218 eqs. B1, B2)
# ----------------------------------------------------------------------------

def directivity_high(mach, theta_deg, phi_deg):
    """High-frequency directivity D_h (trailing-edge, normalized)."""
    mc = 0.8 * mach
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    return (2.0 * np.sin(th / 2.0) ** 2 * np.sin(ph) ** 2
            / ((1.0 + mach * np.cos(th))
               * (1.0 + (mach - mc) * np.cos(th)) ** 2))


def directivity_low(mach, theta_deg, phi_deg):
    """Low-frequency (compact-dipole) directivity D_l."""
    th = np.radians(theta_deg)
    ph = np.radians(phi_deg)
    return (np.sin(th) * np.sin(ph)) ** 2 / (1.0 + mach * np.cos(th)) ** 4


# ----------------------------------------------------------------------------
# Spectral shape functions A and B (RP-1218 eqs. 35-40)
# ----------------------------------------------------------------------------

def _a_min(a):
    x = np.abs(np.asarray(a, dtype=float))
    return np.where(
        x <= 0.204, np.sqrt(np.maximum(67.552 - 886.788 * x ** 2, 0.0)) - 8.219,
        np.where(x <= 0.244, -32.665 * x + 3.981,
                 -142.795 * x ** 3 + 103.656 * x ** 2 - 57.757 * x + 6.006))


def _a_max(a):
    x = np.abs(np.asarray(a, dtype=float))
    return np.where(
        x <= 0.13, np.sqrt(np.maximum(67.552 - 886.788 * x ** 2, 0.0)) - 8.219,
        np.where(x <= 0.321, -15.901 * x + 1.098,
                 -4.669 * x ** 3 + 3.491 * x ** 2 - 16.699 * x + 1.149))


def _b_min(b):
    x = np.abs(np.asarray(b, dtype=float))
    return np.where(
        x <= 0.13, np.sqrt(np.maximum(16.888 - 886.788 * x ** 2, 0.0)) - 4.109,
        np.where(x <= 0.145, -83.607 * x + 8.138,
                 -817.81 * x ** 3 + 355.21 * x ** 2 - 135.024 * x + 10.619))


def _b_max(b):
    x = np.abs(np.asarray(b, dtype=float))
    return np.where(
        x <= 0.10, np.sqrt(np.maximum(16.888 - 886.788 * x ** 2, 0.0)) - 4.109,
        np.where(x <= 0.187, -31.313 * x + 1.854,
                 -80.541 * x ** 3 + 44.174 * x ** 2 - 39.381 * x + 2.344))


def _a0(rc):
    """Strouhal ratio where the A-curve reaches -20 dB, vs. chord Reynolds."""
    if rc < 9.52e4:
        return 0.57
    if rc < 8.57e5:
        return (-9.57e-13) * (rc - 8.57e5) ** 2 + 1.13
    return 1.13


def _shape_a(a, ar):
    return _a_min(a) + ar * (_a_max(a) - _a_min(a))


def _shape_b(b, br):
    return _b_min(b) + br * (_b_max(b) - _b_min(b))


def _interp_ratio_a(rc):
    a0 = _a0(rc)
    return (20.0 + _a_min(a0)) / (_a_min(a0) - _a_max(a0))


def _interp_ratio_b(rc):
    if rc < 9.52e4:
        b0 = 0.30
    elif rc < 8.57e5:
        b0 = (-4.48e-13) * (rc - 8.57e5) ** 2 + 0.56
    else:
        b0 = 0.56
    return (20.0 + _b_min(b0)) / (_b_min(b0) - _b_max(b0))


# ----------------------------------------------------------------------------
# TBL-TE + separation noise (RP-1218 eqs. 24-34)
# ----------------------------------------------------------------------------

def tbl_te(freq, alpha_star, chord, u, span, r, theta_deg, phi_deg,
           c0, nu, dstar_s, dstar_p):
    """Turbulent-boundary-layer trailing-edge and separation noise.

    Parameters
    ----------
    freq : array of 1/3-octave band centers [Hz]
    alpha_star : angle of attack [deg]
    chord, span, r : geometry [m]
    u : local flow speed [m/s]
    c0, nu : speed of sound [m/s], kinematic viscosity [m^2/s]
    dstar_s, dstar_p : suction/pressure-side displacement thickness at the
        trailing edge [m]

    Returns
    -------
    (spl_p, spl_s, spl_alpha) : per-band SPL [dB], floored at -100.
    """
    freq = np.asarray(freq, dtype=float)
    mach = u / c0
    rc = u * chord / nu
    r_dstar_p = dstar_p * u / nu

    d_low = directivity_low(mach, theta_deg, phi_deg)
    d_high = directivity_high(mach, theta_deg, phi_deg)

    # peak Strouhal numbers (eqs. 31-34)
    st1 = 0.02 * mach ** (-0.6)
    if alpha_star <= 1.333:
        st2 = st1
    elif alpha_star <= 12.5:
        st2 = st1 * 10.0 ** (0.0054 * (alpha_star - 1.333) ** 2)
    else:
        st2 = 4.72 * st1
    st1_prime = 0.5 * (st1 + st2)

    ar = _interp_ratio_a(rc)
    ar2 = _interp_ratio_a(3.0 * rc)     # A' curve: A evaluated at 3*Rc
    br = _interp_ratio_b(rc)

    # amplitude functions (eqs. 47-49)
    if rc < 2.47e5:
        k1 = -4.31 * np.log10(rc) + 156.3
    elif rc < 8.0e5:
        k1 = -9.0 * np.log10(rc) + 181.6
    else:
        k1 = 128.5
    if r_dstar_p <= 5000.0:
        delta_k1 = -alpha_star * (5.29 - 1.43 * np.log10(r_dstar_p))
    else:
        delta_k1 = 0.0

    gamma = 27.094 * mach + 3.31
    beta = 72.650 * mach + 10.74
    gamma0 = 23.430 * mach + 4.651
    beta0 = -34.190 * mach - 13.820
    if alpha_star <= gamma0 - gamma:
        k2 = -1000.0
    elif alpha_star <= gamma0 + gamma:
        k2 = np.sqrt(beta ** 2 - (beta / gamma) ** 2
                     * (alpha_star - gamma0) ** 2) + beta0
    else:
        k2 = -12.0
    k2 = k2 + k1

    st_p = freq * dstar_p / u
    st_s = freq * dstar_s / u

    scale_p_h = 10.0 * np.log10(dstar_p * mach ** 5 * d_high * span / r ** 2)
    scale_s_h = 10.0 * np.log10(dstar_s * mach ** 5 * d_high * span / r ** 2)
    scale_s_l = 10.0 * np.log10(dstar_s * mach ** 5 * d_low * span / r ** 2)

    stalled = (alpha_star >= gamma0) or (alpha_star > 12.5)
    if not stalled:
        spl_p = (_shape_a(np.log10(st_p / st1), ar)
                 + k1 - 3.0 + scale_p_h + delta_k1)
        spl_s = (_shape_a(np.log10(st_s / st1_prime), ar)
                 + k1 - 3.0 + scale_s_h)
        spl_alpha = (_shape_b(np.log10(st_s / st2), br) + k2 + scale_s_h)
    else:
        # fully separated: edge sources cut off, compact-dipole stall noise
        spl_p = np.full_like(freq, scale_s_l)
        spl_s = np.full_like(freq, scale_s_l)
        spl_alpha = (_shape_a(np.log10(st_s / st2), ar2) + k2 + scale_s_l)

    spl_p = np.maximum(spl_p, SPL_FLOOR)
    spl_s = np.maximum(spl_s, SPL_FLOOR)
    spl_alpha = np.maximum(spl_alpha, SPL_FLOOR)
    return spl_p, spl_s, spl_alpha


# ----------------------------------------------------------------------------
# LBL-VS noise (RP-1218 eqs. 53-60)
# ----------------------------------------------------------------------------

def lbl_vs(freq, alpha_star, chord, u, span, r, theta_deg, phi_deg,
           c0, nu, delta_p):
    """Laminar-boundary-layer vortex-shedding noise.

    ``delta_p`` is the pressure-side boundary-layer thickness (delta99) at
    the trailing edge [m]. Only meaningful for untripped boundary layers.
    """
    freq = np.asarray(freq, dtype=float)
    mach = u / c0
    rc = u * chord / nu
    d_high = directivity_high(mach, theta_deg, phi_deg)

    if rc <= 1.3e5:
        st1p = 0.18
    elif rc <= 4.0e5:
        st1p = 0.001756 * rc ** 0.3931
    else:
        st1p = 0.28
    st_peak = st1p * 10.0 ** (-0.04 * alpha_star)

    if alpha_star <= 3.0:
        rc0 = 10.0 ** (0.215 * alpha_star + 4.978)
    else:
        rc0 = 10.0 ** (0.120 * alpha_star + 5.263)

    d = rc / rc0
    ld = np.log10(d)
    if d <= 0.3237:
        g2 = 77.852 * ld + 15.328
    elif d <= 0.5689:
        g2 = 65.188 * ld + 9.125
    elif d <= 1.7579:
        g2 = -114.052 * ld ** 2
    elif d <= 3.0889:
        g2 = -65.188 * ld + 9.125
    else:
        g2 = -77.852 * ld + 15.328

    g3 = 171.04 - 3.03 * alpha_star
    scale = 10.0 * np.log10(delta_p * mach ** 5 * d_high * span / r ** 2)

    e = (freq * delta_p / u) / st_peak
    le = np.log10(e)
    g1 = np.where(
        e < 0.5974, 39.8 * le - 11.12,
        np.where(e <= 0.8545, 98.409 * le + 2.0,
        np.where(e < 1.17,
                 -5.076 + np.sqrt(np.maximum(2.484 - 506.25 * le ** 2, 0.0)),
        np.where(e < 1.674, -98.409 * le + 2.0,
                 -39.8 * le - 11.12))))

    return np.maximum(g1 + g2 + g3 + scale, SPL_FLOOR)


# ----------------------------------------------------------------------------
# Trailing-edge bluntness noise (RP-1218 eqs. 70-77)
# ----------------------------------------------------------------------------

def _g5(h_dstar, eta):
    """Bluntness spectral shape at solid angle psi = 14 deg (eq. 76)."""
    h = h_dstar
    if h <= 0.25:
        mu = 0.1211
    elif h <= 0.62:
        mu = -0.2175 * h + 0.1755
    elif h < 1.15:
        mu = -0.0308 * h + 0.0596
    else:
        mu = 0.0242

    if h <= 0.02:
        m = 0.0
    elif h <= 0.5:
        m = 68.724 * h - 1.35
    elif h <= 0.62:
        m = 308.475 * h - 121.23
    elif h <= 1.15:
        m = 224.811 * h - 69.354
    elif h < 1.2:
        m = 1583.28 * h - 1631.592
    else:
        m = 268.344
    m = max(m, 0.0)

    eta0 = -np.sqrt((m * m * mu ** 4) / (6.25 + m * m * mu * mu))
    k = 2.5 * np.sqrt(1.0 - (eta0 / mu) ** 2) - 2.5 - m * eta0

    eta = np.asarray(eta, dtype=float)
    return np.where(
        eta <= eta0, m * eta + k,
        np.where(eta <= 0.0,
                 2.5 * np.sqrt(np.maximum(1.0 - (eta / mu) ** 2, 0.0)) - 2.5,
        np.where(eta <= 0.03616,
                 np.sqrt(np.maximum(1.5625 - 1194.99 * eta ** 2, 0.0)) - 1.25,
                 -155.543 * eta + 4.375)))


def bluntness(freq, u, span, r, theta_deg, phi_deg, c0,
              h_te, psi_deg, dstar_s, dstar_p):
    """Trailing-edge bluntness vortex-shedding noise.

    ``h_te`` is the trailing-edge thickness [m] and ``psi_deg`` the trailing
    edge solid angle between the surface slopes [deg] (14 for a flat plate).
    """
    freq = np.asarray(freq, dtype=float)
    mach = u / c0
    d_high = directivity_high(mach, theta_deg, phi_deg)

    dstar_avg = 0.5 * (dstar_s + dstar_p)
    h_dstar = h_te / dstar_avg

    if h_dstar >= 0.2:
        st_peak = ((0.212 - 0.0045 * psi_deg)
                   / (1.0 + 0.235 / h_dstar - 0.0132 / h_dstar ** 2))
    else:
        st_peak = 0.1 * h_dstar + 0.095 - 0.00243 * psi_deg

    if h_dstar <= 5.0:
        g4 = 17.5 * np.log10(h_dstar) + 157.5 - 1.114 * psi_deg
    else:
        g4 = 169.7 - 1.114 * psi_deg

    eta = np.log10((freq * h_te / u) / st_peak)
    g5_14 = _g5(h_dstar, eta)
    h_dstar_p0 = 6.724 * h_dstar ** 2 - 4.019 * h_dstar + 1.107
    g5_0 = _g5(h_dstar_p0, eta)
    g5 = g5_0 + 0.0714 * psi_deg * (g5_14 - g5_0)
    g5 = np.minimum(g5, 0.0)
    g5 = np.minimum(g5, _g5(0.25, eta))

    scale = 10.0 * np.log10(mach ** 5.5 * h_te * d_high * span / r ** 2)
    return np.maximum(g4 + g5 + scale, SPL_FLOOR)


# ----------------------------------------------------------------------------
# Tip vortex noise (RP-1218 eqs. 61-64)
# ----------------------------------------------------------------------------

def tip_noise(freq, alpha_tip, chord, u, r, theta_deg, phi_deg, c0,
              round_tip=True, alprat=1.0):
    """Tip vortex formation noise for the outermost blade segment.

    ``alpha_tip`` is the tip angle of attack [deg]; ``alprat`` corrects for
    the tip lift-curve slope relative to 2-D (RP-1218 sec. 3.3).
    """
    freq = np.asarray(freq, dtype=float)
    mach = u / c0
    d_high = directivity_high(mach, theta_deg, phi_deg)

    a_tip = alpha_tip * alprat
    if round_tip:
        span_ext = 0.008 * a_tip * chord
    elif abs(a_tip) <= 2.0:
        span_ext = (0.023 + 0.0169 * a_tip) * chord
    else:
        span_ext = (0.0378 + 0.0095 * a_tip) * chord

    if span_ext <= 0.0:
        return np.full_like(freq, SPL_FLOOR)

    m_max = (1.0 + 0.036 * a_tip) * mach
    u_max = m_max * c0
    scale = 10.0 * np.log10(
        mach ** 2 * m_max ** 3 * span_ext ** 2 * d_high / r ** 2)

    st = freq * span_ext / u_max
    return np.maximum(126.0 - 30.5 * (np.log10(st) + 0.3) ** 2 + scale,
                      SPL_FLOOR)


# ----------------------------------------------------------------------------
# BPM boundary-layer thickness correlations (RP-1218 sec. 5, eqs. 2-15)
# ----------------------------------------------------------------------------

def bl_thickness_bpm(chord, u, alpha_star, trip, c0, nu):
    """NACA-0012 boundary-layer correlations at the trailing edge.

    ``trip`` is one of 'none', 'heavy', 'light' (BPM ITRIP = 0, 1, 2).

    Returns a dict with delta_p (pressure-side BL thickness), dstar_s and
    dstar_p (displacement thicknesses), all in meters.
    """
    rc = u * chord / nu
    lrc = np.log10(rc)

    delta0 = 10.0 ** (1.6569 - 0.9045 * lrc + 0.0596 * lrc ** 2) * chord
    if trip == 'light':
        delta0 *= 0.6
    delta_p = 10.0 ** (-0.04175 * alpha_star
                       + 0.00106 * alpha_star ** 2) * delta0

    if trip in ('heavy', 'light'):
        if rc <= 0.3e6:
            dstar0 = 0.0601 * rc ** (-0.114) * chord
        else:
            dstar0 = 10.0 ** (3.411 - 1.5397 * lrc
                              + 0.1059 * lrc ** 2) * chord
        if trip == 'light':
            dstar0 *= 0.6
    elif trip == 'none':
        dstar0 = 10.0 ** (3.0187 - 1.5397 * lrc + 0.1059 * lrc ** 2) * chord
    else:
        raise ValueError("trip must be 'none', 'heavy', or 'light'")

    dstar_p = 10.0 ** (-0.0432 * alpha_star
                       + 0.00113 * alpha_star ** 2) * dstar0

    if trip == 'heavy':
        if alpha_star <= 5.0:
            dstar_s = 10.0 ** (0.0679 * alpha_star) * dstar0
        elif alpha_star <= 12.5:
            dstar_s = 0.381 * 10.0 ** (0.1516 * alpha_star) * dstar0
        else:
            dstar_s = 14.296 * 10.0 ** (0.0258 * alpha_star) * dstar0
    else:
        if alpha_star <= 7.5:
            dstar_s = 10.0 ** (0.0679 * alpha_star) * dstar0
        elif alpha_star <= 12.5:
            dstar_s = 0.0162 * 10.0 ** (0.3066 * alpha_star) * dstar0
        else:
            dstar_s = 52.42 * 10.0 ** (0.0258 * alpha_star) * dstar0

    return {'delta_p': delta_p, 'dstar_s': dstar_s, 'dstar_p': dstar_p}


def tbl_te_total(spl_p, spl_s, spl_alpha):
    """Energetic sum of the three TBL-TE components."""
    return spl_sum(spl_p, spl_s, spl_alpha)
