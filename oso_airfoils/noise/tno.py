"""
TNO-Blake trailing-edge noise model.

A more physics-based alternative to the BPM TBL-TE scaling laws: the
turbulent surface-pressure wavenumber-frequency spectrum is built from a
reconstructed mean-velocity and turbulence profile of the trailing-edge
boundary layer, then propagated to the far field with trailing-edge
diffraction. Follows Parchen (TNO report DDA 1998) as implemented by
Moriarty & Migliore (NREL/TP-500-34478) and NAFNoise's TNO module:

- mean profile: law of the wall plus a cosine wake blend,
  U(y) = u* (ln(u* y / nu)/kappa + C + (Ue-defect) * (1 - cos(pi y/delta))/2)
- mixing length l(y) = 0.085 delta tanh(kappa y / (0.085 delta)) / kappa
- vertical-velocity spectrum: von Karman form; moving-axis spectrum:
  Gaussian with convection speed Uc(y) = 0.7 U(y)
- far field: S(omega) ~ span/(4 pi r^2) * integral over k1 of
  omega/(c0 k1) * P(k1), with P the surface-pressure spectrum integral
  over the boundary layer.

Inputs per surface: delta99, cf (freestream-normalized, as in
boundary_layer.SurfaceBL) — the model needs a positive cf (attached flow at
the TE); it raises otherwise, and callers usually fall back to BPM.

Numerical note: NAFNoise evaluates both nested integrals with a single
61-point Gauss-Kronrod panel, which under-resolves the narrow moving-axis
Gaussian at high frequency; here the k1 integral uses a peak-refined
composite grid and the wall-normal integral 100-point Gauss-Legendre, so
results are converged rather than bit-matched to NAFNoise.
"""

import numpy as np

from .bpm import directivity_high
from .spectrum import SPL_FLOOR, band_width

KAPPA = 0.41
C_NUK = 5.5          # log-law additive constant used by NAFNoise
C_MU = 0.09
ALPHA_SUCTION = 0.45  # vertical turbulence anisotropy factors (Parchen)
ALPHA_PRESSURE = 0.30
GAMMA_RATIO = 0.4213560764  # gamma(5/6)/gamma(1/3)

_N_Y = 100          # wall-normal Gauss-Legendre points
_N_K_BASE = 400     # k1 base grid points
_N_K_PEAK = 320     # k1 peak-refinement points


def _profile(y, u_inf, delta, cf, nu):
    """Mean-flow and turbulence quantities at heights ``y`` (< delta)."""
    u_star = u_inf * np.sqrt(cf / 2.0)
    l_mix = 0.085 * delta / KAPPA * np.tanh(KAPPA * y / (0.085 * delta))
    wake = 1.0 - np.cos(np.pi * y / delta)
    defect = u_inf / u_star - np.log(u_star * delta / nu) / KAPPA - C_NUK
    u = u_star * (np.log(u_star * y / nu) / KAPPA + C_NUK
                  + defect * 0.5 * wake)
    dudy = u_star * (1.0 / (KAPPA * y)
                     + defect * 0.5 * (np.pi / delta) * np.sin(np.pi * y / delta))
    nu_t = (l_mix * KAPPA) ** 2 * np.abs(dudy)
    k_t = np.sqrt((nu_t * dudy) ** 2 / C_MU)
    ke = np.sqrt(np.pi) / l_mix * GAMMA_RATIO
    return u, dudy, l_mix, k_t, ke


def _surface_pressure_integral(omega, k1, y, wy, u_inf, delta, cf, nu, alpha_v):
    """P(k1) = 4 rho^2 k1^2 * integral_0^delta of the TNO source term
    (without the rho^2 factor, which the caller applies)."""
    u, dudy, l_mix, k_t, ke = _profile(y, u_inf, delta, cf, nu)
    ums = alpha_v * k_t
    uc = 0.7 * u
    alpha_gauss = 0.05 * uc / l_mix

    k1c, yc = np.meshgrid(k1, y, indexing='ij')          # (nk, ny)
    phi_m = (1.0 / (alpha_gauss[None, :] * np.sqrt(np.pi))
             * np.exp(-((omega - uc[None, :] * k1c)
                        / alpha_gauss[None, :]) ** 2))
    k1_hat = k1c / ke[None, :]
    phi22 = (4.0 / (9.0 * np.pi) / ke[None, :] ** 2 * k1_hat ** 2
             / (1.0 + k1_hat ** 2) ** (7.0 / 3.0))
    integrand = (l_mix * ums * dudy ** 2)[None, :] * phi22 * phi_m \
        * np.exp(-2.0 * np.abs(k1c) * yc)
    inner = integrand @ wy                                # (nk,)
    # spanwise wavenumber k3 = 0: the k1^2/(k1^2 + k3^2) factor is unity
    return 4.0 * inner


def _k1_grid(omega, k_max, u_inf, delta):
    """Composite k1 grid: coverage of [0, k_max] plus refinement around the
    convective peak k1 ~ omega / (0.7 U)."""
    base = np.linspace(0.0, k_max, _N_K_BASE + 1)[1:]
    kc = omega / (0.7 * u_inf)
    sigma = 0.05 / (0.085 * delta / KAPPA)   # narrowest Gaussian width
    lo = max(kc - 12.0 * sigma, 1e-8)
    hi = min(kc + 12.0 * sigma, k_max)
    peak = np.linspace(lo, hi, _N_K_PEAK) if hi > lo else np.empty(0)
    grid = np.unique(np.concatenate([base, peak]))
    return grid


def tno_te_surface(freq, span, r, theta_deg, phi_deg, u_inf, delta99, cf,
                   c0, rho, nu, suction):
    """TNO trailing-edge noise from ONE surface. Returns per-band SPL [dB].

    ``cf`` is the freestream-normalized TE skin friction; must be > 0.
    """
    if cf <= 0.0:
        raise ValueError("TNO requires attached flow at the TE (cf > 0); "
                         "fall back to the BPM TBL-TE model instead")
    freq = np.asarray(freq, dtype=float)
    mach = u_inf / c0
    dbar_h = directivity_high(mach, theta_deg, phi_deg)
    alpha_v = ALPHA_SUCTION if suction else ALPHA_PRESSURE

    # wall-normal Gauss-Legendre nodes on (0, delta)
    xg, wg = np.polynomial.legendre.leggauss(_N_Y)
    y = 0.5 * delta99 * (xg + 1.0)
    wy = 0.5 * delta99 * wg

    spl = np.empty_like(freq)
    p_ref2 = (2.0e-5) ** 2
    for i, f in enumerate(freq):
        omega = 2.0 * np.pi * f
        k_max = 10.0 * omega / (mach * c0)
        k1 = _k1_grid(omega, k_max, u_inf, delta99)
        p_k1 = rho ** 2 * _surface_pressure_integral(
            omega, k1, y, wy, u_inf, delta99, cf, nu, alpha_v)
        outer = np.trapz(omega / (c0 * k1) * p_k1, k1)
        spectrum = span / (4.0 * np.pi * r ** 2) * outer
        val = spectrum * dbar_h / p_ref2
        spl[i] = 10.0 * np.log10(max(val, 1e-30))
    spl = spl + 10.0 * np.log10(band_width(freq))
    return np.maximum(spl, SPL_FLOOR)


def tno_te(freq, span, r, theta_deg, phi_deg, bl, u_inf, c0, rho, nu):
    """TNO TBL-TE noise for both surfaces of a `boundary_layer.BLInputs`.

    Returns (spl_pressure_side, spl_suction_side). The separation/stall
    contribution is not part of TNO; combine with BPM's spl_alpha as
    NAFNoise does.
    """
    spl_s = tno_te_surface(freq, span, r, theta_deg, phi_deg, u_inf,
                           bl.suction.delta99, bl.suction.cf, c0, rho, nu,
                           suction=True)
    spl_p = tno_te_surface(freq, span, r, theta_deg, phi_deg, u_inf,
                           bl.pressure.delta99, bl.pressure.cf, c0, rho, nu,
                           suction=False)
    return spl_p, spl_s
