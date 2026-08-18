"""
Frequency bands, A-weighting, and decibel arithmetic.

All spectra in this package are one-third-octave band sound pressure levels
(dB re 20 uPa) evaluated at the nominal band-center frequencies.
"""

import numpy as np

# Nominal 1/3-octave band centers (IEC 61260 preferred numbers), one decade.
_NOMINAL_THIRDS = np.array([1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0])

# dB floor used throughout (matches NAFNoise / the BPM reference code).
SPL_FLOOR = -100.0

P_REF = 2.0e-5  # reference pressure [Pa]


def third_octave_bands(f_low=10.0, f_high=20000.0):
    """Nominal 1/3-octave band-center frequencies covering [f_low, f_high].

    The default range reproduces NAFNoise's 34 bands (10 Hz .. 20 kHz).
    """
    decades = np.arange(0, 8)
    freqs = np.concatenate([_NOMINAL_THIRDS * 10.0 ** d for d in decades])
    return freqs[(freqs >= f_low * (1 - 1e-9)) & (freqs <= f_high * (1 + 1e-9))]


def band_width(freq):
    """Bandwidth of each 1/3-octave band centered at ``freq``."""
    ratio = 2.0 ** (1.0 / 3.0)
    return np.asarray(freq) * (np.sqrt(ratio) - 1.0 / np.sqrt(ratio))


def a_weighting(freq):
    """A-weighting in dB (IEC 61672) at frequencies ``freq`` [Hz]."""
    f = np.asarray(freq, dtype=float)
    f2 = f ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    return 20.0 * np.log10(ra) + 2.0


def spl_sum(*spls):
    """Energetic (mean-square-pressure) sum of SPL spectra, elementwise."""
    stacked = np.vstack([np.asarray(s, dtype=float) for s in spls])
    total = 10.0 * np.log10(np.sum(10.0 ** (stacked / 10.0), axis=0))
    return np.maximum(total, SPL_FLOOR)


def oaspl(spl, a_weighted=False, freq=None):
    """Overall SPL from a band spectrum; A-weighted if requested."""
    spl = np.asarray(spl, dtype=float)
    if a_weighted:
        if freq is None:
            raise ValueError("freq is required for A-weighted OASPL")
        spl = spl + a_weighting(freq)
    return 10.0 * np.log10(np.sum(10.0 ** (spl / 10.0)))


def apply_floor(spl, floor=SPL_FLOOR):
    """Clip an SPL spectrum at the standard floor, preserving array shape."""
    return np.maximum(np.asarray(spl, dtype=float), floor)
