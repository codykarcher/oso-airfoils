"""
Tests for oso_airfoils.noise.

Golden data:
- data/bpm_golden.csv.gz : outputs of the NAFNoise/BPM Fortran kernels
  (bpm/PREDICT.FOR), compiled in double precision with full line length and
  with the G5COMP interval gaps closed (see data/kernels.f + driver.f90).
- data/tno_golden.csv : outputs of the NAFNoise TNO module compiled as-is
  (data/tno_driver.f90) at the S809 case with qfoil TE BL values.

Run:  pytest oso_airfoils/noise/tests -q
"""

import gzip
import pathlib

import numpy as np
import pytest

from oso_airfoils.noise import bpm
from oso_airfoils.noise.spectrum import (a_weighting, oaspl, spl_sum,
                                         third_octave_bands)

DATA = pathlib.Path(__file__).parent / "data"

C0, VISC, L, R = 337.7559, 1.4529e-5, 0.509, 1.22
FREQ = third_octave_bands()
TRIPS = {0: "none", 1: "heavy", 2: "light"}


def _floor(x):
    return np.maximum(np.asarray(x, float), -100.0)


def _golden_rows():
    with gzip.open(DATA / "bpm_golden.csv.gz", "rt") as fh:
        for line in fh:
            if line.strip():
                yield line.strip().split(",")


# ---------------------------------------------------------------------------
# spectrum utilities
# ---------------------------------------------------------------------------

def test_band_centers():
    assert len(FREQ) == 34
    assert FREQ[0] == 10.0 and FREQ[-1] == 20000.0


def test_a_weighting_anchor():
    assert abs(a_weighting(np.array([1000.0]))[0]) < 0.02


def test_spl_sum_doubling():
    two = spl_sum(np.array([60.0]), np.array([60.0]))
    assert abs(two[0] - 63.0103) < 1e-3


# ---------------------------------------------------------------------------
# BPM kernels vs Fortran golden data
# ---------------------------------------------------------------------------

def test_bpm_golden():
    worst = {}
    for row in _golden_rows():
        kind = row[0]
        vals = [float(v) for v in row[1:] if v.strip()]
        if kind == "THICK":
            c, u, alp, itrip = vals[0], vals[1], vals[2], int(vals[3])
            got = bpm.bl_thickness_bpm(c, u, alp, TRIPS[itrip], C0, VISC)
            mine = np.array([got["delta_p"], got["dstar_s"],
                             got["dstar_p"]])
            diff = np.max(np.abs(mine / np.array(vals[4:7]) - 1.0))
        elif kind == "DIRECT":
            m, th, ph = vals[0], vals[1], vals[2]
            mine = np.array([bpm.directivity_high(m, th, ph),
                             bpm.directivity_low(m, th, ph)])
            diff = np.max(np.abs(mine / np.array(vals[3:5]) - 1.0))
        elif kind in ("TBLTE_P", "TBLTE_S", "TBLTE_A"):
            c, u, alp = vals[:3]
            blp = bpm.bl_thickness_bpm(c, u, alp, "none", C0, VISC)
            p, s, a = bpm.tbl_te(FREQ, alp, c, u, L, R, 90.0, 90.0, C0,
                                 VISC, blp["dstar_s"], blp["dstar_p"])
            mine = {"TBLTE_P": p, "TBLTE_S": s, "TBLTE_A": a}[kind]
            diff = np.max(np.abs(_floor(mine) - _floor(vals[3:])))
        elif kind == "LBLVS":
            c, u, alp = vals[:3]
            blp = bpm.bl_thickness_bpm(c, u, alp, "none", C0, VISC)
            mine = bpm.lbl_vs(FREQ, alp, c, u, L, R, 90.0, 90.0, C0, VISC,
                              blp["delta_p"])
            diff = np.max(np.abs(_floor(mine) - _floor(vals[3:])))
        elif kind == "BLUNT":
            c, u, alp, h, psi = vals[:5]
            blp = bpm.bl_thickness_bpm(c, u, alp, "none", C0, VISC)
            mine = bpm.bluntness(FREQ, u, L, R, 90.0, 90.0, C0, h, psi,
                                 blp["dstar_s"], blp["dstar_p"])
            diff = np.max(np.abs(_floor(mine) - _floor(vals[5:])))
        elif kind == "TBLTE_DIR":
            c, u, alp, th, ph = vals[:5]
            blp = bpm.bl_thickness_bpm(c, u, alp, "none", C0, VISC)
            p, s, a = bpm.tbl_te(FREQ, alp, c, u, L, R, th, ph, C0, VISC,
                                 blp["dstar_s"], blp["dstar_p"])
            diff = np.max(np.abs(_floor(bpm.tbl_te_total(p, s, a))
                                 - _floor(vals[5:])))
        elif kind == "TIP":
            c, u, alp, iround = vals[0], vals[1], vals[2], int(vals[3])
            mine = bpm.tip_noise(FREQ, alp, c, u, R, 90.0, 90.0, C0,
                                 round_tip=(iround == 0))
            diff = np.max(np.abs(_floor(mine) - _floor(vals[4:])))
        else:
            continue
        worst[kind] = max(worst.get(kind, 0.0), float(diff))

    assert worst["THICK"] < 5e-4
    # the Fortran reference uses a truncated deg->rad constant (0.017453)
    assert worst["DIRECT"] < 1e-2
    for kind in ("TBLTE_P", "TBLTE_S", "TBLTE_A", "TBLTE_DIR", "LBLVS",
                 "BLUNT", "TIP"):
        assert worst[kind] < 5e-3, f"{kind}: {worst[kind]:.3e} dB"


# ---------------------------------------------------------------------------
# TNO vs the compiled NAFNoise TNO module
# ---------------------------------------------------------------------------

def test_tno_golden():
    from oso_airfoils.noise import tno
    gold = {1: {}, 2: {}}
    for line in open(DATA / "tno_golden.csv"):
        f, side, v = line.split(",")
        gold[int(side)][float(f)] = float(v)
    ss = tno.tno_te_surface(FREQ, 0.509, 1.22, 90.0, 90.0, 63.92, 11.06e-3,
                            0.00035, C0, 1.225, VISC, suction=True)
    sp = tno.tno_te_surface(FREQ, 0.509, 1.22, 90.0, 90.0, 63.92, 7.48e-3,
                            0.00193, C0, 1.225, VISC, suction=False)
    for spl, side in ((ss, 1), (sp, 2)):
        ref = np.array([gold[side][f] for f in FREQ])
        mask = ref > 0.0        # audible bands
        # residual is dominated by NAFNoise's single 61-point Gauss-Kronrod
        # panel vs our converged composite grid
        assert np.max(np.abs(spl[mask] - ref[mask])) < 0.25


# ---------------------------------------------------------------------------
# NAFNoise S809 case (needs metafoil)
# ---------------------------------------------------------------------------

def test_inflow_nafnoise_s809():
    from oso_airfoils.noise import inflow
    from oso_airfoils.noise.validation import compare_nafnoise as v
    freq, ref = v.load_reference()
    spl = inflow.turbulent_inflow(
        freq, v.ALPHA, v.U, v.CHORD, v.SPAN, v.R_OBS, v.THETA, v.PHI,
        v.C0, v.RHO, v.TI_FRACTION, v.L_TURB,
        thickness_1pc=v.THICK_1PC, thickness_10pc=v.THICK_10PC)
    assert np.max(np.abs(spl - ref["inflow"])) < 0.01


def test_section_s809_vs_nafnoise_qfoil():
    pytest.importorskip("metafoil")
    from oso_airfoils.noise.validation import compare_nafnoise as v
    freq, ref = v.load_reference()
    res = v.run(bl_source="qfoil")
    diff = res.total - ref["total"]
    assert np.max(np.abs(diff)) < 0.5


def test_section_s809_vs_nafnoise_nqfoil():
    pytest.importorskip("metafoil")
    from oso_airfoils.noise.validation import compare_nafnoise as v
    freq, ref = v.load_reference()
    res = v.run(bl_source="nqfoil")
    diff = res.total - ref["total"]
    assert np.max(np.abs(diff)) < 2.0


def test_result_summaries():
    pytest.importorskip("metafoil")
    from oso_airfoils.noise.validation import compare_nafnoise as v
    res = v.run(bl_source="nqfoil")
    assert 60.0 < res.oaspl < 90.0
    assert np.isfinite(res.oaspl_a)
    d = res.as_dict()
    assert "total" in d and len(d["freq"]) == 34
