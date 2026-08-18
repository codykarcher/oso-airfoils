"""
Validation against the NAFNoise-distributed S809 example case.

NAFNoise (NREL, Moriarty 2005) ships an example: S809 section, chord
0.2286 m, span 0.509 m, U = 63.92 m/s, alpha = 3 deg, tripped at
x/c = 0.02 (upper) / 0.10 (lower), TE thickness 0.21 mm with psi = 12.5 deg,
observer at r = 1.22 m, theta = phi = 90 deg. Its nafnoise.out was produced
with XFOIL-derived BL thicknesses, BPM TBL-TE, BPM bluntness, and the
simplified Guidati inflow model (turbulence "0.050 (%)" -> 0.0005 fraction,
L = 0.06 m).

Run:  python -m oso_airfoils.noise.validation.compare_nafnoise

Expected agreement:
- inflow: < 0.01 dB (identical model, no BL dependence)
- TBL-TE and bluntness: within ~1-2 dB where levels are audible. The
  remaining difference is dominated by (a) qfoil/nqfoil vs the 2005 XFOIL
  build's TE boundary-layer values and (b) known quirks of the distributed
  NAFNoise binary (single precision; four constants silently truncated by
  the Fortran 72-column limit; an uninitialized-variable path in the G5
  bluntness helper) which our implementation intentionally corrects.
"""

import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

# NAFNoise s809 case inputs (nafnoise.ipt)
C0, NU, RHO = 337.7559, 1.4529e-5, 1.225
CHORD, SPAN, U, ALPHA = 0.2286, 0.509, 63.92, 3.0
H_TE, PSI = 0.00021, 12.5
R_OBS, THETA, PHI = 1.22, 90.0, 90.0
XTR_U, XTR_L = 0.02, 0.1
TI_FRACTION, L_TURB = 0.05 / 100.0, 0.06
THICK_1PC, THICK_10PC = 0.02, 0.12


def load_reference():
    """Parse nafnoise.out into (freq, spl[7]) arrays."""
    rows = []
    for line in open(HERE / "nafnoise.out"):
        parts = line.split()
        if len(parts) == 8 and parts[0][0].isdigit():
            rows.append([float(v) for v in parts])
    data = np.array(rows)
    names = ["tbl_te_pressure", "tbl_te_suction", "separation",
             "lbl_vs", "bluntness", "inflow", "total"]
    return data[:, 0], {n: data[:, i + 1] for i, n in enumerate(names)}


def s809_kulfan():
    from metafoil.core.kulfan import Kulfan
    dat = (HERE / "s809.dat").read_text().splitlines()
    xy = np.array([l.split() for l in dat[1:] if len(l.split()) == 2], float)
    k = Kulfan.fit_to_coordinates(xy[:, 0], xy[:, 1], fit_order=8)
    return (np.asarray(k.upperCoefficients, float).ravel(),
            np.asarray(k.lowerCoefficients, float).ravel())


def run(bl_source="qfoil"):
    from oso_airfoils.noise import (Atmosphere, Observer,
                                    predict_section_noise)
    uw, lw = s809_kulfan()
    return predict_section_noise(
        uw, lw, te_gap=0.0,
        alpha=ALPHA, u=U, chord=CHORD, span=SPAN,
        observer=Observer(r=R_OBS, theta_deg=THETA, phi_deg=PHI),
        atmosphere=Atmosphere(c0=C0, nu=NU, rho=RHO),
        bl_source=bl_source, tbl_te_method="bpm",
        include_lbl=False, include_bluntness=True,
        h_te=H_TE, te_psi_deg=PSI,
        xtr_upper=XTR_U, xtr_lower=XTR_L,
        inflow_turbulence={"intensity": TI_FRACTION, "length_scale": L_TURB,
                           "thickness_1pc": THICK_1PC,
                           "thickness_10pc": THICK_10PC})


def main():
    freq, ref = load_reference()
    for bl_source in ("qfoil", "nqfoil"):
        res = run(bl_source)
        print(f"\n=== bl_source = {bl_source} ===")
        print(f"{'freq':>8} | " + " | ".join(
            f"{n[:12]:>21}" for n in
            ("tbl_te_pressure", "tbl_te_suction", "bluntness", "inflow")))
        print(f"{'':>8} | " + " | ".join(
            f"{'ref':>10} {'oso':>10}" for _ in range(4)))
        for i in range(0, len(freq), 4):
            cells = []
            for n in ("tbl_te_pressure", "tbl_te_suction", "bluntness",
                      "inflow"):
                cells.append(f"{ref[n][i]:10.2f} {res.spl[n][i]:10.2f}")
            print(f"{freq[i]:8.0f} | " + " | ".join(cells))
        # summary over audible bands
        for n in ("tbl_te_pressure", "tbl_te_suction", "bluntness",
                  "inflow", "total"):
            mine = res.total if n == "total" else res.spl[n]
            mask = ref[n] > 0.0
            if mask.any():
                d = mine[mask] - ref[n][mask]
                print(f"  {n:<16} mean diff {np.mean(d):+6.2f} dB, "
                      f"max |diff| {np.max(np.abs(d)):5.2f} dB "
                      f"({mask.sum()} bands with ref > 0 dB)")


if __name__ == "__main__":
    main()
