"""
Example: noise spectra for an airfoil section from Kulfan coefficients.

Runs the NAFNoise S809 demonstration case through the section pipeline and
plots the per-source spectra, then compares BPM vs TNO trailing-edge noise.

    python -m oso_airfoils.noise.example
"""

import matplotlib.pyplot as plt
import numpy as np

from oso_airfoils.noise import (Atmosphere, Observer, predict_section_noise)
from oso_airfoils.noise.validation.compare_nafnoise import s809_kulfan

LABELS = {
    "tbl_te_pressure": "TBL-TE pressure side",
    "tbl_te_suction": "TBL-TE suction side",
    "separation": "separation",
    "lbl_vs": "LBL vortex shedding",
    "bluntness": "TE bluntness",
    "inflow": "turbulent inflow",
}


def main():
    uw, lw = s809_kulfan()
    common = dict(
        alpha=3.0, u=63.92, chord=0.2286, span=0.509,
        observer=Observer(r=1.22),
        atmosphere=Atmosphere.nafnoise_default(),
        h_te=0.00021, te_psi_deg=12.5,
        xtr_upper=0.02, xtr_lower=0.1,
        inflow_turbulence={"intensity": 0.0005, "length_scale": 0.06})

    res = predict_section_noise(uw, lw, te_gap=0.0, bl_source="nqfoil",
                                **common)
    res_tno = predict_section_noise(uw, lw, te_gap=0.0, bl_source="qfoil",
                                    tbl_te_method="tno", **common)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name, spl in res.spl.items():
        ax1.semilogx(res.freq, spl, label=LABELS.get(name, name))
    ax1.semilogx(res.freq, res.total, "k-", lw=2.2, label="total")
    ax1.set_title(f"S809 sources (BPM, nqfoil BL)\n"
                  f"OASPL {res.oaspl:.1f} dB / {res.oaspl_a:.1f} dBA")
    ax1.legend(fontsize=7)

    for r, style, tag in ((res, "-", "BPM"), (res_tno, "--", "TNO")):
        ax2.semilogx(r.freq, r.spl["tbl_te_suction"], "C0" + style,
                     label=f"suction ({tag})")
        ax2.semilogx(r.freq, r.spl["tbl_te_pressure"], "C1" + style,
                     label=f"pressure ({tag})")
    ax2.set_title("TBL-TE: BPM vs TNO")
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlabel("frequency [Hz]")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(-10, 70)
    ax1.set_ylabel("1/3-octave SPL [dB re 20 $\\mu$Pa]")
    fig.tight_layout()
    out = "s809_noise_example.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
