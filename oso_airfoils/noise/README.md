# oso_airfoils.noise — airfoil and rotor noise prediction

Semi-empirical aeroacoustic prediction for wind-turbine airfoils: **Kulfan
coefficients + TE gap in, 1/3-octave SPL spectra out**, with boundary-layer
inputs supplied by metafoil (the `nqfoil` neural surrogate by default, `qfoil`
for a real viscous solve). A clean-room reimplementation of the models in
NREL's NAFNoise (which has no maintained theory guide — the references below
are the theory guide), verified kernel-by-kernel against its Fortran.

## Quickstart

```python
from oso_airfoils.noise import predict_section_noise, Observer

res = predict_section_noise(
    upper_weights, lower_weights, te_gap=0.002,   # metafoil-convention Kulfan
    alpha=3.0, u=63.92, chord=0.2286, span=0.509, # deg, m/s, m, m
    observer=Observer(r=1.22),                    # theta=phi=90 default
    inflow_turbulence={"intensity": 0.05, "length_scale": 0.06})

res.freq                      # 34 bands, 10 Hz .. 20 kHz
res.spl["tbl_te_suction"]     # per-source spectra [dB re 20 uPa]
res.total, res.oaspl, res.oaspl_a
```

Rotor level (the path to full-turbine noise):

```python
from oso_airfoils.noise import BladeStation, Rotor, predict_rotor_noise

rotor = Rotor(stations=[BladeStation(r=..., chord=..., twist_deg=...,
                                     upper_weights=..., lower_weights=...,
                                     te_gap=...), ...],
              n_blades=3, hub_height=150.0, hub_radius=3.0)
res = predict_rotor_noise(rotor, v_wind=10.0, rpm=7.0,
                          inflow_turbulence={"intensity": 0.10,
                                             "length_scale": 42.0})
res.oaspl_a   # dBA at the (default IEC-style) ground observer
```

## Noise sources

| source | model | needs |
|---|---|---|
| TBL-TE (pressure/suction) | BPM scaling laws (default) or TNO-Blake (`tbl_te_method='tno'`) | delta* at TE (BPM); delta99 + Cf (TNO) |
| separation / stall | BPM angle-of-attack term | delta* at TE |
| LBL vortex shedding | BPM (untripped only) | pressure-side delta99 |
| TE bluntness | BPM | TE thickness h (default `te_gap*chord`), psi angle |
| turbulent inflow | Amiet + LFC, optional simplified-Guidati thickness correction (+10 dB NLR calibration) | TI, length scale, t/c at 1% & 10% chord (computed from the Kulfan automatically) |
| tip vortex | BPM (rotor level) | tip alpha |

Boundary-layer sources (`bl_source`): `'nqfoil'` (default, ~30 ms/case,
differentiable-friendly), `'qfoil'` (viscous solve, Cf from the solver),
`'bpm'` (NACA-0012 correlations; no TNO). TE values are taken at the last
station (qfoil) or extrapolated to x/c=1 from the 32-station nqfoil output;
delta99 = theta*(3.15 + 1.72/(Hk-1)) + delta* (Drela), Cf from the XFOIL
Coles closure scaled to freestream q (both exactly what NAFNoise does with
XFOIL output).

## Validation

`python -m oso_airfoils.noise.validation.compare_nafnoise` reproduces the
NAFNoise-distributed S809 case (0.2286 m chord, U=63.92 m/s, alpha=3 deg,
tripped 0.02/0.10). Current status:

- BPM kernels: match the NAFNoise Fortran to **< 5e-3 dB** on a 900-case
  grid (tests/data/bpm_golden.csv.gz, regenerable via driver.f90).
- TNO: matches the compiled NAFNoise TNO module to **< 0.25 dB** in audible
  bands (residual = their single 61-pt Gauss-Kronrod panel vs our converged
  grid).
- Inflow: matches nafnoise.out to **< 0.01 dB**.
- Full S809 section vs nafnoise.out totals: **< 0.15 dB** (qfoil BL),
  **< 0.6 dB** (nqfoil BL).

### Known quirks of the reference (we correct them)

The distributed NAFNoise/BPM Fortran, compiled with standard fixed-form
72-column rules, silently truncates four constants (A-min cubic `+6.006` ->
`+6.0`; B-min `10.619` -> `10.61`; two G5 bluntness branches lose `-2.5` /
`-1.25`), and its `G5COMP` has open interval boundaries that read an
uninitialized `mu` exactly at h/delta*=0.25 — which the F4TEMP cap evaluates
on *every* call. Combined with single precision, historic NAFNoise binaries
can differ from the published RP-1218 equations by up to ~2.5 dB in
individual bluntness bands and ~0.05 dB elsewhere. This package implements
the report-intent equations; the golden data was generated with
`-ffixed-line-length-none -fdefault-real-8` and closed intervals.

## Theory references

- Brooks, Pope & Marcolini, *Airfoil Self-Noise and Prediction*, NASA
  RP-1218, 1989 (BPM: TBL-TE, separation, LBL-VS, bluntness, tip, BL
  correlations).
- Moriarty & Migliore, *Semi-Empirical Aeroacoustic Noise Prediction Code
  for Wind Turbines*, NREL/TP-500-34478, 2003 (XFOIL coupling, TNO).
- Parchen, TNO report DDA 1998 (TNO-Blake surface-pressure model).
- Amiet, J. Sound Vib. 41(4), 1975 (inflow noise).
- Moriarty, Guidati & Migliore, AIAA 2004-3041 and AIAA 2005-2881
  (Guidati corrections; the simplified thickness correction + NLR offset).
- Moriarty, *NAFNoise User's Guide*, NREL, 2005.

## Limitations / next steps

- BPM constants were fit to NACA-0012 data at Rc <= ~3M and alpha <= 12.5
  deg; megawatt-scale sections (Re 6-15M) extrapolate. Prefer solver BL
  inputs (default) and consider TNO for shape-sensitivity studies (BPM only
  sees the airfoil through delta*).
- Above the stall switch (alpha >= gamma0(M) or 12.5 deg) BPM's crude
  separated-flow model takes over; treat results as qualitative.
- Rotor level: no Doppler/retarded time, no atmospheric absorption or
  ground reflection, steady uniform inflow (no shear/yaw/tower shadow),
  local-frame directivity without wake skew. These are the standard
  OpenFAST-AeroAcoustics upgrades if certification-grade output is needed.
- alpha < 0 is handled by mirroring (|alpha| with surfaces swapped).
- nqfoil BL values inherit surrogate error (~1-2 dB in TBL-TE components
  vs a qfoil solve on S809); use `bl_source='qfoil'` when it matters.

## Files

- `bpm.py`, `tno.py`, `inflow.py` — acoustic models
- `boundary_layer.py` — TE BL inputs from nqfoil / qfoil
- `section.py` — `predict_section_noise` (main entry point)
- `turbine.py` — BEM + rotor integration (`predict_rotor_noise`)
- `spectrum.py`, `atmosphere.py` — bands, A-weighting, dB utils, dataclasses
- `validation/` — NAFNoise S809 case + comparison script
- `tests/` — golden-data tests (`pytest oso_airfoils/noise/tests`)
