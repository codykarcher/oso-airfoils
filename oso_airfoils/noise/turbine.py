"""
Rotor-level noise: integrate section predictions along the blade.

This is the "path to full turbine noise": each blade station is analyzed
with the section pipeline (BPM/TNO trailing-edge sources + optional inflow
noise) at its local inflow speed and angle of attack, with the observer
distance and BPM directivity angles evaluated per station and per blade
azimuth; contributions are summed on a mean-square-pressure basis over
stations, azimuth positions, and blades. The outermost station optionally
adds BPM tip-vortex noise.

Local flow conditions come from a built-in blade-element-momentum solve
(axial/tangential induction with Prandtl tip and hub loss and the Glauert
high-induction correction), with airfoil polars evaluated on the fly by
metafoil's nqfoil — or the caller can pass precomputed induction factors.

Deliberate v1 simplifications (all standard next steps, all noted in the
README): no Doppler/convective retarded-time effects, no atmospheric
absorption or ground reflection, directivity angles from the local blade
frame without wake skew, BL and loads steady and azimuth-independent
(uniform inflow — no shear, yaw, or tower shadow).

Geometry: wind along +x, z up, hub at (0, 0, hub_height). Blade azimuth
psi = 0 points up; the blade element at radius r sits at
(0, -r sin psi, hub_height + r cos psi). The observer position is given in
ground coordinates (x downwind of the tower base).
"""

from dataclasses import dataclass, field

import numpy as np

from .atmosphere import Atmosphere, Observer
from .section import predict_section_noise
from .spectrum import a_weighting, oaspl, spl_sum, third_octave_bands
from . import bpm


@dataclass
class BladeStation:
    """One aerodynamic blade station (SI units, angles in degrees)."""
    r: float                    # radius from hub center [m]
    chord: float                # [m]
    twist_deg: float            # aerodynamic twist, + toward feather
    upper_weights: np.ndarray   # metafoil-convention Kulfan weights
    lower_weights: np.ndarray
    te_gap: float = 0.0         # TE thickness / chord
    xtr_upper: float = 1.0      # forced-transition locations
    xtr_lower: float = 1.0


@dataclass
class Rotor:
    stations: list
    n_blades: int = 3
    hub_height: float = 100.0
    hub_radius: float = None    # default: first station radius

    @property
    def tip_radius(self):
        return max(s.r for s in self.stations)


@dataclass
class RotorNoiseResult:
    freq: np.ndarray
    spl: dict                   # per-source rotor spectra [dB]
    stations: list = field(default_factory=list)   # per-station diagnostics
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


# ---------------------------------------------------------------------------
# Blade-element momentum
# ---------------------------------------------------------------------------

def _polar(station, alpha, re, model_size):
    """(cl, cd) from nqfoil at one condition."""
    from metafoil.nqfoil import full_bl
    out = full_bl.get_aero_from_kulfan_parameters(
        {"upper_weights": np.asarray(station.upper_weights, float).ravel(),
         "lower_weights": np.asarray(station.lower_weights, float).ravel(),
         "TE_thickness": float(station.te_gap)},
        alpha=float(alpha), Re=float(re),
        xtr_upper=station.xtr_upper, xtr_lower=station.xtr_lower,
        model_size=model_size)
    return float(out["CL"]), float(out["CD"])


def _bem_station(station, v_wind, omega, pitch_deg, n_blades, r_tip, r_hub,
                 nu, model_size, n_iter=40, relax=0.5):
    """Classic BEM at one station. Returns (w, alpha_deg, a, ap)."""
    r, chord = station.r, station.chord
    sigma = n_blades * chord / (2.0 * np.pi * r)
    a, ap = 0.3, 0.0
    lam_r = omega * r / v_wind
    for _ in range(n_iter):
        phi = np.arctan2(v_wind * (1.0 - a), omega * r * (1.0 + ap))
        sin_phi = max(np.sin(phi), 1e-4)
        # Prandtl tip and hub loss
        f_tip = (2.0 / np.pi) * np.arccos(np.exp(
            -n_blades * (r_tip - r) / (2.0 * r * sin_phi)))
        f_hub = (2.0 / np.pi) * np.arccos(np.exp(
            -n_blades * (r - r_hub) / (2.0 * r_hub * sin_phi))) \
            if r_hub > 0 else 1.0
        f_loss = max(f_tip * f_hub, 1e-3)

        alpha = np.degrees(phi) - station.twist_deg - pitch_deg
        w = np.hypot(v_wind * (1.0 - a), omega * r * (1.0 + ap))
        re = w * chord / nu
        cl, cd = _polar(station, alpha, re, model_size)
        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)

        ct_thrust = sigma * (1.0 - a) ** 2 * cn / sin_phi ** 2
        # Glauert correction (Buhl) above a ~ 0.4
        if ct_thrust < 0.96 * f_loss:
            a_new = 0.5 * (1.0 - np.sqrt(max(1.0 - ct_thrust / f_loss, 0.0)))
        else:
            a_new = (18.0 * f_loss - 20.0 - 3.0 * np.sqrt(
                max(ct_thrust * (50.0 - 36.0 * f_loss)
                    + 12.0 * f_loss * (3.0 * f_loss - 4.0), 0.0))) \
                / (36.0 * f_loss - 50.0)
        kt = sigma * ct / (4.0 * f_loss * sin_phi * np.cos(phi))
        ap_new = kt * (1.0 - a) / (lam_r * (1.0 + kt / lam_r)) \
            if lam_r > 0 else 0.0
        da = a_new - a
        a += relax * da
        ap += relax * (ap_new - ap)
        a = min(max(a, 0.0), 0.95)
        if abs(da) < 1e-5:
            break
    phi = np.arctan2(v_wind * (1.0 - a), omega * r * (1.0 + ap))
    alpha = np.degrees(phi) - station.twist_deg - pitch_deg
    w = np.hypot(v_wind * (1.0 - a), omega * r * (1.0 + ap))
    return w, alpha, a, ap


# ---------------------------------------------------------------------------
# Geometry: observer angles in the local blade frame
# ---------------------------------------------------------------------------

def _observer_geometry(r_station, psi, hub_height, obs_xyz):
    """(distance, theta_deg, phi_deg) for one element/azimuth.

    Local frame: span = radial (hub to tip), chord = tangential pointing
    from LE to TE (opposite the direction of motion), normal completes the
    triad. theta is the angle of the observer vector from the chord axis,
    phi from the span axis (BPM convention; 90/90 = broadside).
    """
    pos = np.array([0.0, -r_station * np.sin(psi),
                    hub_height + r_station * np.cos(psi)])
    span_hat = np.array([0.0, -np.sin(psi), np.cos(psi)])
    motion_hat = np.array([0.0, -np.cos(psi), -np.sin(psi)])  # leading-edge-first
    chord_hat = -motion_hat                                    # LE -> TE
    v = np.asarray(obs_xyz, float) - pos
    dist = np.linalg.norm(v)
    v_hat = v / dist
    theta = np.degrees(np.arccos(np.clip(np.dot(v_hat, chord_hat), -1, 1)))
    phi = np.degrees(np.arccos(np.clip(np.dot(v_hat, span_hat), -1, 1)))
    return dist, theta, phi


# ---------------------------------------------------------------------------
# Rotor prediction
# ---------------------------------------------------------------------------

def predict_rotor_noise(rotor, v_wind, rpm, pitch_deg=0.0,
                        observer_xyz=None, atmosphere=None,
                        n_azimuth=8, induction=None,
                        include_tip=True, round_tip=True,
                        inflow_turbulence=None,
                        bl_source="nqfoil", tbl_te_method="bpm",
                        freq=None, model_size="xxlarge", **section_kwargs):
    """Total rotor noise spectra at a ground observer.

    Parameters
    ----------
    rotor : `Rotor`. v_wind [m/s], rpm, pitch_deg : operating point.
    observer_xyz : ground-frame observer position [m]; default is the
        IEC-style reference position, hub_height downwind at ground level.
    induction : None (run the built-in BEM) or a list of (a, a_prime)
        per station to bypass it.
    inflow_turbulence : as in `predict_section_noise` (applied per station).
    section_kwargs : forwarded to `predict_section_noise` (e.g. n_crit).

    Returns
    -------
    RotorNoiseResult with per-source rotor spectra and per-station
    diagnostics (w, alpha, a, ap, segment OASPL).
    """
    atm = atmosphere or Atmosphere()
    freq = np.asarray(freq if freq is not None else third_octave_bands(),
                      dtype=float)
    if observer_xyz is None:
        observer_xyz = (rotor.hub_height, 0.0, 0.0)
    omega = rpm * 2.0 * np.pi / 60.0
    stations = sorted(rotor.stations, key=lambda s: s.r)
    r_tip = rotor.tip_radius
    r_hub = rotor.hub_radius if rotor.hub_radius is not None \
        else stations[0].r

    # segment widths from station midpoints
    radii = np.array([s.r for s in stations])
    edges = np.concatenate([[r_hub], 0.5 * (radii[1:] + radii[:-1]),
                            [r_tip]])
    widths = np.diff(edges)

    psis = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    power = {}          # per-source mean-square pressure accumulator
    diagnostics = []

    for i, st in enumerate(stations):
        if induction is not None:
            a, ap = induction[i]
            phi_flow = np.arctan2(v_wind * (1.0 - a),
                                  omega * st.r * (1.0 + ap))
            alpha = np.degrees(phi_flow) - st.twist_deg - pitch_deg
            w = np.hypot(v_wind * (1.0 - a), omega * st.r * (1.0 + ap))
        else:
            w, alpha, a, ap = _bem_station(
                st, v_wind, omega, pitch_deg, rotor.n_blades, r_tip, r_hub,
                atm.nu, model_size)

        # solver BL once per station; reused across azimuth positions
        seg_result = None
        for psi in psis:
            dist, theta, phi = _observer_geometry(
                st.r, psi, rotor.hub_height, observer_xyz)
            res = predict_section_noise(
                st.upper_weights, st.lower_weights, st.te_gap,
                alpha=alpha, u=w, chord=st.chord, span=widths[i],
                observer=Observer(r=dist, theta_deg=theta, phi_deg=phi),
                atmosphere=atm, bl_source=bl_source,
                tbl_te_method=tbl_te_method,
                xtr_upper=st.xtr_upper, xtr_lower=st.xtr_lower,
                inflow_turbulence=inflow_turbulence, freq=freq,
                model_size=model_size,
                bl_inputs=None if seg_result is None else seg_result.bl,
                **section_kwargs)
            seg_result = res
            for name, spl in res.spl.items():
                power.setdefault(name, np.zeros_like(freq))
                power[name] += 10.0 ** (spl / 10.0) / n_azimuth

            if include_tip and st.r == r_tip:
                spl_tip = bpm.tip_noise(freq, max(alpha, 0.0), st.chord, w,
                                        dist, theta, phi, atm.c0,
                                        round_tip=round_tip)
                power.setdefault("tip", np.zeros_like(freq))
                power["tip"] += 10.0 ** (spl_tip / 10.0) / n_azimuth

        diagnostics.append({
            "r": st.r, "width": widths[i], "w": w, "alpha": alpha,
            "a": a, "ap": ap,
            "segment_oaspl": res.oaspl if seg_result else None})

    n_b = rotor.n_blades
    spl = {name: 10.0 * np.log10(np.maximum(p * n_b, 1e-30))
           for name, p in power.items()}
    meta = {"v_wind": v_wind, "rpm": rpm, "pitch_deg": pitch_deg,
            "observer_xyz": tuple(observer_xyz), "n_azimuth": n_azimuth,
            "bl_source": bl_source, "tbl_te_method": tbl_te_method}
    return RotorNoiseResult(freq=freq, spl=spl, stations=diagnostics,
                            meta=meta)
