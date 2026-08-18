"""
Atmospheric constants and observer geometry for airfoil noise prediction.

Angle conventions follow Brooks, Pope & Marcolini (NASA RP-1218, 1989),
Fig. 11 / Appendix B, as used by NREL NAFNoise:

- ``theta_deg`` is measured from the chordline extending downstream of the
  trailing edge, positive toward the suction side. ``theta = 90`` is directly
  above/below the trailing edge.
- ``phi_deg`` is measured from the spanline. ``phi = 90`` is in the plane
  normal to the span (the usual observer position for a 2-D section).
- ``r`` is the straight-line distance from the trailing edge to the observer
  in meters.
"""

from dataclasses import dataclass


@dataclass
class Atmosphere:
    """Ambient air properties (SI units)."""
    c0: float = 340.29       # speed of sound            [m/s]
    nu: float = 1.4607e-5    # kinematic viscosity       [m^2/s]
    rho: float = 1.225       # density                   [kg/m^3]

    @classmethod
    def nafnoise_default(cls):
        """The constants distributed in the NAFNoise example input file."""
        return cls(c0=337.7559, nu=1.4529e-5, rho=1.225)


@dataclass
class Observer:
    """Observer location relative to the trailing edge (BPM convention)."""
    r: float                  # distance from trailing edge         [m]
    theta_deg: float = 90.0   # angle from chordline                [deg]
    phi_deg: float = 90.0     # angle from spanline                 [deg]
