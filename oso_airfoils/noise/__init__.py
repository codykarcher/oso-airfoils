"""
oso_airfoils.noise — semi-empirical airfoil (and rotor) noise prediction.

Kulfan coefficients + trailing-edge gap in, 1/3-octave noise spectra out,
with boundary-layer inputs supplied by metafoil (nqfoil surrogate by
default). See README.md in this directory for theory references and
validation status.

    from oso_airfoils.noise import predict_section_noise, Observer
"""

from .atmosphere import Atmosphere, Observer
from .boundary_layer import (BLInputs, SurfaceBL, bl_from_nqfoil,
                             bl_from_qfoil)
from .section import SectionNoiseResult, predict_section_noise
from .spectrum import (a_weighting, band_width, oaspl, spl_sum,
                       third_octave_bands)
from .turbine import (BladeStation, Rotor, RotorNoiseResult,
                      predict_rotor_noise)
from . import bpm, inflow, tno

__all__ = [
    "Atmosphere", "Observer", "BLInputs", "SurfaceBL",
    "bl_from_nqfoil", "bl_from_qfoil",
    "SectionNoiseResult", "predict_section_noise",
    "BladeStation", "Rotor", "RotorNoiseResult", "predict_rotor_noise",
    "a_weighting", "band_width", "oaspl", "spl_sum", "third_octave_bands",
    "bpm", "inflow", "tno",
]
