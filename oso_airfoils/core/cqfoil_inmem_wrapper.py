"""
cqfoil_inmem_wrapper.py  —  in-memory CQFOIL wrapper (complex-step gradients).

Uses libcqfoil.so (CQFOIL's src_cs/ complex-step Qfoil/RFOIL compiled as an
f2py extension) to compute CL, CD and all their gradients via complex-step
differentiation, carrying Qfoil's RFOIL boundary-layer physics (LIPAN/LBLINI
cold reset every call via batch_oper, and the GWAKE wake-correction factor).

Geometry is loaded once via a short-lived real-valued temp DAT file (LOAD ->
AREAD/SCALC/SEGSPL/GEOPAR -> PANGEN).  This is the only correct path for
Qfoil's PANGEN (IPFAC=5); direct XB/YB injection produces NaN panel coords
(see qfoil_inmem_wrapper.py).  Because the geometry file is always real,
'coords' is not a supported gradient input here — use cfoil_inmem_wrapper /
cxfoil_inmem_wrapper for coordinate-gradient use cases.

For standard XFOIL (non-RFOIL) boundary-layer physics use
cxfoil_inmem_wrapper instead.

API mirrors cxfoil_inmem_wrapper (subroutine names, common-block layout, and
grads/bl/gradbl kwargs are identical) but is not file-based per call beyond
the initial geometry load.
"""

import os
import sys
import tempfile
import numpy as np

# Ensure the repo root is on sys.path so cqfoil_lib is findable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cmplxfoil.MExt import MExt

# Complex step size
_H_DEFAULT = 1.0e-20

# Default Qfoil GWAKE parameter (wake correction factor)
_GWAKE_DEFAULT = 0.40

# Scalar output names
_SCALAR_KEYS = ['cl', 'cd', 'cdp', 'cdf', 'cm', 'xtr_top', 'xtr_bot', 'cpmin']


def _load_module():
    """Load a unique libcqfoil instance via MExt."""
    return MExt('libcqfoil', 'cqfoil_lib')._module


def _load_geometry(xf, coords: np.ndarray):
    """
    Load airfoil coordinates via LOAD (short-lived real-valued temp DAT
    file), then run XFOIL() for INIT + PANGEN.

    Only called once per airfoil shape, not per solve — Qfoil's PANGEN
    (IPFAC=5) requires the buffer to be preprocessed by AREAD/SCALC/SEGSPL/
    GEOPAR before PANGEN runs; direct NB/XB/YB injection is not supported.
    """
    # Remove only consecutive duplicate points (e.g. LE (0,0) appearing
    # twice); do NOT touch first==last (closed-TE airfoil is valid).
    mask = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords)):
        if (abs(coords[i, 0] - coords[i-1, 0]) < 1e-12 and
                abs(coords[i, 1] - coords[i-1, 1]) < 1e-12):
            mask[i] = False
    coords = coords[mask]

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.dat',
                                      prefix='cqfoil_', delete=False)
    tmp.write('airfoil\n')
    for x, y in coords:
        tmp.write(f'  {x:.10f}  {y:.10f}\n')
    tmp.close()

    try:
        itype = np.array(1, dtype=np.int32)
        xf.load(tmp.name, itype)   # AREAD + SCALC + SEGSPL + GEOPAR
    finally:
        os.unlink(tmp.name)

    xf.xfoil()   # INIT + ABCOPY + PANGEN


def _set_conditions(xf, alpha_deg: float, Re: float, Ncrit: float,
                    Mach: float, xtp_u: float, xtp_l: float,
                    h_alpha=0.0, h_re=0.0, h_ncrit=0.0,
                    h_mach=0.0, h_xtop=0.0, h_xbot=0.0,
                    itmax: int = 100):
    """Set operating conditions with optional imaginary perturbations."""
    xf.cr09.adeg    = complex(alpha_deg, h_alpha)
    xf.cr15.reinf1  = complex(Re,        h_re)
    ncrit_c         = complex(Ncrit,     h_ncrit)
    xf.cr15.acrit   = [ncrit_c, ncrit_c]   # Qfoil: ACRIT(ISX)
    xf.cr09.minf1   = complex(Mach,      h_mach)
    xf.cr15.xstrip  = [complex(xtp_u, h_xtop), complex(xtp_l, h_xbot)]
    xf.ci04.itmax   = itmax
    xf.cr09.xcmref  = complex(0.25, 0.0)
    xf.cr09.ycmref  = complex(0.0,  0.0)
    xf.cl01.lvconv  = False


def _extract_forward(xf) -> dict:
    """Extract real-part (forward) scalar results."""
    cl   = float(np.real(xf.cr09.cl))
    cd   = float(np.real(xf.cr09.cd))
    cdp  = float(np.real(xf.cr09.cdp))
    cdf  = cd - cdp
    cm   = float(np.real(xf.cr09.cm))
    xtr_top = float(np.real(xf.cr15.xoctr[0]))
    xtr_bot = float(np.real(xf.cr15.xoctr[1]))
    cpmin   = float(np.real(xf.cr09.cpmn))
    conv    = (not bool(xf.cl01.lexitflag)) and bool(xf.cl01.lvconv)
    return dict(cl=cl, cd=cd, cdp=cdp, cdf=cdf, cm=cm,
                xtr_top=xtr_top, xtr_bot=xtr_bot, cpmin=cpmin,
                converged=conv)


def _extract_grad(xf, h: float) -> dict:
    """Extract imaginary-part / h (gradient) scalar results."""
    cl   = float(np.imag(xf.cr09.cl))   / h
    cd   = float(np.imag(xf.cr09.cd))   / h
    cdp  = float(np.imag(xf.cr09.cdp))  / h
    cdf  = (float(np.imag(xf.cr09.cd)) - float(np.imag(xf.cr09.cdp))) / h
    cm   = float(np.imag(xf.cr09.cm))   / h
    xtr_top = float(np.imag(xf.cr15.xoctr[0])) / h
    xtr_bot = float(np.imag(xf.cr15.xoctr[1])) / h
    cpmin   = float(np.imag(xf.cr09.cpmn))      / h
    return dict(cl=cl, cd=cd, cdp=cdp, cdf=cdf, cm=cm,
                xtr_top=xtr_top, xtr_bot=xtr_bot, cpmin=cpmin)


def _extract_bl_forward(xf, is_surface: int) -> dict:
    """Extract forward BL distribution for one surface (1=upper, 2=lower)."""
    is_ = is_surface - 1   # 0-indexed for numpy
    nbl = int(xf.ci05.nbl[is_])
    qinf2 = 2.0 * float(np.real(xf.cr09.qinf)) ** 2
    idx = np.array([int(xf.ci05.ipan[i, is_]) - 1 for i in range(nbl)])
    data = dict(
        i    = np.arange(1, nbl + 1),
        xssi = np.real(xf.cr15.xssi[:nbl, is_]).astype(float),
        x    = np.array([float(np.real(xf.cr05.x[j])) for j in idx]),
        ue   = np.real(xf.cr15.uedg[:nbl, is_]).astype(float),
        thet = np.real(xf.cr15.thet[:nbl, is_]).astype(float),
        dstr = np.real(xf.cr15.dstr[:nbl, is_]).astype(float),
        cf   = np.real(xf.cr15.tau[:nbl, is_]).astype(float) / max(qinf2, 1e-30),
    )
    return data


def _extract_bl_grad(xf, is_surface: int, h: float) -> dict:
    """Extract BL gradient distribution for one surface."""
    is_ = is_surface - 1
    nbl = int(xf.ci05.nbl[is_])
    qinf2 = 2.0 * float(np.real(xf.cr09.qinf)) ** 2
    idx = np.array([int(xf.ci05.ipan[i, is_]) - 1 for i in range(nbl)])
    data = dict(
        i    = np.arange(1, nbl + 1),
        xssi = np.imag(xf.cr15.xssi[:nbl, is_]).astype(float) / h,
        x    = np.array([float(np.imag(xf.cr05.x[j])) for j in idx]) / h,
        ue   = np.imag(xf.cr15.uedg[:nbl, is_]).astype(float) / h,
        thet = np.imag(xf.cr15.thet[:nbl, is_]).astype(float) / h,
        dstr = np.imag(xf.cr15.dstr[:nbl, is_]).astype(float) / h,
        cf   = np.imag(xf.cr15.tau[:nbl, is_]).astype(float) / max(qinf2, 1e-30) / h,
    )
    return data


def run(coords: np.ndarray,
        alpha: float,
        Re: float   = 1e6,
        Ncrit: float = 9.0,
        Mach: float  = 0.0,
        xtp_u: float = 1.0,
        xtp_l: float = 1.0,
        itmax: int   = 100,
        h: float     = _H_DEFAULT,
        grads: list  = None,
        bl: bool     = False,
        gradbl: bool = False,
        gwake: float = _GWAKE_DEFAULT) -> dict:
    """
    Run one complex-step Qfoil/RFOIL solve (CQFOIL physics).

    Parameters
    ----------
    coords : (N,2) array of x,y airfoil coordinates
    alpha  : angle of attack [deg]
    Re     : Reynolds number
    Ncrit  : e^N transition criterion
    Mach   : freestream Mach number
    xtp_u/l: forced transition x/c (1.0 = free transition)
    itmax  : max Newton iterations
    h      : complex step size
    grads  : list of gradient inputs to compute, e.g.
             ['alpha','re','ncrit','mach','xtp_top','xtp_bot']
             ('coords' is not supported — see module docstring)
    bl     : if True, include BL distributions in result
    gradbl : if True and grads non-empty, include BL gradient distributions
    gwake  : Qfoil wake correction factor (GWAKE in BLPAR common block)

    Returns
    -------
    dict with keys:
        forward: {'cl','cd','cdp','cdf','cm','xtr_top','xtr_bot','cpmin','converged'}
        bl_up, bl_lo  (if bl=True): BL arrays for upper/lower surface
        d<key>_d<input>  for each requested gradient
        dbl_up_d<input>, dbl_lo_d<input>  (if gradbl=True)
    """
    if grads is None:
        grads = []
    grads_lower = [g.lower() for g in grads]
    if 'coords' in grads_lower:
        raise NotImplementedError(
            "cqfoil_inmem_wrapper does not support coordinate gradients — "
            "Qfoil's PANGEN requires the real-valued LOAD file path. "
            "Use cfoil_inmem_wrapper or cxfoil_inmem_wrapper for that."
        )

    xf = _load_module()
    xf.blpar.gwake = float(gwake)
    _load_geometry(xf, coords)   # LOAD + PANGEN (once, real-valued)

    # ── forward solve ─────────────────────────────────────────────
    _set_conditions(xf, alpha, Re, Ncrit, Mach, xtp_u, xtp_l, itmax=itmax)
    xf.batch_oper()   # SPECAL + VISCAL (LIPAN/LBLINI reset) + FCPMIN
    result = _extract_forward(xf)

    if bl:
        result['bl_up'] = _extract_bl_forward(xf, 1)
        result['bl_lo'] = _extract_bl_forward(xf, 2)

    # ── scalar input gradients ────────────────────────────────────
    _scalar_inputs = [
        ('alpha',   dict(h_alpha=h)),
        ('re',      dict(h_re=h)),
        ('ncrit',   dict(h_ncrit=h)),
        ('mach',    dict(h_mach=h)),
        ('xtp_top', dict(h_xtop=h)),
        ('xtp_bot', dict(h_xbot=h)),
    ]
    for inp_name, h_kwargs in _scalar_inputs:
        if inp_name in grads_lower:
            _set_conditions(xf, alpha, Re, Ncrit, Mach, xtp_u, xtp_l,
                            itmax=itmax, **h_kwargs)
            xf.batch_oper()
            g = _extract_grad(xf, h)
            for key, val in g.items():
                result[f'd{key}_d{inp_name}'] = val
            if gradbl:
                result[f'dbl_up_d{inp_name}'] = _extract_bl_grad(xf, 1, h)
                result[f'dbl_lo_d{inp_name}'] = _extract_bl_grad(xf, 2, h)

    return result
