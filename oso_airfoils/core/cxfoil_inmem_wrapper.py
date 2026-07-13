"""
cxfoil_inmem_wrapper.py  —  in-memory CXFOIL wrapper (complex-step gradients).

Uses libcxfoil.so (CMPLXFOIL src_cs/ complex-step XFOIL compiled as f2py
extension) to compute CL, CD and all their gradients via complex-step
differentiation.  No file I/O per solve.

Standard XFOIL (non-RFOIL) boundary layer model.
For Qfoil RFOIL physics use cqfoil_inmem_wrapper.

API mirrors cfoil_wrapper (subprocess version) but is significantly faster.
"""

import os
import sys
import numpy as np

# Ensure the repo root is on sys.path so cxfoil_lib is findable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cmplxfoil.MExt import MExt

# Complex step size
_H_DEFAULT = 1.0e-20

# Scalar output names (index → name mapping)
_SCALAR_KEYS = ['cl', 'cd', 'cdp', 'cdf', 'cm', 'xtr_top', 'xtr_bot', 'cpmin']
_N_SCALAR    = len(_SCALAR_KEYS)   # 8


def _load_module():
    """Load a unique libcxfoil instance via MExt."""
    return MExt('libcxfoil', 'cxfoil_lib')._module


def _set_geometry(xf, coords: np.ndarray):
    """
    Write airfoil coordinates directly into XFOIL common blocks,
    then run XFOIL() (INIT + ABCOPY + PANGEN).
    """
    n = len(coords)
    xf.ci02.nb = n
    xb = xf.cr14.xb
    yb = xf.cr14.yb
    for i in range(n):
        xb[i] = complex(coords[i, 0], 0.0)
        yb[i] = complex(coords[i, 1], 0.0)
    xf.cr14.xb = xb
    xf.cr14.yb = yb
    xf.xfoil()   # INIT + ABCOPY + PANGEN


def _set_conditions(xf, alpha_deg: float, Re: float, Ncrit: float,
                    Mach: float, xtp_u: float, xtp_l: float,
                    h_alpha=0.0, h_re=0.0, h_ncrit=0.0,
                    h_mach=0.0, h_xtop=0.0, h_xbot=0.0,
                    itmax: int = 100):
    """Set operating conditions with optional imaginary perturbations."""
    xf.cr09.adeg    = complex(alpha_deg, h_alpha)
    xf.cr15.reinf1  = complex(Re,        h_re)
    xf.cr15.acrit   = complex(Ncrit,     h_ncrit)
    xf.cr09.minf1   = complex(Mach,      h_mach)
    xf.cr15.xstrip  = [complex(xtp_u, h_xtop), complex(xtp_l, h_xbot)]
    xf.ci04.itmax   = itmax
    xf.cr09.xcmref  = complex(0.25, 0.0)
    xf.cr09.ycmref  = complex(0.0,  0.0)
    xf.cl01.lvconv  = False
    xf.cl01.lblini  = False
    xf.cl01.lipan   = False
    xf.cl01.lvisc   = True
    xf.cl01.lalfa   = True


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
    conv    = bool(xf.cl01.lvconv)
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
    nbl = int(xf.ci02.nbl[is_])
    qinf2 = 2.0 * float(np.real(xf.cr09.qinf)) ** 2
    data = dict(
        i    = np.arange(1, nbl + 1),
        xssi = np.real(xf.cr15.xssi[:nbl, is_]).astype(float),
        x    = np.array([float(np.real(xf.cr05.x[int(xf.ci02.ipan[i, is_]) - 1]))
                         for i in range(nbl)]),
        ue   = np.real(xf.cr15.uedg[:nbl, is_]).astype(float),
        thet = np.real(xf.cr15.thet[:nbl, is_]).astype(float),
        dstr = np.real(xf.cr15.dstr[:nbl, is_]).astype(float),
        cf   = np.real(xf.cr15.tau[:nbl, is_]).astype(float) / max(qinf2, 1e-30),
    )
    return data


def _extract_bl_grad(xf, is_surface: int, h: float) -> dict:
    """Extract BL gradient distribution for one surface."""
    is_ = is_surface - 1
    nbl = int(xf.ci02.nbl[is_])
    qinf2 = 2.0 * float(np.real(xf.cr09.qinf)) ** 2
    data = dict(
        i    = np.arange(1, nbl + 1),
        xssi = np.imag(xf.cr15.xssi[:nbl, is_]).astype(float) / h,
        x    = np.array([float(np.imag(xf.cr05.x[int(xf.ci02.ipan[i, is_]) - 1]))
                         for i in range(nbl)]) / h,
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
        gradbl: bool = False) -> dict:
    """
    Run one complex-step XFOIL solve (CXFOIL physics).

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
             ['alpha','re','ncrit','mach','xtp_top','xtp_bot','coords']
    bl     : if True, include BL distributions in result
    gradbl : if True and grads non-empty, include BL gradient distributions

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

    xf = _load_module()
    _set_geometry(xf, coords)

    # ── forward solve ─────────────────────────────────────────────
    _set_conditions(xf, alpha, Re, Ncrit, Mach, xtp_u, xtp_l, itmax=itmax)
    xf.oper()
    result = _extract_forward(xf)
    result['converged'] = bool(xf.cl01.lvconv)

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
            xf.oper()
            g = _extract_grad(xf, h)
            for key, val in g.items():
                result[f'd{key}_d{inp_name}'] = val
            if gradbl:
                result[f'dbl_up_d{inp_name}'] = _extract_bl_grad(xf, 1, h)
                result[f'dbl_lo_d{inp_name}'] = _extract_bl_grad(xf, 2, h)

    # ── coordinate gradients ──────────────────────────────────────
    if 'coords' in grads_lower:
        n_coords = len(coords)
        coords_base = coords.copy()
        dout_dx = {k: np.zeros(n_coords) for k in _SCALAR_KEYS}
        dout_dy = {k: np.zeros(n_coords) for k in _SCALAR_KEYS}

        for i in range(n_coords):
            # Perturb x_i
            xf.ci02.nb = n_coords
            xb = np.array([complex(coords_base[j, 0], h if j == i else 0.0)
                           for j in range(n_coords)])
            yb = np.array([complex(coords_base[j, 1], 0.0)
                           for j in range(n_coords)])
            xf.cr14.xb[:n_coords] = xb
            xf.cr14.yb[:n_coords] = yb
            xf.xfoil()
            _set_conditions(xf, alpha, Re, Ncrit, Mach, xtp_u, xtp_l, itmax=itmax)
            xf.oper()
            g = _extract_grad(xf, h)
            for key in _SCALAR_KEYS:
                dout_dx[key][i] = g[key]

            # Perturb y_i
            xb = np.array([complex(coords_base[j, 0], 0.0)
                           for j in range(n_coords)])
            yb = np.array([complex(coords_base[j, 1], h if j == i else 0.0)
                           for j in range(n_coords)])
            xf.cr14.xb[:n_coords] = xb
            xf.cr14.yb[:n_coords] = yb
            xf.xfoil()
            _set_conditions(xf, alpha, Re, Ncrit, Mach, xtp_u, xtp_l, itmax=itmax)
            xf.oper()
            g = _extract_grad(xf, h)
            for key in _SCALAR_KEYS:
                dout_dy[key][i] = g[key]

        for key in _SCALAR_KEYS:
            result[f'd{key}_dx_coords'] = dout_dx[key]
            result[f'd{key}_dy_coords'] = dout_dy[key]

    return result
