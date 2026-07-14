"""
_xfoil_inmem_base.py  —  shared helpers for in-memory XFOIL-family wrappers.

Not intended for direct import by users.  Used by:
  xfoil_inmem_wrapper.py  (libcmplxfoil.so  — real XFOIL)
  qfoil_inmem_wrapper.py  (libqfoil.so      — RFOIL/Qfoil)
  cfoil_inmem_wrapper.py  (libcmplxfoil_cs.so — complex-step XFOIL)
"""
import sys
import os
import pathlib
import numpy as np

# ── MExt import ────────────────────────────────────────────────────────────
_CMPLXFOIL_PATH = str(
    pathlib.Path(__file__).parent.parent.parent / 'CMPLXFOIL'
)
if _CMPLXFOIL_PATH not in sys.path:
    sys.path.insert(0, _CMPLXFOIL_PATH)

# qfoil_lib lives at the oso-airfoils repo root
_OSO_AIRFOILS_PATH = str(pathlib.Path(__file__).parent.parent.parent)
if _OSO_AIRFOILS_PATH not in sys.path:
    sys.path.insert(0, _OSO_AIRFOILS_PATH)

from cmplxfoil import MExt  # noqa: E402  (after path setup)

# ── IBX constant (max buffer-airfoil points in XFOIL common block) ─────────
IBX = 572
IVX = 229
ISX = 2


def _load_module(lib_name: str, package_name: str = 'cmplxfoil'):
    """Load a copy of lib_name.so from package_name via MExt."""
    return MExt.MExt(lib_name, package_name)._module

def _kulfan_to_xy(upper_coeffs, lower_coeffs, n_pts: int = 100,
                  te_gap: float = 0.0):
    """
    Convert Kulfan CST coefficients → (x, y) numpy arrays.

    Returns a (N, 2) float64 array of airfoil coordinates in the same
    format as Kulfan.coordinates (upper surface TE→LE→ lower surface TE).
    """
    from metafoil.core.kulfan import Kulfan  # lazy import
    afl = Kulfan(TE_gap=te_gap)
    afl.upperCoefficients = upper_coeffs
    afl.lowerCoefficients = lower_coeffs
    if n_pts is not None:
        afl.utility.Npoints = n_pts
    return np.array(afl.coordinates, dtype=np.float64)


def _set_geometry(xf, coords: np.ndarray):
    """
    Load airfoil coordinates into NB/XB/YB common blocks and call
    XFOIL() to run PANGEN (panel setup).  Works for both real and
    complex-step modules.

    Parameters
    ----------
    xf  : loaded f2py module (libcmplxfoil, libqfoil, or libcmplxfoil_cs)
    coords : (N, 2) float64 array  [x, y]
    """
    # Remove duplicate consecutive points (Qfoil's SEGSPL is stricter)
    mask = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords)):
        if (abs(coords[i, 0] - coords[i-1, 0]) < 1e-12 and
                abs(coords[i, 1] - coords[i-1, 1]) < 1e-12):
            mask[i] = False
    coords = coords[mask]
    # Note: do NOT remove first==last (closed airfoil TE); Qfoil's PANGEN
    # handles that correctly. Only consecutive duplicates need removal.

    nb = len(coords)
    x_full = np.zeros(IBX, dtype=np.float64)
    y_full = np.zeros(IBX, dtype=np.float64)
    x_full[:nb] = coords[:, 0]
    y_full[:nb] = coords[:, 1]

    xf.ci04.nb = nb
    xf.cr14.xb = x_full
    xf.cr14.yb = y_full
    xf.xfoil()          # runs PANGEN — sets up panel geometry


def _set_conditions(xf, alpha_deg: float, Re: float, Ncrit: float,
                    Mach: float, xtp_u: float, xtp_l: float,
                    itmax: int = 100):
    """Set operating conditions in XFOIL common blocks."""
    xf.cr09.adeg   = alpha_deg
    xf.cr15.reinf1 = Re
    # ACRIT may be scalar (CMPLXFOIL/xfoil_inmem) or array(2) (Qfoil)
    try:
        xf.cr15.acrit  = [Ncrit, Ncrit]   # Qfoil: ACRIT(ISX)
    except (TypeError, ValueError):
        xf.cr15.acrit  = Ncrit             # CMPLXFOIL: scalar ACRIT
    xf.cr09.minf1  = Mach
    xf.cr15.xstrip = [xtp_u, xtp_l]
    xf.ci04.itmax  = itmax
    xf.cr09.xcmref = 0.25   # standard moment reference
    xf.cr09.ycmref = 0.0
    xf.cl01.lvconv = False   # force re-solve
    try:
        xf.cl01.printconv = False   # suppress convergence output
    except Exception:
        pass


def _extract_results(xf, converged: bool) -> dict:
    """
    Extract scalar aerodynamic results from XFOIL common blocks.

    Returns a dict with keys: cl, cd, cdp, cdf, cm, cpmin,
    xtr_top, xtr_bot, converged.
    """
    nan = float('nan')
    if not converged:
        return dict(cl=nan, cd=nan, cdp=nan, cdf=nan, cm=nan,
                    cpmin=nan, xtr_top=nan, xtr_bot=nan,
                    converged=False)

    cl   = float(xf.cr09.cl)
    cd   = float(xf.cr09.cd)
    cdp  = float(xf.cr09.cdp)
    cdf  = cd - cdp
    cm   = float(xf.cr09.cm)
    cpmin = float(xf.cr09.cpmn)
    xtr_top = float(xf.cr15.xoctr[0])
    xtr_bot = float(xf.cr15.xoctr[1])
    return dict(cl=cl, cd=cd, cdp=cdp, cdf=cdf, cm=cm,
                cpmin=cpmin, xtr_top=xtr_top, xtr_bot=xtr_bot,
                converged=True)


def _extract_bl(xf) -> dict:
    """
    Extract boundary-layer distributions from XFOIL common blocks.

    Returns {'upper': dict, 'lower': dict} where each sub-dict has keys
    matching xfoil_wrapper bl_data:  s, x, Ue/Vinf, Dstar, Theta, Cf, H.
    """
    result = {}
    qinf_sq = float(xf.cr09.qinf) ** 2
    if qinf_sq == 0.0:
        qinf_sq = 1.0

    nbl  = xf.ci05.nbl      # shape (2,)  — number of BL points per surface
    ipan = xf.ci05.ipan     # shape (IVX, 2)  — panel index for each BL pt
    x_panel = xf.cr05.x     # shape (IZX,)

    for k, label in enumerate(['upper', 'lower']):
        n = int(nbl[k])
        if n <= 0:
            result[label] = None
            continue

        s    = xf.cr15.xssi[:n, k].copy()
        ue   = xf.cr15.uedg[:n, k].copy()
        thet = xf.cr15.thet[:n, k].copy()
        dstr = xf.cr15.dstr[:n, k].copy()
        tau  = xf.cr15.tau[:n, k].copy()
        cf   = tau / (0.5 * qinf_sq)
        # x via panel index mapping
        idx  = xf.ci05.ipan[:n, k].astype(int) - 1   # 0-based
        x    = x_panel[idx].copy()
        h    = np.where(thet > 0, dstr / thet, np.nan)

        result[label] = {
            's':       list(s.astype(float)),
            'x':       list(x.astype(float)),
            'Ue/Vinf': list(ue.astype(float)),
            'Dstar':   list(dstr.astype(float)),
            'Theta':   list(thet.astype(float)),
            'Cf':      list(cf.astype(float)),
            'H':       list(h.astype(float)),
        }
    return result


def _is_converged(xf) -> bool:
    """Check XFOIL convergence flags."""
    return (not bool(xf.cl01.lexitflag)) and bool(xf.cl01.lvconv)


def run_single(xf, coords: np.ndarray, alpha_deg: float, Re: float,
               Ncrit: float, Mach: float, xtp_u: float, xtp_l: float,
               itmax: int = 100, get_bl: bool = False,
               oper_name: str = 'oper') -> dict:
    """
    Run one XFOIL solve (forward, real arithmetic).

    Parameters
    ----------
    xf      : loaded f2py module
    coords  : (N, 2) airfoil coordinates
    ...     : operating conditions

    Returns
    -------
    dict with keys: cl, cd, cdp, cdf, cm, cpmin, xtr_top, xtr_bot,
                    converged, bl_data (if get_bl=True)
    """
    _set_geometry(xf, coords)
    _set_conditions(xf, alpha_deg, Re, Ncrit, Mach, xtp_u, xtp_l, itmax)
    getattr(xf, oper_name)()    # call 'oper' or 'batch_oper'
    conv   = _is_converged(xf)
    result = _extract_results(xf, conv)
    if get_bl and conv:
        result['bl_data'] = _extract_bl(xf)
    else:
        result['bl_data'] = None
    return result


def _build_return(results_list: list, alpha_list: list,
                  is_iterable: bool, xtp_u: float, xtp_l: float,
                  Re: float, Mach: float, Ncrit: float,
                  n_pts: int) -> dict:
    """Assemble the final return dict from a list of single-point results."""
    def _extract(key):
        vals = [r[key] for r in results_list]
        return vals[0] if not is_iterable else np.array(vals)

    res = {
        'cl':        _extract('cl'),
        'cd':        _extract('cd'),
        'cdp':       _extract('cdp'),
        'cdf':       _extract('cdf'),
        'cm':        _extract('cm'),
        'cpmin':     _extract('cpmin'),
        'xtr_top':   _extract('xtr_top'),
        'xtr_bot':   _extract('xtr_bot'),
        'alpha':     alpha_list[0] if not is_iterable else np.array(alpha_list),
        'xtp_top':   xtp_u,
        'xtp_bot':   xtp_l,
        'Re':        Re,
        'M':         Mach,
        'N_crit':    Ncrit,
        'N_panels_kulfan': n_pts,
        'converged': results_list[0]['converged'] if not is_iterable
                     else [r['converged'] for r in results_list],
        'cp_data':   None,   # not supported
        'bl_data':   results_list[0]['bl_data'] if not is_iterable
                     else [r['bl_data'] for r in results_list],
    }
    return res
