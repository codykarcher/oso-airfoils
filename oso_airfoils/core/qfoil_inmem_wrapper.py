"""
qfoil_inmem_wrapper.py  —  in-memory Qfoil wrapper (minimal file I/O).

Uses libqfoil.so (the Qfoil RFOIL-modified XFOIL compiled as a Python f2py
extension).

Geometry loading uses a short-lived temp DAT file so that Qfoil's LOAD→PANGEN
path is used (Qfoil's full PANGEN with IPFAC=5 requires the buffer to be
preprocessed by AREAD/SCALC/SEGSPL/GEOPAR before PANGEN runs).  Panels are
set up only once per airfoil shape.  All aerodynamic outputs are read from
Fortran common blocks — no polar file, no BL file, no stdout file.

The key Qfoil convergence improvement (LIPAN=.FALSE., LBLINI=.FALSE. cold
reset at the start of every VISCAL call) is already present in Qfoil's
xoper.f and is therefore active in every batch_oper() call.

API mirrors qfoil_wrapper.run() exactly.  Key differences:
  • No polar/BL/exec-script temp files — only a short-lived DAT geometry file.
  • Geometry is loaded and panelled once; every alpha reuses those panels.
  • GWAKE is exposed as a first-class parameter.
  • CL mode is not supported (raises NotImplementedError).
"""

import os
import tempfile
import warnings
import numpy as np

from oso_airfoils.core._xfoil_inmem_base import (
    _load_module, _kulfan_to_xy,
    _set_conditions, _extract_results, _extract_bl, _is_converged,
    _build_return, IBX,
)

# Default Qfoil GWAKE parameter
_GWAKE_DEFAULT = 0.40


def _qfoil_load_geometry(xf, coords: np.ndarray):
    """
    Load airfoil coordinates into Qfoil via LOAD (writes a short-lived temp
    DAT file), then runs XFOIL() for INIT + PANGEN.

    This is the only correct path for Qfoil's PANGEN (IPFAC=5): direct
    NB/XB/YB injection without AREAD preprocessing produces NaN panel coords.

    Only called once per airfoil shape, not per alpha.
    """
    # Remove only consecutive duplicate points (e.g., LE (0,0) appearing
    # twice).  Do NOT touch first==last (closed-TE airfoil is valid).
    mask = np.ones(len(coords), dtype=bool)
    for i in range(1, len(coords)):
        if (abs(coords[i, 0] - coords[i-1, 0]) < 1e-12 and
                abs(coords[i, 1] - coords[i-1, 1]) < 1e-12):
            mask[i] = False
    coords = coords[mask]

    # Write to a temp DAT file (short-lived — deleted immediately after LOAD)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.dat',
                                      prefix='qfoil_', delete=False)
    tmp.write('airfoil\n')
    for x, y in coords:
        tmp.write(f'  {x:.10f}  {y:.10f}\n')
    tmp.close()

    try:
        itype = np.array(1, dtype=np.int32)
        xf.load(tmp.name, itype)   # AREAD + SCALC + SEGSPL + GEOPAR
    finally:
        os.unlink(tmp.name)

    # Panel setup: INIT (sets PI, DTOR, NPAN=160) + ABCOPY + PANGEN
    xf.xfoil()


def run(mode,
        upperKulfanCoefficients,
        lowerKulfanCoefficients,
        val=0.0,
        Re=1e7,
        M=0.0,
        xtp_u=1.0,
        xtp_l=1.0,
        N_crit=9.0,
        N_panels_kulfan=100,
        TE_gap=0.0,
        timelimit=None,
        max_iter=100,
        save_boundary_layer_data=False,
        force_list=False,
        gwake=_GWAKE_DEFAULT,
        **kwargs):
    """
    In-memory Qfoil solve via libqfoil.so.

    Parameters match qfoil_wrapper.run().  Additional parameter:
        gwake : float, default 0.40
            Qfoil wake correction factor (GWAKE in BLPAR common block).
            Set to 0.0 to disable wake correction.
    """
    mode = mode.lower()
    if mode in ('alfa', 'alpha'):
        mode = 'alpha'
    else:
        raise NotImplementedError(
            "qfoil_inmem_wrapper only supports mode='alpha'."
        )

    # ── alpha list ────────────────────────────────────────────────────
    try:
        iter(val)
        is_iterable = True
    except TypeError:
        is_iterable = False

    if is_iterable:
        val = list(val)
        _is_triplet = (len(val) == 3 and not force_list)
        if _is_triplet:
            start, stop, step = val
            n = int(round((stop - start) / step)) + 1
            alpha_list = [start + i * step for i in range(n)]
        else:
            alpha_list = [float(v) for v in val]
    else:
        alpha_list = [float(val)]
        is_iterable = False

    # ── geometry (load once, panel once) ──────────────────────────────
    coords = _kulfan_to_xy(upperKulfanCoefficients, lowerKulfanCoefficients,
                           n_pts=N_panels_kulfan, te_gap=TE_gap)

    xf = _load_module('libqfoil', 'qfoil_lib')   # independent qfoil_lib package
    xf.blpar.gwake = float(gwake)
    _qfoil_load_geometry(xf, coords)   # LOAD + PANGEN (once)

    # ── solve each alpha (no re-panel) ────────────────────────────────
    results = []
    for alpha_deg in alpha_list:
        _set_conditions(xf, alpha_deg, Re, N_crit, M,
                        xtp_u, xtp_l, itmax=max_iter)
        xf.batch_oper()    # SPECAL + VISCAL (LIPAN/LBLINI reset) + FCPMIN

        conv   = _is_converged(xf)
        result = _extract_results(xf, conv)
        if save_boundary_layer_data and conv:
            result['bl_data'] = _extract_bl(xf)
        else:
            result['bl_data'] = None
        results.append(result)

    return _build_return(results, alpha_list, is_iterable,
                         xtp_u, xtp_l, Re, M, N_crit, N_panels_kulfan)


def run_sweep(upperKulfanCoefficients, lowerKulfanCoefficients,
              alpha_start, alpha_stop, alpha_step, **kwargs):
    """Convenience wrapper for an alpha sweep."""
    return run('alpha', upperKulfanCoefficients, lowerKulfanCoefficients,
               val=[alpha_start, alpha_stop, alpha_step], **kwargs)


if __name__ == '__main__':
    import sys
