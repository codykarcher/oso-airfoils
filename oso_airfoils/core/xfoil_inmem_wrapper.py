"""
xfoil_inmem_wrapper.py  —  in-memory XFOIL wrapper (no file I/O).

Uses libcmplxfoil.so (the real XFOIL compiled as a Python f2py extension)
instead of the subprocess + temp-file approach in xfoil_wrapper.py.

API mirrors xfoil_wrapper.run() exactly.  Key differences:
  • Zero temp files for the aerodynamic solve.
  • Significantly lower overhead per call (~10× faster for sweeps).
  • CL mode is not supported (raises NotImplementedError).
  • Flap deflections are not supported.
  • cp_data is always None (Cp distributions not extracted).
"""

import warnings
import numpy as np

from oso_airfoils.core._xfoil_inmem_base import (
    _load_module, _kulfan_to_xy, _set_geometry,
    _set_conditions, _extract_results, _extract_bl, _is_converged,
    run_single, _build_return,
)


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
        timelimit=None,    # ignored — in-memory solve has no process timeout
        max_iter=100,
        save_boundary_layer_data=False,
        force_list=False,
        **kwargs):          # absorbs unused xfoil_wrapper args silently
    """
    In-memory XFOIL solve via libcmplxfoil.so.

    Parameters match xfoil_wrapper.run().  Unsupported args (N_panels_xfoil,
    flapLocation, flapDeflection, path_to_XFOIL, stdout_log_path, etc.) are
    silently ignored.

    Returns
    -------
    dict with same keys as xfoil_wrapper.run():
        cl, cd, cdp, cdf, cm, cpmin, xtr_top, xtr_bot,
        alpha, xtp_top, xtp_bot, Re, M, N_crit,
        N_panels_kulfan, converged, cp_data, bl_data
    """
    mode = mode.lower()
    if mode in ('alfa', 'alpha'):
        mode = 'alpha'
    else:
        raise NotImplementedError(
            "xfoil_inmem_wrapper only supports mode='alpha'. "
            "CL mode is not implemented."
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

    # ── geometry ──────────────────────────────────────────────────────
    coords = _kulfan_to_xy(upperKulfanCoefficients, lowerKulfanCoefficients,
                           n_pts=N_panels_kulfan, te_gap=TE_gap)

    # ── load a fresh module instance ──────────────────────────────────
    xf = _load_module('libcmplxfoil')
    # Note: init() is called internally by xfoil() during _set_geometry

    # ── run each alpha point ──────────────────────────────────────────
    results = []
    for alpha_deg in alpha_list:
        r = run_single(xf, coords, alpha_deg, Re, N_crit, M,
                       xtp_u, xtp_l, itmax=max_iter,
                       get_bl=save_boundary_layer_data)
        results.append(r)

    return _build_return(results, alpha_list, is_iterable,
                         xtp_u, xtp_l, Re, M, N_crit, N_panels_kulfan)


def run_sweep(upperKulfanCoefficients, lowerKulfanCoefficients,
              alpha_start, alpha_stop, alpha_step, **kwargs):
    """Convenience wrapper for an alpha sweep."""
    return run('alpha', upperKulfanCoefficients, lowerKulfanCoefficients,
               val=[alpha_start, alpha_stop, alpha_step], **kwargs)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))

    upper = np.array([0.2, 0.3, 0.35, 0.25, 0.15, 0.05])
    lower = np.array([-0.1, -0.15, -0.1, -0.05, 0.0, 0.05])

    print("── scalar, no BL ─────────────────────────────────────────────")
    r = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0)
    print(f"  CL={r['cl']:.4f}  CD={r['cd']:.5f}  conv={r['converged']}")

    print("\n── sweep [-2, 8, 2] ──────────────────────────────────────────")
    r2 = run_sweep(upper, lower, -2.0, 8.0, 2.0, Re=1e6, N_crit=9.0)
    for a, cl, cd in zip(r2['alpha'], r2['cl'], r2['cd']):
        print(f"  α={a:5.1f}°  CL={cl:.4f}  CD={cd:.5f}")

    print("\n── scalar, with BL ───────────────────────────────────────────")
    r3 = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0,
             save_boundary_layer_data=True)
    n = len(r3['bl_data']['upper']['x'])
    print(f"  BL upper: {n} stations, Cf[5]={r3['bl_data']['upper']['Cf'][5]:.4e}")
