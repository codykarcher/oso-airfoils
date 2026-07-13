"""
cfoil_inmem_wrapper.py  —  in-memory complex-step XFOIL wrapper (no file I/O).

Uses libcmplxfoil_cs.so (XFOIL with complex arithmetic) to compute aerodynamic
outputs AND their exact gradients w.r.t. any input via complex-step
differentiation — all in-process, without any subprocesses or files.

API mirrors cfoil_wrapper.run() but eliminates subprocess overhead:
  • Zero files at any step.
  • Each gradient costs exactly one Fortran OPER call (~5–10 ms).
  • BL gradients are array-valued (Ue, θ, δ*, Cf distributions).

Limitations:
  • CL mode not supported.
  • Coordinate gradients for large airfoils are slow (N×2 OPER calls);
    use cfoil_wrapper for that case.
"""

import numpy as np

from oso_airfoils.core._xfoil_inmem_base import (
    _load_module, _kulfan_to_xy, _set_geometry, _is_converged,
    _extract_results, _extract_bl, _build_return,
)

# Complex-step perturbation size
_H_DEFAULT = 1.0e-20

# All differentiable scalar inputs
_ALL_GRAD_INPUTS = ['alpha', 're', 'ncrit', 'mach', 'xtp_top', 'xtp_bot']

# Scalar output names (matching cfoil_wrapper)
_SCALAR_OUTS = ['cl', 'cd', 'cdp', 'cdf', 'cm',
                'xtr_top', 'xtr_bot', 'cpmin']


# ── low-level: one complex-step solve ─────────────────────────────────────
def _cs_solve(xf_cs, coords, alpha, Re, Ncrit, Mach, xtp_u, xtp_l,
              itmax, h, perturb_input):
    """
    Run one XFOIL solve with the complex-step perturbation on `perturb_input`.

    perturb_input : str  one of _ALL_GRAD_INPUTS, or None for forward solve
    h             : float  complex step size

    Returns (fwd_dict, grad_dict, bl_fwd, bl_grad) where bl_* are None
    unless bl is requested at call site.
    """
    # Set geometry (real, h=0 for coordinates)
    from oso_airfoils.core._xfoil_inmem_base import IBX
    nb = len(coords)
    x_full = np.zeros(IBX)
    y_full = np.zeros(IBX)
    x_full[:nb] = coords[:, 0]
    y_full[:nb] = coords[:, 1]
    xf_cs.ci04.nb = nb
    xf_cs.cr14.xb = x_full
    xf_cs.cr14.yb = y_full
    xf_cs.xfoil()

    # Set operating conditions with perturbation
    a_  = complex(alpha, h if perturb_input == 'alpha'   else 0.0)
    re_ = complex(Re,    h if perturb_input == 're'      else 0.0)
    nc_ = complex(Ncrit, h if perturb_input == 'ncrit'   else 0.0)
    ma_ = complex(Mach,  h if perturb_input == 'mach'    else 0.0)
    xt_ = complex(xtp_u, h if perturb_input == 'xtp_top' else 0.0)
    xb_ = complex(xtp_l, h if perturb_input == 'xtp_bot' else 0.0)

    xf_cs.cr09.adeg   = a_
    xf_cs.cr15.reinf1 = re_
    xf_cs.cr15.acrit  = nc_
    xf_cs.cr09.minf1  = ma_
    xf_cs.cr15.xstrip = [xt_, xb_]
    xf_cs.ci04.itmax  = itmax
    xf_cs.cl01.lvconv = False
    xf_cs.oper()

    conv = _is_converged(xf_cs)
    nan  = float('nan')

    def _re(z): return float(z.real) if hasattr(z, 'real') else float(z)
    def _im(z): return float(z.imag) / h if hasattr(z, 'imag') else 0.0

    if not conv:
        fwd  = {k: nan for k in _SCALAR_OUTS}
        fwd['converged'] = False
        grad = {k: nan for k in _SCALAR_OUTS} if perturb_input else None
        return fwd, grad

    cl   = xf_cs.cr09.cl
    cd   = xf_cs.cr09.cd
    cdp  = xf_cs.cr09.cdp
    cm   = xf_cs.cr09.cm
    cpmin = xf_cs.cr09.cpmn
    xot  = xf_cs.cr15.xoctr[0]
    xob  = xf_cs.cr15.xoctr[1]

    fwd = dict(
        cl=_re(cl), cd=_re(cd), cdp=_re(cdp), cdf=_re(cd)-_re(cdp),
        cm=_re(cm), cpmin=_re(cpmin),
        xtr_top=_re(xot), xtr_bot=_re(xob),
        converged=conv,
    )

    grad = None
    if perturb_input is not None:
        grad = dict(
            cl=_im(cl), cd=_im(cd), cdp=_im(cdp),
            cdf=_im(cd)-_im(cdp),
            cm=_im(cm), cpmin=_im(cpmin),
            xtr_top=_im(xot), xtr_bot=_im(xob),
        )

    return fwd, grad


def _extract_bl_cs(xf_cs, h, is_grad: bool) -> dict:
    """Extract BL distributions from the complex-step module."""
    qinf = complex(xf_cs.cr09.qinf)
    qinf_sq = abs(qinf) ** 2 or 1.0
    result = {}
    nbl  = xf_cs.ci05.nbl
    ipan = xf_cs.ci05.ipan
    x_panel = xf_cs.cr05.x

    for k, label in enumerate(['upper', 'lower']):
        n = int(nbl[k].real) if hasattr(nbl[k], 'real') else int(nbl[k])
        if n <= 0:
            result[label] = None
            continue

        def _v(arr): return np.array([complex(v) for v in arr[:n, k]])
        s    = _v(xf_cs.cr15.xssi)
        ue   = _v(xf_cs.cr15.uedg)
        thet = _v(xf_cs.cr15.thet)
        dstr = _v(xf_cs.cr15.dstr)
        tau  = _v(xf_cs.cr15.tau)
        cf   = tau / (0.5 * qinf_sq)
        idx  = np.array([int(v.real) if hasattr(v,'real') else int(v)
                         for v in ipan[:n, k]]) - 1
        x    = np.array([complex(x_panel[i]) for i in idx])
        h_bl = np.where(thet.real > 0, dstr / thet, complex(float('nan')))

        if not is_grad:
            result[label] = {
                's':       list(s.real),
                'x':       list(x.real),
                'Ue/Vinf': list(ue.real),
                'Dstar':   list(dstr.real),
                'Theta':   list(thet.real),
                'Cf':      list(cf.real),
                'H':       list(h_bl.real),
            }
        else:
            result[label] = {
                's':       list(s.imag / h),
                'x':       list(x.imag / h),
                'Ue/Vinf': list(ue.imag / h),
                'Dstar':   list(dstr.imag / h),
                'Theta':   list(thet.imag / h),
                'Cf':      list(cf.imag / h),
            }
    return result


# ── public run() ──────────────────────────────────────────────────────────
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
        compute_gradients=False,
        gradient_inputs=None,
        h_cs=_H_DEFAULT,
        force_list=False,
        **kwargs):
    """
    In-memory complex-step XFOIL solve via libcmplxfoil_cs.so.

    Parameters match cfoil_wrapper.run().  Returns the same dict keys.
    When compute_gradients=True, also returns:
        grad     : dict[input_name → dict[output_name → float]]
        bl_grad  : dict[input_name → dict['upper'/'lower' → BL dict]]
                   (only when save_boundary_layer_data=True)
    """
    mode = mode.lower()
    if mode in ('alfa', 'alpha'):
        mode = 'alpha'
    else:
        raise NotImplementedError("cfoil_inmem_wrapper only supports mode='alpha'.")

    if gradient_inputs is None:
        gradient_inputs = _ALL_GRAD_INPUTS

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

    # ── load complex-step module ──────────────────────────────────────
    xf_cs = _load_module('libcmplxfoil_cs')
    # Note: init() is called internally by xfoil() during _cs_solve

    # ── run each alpha ─────────────────────────────────────────────────
    all_fwd   = []
    all_grad  = []    # list of {input: {output: float}}
    all_bl_fwd  = []
    all_bl_grad = []  # list of {input: {surface: BL dict}}

    for alpha_deg in alpha_list:
        # Forward solve (h=0 on all inputs)
        fwd, _ = _cs_solve(xf_cs, coords, alpha_deg, Re, N_crit, M,
                            xtp_u, xtp_l, max_iter, h_cs, None)
        bl_fwd = None
        if save_boundary_layer_data and fwd['converged']:
            # Re-run with h=0 to get clean real BL data
            _cs_solve(xf_cs, coords, alpha_deg, Re, N_crit, M,
                      xtp_u, xtp_l, max_iter, h_cs, None)
            bl_fwd = _extract_bl_cs(xf_cs, h_cs, is_grad=False)
        fwd['bl_data'] = bl_fwd
        all_fwd.append(fwd)

        # Gradient solves
        point_grad = {}
        point_bl_grad = {}
        if compute_gradients:
            for inp in gradient_inputs:
                _, g = _cs_solve(xf_cs, coords, alpha_deg, Re, N_crit, M,
                                  xtp_u, xtp_l, max_iter, h_cs, inp)
                point_grad[inp] = g
                if save_boundary_layer_data and fwd['converged']:
                    point_bl_grad[inp] = _extract_bl_cs(xf_cs, h_cs, is_grad=True)
        all_grad.append(point_grad)
        all_bl_grad.append(point_bl_grad)

    # ── assemble result ──────────────────────────────────────────────
    res = _build_return(all_fwd, alpha_list, is_iterable,
                        xtp_u, xtp_l, Re, M, N_crit, N_panels_kulfan)

    if compute_gradients:
        combined_grad = {inp: {} for inp in gradient_inputs}
        for inp in gradient_inputs:
            for out in _SCALAR_OUTS:
                vals = [pt[inp][out] if pt.get(inp) else float('nan')
                        for pt in all_grad]
                combined_grad[inp][out] = vals[0] if not is_iterable \
                                          else np.array(vals)
        res['grad'] = combined_grad

        if save_boundary_layer_data:
            combined_bl_grad = {}
            for inp in gradient_inputs:
                combined_bl_grad[inp] = (all_bl_grad[0].get(inp)
                                         if not is_iterable
                                         else [pt.get(inp) for pt in all_bl_grad])
            res['bl_grad'] = combined_bl_grad

    return res


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

    print("── forward only ──────────────────────────────────────────────")
    r = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0)
    print(f"  CL={r['cl']:.4f}  CD={r['cd']:.5f}  conv={r['converged']}")

    print("\n── forward + all scalar gradients ────────────────────────────")
    r2 = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0,
             compute_gradients=True)
    print(f"  CL={r2['cl']:.4f}  CD={r2['cd']:.5f}")
    print(f"  dCL/dalpha = {r2['grad']['alpha']['cl']:+.5f}")
    print(f"  dCD/dalpha = {r2['grad']['alpha']['cd']:+.5f}")
    print(f"  dCL/dre    = {r2['grad']['re']['cl']:+.5e}")
    print(f"  dCL/dncrit = {r2['grad']['ncrit']['cl']:+.5f}")

    print("\n── sweep [-2, 8, 2] with gradients ───────────────────────────")
    r3 = run_sweep(upper, lower, -2.0, 8.0, 2.0, Re=1e6, N_crit=9.0,
                   compute_gradients=True, gradient_inputs=['alpha'])
    for a, cl, dclda in zip(r3['alpha'], r3['cl'], r3['grad']['alpha']['cl']):
        print(f"  α={a:5.1f}°  CL={cl:.4f}  dCL/dα={dclda:+.4f}")
