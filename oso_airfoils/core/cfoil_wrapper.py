"""
cfoil_wrapper.py  —  Python wrapper for the cfoil complex-step executable.

API mirrors xfoil_wrapper.run() and qfoil_wrapper.run() so it can be used as
a drop-in for most single-point and sweep calls.

Key additions over xfoil/qfoil wrappers:
  • Returns 'cdf' (friction drag) as a first-class output.
  • Optional `compute_gradients` flag returns dOutputs/dInputs via complex step.
  • BL distributions are returned in the same format as xfoil/qfoil 'bl_data'.
  • Zero temp files for the aerodynamic solve; geometry still uses a temp DAT
    file (same as the other wrappers) until a full in-memory port is done.

Limitations compared to xfoil/qfoil wrappers:
  • CL mode is not currently supported (raises NotImplementedError).
  • Flap deflections are not supported.
  • Panel count cannot be specified via ppar; use N_panels_kulfan instead.
  • Alpha sweeps call cfoil once per point (no native aseq); performance is
    dominated by subprocess startup for large sweeps.
"""

import subprocess
import warnings
import tempfile
import os
import sys
import shutil
import pathlib
import platform
import time

import numpy as np
import pandas as pd
from oso_airfoils.geometry.kulfan import Kulfan

path_to_here = pathlib.Path(__file__).parent.resolve()

# ── locate cfoil binary ────────────────────────────────────────────────────
_CFOIL_DEFAULT = str(
    path_to_here.parent.parent / 'CMPLXFOIL' / 'cfoil' / 'cfoil'
)


# ── scalar output names ────────────────────────────────────────────────────
_SCALAR_OUTS = ['CL', 'CD', 'CDP', 'CDF', 'CM',
                'XOCTR_TOP', 'XOCTR_BOT', 'CPMIN']

_GRAD_SUFFIXES = {
    'alpha':   '_dalpha',
    're':      '_dre',
    'ncrit':   '_dncrit',
    'mach':    '_dmach',
    'xtp_top': '_dxtp_top',
    'xtp_bot': '_dxtp_bot',
}

_ALL_GRAD_INPUTS = list(_GRAD_SUFFIXES.keys())


# ── low-level output parsers ───────────────────────────────────────────────
def _parse_scalar(lines, key):
    for line in lines:
        tok = line.split()
        if tok and tok[0] == key:
            try:
                return float(tok[1])
            except (IndexError, ValueError):
                pass
    return float('nan')


def _parse_converged(lines):
    for line in lines:
        tok = line.split()
        if tok and tok[0] == 'CONVERGED':
            return tok[1] == 'T'
    return False


def _parse_bl_surface(lines, prefix):
    """
    Parse BL distribution rows for one surface.
    Returns a dict matching xfoil/qfoil bl_data column names:
      s, x, Ue/Vinf, Dstar, Theta, Cf, H
    (y, H*, P, m, K, tau, Di are not output by cfoil and are omitted)
    """
    rows = [l.split() for l in lines if l.startswith(prefix + '  ')]
    if not rows:
        return None
    # cfoil columns: i  xssi  x  Ue  theta  dstr  CF
    arr = np.array([[float(v) for v in r[1:]] for r in rows])
    xssi  = arr[:, 0]
    x     = arr[:, 1]
    ue    = arr[:, 2]
    theta = arr[:, 3]
    dstr  = arr[:, 4]
    cf    = arr[:, 5]
    h     = np.where(theta > 0, dstr / theta, float('nan'))
    return {
        's':        list(xssi),
        'x':        list(x),
        'Ue/Vinf':  list(ue),
        'Dstar':    list(dstr),
        'Theta':    list(theta),
        'Cf':       list(cf),
        'H':        list(h),
    }


def _parse_bl_grad_surface(lines, prefix, h_cs):
    """
    Parse d(BL)/d(input) rows for one surface (complex-step gradient output).
    Returns the same column structure as _parse_bl_surface but with gradient values.
    """
    rows = [l.split() for l in lines if l.startswith(prefix)]
    if not rows:
        return None
    # cfoil columns: i  d(xssi)  d(x)  d(Ue)  d(theta)  d(dstr)  d(CF)
    arr = np.array([[float(v) for v in r[2:]] for r in rows])
    dxssi  = arr[:, 0]
    dx     = arr[:, 1]
    due    = arr[:, 2]
    dtheta = arr[:, 3]
    ddstr  = arr[:, 4]
    dcf    = arr[:, 5]
    return {
        's':        list(dxssi),
        'x':        list(dx),
        'Ue/Vinf':  list(due),
        'Dstar':    list(ddstr),
        'Theta':    list(dtheta),
        'Cf':       list(dcf),
    }


# ── main run function ──────────────────────────────────────────────────────
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
        timelimit=30,
        max_iter=100,
        path_to_cfoil=None,
        save_boundary_layer_data=False,
        compute_gradients=False,
        gradient_inputs=None,
        h_cs=1e-20,
        force_list=False):
    """
    Run the cfoil complex-step executable for one or more operating points.

    Parameters
    ----------
    mode : str
        Operating mode.  Only 'alpha' (or 'alfa') is supported.
    upperKulfanCoefficients, lowerKulfanCoefficients : array-like
        Kulfan CST coefficients for upper and lower surfaces.
    val : float or list-like
        Angle-of-attack value(s) in degrees.
        • Scalar → single-point solve.
        • List   → solved sequentially; returns arrays.
        • 3-element list [start, stop, step] → alpha sweep (unless force_list=True).
    Re : float
        Reynolds number.
    M : float
        Mach number (default 0.0 = incompressible).
    xtp_u, xtp_l : float
        Forced transition x/c on upper/lower surfaces (1.0 = free transition).
    N_crit : float
        e^N transition criterion (default 9.0).
    N_panels_kulfan : int
        Number of points used when sampling the Kulfan geometry.
    TE_gap : float
        Trailing-edge gap (passed to Kulfan geometry generator).
    timelimit : int
        Per-point wall-time limit in seconds.
    max_iter : int
        Maximum XFOIL Newton iterations per point.
    path_to_cfoil : str or None
        Explicit path to the cfoil binary.  Defaults to the binary built in
        CMPLXFOIL/cfoil/cfoil relative to this file.
    save_boundary_layer_data : bool
        If True, BL distributions (Ue, θ, δ*, Cf, H) are included in the
        return dict under 'bl_data'.
    compute_gradients : bool
        If True, complex-step gradients are computed for every requested input
        and returned in the 'grad' (and optionally 'bl_grad') keys.
    gradient_inputs : list of str or None
        Which inputs to differentiate with respect to.
        Choices: 'alpha', 're', 'ncrit', 'mach', 'xtp_top', 'xtp_bot'.
        Default when None: all six scalar inputs.
    h_cs : float
        Complex-step perturbation size (default 1e-20).
    force_list : bool
        If True, treat a 3-element val as an explicit list rather than
        [start, stop, step].

    Returns
    -------
    dict with keys matching xfoil_wrapper / qfoil_wrapper:
        cd, cdp, cdf, cl, alpha, cm, cpmin, xtr_top, xtr_bot,
        xtp_top, xtp_bot, Re, M, N_crit, N_panels_kulfan,
        cp_data (always None — not output by cfoil),
        bl_data (list of dicts if save_boundary_layer_data else None),
        converged (bool or list of bool),

    Additional keys when compute_gradients=True:
        grad : dict keyed by input name →
               dict keyed by output name → float (or array for sweeps).
        bl_grad : dict keyed by input name →
                  dict with 'up'/'lo' → bl distribution dict
                  (only present when save_boundary_layer_data=True).
    """
    # ── validate ────────────────────────────────────────────────────────────
    mode = mode.lower()
    if mode == 'alfa':
        mode = 'alpha'
    if mode != 'alpha':
        raise NotImplementedError(
            "cfoil_wrapper only supports mode='alpha'.  "
            "CL mode is not implemented in the cfoil executable."
        )

    if path_to_cfoil is None:
        path_to_cfoil = _CFOIL_DEFAULT
    if not os.path.isfile(path_to_cfoil):
        raise FileNotFoundError(
            f"cfoil binary not found at: {path_to_cfoil}\n"
            "Build it with:  cd CMPLXFOIL/cfoil && make"
        )

    if gradient_inputs is None:
        gradient_inputs = _ALL_GRAD_INPUTS
    else:
        unknown = [g for g in gradient_inputs if g not in _GRAD_SUFFIXES]
        if unknown:
            raise ValueError(
                f"Unknown gradient_inputs: {unknown}. "
                f"Valid choices: {_ALL_GRAD_INPUTS}"
            )

    # ── determine alpha values to run ───────────────────────────────────────
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
            n_steps = int(round((stop - start) / step)) + 1
            alpha_list = [start + i * step for i in range(n_steps)]
        else:
            alpha_list = [float(v) for v in val]
    else:
        alpha_list = [float(val)]
        is_iterable = False

    # ── build temporary DAT file from Kulfan coefficients ───────────────────
    airfoil = Kulfan(TE_gap=TE_gap)
    airfoil.upperCoefficients = upperKulfanCoefficients
    airfoil.lowerCoefficients = lowerKulfanCoefficients
    if N_panels_kulfan is not None:
        airfoil.utility.Npoints = N_panels_kulfan

    tmp_dat = tempfile.NamedTemporaryFile(
        mode='w', suffix='.dat', prefix='cfoil_', delete=False
    )
    tmp_dat.close()
    airfoil.write2file(tmp_dat.name)

    # ── helper: build cfoil stdin for one alpha ──────────────────────────────
    def _build_input(alpha_deg, get_bl, grad_inputs_list):
        lines = [
            f'dat {tmp_dat.name}',
            f'alpha {alpha_deg:.6f}',
            f're {Re:.6g}',
            f'ncrit {N_crit:.6f}',
            f'mach {M:.6f}',
            f'xtp_top {xtp_u:.6f}',
            f'xtp_bot {xtp_l:.6f}',
            f'itmax {int(max_iter)}',
            f'h {h_cs:.6e}',
        ]
        if get_bl:
            lines.append('bl')
        if grad_inputs_list:
            lines.append('grads ' + ' '.join(grad_inputs_list))
            if get_bl:
                lines.append('gradbl')
        return '\n'.join(lines) + '\n'

    # ── helper: run cfoil with a timeout ────────────────────────────────────
    def _run_one(input_str):
        try:
            result = subprocess.run(
                [path_to_cfoil],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=timelimit,
            )
            return result.stdout.splitlines()
        except subprocess.TimeoutExpired:
            return []

    # ── helper: parse one set of cfoil output lines ─────────────────────────
    def _parse_one(lines, alpha_deg, grad_inputs_list, get_bl):
        converged = _parse_converged(lines)
        nan = float('nan')

        fwd = {k: _parse_scalar(lines, k) for k in _SCALAR_OUTS}
        if not converged:
            fwd = {k: nan for k in _SCALAR_OUTS}

        # bl data
        bl = None
        if get_bl and converged:
            bl_up = _parse_bl_surface(lines, 'BL_UP')
            bl_lo = _parse_bl_surface(lines, 'BL_LO')
            bl = {'upper': bl_up, 'lower': bl_lo}

        # scalar gradients
        grad = {}
        for inp in grad_inputs_list:
            suf = _GRAD_SUFFIXES[inp]
            row = {}
            for out in _SCALAR_OUTS:
                key = ('d' + out + suf + '   ')[:len('d' + out + suf) + 4].rstrip()
                # match the padded keys cfoil actually emits
                key_padded = 'd' + out + suf
                row[out.lower()] = _parse_scalar(lines, key_padded.rstrip())
            grad[inp] = row

        # bl gradients
        bl_grad = {}
        if get_bl and grad_inputs_list:
            for inp in grad_inputs_list:
                up_prefix = f'dBL_UP{_GRAD_SUFFIXES[inp]}'
                lo_prefix = f'dBL_LO{_GRAD_SUFFIXES[inp]}'
                g_up = _parse_bl_grad_surface(lines, up_prefix, h_cs)
                g_lo = _parse_bl_grad_surface(lines, lo_prefix, h_cs)
                bl_grad[inp] = {'upper': g_up, 'lower': g_lo}

        return fwd, converged, bl, grad, bl_grad

    # ── run all alpha points ─────────────────────────────────────────────────
    try:
        results_fwd   = []
        results_conv  = []
        results_bl    = []
        results_grad  = []
        results_bl_grad = []

        _grad_list = gradient_inputs if compute_gradients else []

        for alpha_deg in alpha_list:
            inp_str = _build_input(alpha_deg, save_boundary_layer_data, _grad_list)
            lines   = _run_one(inp_str)
            fwd, conv, bl, grad, bl_grad = _parse_one(
                lines, alpha_deg, _grad_list, save_boundary_layer_data
            )
            results_fwd.append(fwd)
            results_conv.append(conv)
            results_bl.append(bl)
            results_grad.append(grad)
            results_bl_grad.append(bl_grad)

    finally:
        try:
            os.remove(tmp_dat.name)
        except OSError:
            pass

    # ── assemble return dict ─────────────────────────────────────────────────
    def _extract(key):
        vals = [r[key] for r in results_fwd]
        return vals[0] if not is_iterable else np.array(vals)

    res = {
        'cl':        _extract('CL'),
        'cd':        _extract('CD'),
        'cdp':       _extract('CDP'),
        'cdf':       _extract('CDF'),
        'cm':        _extract('CM'),
        'cpmin':     _extract('CPMIN'),
        'xtr_top':   _extract('XOCTR_TOP'),
        'xtr_bot':   _extract('XOCTR_BOT'),
        'alpha':     alpha_list[0] if not is_iterable else np.array(alpha_list),
        'xtp_top':   xtp_u,
        'xtp_bot':   xtp_l,
        'Re':        Re,
        'M':         M,
        'N_crit':    N_crit,
        'N_panels_kulfan': N_panels_kulfan,
        'converged': results_conv[0] if not is_iterable else results_conv,
        'cp_data':   None,   # cfoil does not output Cp distributions
        'bl_data':   None,
    }

    # bl_data: list for sweeps, single dict for scalar
    if save_boundary_layer_data:
        bl_list = results_bl
        res['bl_data'] = bl_list[0] if not is_iterable else bl_list

    # gradient dict: {input_name: {output_name: float_or_array}}
    if compute_gradients:
        combined_grad = {inp: {} for inp in _grad_list}
        for inp in _grad_list:
            for out_key in _SCALAR_OUTS:
                out_lower = out_key.lower()
                vals = [r[inp][out_lower] for r in results_grad]
                combined_grad[inp][out_lower] = (
                    vals[0] if not is_iterable else np.array(vals)
                )
        res['grad'] = combined_grad

        if save_boundary_layer_data:
            combined_bl_grad = {inp: [] for inp in _grad_list}
            for inp in _grad_list:
                combined_bl_grad[inp] = (
                    results_bl_grad[0][inp]
                    if not is_iterable
                    else [r[inp] for r in results_bl_grad]
                )
            res['bl_grad'] = combined_bl_grad

    return res


# ── convenience: alpha sweep shorthand ────────────────────────────────────
def run_sweep(upperKulfanCoefficients,
              lowerKulfanCoefficients,
              alpha_start, alpha_stop, alpha_step,
              **kwargs):
    """Convenience wrapper for an alpha sweep. Passes [start, stop, step] to run()."""
    return run('alpha',
                upperKulfanCoefficients,
                lowerKulfanCoefficients,
                val=[alpha_start, alpha_stop, alpha_step],
                **kwargs)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

    # ── quick smoke test ───────────────────────────────────────────────────
    upper = np.array([0.2,  0.3,  0.35, 0.25, 0.15, 0.05])
    lower = np.array([-0.1, -0.15, -0.1, -0.05, 0.0, 0.05])

    print("── scalar alpha, no gradients ───────────────────────────────")
    r = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0,
            save_boundary_layer_data=True)
    print(f"  CL={r['cl']:.4f}  CD={r['cd']:.5f}  CDP={r['cdp']:.5f}"
          f"  CDF={r['cdf']:.5f}  CM={r['cm']:.4f}")
    print(f"  xtr_top={r['xtr_top']:.3f}  xtr_bot={r['xtr_bot']:.3f}"
          f"  cpmin={r['cpmin']:.3f}  converged={r['converged']}")
    if r['bl_data']:
        nbl = len(r['bl_data']['upper']['x'])
        print(f"  BL upper: {nbl} stations,"
              f"  Cf[10]={r['bl_data']['upper']['Cf'][10]:.4e}")

    print("\n── scalar alpha, with all gradients ─────────────────────────")
    r2 = run('alpha', upper, lower, val=5.0, Re=1e6, N_crit=9.0,
             compute_gradients=True, save_boundary_layer_data=True)
    print(f"  dCL/dalpha  = {r2['grad']['alpha']['cl']:+.5f}")
    print(f"  dCD/dalpha  = {r2['grad']['alpha']['cd']:+.5f}")
    print(f"  dCL/dre     = {r2['grad']['re']['cl']:+.5e}")
    print(f"  dCL/dncrit  = {r2['grad']['ncrit']['cl']:+.5f}")

    print("\n── alpha sweep  [-2, 8, 2] ──────────────────────────────────")
    r3 = run_sweep(upper, lower, -2.0, 8.0, 2.0, Re=1e6, N_crit=9.0)
    for a, cl, cd, c in zip(r3['alpha'], r3['cl'], r3['cd'], r3['converged']):
        print(f"  α={a:5.1f}°  CL={cl:.4f}  CD={cd:.5f}  conv={c}")
