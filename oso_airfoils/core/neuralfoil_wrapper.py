import subprocess
import warnings
import tempfile
import numpy as np
import pandas as pd
from metafoil.core.kulfan import Kulfan
import os
import sys
import math
import shutil
path_to_XFOIL = shutil.which('xfoil')
import pathlib
import random
import string
from datetime import datetime
# path_to_here = pathlib.Path(__file__).parent.resolve()
import platform
from collections.abc import Iterable

import neuralfoil as nf


def _find_alpha_for_cl(target_cl, alpha_coarse, cl_coarse, kp,
                       Re, N_crit, xtp_u, xtp_l, model, cl_tol=1e-4):
    """
    Find alpha [deg] such that NeuralFoil gives CL = target_cl.

    The caller pre-computes (alpha_coarse, cl_coarse) once and passes them in
    so that multiple CL targets can share a single coarse sweep.

    Raises ValueError if target_cl is outside the achievable range.
    """
    cl_min = float(cl_coarse.min())
    cl_max = float(cl_coarse.max())
    if not (cl_min - abs(cl_tol) <= target_cl <= cl_max + abs(cl_tol)):
        raise ValueError(
            f"Target CL={target_cl:.4f} is outside the NeuralFoil-predicted "
            f"achievable range [{cl_min:.4f}, {cl_max:.4f}] "
            f"for alpha in [-30, 30] deg."
        )

    diff = cl_coarse - target_cl

    # Find the first sign change in diff to bracket the root
    a_lo = a_hi = f_lo = f_hi = None
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0.0:
            a_lo, a_hi = float(alpha_coarse[i]), float(alpha_coarse[i + 1])
            f_lo, f_hi = float(diff[i]),           float(diff[i + 1])
            break

    if a_lo is None:
        # CL exactly at a coarse-grid point
        return float(alpha_coarse[int(np.argmin(np.abs(diff)))])

    # Classic bisection — works for both ascending and descending brackets
    for _ in range(60):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = float(np.asarray(nf.get_aero_from_kulfan_parameters(
            kp, alpha=a_mid, Re=Re, n_crit=N_crit,
            xtr_upper=xtp_u, xtr_lower=xtp_l, model_size=model,
        )['CL']).flat[0]) - target_cl
        if abs(f_mid) <= cl_tol:
            return a_mid
        if f_lo * f_mid <= 0.0:
            a_hi, f_hi = a_mid, f_mid
        else:
            a_lo, f_lo = a_mid, f_mid

    return 0.5 * (a_lo + a_hi)


def run(mode, 
        upperKulfanCoefficients,
        lowerKulfanCoefficients,
        val = 0.0, 
        Re = 1e7,
        M = 0.0,
        xtp_u=1.0,
        xtp_l=1.0,
        N_crit=9.0,
        N_panels = 160,
        flapLocation = None,
        flapDeflection = 0.0,
        polarfile      = None,
        cpDatafile     = None,
        blDatafile     = None,
        defaultDatfile = None,
        executionFile  = None,
        stdoutFile     = None,
        TE_gap = 0.0,
        timelimit = 10,
        max_iter=100,
        file_system = None,
        cl_tol = 1e-8,
        model = 'xxxlarge',
        save_boundary_layer_data = True):

    mode = mode.lower()
    if mode == 'alfa':
        mode = 'alpha'
    if mode not in ['alpha', 'cl']:
        raise ValueError("Neuralfoil mode must be 'alpha'/'alfa' (angle of attack) or 'cl' (lift coefficient).")

    if model not in ["xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]:
        raise ValueError('Invalid Model, must be one of ["xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]')

    # Build Kulfan object and parameter dict (shared by both modes)
    afl = Kulfan(TE_gap=TE_gap)
    afl.upperCoefficients = upperKulfanCoefficients
    afl.lowerCoefficients = lowerKulfanCoefficients
    afl.changeOrder(8)
    kp = {
        'upper_weights':      afl.upperCoefficients.magnitude,
        'lower_weights':      afl.lowerCoefficients.magnitude,
        'TE_thickness':       TE_gap,
        'leading_edge_weight': 0.0,
    }

    is_iterable = isinstance(val, Iterable)
    # A 3-element iterable is treated as [start, stop, step]; any other length
    # is treated as an explicit list of values.

    if mode == 'cl':
        # ── CL mode: coarse sweep (shared) then bisection per target ─────────
        if is_iterable:
            if len(val) == 3:
                target_cls = np.linspace(val[0], val[1], int((val[1] - val[0]) / val[2]) + 1)
            else:
                target_cls = np.asarray(val, dtype=float)
        else:
            target_cls = np.array([float(val)])

        alpha_coarse = np.linspace(-30.0, 30.0, 121)
        cl_coarse    = nf.get_aero_from_kulfan_parameters(
            kp, alpha=alpha_coarse, Re=Re,
            n_crit=N_crit, xtr_upper=xtp_u, xtr_lower=xtp_l, model_size=model,
        )['CL']

        alpha = np.array([
            _find_alpha_for_cl(tc, alpha_coarse, cl_coarse, kp,
                               Re, N_crit, xtp_u, xtp_l, model, cl_tol)
            for tc in target_cls
        ])
    else:
        # ── alpha mode ────────────────────────────────────────────────────────
        if is_iterable:
            if len(val) == 3:
                alpha = np.linspace(val[0], val[1], int((val[1] - val[0]) / val[2]) + 1)
            else:
                alpha = np.asarray(val, dtype=float)
        else:
            alpha = np.array([float(val)])

    res1 = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=kp,
        alpha=alpha,
        Re=Re,
        n_crit=N_crit,
        xtr_upper=xtp_u,
        xtr_lower=xtp_l,
        model_size=model,
    )

    # --- Boundary layer and cp packing ---
    # NeuralFoil provides 32 stations at nf.bl_x_points (mid-points of equal x-intervals)
    n_stations = 32
    x_bl = nf.bl_x_points  # shape (32,)

    # Gather BL arrays: shape (n_stations, n_alpha)
    upper_ue    = np.array([res1['upper_bl_ue/vinf_' + str(i)] for i in range(n_stations)])
    upper_theta = np.array([res1['upper_bl_theta_'   + str(i)] for i in range(n_stations)])
    upper_H     = np.array([res1['upper_bl_H_'       + str(i)] for i in range(n_stations)])
    lower_ue    = np.array([res1['lower_bl_ue/vinf_' + str(i)] for i in range(n_stations)])
    lower_theta = np.array([res1['lower_bl_theta_'   + str(i)] for i in range(n_stations)])
    lower_H     = np.array([res1['lower_bl_H_'       + str(i)] for i in range(n_stations)])

    # cp = 1 - (Ue/Vinf)^2  (incompressible Bernoulli)
    upper_cp = 1.0 - upper_ue**2
    lower_cp = 1.0 - lower_ue**2

    # Wrapped x: upper surface TE→LE (reversed), then lower surface LE→TE
    # This matches the XFoil convention for cp_data and bl_data
    x_upper_rev = np.flip(x_bl)
    x_wrap = np.concatenate([x_upper_rev, x_bl])  # shape (64,)
    n_wrap = len(x_wrap)

    def _build_cp_data(ix):
        cp_upper_rev = np.flip(upper_cp[:, ix])
        cp_lower     = lower_cp[:, ix]
        return {
            'x':  list(x_wrap),
            'cp': list(np.concatenate([cp_upper_rev, cp_lower])),
        }

    def _build_bl_data(ix):
        nan_arr = [np.nan] * n_wrap
        ue    = np.concatenate([np.flip(upper_ue[:,    ix]), lower_ue[:,    ix]])
        theta = np.concatenate([np.flip(upper_theta[:, ix]), lower_theta[:, ix]])
        H     = np.concatenate([np.flip(upper_H[:,     ix]), lower_H[:,     ix]])
        return {
            's':       nan_arr,
            'x':       list(x_wrap),
            'y':       nan_arr,
            'Ue/Vinf': list(ue),
            'Dstar':   nan_arr,
            'Theta':   list(theta),
            'Cf':      nan_arr,
            'H':       list(H),
            'H*':      nan_arr,
            'P':       nan_arr,
            'm':       nan_arr,
            'K':       nan_arr,
            'tau':     nan_arr,
            'Di':      nan_arr,
        }

    # cpmin: most negative cp across all stations and both surfaces, per alpha
    cpmin_array = np.zeros(len(alpha))
    for ix in range(len(alpha)):
        max_vel = max(np.max(np.abs(upper_ue[:, ix])), np.max(np.abs(lower_ue[:, ix])))
        cpmin_array[ix] = 1.0 - max_vel**2

    if is_iterable:
        res = {}
        res['cd']      = res1['CD']
        res['cl']      = res1['CL']
        res['alpha']   = alpha
        res['cm']      = res1['CM']
        res['cpmin']   = cpmin_array
        res['xtr_top'] = res1['Top_Xtr']
        res['xtr_bot'] = res1['Bot_Xtr']
        res['xtp_top'] = xtp_u
        res['xtp_bot'] = xtp_l
        res['Re']      = Re
        res['M']       = 0.0
        res['N_crit']  = N_crit
        res['N_panels'] = None
        res['cp_data'] = [_build_cp_data(ix) for ix in range(len(alpha))]
        res['bl_data'] = [_build_bl_data(ix) for ix in range(len(alpha))]
    else:
        res = {}
        res['cd']      = res1['CD'][0]
        res['cl']      = res1['CL'][0]
        res['alpha']   = alpha[0]
        res['cm']      = res1['CM'][0]
        res['cpmin']   = cpmin_array[0]
        res['xtr_top'] = res1['Top_Xtr'][0]
        res['xtr_bot'] = res1['Bot_Xtr'][0]
        res['xtp_top'] = xtp_u
        res['xtp_bot'] = xtp_l
        res['Re']      = Re
        res['M']       = 0.0
        res['N_crit']  = N_crit
        res['N_panels'] = None
        res['cp_data'] = _build_cp_data(0)
        res['bl_data'] = _build_bl_data(0)

    return res

if __name__ == '__main__':
    res = run('alpha', [0.1, 0.1], [-0.1, -0.1], [0, 5, 1])
    print('alpha:', res['alpha'])
    print('cl:   ', res['cl'])
    print('cd:   ', res['cd'])
    print('cpmin:', res['cpmin'])
    print('cp_data[0] x[:4]:', res['cp_data'][0]['x'][:4])
    print('cp_data[0] cp[:4]:', res['cp_data'][0]['cp'][:4])
    print('bl_data[0] Ue/Vinf[:4]:', res['bl_data'][0]['Ue/Vinf'][:4])
    print('bl_data[0] Theta[:4]:', res['bl_data'][0]['Theta'][:4])
    print('bl_data[0] H[:4]:', res['bl_data'][0]['H'][:4])

    res2 = run('cl', [0.1, 0.1], [-0.1, -0.1], [0.2, 0.8, 0.2])
    print('\ncl mode (vector):')
    print('alpha:', res2['alpha'])
    print('cl:   ', res2['cl'])
    print('cd:   ', res2['cd'])
