import subprocess
import warnings
import tempfile
import numpy as np
import pandas as pd
from oso_airfoils.geometry.kulfan import Kulfan
import os
import sys
import math
import shutil
# path_to_XFOIL_default = shutil.which('xfoil')
import pathlib
import random
import string
from datetime import datetime
path_to_here = pathlib.Path(__file__).parent.resolve()
import platform

class FNM(object):
    def __init__(self,ldr,N=5):
        # ldr = '/tmp/t_'
        x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(N))
        timestamp = datetime.now().strftime("%M%S%f")
        self.name = ldr + timestamp + x


def _nf_cl_check(target_cl, upperKulfanCoefficients, lowerKulfanCoefficients,
                 Re, N_crit, xtp_u, xtp_l, TE_gap, cl_margin=1.05):
    """
    Use NeuralFoil to verify that *target_cl* is achievable and return an
    alpha estimate for XFoil inviscid seeding.

    NeuralFoil is required for XFoil CL mode.  An ImportError is raised if
    it is not installed.

    Returns
    -------
    alpha_est : float
        AoA [deg] estimate for the target CL from a coarse NF sweep.
    achievable : bool
        True if target_cl <= cl_margin * cl_max_nf.
    cl_max_nf : float
        Maximum CL predicted by NeuralFoil for alpha in [-30, 30] deg.
    """
    import neuralfoil as nf  # required; raises ImportError if not installed

    afl = Kulfan(TE_gap=TE_gap)
    afl.upperCoefficients = upperKulfanCoefficients
    afl.lowerCoefficients = lowerKulfanCoefficients
    afl.changeOrder(8)
    kp = {
        'upper_weights':       afl.upperCoefficients.magnitude,
        'lower_weights':       afl.lowerCoefficients.magnitude,
        'TE_thickness':        TE_gap,
        'leading_edge_weight': 0.0,
    }
    alpha_coarse = np.linspace(-30.0, 30.0, 121)
    cl_coarse = nf.get_aero_from_kulfan_parameters(
        kp, alpha=alpha_coarse, Re=Re, n_crit=N_crit,
        xtr_upper=xtp_u, xtr_lower=xtp_l,
    )['CL']

    cl_max_nf  = float(cl_coarse.max())
    achievable = (target_cl <= cl_margin * cl_max_nf)

    # Linear interpolation in the first sign-change bracket for alpha estimate
    diff = cl_coarse - target_cl
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0.0:
            span = float(diff[i + 1]) - float(diff[i])
            frac = (-float(diff[i]) / span) if span != 0.0 else 0.5
            alpha_est = float(alpha_coarse[i]) + frac * (
                float(alpha_coarse[i + 1]) - float(alpha_coarse[i])
            )
            return alpha_est, achievable, cl_max_nf

    idx = int(np.argmin(np.abs(diff)))
    return float(alpha_coarse[idx]), achievable, cl_max_nf

def run(mode, 
        upperKulfanCoefficients,
        lowerKulfanCoefficients,
        val = 0.0, 
        Re = 1e7,
        M = 0.0,
        xtp_u=1.0,
        xtp_l=1.0,
        N_crit=9.0,
        N_panels_xfoil = None,
        N_panels_kulfan = 100,
        flapLocation = None,
        flapDeflection = 0.0,
        TE_gap = 0.0,
        timelimit = 10,
        max_iter=100,
        path_to_XFOIL = None,
        tfpre = None,
        cl_margin = 1.05,
        alpha_margin = 1.05,
        save_boundary_layer_data = False,
        force_list = False,
        stdout_log_path = None,
        exec_script_path = None,
        airfoil_name = None):

    try:
        iter(val)
        is_iterable = True
    except TypeError:
        is_iterable = False

    path_to_XFOIL_default = shutil.which('xfoil')

    if path_to_XFOIL is None:
        path_to_XFOIL = path_to_XFOIL_default

    if tfpre is None:
        tfpre = 't_'

    tempDatfile    = FNM(tfpre,5)
    tempPolarfile  = FNM(tfpre,5)
    tempStdoutFile = FNM(tfpre,5)
    tempExecFile   = FNM(tfpre,5)
    tempCpDatafile_array = None
    tempBlDatafile_array = None

    # make sure we dont accidently reuse
    if os.path.exists(tempDatfile.name):
        os.remove(tempDatfile.name)
    if os.path.exists(tempPolarfile.name):
        os.remove(tempPolarfile.name)
    if os.path.exists(tempStdoutFile.name):
        os.remove(tempStdoutFile.name)
    if os.path.exists(tempExecFile.name):
        os.remove(tempExecFile.name)

    mode = mode.lower()
    if mode == 'alpha':
        mode = 'alfa'

    if mode not in ['alfa','cl']:
        raise ValueError('Invalid input mode.  Must be one of: alfa, cl ')

    # ── NeuralFoil pre-check for CL mode (achievability + inviscid seed) ────────
    _nf_alpha_est  = None
    _nf_achievable = True
    _nf_cl_max     = None
    if mode == 'cl':
        _cl_seed = val[0] if is_iterable else float(val)
        _nf_alpha_est, _nf_achievable, _nf_cl_max = _nf_cl_check(
            _cl_seed, upperKulfanCoefficients, lowerKulfanCoefficients,
            Re, N_crit, xtp_u, xtp_l, TE_gap, cl_margin=cl_margin,
        )
        if not _nf_achievable:
            warnings.warn(
                f"Skipping XFoil: CL={_cl_seed:.4f} exceeds {cl_margin}×"
                f" NeuralFoil CL_max ({_nf_cl_max:.4f}).",
                stacklevel=2,
            )
            nan = float('nan')
            return {
                'cd': nan, 'cdp': nan, 'cl': nan, 'alpha': nan,
                'cm': nan, 'cpmin': nan, 'xtr_top': nan, 'xtr_bot': nan,
                'xtp_top': xtp_u, 'xtp_bot': xtp_l,
                'Re': Re, 'M': M, 'N_crit': N_crit,
                'N_panels_xfoil': N_panels_xfoil, 'N_panels_kulfan': N_panels_kulfan,
                'cp_data': None, 'bl_data': None,
            }
    # ───────────────────────────────────────────────────────────────

    if N_panels_xfoil is not None and N_panels_kulfan is not None:
        warnings.warn(
            f"Both N_panels_xfoil={N_panels_xfoil} and N_panels_kulfan={N_panels_kulfan} "
            "are set.  xfoil will repanel after loading the Kulfan-generated dat file.",
            stacklevel=2,
        )

    airfoil = Kulfan(TE_gap=TE_gap)
    airfoil.upperCoefficients = upperKulfanCoefficients
    airfoil.lowerCoefficients = lowerKulfanCoefficients
    if N_panels_kulfan is not None:
        airfoil.utility.Npoints = N_panels_kulfan
    airfoil.write2file(tempDatfile.name)
    
    assert(os.path.isfile(tempDatfile.name))

    topline = 'load ' + tempDatfile.name + ' \n' + 'airfoil \n'
    
    estr = ''
    estr += 'plop\n'
    estr += 'g\n'
    estr += '\n'
    estr += topline
    if N_panels_xfoil is not None:
        estr += 'ppar\n'
        estr += 'n %d\n'%(N_panels_xfoil)
        estr += '\n'
        estr += '\n'
    if flapLocation is not None:
        ck1 = flapLocation >= 0.0
        ck2 = flapLocation <= 1.0
        if ck1 and ck2:
            estr += 'gdes \n'
            estr += 'flap \n'
            estr += '%f \n'%(flapLocation)
            estr += '999 \n'
            estr += '0.5 \n'
            estr += '%f \n'%(flapDeflection)
            estr += 'x \n'
            estr += '\n'
        else:
            raise ValueError('Invalid flapLocation.  Must be between 0.0 and 1.0')
    estr += 'oper \n'
    estr += "iter %d\n" %(max_iter)
    #run inviscid first
    if is_iterable:
        if mode == 'alfa':
            estr += "alfa %.2f \n" %(val[0])
        if mode == 'cl':
            if _nf_alpha_est is not None:
                estr += "alfa %.2f \n" %(_nf_alpha_est)
            else:
                estr += "cl %.3f \n" %(val[0])
    else:
        if mode == 'alfa':
            estr += "alfa %.2f \n" %(val)
        if mode == 'cl':
            if _nf_alpha_est is not None:
                estr += "alfa %.2f \n" %(_nf_alpha_est)
            else:
                estr += "cl %.3f \n" %(val)
    estr += 'visc \n'
    estr += "%.0f \n" %(float(Re))
    estr += "M \n"
    estr += "%.2f \n" %(M)
    if N_crit < 9.0:
        # try to pre-seed rough cases
        if is_iterable:
            if mode == 'alfa':
                estr += "alfa %.2f \n" %(val[0])
            if mode == 'cl':
                if _nf_alpha_est is not None:
                    estr += "alfa %.2f \n" %(_nf_alpha_est)
                else:
                    estr += "cl %.3f \n" %(val[0])
        else:
            if mode == 'alfa':
                estr += "alfa %.2f \n" %(val)
            if mode == 'cl':
                if _nf_alpha_est is not None:
                    estr += "alfa %.2f \n" %(_nf_alpha_est)
                else:
                    estr += "cl %.3f \n" %(val)
    estr += 'vpar \n'
    estr += 'xtr \n'
    estr += '%f \n'%(xtp_u)
    estr += '%f \n'%(xtp_l)
    estr += 'n \n'
    estr += '%f \n'%(N_crit)
    estr += '\n'
    if N_crit < 9.0:
        # do the pre-seeding again for rough cases
        if is_iterable:
            if mode == 'alfa':
                estr += "alfa %.2f \n" %(val[0])
            if mode == 'cl':
                if _nf_alpha_est is not None:
                    estr += "alfa %.2f \n" %(_nf_alpha_est)
                else:
                    estr += "cl %.3f \n" %(val[0])
        else:
            if mode == 'alfa':
                estr += "alfa %.2f \n" %(val)
            if mode == 'cl':
                if _nf_alpha_est is not None:
                    estr += "alfa %.2f \n" %(_nf_alpha_est)
                else:
                    estr += "cl %.3f \n" %(val)
    # to include cpmin:
    estr += 'cinc\n'
    estr += 'pacc \n'
    estr += tempPolarfile.name + ' \n'    #estr += '\n'
    estr += '\n'

    if is_iterable:
        # A 3-element list is treated as [start, stop, step]; any other length
        # is treated as an explicit list of values to run individually.
        _is_triplet = (len(val) == 3 and not force_list)
        _iter_vals = (
            [val[0] + i * val[2] for i in range(int((val[1] - val[0]) / val[2]) + 1)]
            if _is_triplet else list(val)
        )
        if save_boundary_layer_data:
            # raise NotImplementedError('Boundary layer data saving is not currently implemented for iterative runs')
            tempCpDatafile_array = []
            tempBlDatafile_array = []

            kwd = 'alfa ' if mode == 'alfa' else 'cl '

            for v in _iter_vals:
                estr += kwd + '%.3f \n' %(v)
                tempCpDatafile = FNM(tfpre,5)
                tempBlDatafile = FNM(tfpre,5)
                estr += 'cpwr \n'
                estr += tempCpDatafile.name + '\n'
                estr += 'dump \n'
                estr += tempBlDatafile.name + '\n'
                tempCpDatafile_array.append(tempCpDatafile.name)
                tempBlDatafile_array.append(tempBlDatafile.name)

        else:
            if _is_triplet:
                if mode == 'alfa':
                    estr += "aseq %.2f %.2f %.2f \n" %(val[0],val[1],val[2])
                if mode == 'cl':
                    estr += "cseq %.3f %.3f %.3f \n" %(val[0],val[1],val[2])
            else:
                kwd = 'alfa ' if mode == 'alfa' else 'cl '
                for v in _iter_vals:
                    estr += kwd + '%.3f \n' %(v)

        # estr += 'pwrt \n'
        # estr += tempPolarfile.name + ' \n'    
        estr += '\n'
        estr += 'q \n'

    else:
        if mode == 'alfa':
            estr += "alfa %.2f \n" %(val)
        if mode == 'cl':
            estr += "cl %.3f \n" %(val)
        # estr += 'pwrt \n'
        # estr += tempPolarfile.name + ' \n'
        tempCpDatafile_array = None
        tempBlDatafile_array = None
        if save_boundary_layer_data:
            tempCpDatafile = FNM(tfpre,5)
            tempBlDatafile = FNM(tfpre,5)
            estr += 'cpwr \n'
            estr += tempCpDatafile.name + '\n'
            estr += 'dump \n'
            estr += tempBlDatafile.name + '\n'
            tempCpDatafile_array = [tempCpDatafile.name]
            tempBlDatafile_array = [tempBlDatafile.name]
        estr += '\n'
        estr += 'q \n'

    exFile = open(tempExecFile.name,'w')
    exFile.write(estr)
    exFile.close()


    cmd = ''
    if sys.platform == "linux" or sys.platform == "linux2":
        # linux
        cmd += 'timeout %d '%(timelimit)
    else:
        # OS X
        assert(sys.platform == "darwin")
        cmd += 'timelimit -t%d '%(timelimit)
    # elif sys.platform == "win32":
    # Windows...


    cmd += path_to_XFOIL
    cmd += ' <' + tempExecFile.name
    cmd += ' >'+tempStdoutFile.name
    # print(estr)

    assert(os.path.isfile(tempDatfile.name))

    try:
        # stdout_val = subprocess.check_output(cmd, shell=True, timeout=5)
        subprocess.run(cmd, shell=True)
    except:
        # process failed or timed out, will be handled below as a normal failure
        # print( upperKulfanCoefficients, lowerKulfanCoefficients, val)
        pass

    try:
        with warnings.catch_warnings():
            # catch warning for empty file
            warnings.simplefilter('ignore')
            data = np.genfromtxt(tempPolarfile.name, skip_header=12)

        if mode == 'cl' and (data.ndim == 0 or data.size == 0):
            warnings.warn(
                f"XFoil returned no converged solution for CL={_cl_seed:.4f} "
                f"(NeuralFoil predicted alpha ≈ {_nf_alpha_est:.2f}°, "
                f"CL_max ≈ {_nf_cl_max:.4f}).",
                stacklevel=2,
            )

        if not is_iterable:
            alpha   = data[0]
            cl      = data[1]
            cd      = data[2]
            cdp     = data[3]
            cm      = data[4]
            cpmin   = data[5]
            xtr_top = data[6]
            xtr_bot = data[7]
            Reval   = Re
            Mval    = M

            cpData = None
            blData = None
            if save_boundary_layer_data:
                cpData = pd.read_csv(tempCpDatafile_array[0], sep="\\s+",skiprows=1, names = ['x' , 'cp']).to_dict('list')
                blData = pd.read_csv(tempBlDatafile_array[0], sep="\\s+",skiprows=1, names = ['s', 'x', 'y', 'Ue/Vinf', 'Dstar', 'Theta', 'Cf', 'H', 'H*', 'P', 'm', 'K', 'tau', 'Di']).to_dict('list')

            # ── alpha match check (CL mode only) ──────────────────────────────
            if mode == 'cl' and _nf_alpha_est is not None and not np.isnan(alpha):
                _alpha_tol = (alpha_margin - 1.0) * abs(_nf_alpha_est) + 1.0
                if abs(float(alpha) - _nf_alpha_est) > _alpha_tol:
                    warnings.warn(
                        f"XFoil alpha={float(alpha):.2f}° deviates from NeuralFoil "
                        f"estimate {_nf_alpha_est:.2f}° by more than the tolerance "
                        f"{_alpha_tol:.2f}° (alpha_margin={alpha_margin}). "
                        f"Possible non-unique solution (e.g. post-stall convergence).",
                        stacklevel=2,
                    )
            # ───────────────────────────────────────────────────────────────

        else:
            # np.genfromtxt returns shape (0,) for 0 rows and (n_cols,) for
            # exactly 1 row. Normalise to (n_rows, n_cols) in all cases so
            # that the column slices below always work; an empty (0, 8) array
            # propagates naturally and _iterable_to_records returns [].
            if data.ndim == 1 and data.size == 0:
                data = np.empty((0, 8))
            else:
                data = np.atleast_2d(data)
            alpha   = data[:,0]
            cl      = data[:,1]
            cd      = data[:,2]
            cdp     = data[:,3]
            cm      = data[:,4]
            cpmin   = data[:,5]
            xtr_top = data[:,6]
            xtr_bot = data[:,7]
            Reval   = Re
            Mval    = M

            cpData = None
            blData = None
            if save_boundary_layer_data:
                cpData = []
                blData = []
                for i in range(len(tempCpDatafile_array)):
                    cpData.append(pd.read_csv(tempCpDatafile_array[i], sep="\\s+",skiprows=1, names = ['x' , 'cp']).to_dict('list'))
                    blData.append(pd.read_csv(tempBlDatafile_array[i], sep="\\s+",skiprows=1, names = ['s', 'x', 'y', 'Ue/Vinf', 'Dstar', 'Theta', 'Cf', 'H', 'H*', 'P', 'm', 'K', 'tau', 'Di']).to_dict('list'))
                assert(len(cpData) == len(alpha))

        res = {}
        res['cd'] = cd
        res['cdp'] = cdp
        res['cl'] = cl
        res['alpha'] = alpha
        res['cm'] = cm
        res['cpmin'] = cpmin
        res['xtr_top'] = xtr_top
        res['xtr_bot'] = xtr_bot
        res['xtp_top'] = xtp_u
        res['xtp_bot'] = xtp_l
        res['Re'] = Reval
        res['M'] = Mval
        res['N_crit'] = N_crit
        res['N_panels_xfoil'] = N_panels_xfoil
        res['N_panels_kulfan'] = N_panels_kulfan
        res['cp_data'] = cpData
        res['bl_data'] = blData

        return res

    finally:
        def _safe_remove(path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        if stdout_log_path is not None and os.path.exists(tempStdoutFile.name):
            import shutil as _shutil
            if os.path.isdir(stdout_log_path):
                _name_part = f'{airfoil_name}_' if airfoil_name else ''
                _fname = f'xfoil_stdout_{_name_part}Re{Re:.0f}_Ncrit{N_crit:.1f}_xtp{xtp_u:.2f}_{xtp_l:.2f}.txt'
                _shutil.copy2(tempStdoutFile.name, os.path.join(stdout_log_path, _fname))
            else:
                _shutil.copy2(tempStdoutFile.name, stdout_log_path)
        if exec_script_path is not None and os.path.exists(tempExecFile.name):
            import shutil as _shutil
            if os.path.isdir(exec_script_path):
                _name_part = f'{airfoil_name}_' if airfoil_name else ''
                _fname = f'xfoil_exec_{_name_part}Re{Re:.0f}_Ncrit{N_crit:.1f}_xtp{xtp_u:.2f}_{xtp_l:.2f}.txt'
                _shutil.copy2(tempExecFile.name, os.path.join(exec_script_path, _fname))
            else:
                _shutil.copy2(tempExecFile.name, exec_script_path)
        for f in [tempDatfile.name, tempPolarfile.name,
                  tempStdoutFile.name, tempExecFile.name]:
            _safe_remove(f)
        for arr in (tempCpDatafile_array, tempBlDatafile_array):
            if arr:
                for f in arr:
                    _safe_remove(f)

if __name__ == '__main__':
    res = run('alpha',[0.2,0.2],[-0.2,-0.2],0, save_boundary_layer_data=True)
    print('\n')
    print(res)
    print(res['cp_data'])


    # res = run('alpha',[0.1,0.1],[-0.1,-0.1],[0,5,1], save_boundary_layer_data=True)
    # print('\n')
    # # print(res)
    # print(res['alpha'])
    # print(res['cl'])
    # print(res['cd'])
    # print(res['cpmin'])
    # print([res['cp_data'][i]['cp'][0] for i in range(len(res['cp_data']))])
    
    # res = run('alpha',[0.1,0.1],[-0.1,-0.1],[0,-30,-0.25], file_system=1)
    # print('\n')
    # print(res)

    # res = run('alpha',[0.1,0.1],[-0.1,-0.1],[0,-30,-0.25], file_system=2)
    # print('\n')
    # print(res)

    # res = run('alpha',[0.1,0.1],[-0.1,-0.1],[0,-30,-0.25], file_system=3)
    # print('\n')
    # print(res)

    res2 = run('cl', [0.2,0.2], [-0.2,-0.2], 0.5)
    print('\ncl mode (scalar):')
    print('alpha:', res2['alpha'])
    print('cl:   ', res2['cl'])
    print('cd:   ', res2['cd'])
