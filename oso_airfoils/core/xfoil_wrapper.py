"""
xfoil_wrapper.py  --  oso's adapter over metafoil's XFOIL.

The solver itself lives in metafoil. This module is a thin dispatcher that keeps
oso's historical call signature and return contract:

  * alpha mode goes to metafoil's in-memory XFOIL (``libxfoil.so`` via f2py) --
    no temp files, ~10x faster for sweeps;
  * CL mode, flap deflections and an explicitly-pointed xfoil binary go to
    metafoil's subprocess/file-I/O XFOIL, which additionally does the NeuralFoil
    CL-achievability seeding.

Both implementations used to be duplicated here (a ~430-line ``_run_fileio``
alongside the in-memory delegation). They now live only in metafoil; what is left
here is the argument reconciliation oso's call sites depend on -- ``N_panels`` as an
alias for ``N_panels_kulfan``, and the ``N_panels`` / ``N_panels_xfoil`` keys that
``core.sweep`` reads back off the result.
"""

import metafoil.xfoil as _xfoil

from oso_airfoils.core.result_schema import to_oso_schema
from metafoil.xfoil.wrappers import xfoil_fileio as _xfoil_fileio


def run(mode,
        upperKulfanCoefficients,
        lowerKulfanCoefficients,
        val=0.0,
        Re=1e7,
        M=0.0,
        xtp_u=1.0,
        xtp_l=1.0,
        N_crit=9.0,
        N_panels_xfoil=None,
        N_panels_kulfan=100,
        N_panels=None,
        flapLocation=None,
        flapDeflection=0.0,
        TE_gap=0.0,
        timelimit=10,
        max_iter=100,
        path_to_XFOIL=None,
        tfpre=None,
        cl_margin=1.05,
        alpha_margin=1.05,
        save_boundary_layer_data=False,
        force_list=False,
        stdout_log_path=None,
        exec_script_path=None,
        airfoil_name=None,
        version=None,
        **kwargs):
    """
    Run XFOIL for a Kulfan airfoil.

    ``N_panels`` is accepted as an alias for ``N_panels_kulfan`` (the name
    ``core.sweep`` passes); ``version`` and any other unknown kwargs are absorbed.
    """
    if N_panels is not None:
        N_panels_kulfan = N_panels

    m = mode.lower()
    if m == 'alfa':
        m = 'alpha'

    # Features the in-memory solver doesn't cover -> metafoil's file-I/O path.
    needs_fileio = (m == 'cl'
                    or flapLocation is not None
                    or path_to_XFOIL is not None)

    if needs_fileio:
        res = _xfoil_fileio.run_from_kulfan(
            m, upperKulfanCoefficients, lowerKulfanCoefficients,
            val=val, N_panels_kulfan=N_panels_kulfan, TE_gap=TE_gap,
            Re=Re, M=M, xtp_u=xtp_u, xtp_l=xtp_l, N_crit=N_crit,
            N_panels_xfoil=N_panels_xfoil,
            flapLocation=flapLocation, flapDeflection=flapDeflection,
            timelimit=timelimit, max_iter=max_iter,
            path_to_XFOIL=path_to_XFOIL, tfpre=tfpre,
            cl_margin=cl_margin, alpha_margin=alpha_margin,
            save_boundary_layer_data=save_boundary_layer_data,
            force_list=force_list, stdout_log_path=stdout_log_path,
            exec_script_path=exec_script_path, airfoil_name=airfoil_name,
        )
    else:
        res = _xfoil.run_from_kulfan(
            'alpha', upperKulfanCoefficients, lowerKulfanCoefficients,
            val=val, N_panels_kulfan=N_panels_kulfan, TE_gap=TE_gap,
            Re=Re, M=M, xtp_u=xtp_u, xtp_l=xtp_l, N_crit=N_crit,
            max_iter=max_iter,
            save_boundary_layer_data=save_boundary_layer_data,
            force_list=force_list,
        )

    # Reconcile return keys with the oso contract / core.sweep consumer, which
    # reads res['N_panels'] and (historically) res['N_panels_xfoil'].
    res = to_oso_schema(res)
    res.setdefault('N_panels_kulfan', N_panels_kulfan)
    res['N_panels'] = res.get('N_panels_kulfan', N_panels_kulfan)
    res.setdefault('N_panels_xfoil', N_panels_xfoil)
    return res


if __name__ == '__main__':
    res = run('alpha', [0.2, 0.2], [-0.2, -0.2], 0, save_boundary_layer_data=True)
    print(res)
    print(res['cp_data'])
