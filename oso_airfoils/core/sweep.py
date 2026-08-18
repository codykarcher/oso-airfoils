"""
sweep.py  —  MPI-aware aerodynamic parameter sweep for a single airfoil.

The public entry point is :func:`run_sweep`.  When launched under ``mpirun``,
cases are distributed across all ranks and results are gathered back to rank 0.
Non-rank-0 ranks return an empty list.
"""


def run_sweep(solver, airfoil, cases, *, comm=None, rank=0, size=1,
              nf_model='xxlarge', xfoil_bl=True):
    """
    Run an aerodynamic parameter sweep for a single Kulfan airfoil.

    Parameters
    ----------
    solver : {'xfoil', 'qfoil', 'neuralfoil'}
        Which aerodynamic solver to use.
    airfoil : Kulfan
        Airfoil geometry object with ``upperCoefficients`` and
        ``lowerCoefficients`` attributes.
    cases : list of dict
        Each dict defines one run condition.  Recognised keys:
        ``alpha`` (required), ``Re`` (required), ``M`` (default 0.0),
        ``N_crit`` (default 9.0), ``N_panels`` (default 160, xfoil only),
        ``xtp_u`` (default 1.0), ``xtp_l`` (default 1.0).
        NeuralFoil cases may additionally include ``model`` to override
        *nf_model* for that specific case.
    comm : MPI communicator, optional
        Pass ``MPI.COMM_WORLD`` when running under mpi4py.
    rank : int, default 0
        MPI rank of this process.
    size : int, default 1
        Total number of MPI ranks.
    nf_model : str, default 'xxlarge'
        NeuralFoil model size used when ``model`` is not in the case dict.
    xfoil_bl : bool, default True
        Save XFoil boundary-layer / Cp data.

    Returns
    -------
    list of dict
        Run records on rank 0.  Empty list on all other ranks.
    """
    from oso_airfoils.core.ingest_data import (
        _is_sweep_spec, _wrapper_result_to_records, _get_xfoil_version,
    )

    if solver == 'xfoil':
        from oso_airfoils.core.xfoil_wrapper import run as _runner
        extra_kw = dict(version=_get_xfoil_version())
    elif solver == 'qfoil':
        from oso_airfoils.core.qfoil_wrapper import run as _runner
        extra_kw = {}
    elif solver == 'neuralfoil':
        from oso_airfoils.core.neuralfoil_wrapper import run as _runner
        extra_kw = {}
    else:
        raise ValueError(f"solver must be 'xfoil', 'qfoil', or 'neuralfoil', got '{solver}'")

    kw_geom = dict(
        upperKulfanCoefficients=airfoil.upperCoefficients,
        lowerKulfanCoefficients=airfoil.lowerCoefficients,
        TE_gap=airfoil.constants.TE_gap,
    )

    def _valid(recs):
        """Drop records with NaN cd (e.g. XFoil CL-mode achievability rejects)."""
        import math
        return [r for r in recs
                if r.get('cd') is not None and not math.isnan(r['cd'])]

    records = []
    for case in cases[rank::size]:
        c    = dict(case)
        mode = c.pop('mode', 'alpha')
        av   = c.pop('cl') if mode == 'cl' else c.pop('alpha')
        Re = c.pop('Re')
        M  = c.pop('M')
        Nc = c.pop('N_crit')
        np_= c.pop('N_panels')
        xu = c.pop('xtp_u')
        xl = c.pop('xtp_l')

        if solver in ('xfoil', 'qfoil'):
            shared = dict(**kw_geom, mode=mode, Re=Re, M=M, N_crit=Nc,
                          N_panels=np_, xtp_u=xu, xtp_l=xl,
                          save_boundary_layer_data=xfoil_bl)
            kw = extra_kw
        else:
            model  = c.pop('model', nf_model)
            shared = dict(**kw_geom, mode=mode, Re=Re, M=M, N_crit=Nc,
                          xtp_u=xu, xtp_l=xl, model=model)
            kw = dict(model=model)

        if _is_sweep_spec(av):
            try:
                records += _valid(_wrapper_result_to_records(solver, _runner(val=av, **shared), **kw))
            except Exception:
                pass
        else:
            for a in ([av] if not hasattr(av, '__iter__') else av):
                try:
                    records += _valid(_wrapper_result_to_records(solver, _runner(val=float(a), **shared), **kw))
                except Exception:
                    pass

    # MPI gather — all ranks must call this together; only rank 0 gets data back
    if comm is not None and size > 1:
        all_records = comm.gather(records, root=0)
        if rank == 0:
            return [rec for chunk in all_records for rec in chunk]
        return []

    return records


if __name__ == "__main__":
    import numpy as np
    from metafoil.core.kulfan import Kulfan

    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _comm, _rank, _size = None, 0, 1

    afl = Kulfan()
    afl.naca4(4412)

    cases = [
        {'alpha': float(a), 'Re': Re, 'M': 0.0,
         'N_crit': Nc, 'N_panels': 160, 'xtp_u': xu, 'xtp_l': xl}
        for Re  in [1e6, 6e6]
        for Nc, xu, xl in [(9.0, 1.0, 1.0), (3.0, 0.05, 0.05)]
        for a   in np.linspace(-5, 15, 21)
    ]

    for solver in ('xfoil', 'qfoil', 'neuralfoil'):
        records = run_sweep(solver, afl, cases,
                            comm=_comm, rank=_rank, size=_size)
        if _rank == 0:
            print(f"{solver}: {len(records)} records")
            if records:
                r = records[0]
                print(f"  alpha={r['alpha']:.1f}  cl={r['cl']:.4f}  cd={r['cd']:.5f}"
                      f"  Re={r['Re']:.2e}  N_crit={r['N_crit']}")
