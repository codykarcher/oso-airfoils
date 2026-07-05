import pathlib
from oso_airfoils.core.airfoil_family import get_geometry_from_dir, get_all_geometry_from_dir, plot_family_from_dir
from oso_airfoils.core.colors import default_color_cycle as dcc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=dcc)

_datfiles = pathlib.Path(__file__).parent.resolve() / 'datfiles'
_perf_dir = pathlib.Path(__file__).resolve().parents[2] / 'data' / pathlib.Path(__file__).parent.name / 'performance_data'
_FAMILY = 'RISO-A'


def get_geometry(name):
    return get_geometry_from_dir(name, _datfiles, _FAMILY)


def get_coordinates(name):
    result = get_geometry(name)
    if isinstance(result, dict):
        return {n: (afl.xcoordinates, afl.ycoordinates) if afl is not None else None
                for n, afl in result.items()}
    return result.xcoordinates, result.ycoordinates


def plot_family():
    plot_family_from_dir(
        _datfiles,
        title='Risø-A Airfoil Family',
        save_path=pathlib.Path(__file__).parent.resolve() / 'Family_RISO-A.png',
    )



def ingest_performance_data(mode, name=None, **kwargs):
    from oso_airfoils.core.ingest_data import ingest, find_json_path
    json_path = find_json_path(name, _datfiles, _perf_dir) if name is not None else None
    ingest(mode, name, json_path, _datfiles, perf_dir=_perf_dir, **kwargs)


if __name__ == "__main__":
    import numpy as np

    _PLOT_FAMILY       = True
    _INGEST_RFOIL_DATA = False
    _INGEST_XFOIL_DATA_ALPHA = False
    _INGEST_XFOIL_DATA_CL  = False

    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _comm, _rank, _size = None, 0, 1

    if _rank == 0:
        if _PLOT_FAMILY:
            plot_family()

        if _INGEST_RFOIL_DATA:
            rfoil_dir = _perf_dir / 'rfoil'
            ingest_performance_data('rfoil_data', polar_files=sorted(rfoil_dir.glob('*.dat')))

    if _INGEST_XFOIL_DATA_ALPHA:
        from itertools import product
        from oso_airfoils.core.sweep import run_sweep
        from oso_airfoils.core.ingest_data import find_json_path, _merge_runs, _NumpyEncoder
        import json

        # ── parameter decisions ─────────────────────────────────────────
        airfoils   = sorted(p.stem for p in _datfiles.glob('*.dat'))
        Re_list    = [0.5e6, 1e6, 5e6, 10e6, 15e6, 20e6]
        alpha_list = list(np.linspace(-30, 30, 61))
        M_list     = [0.0]
        conditions = [(9.0, 1.0,   1.0  ),   # (N_crit, xtp_u, xtp_l) clean
                      (3.0, 0.05,  0.05 )]    #                         rough
        N_panels   = 160
        nf_model   = 'xxxlarge'
        solvers    = ('xfoil', 'neuralfoil')
        # ───────────────────────────────────────────────────────────────

        run_cases = [
            {'alpha': float(a), 'Re': Re, 'M': M, 'N_crit': Nc,
             'xtp_u': xu, 'xtp_l': xl, 'N_panels': N_panels}
            for Re, M, (Nc, xu, xl), a in product(Re_list, M_list, conditions, alpha_list)
        ]

        for afl_name in airfoils:
            afl = get_geometry(afl_name)
            for solver in solvers:
                records = run_sweep(solver, afl, run_cases,
                                    comm=_comm, rank=_rank, size=_size,
                                    nf_model=nf_model)
                if _rank == 0 and records:
                    jpath = find_json_path(afl_name, _datfiles, _perf_dir)
                    with open(jpath) as f:
                        jdata = json.load(f)
                    merged, na, ns, nu, nc = _merge_runs(jdata['runs'], records, source_hint=jpath.name)
                    jdata['runs'] = merged
                    with open(jpath, 'w') as f:
                        json.dump(jdata, f, indent=2, cls=_NumpyEncoder)
                    parts = [f"{na} added"]
                    if ns:
                        parts.append(f"{ns} skipped (duplicate)")
                    if nu:
                        parts.append(f"{nu} null-populated")
                    if nc:
                        parts.append(f"{nc} conflict(s)")
                    print(f"  [{solver}] {', '.join(parts)} → {jpath.name}")

    if _INGEST_XFOIL_DATA_CL:
        from itertools import product
        from oso_airfoils.core.sweep import run_sweep
        from oso_airfoils.core.ingest_data import find_json_path, _merge_runs, _NumpyEncoder
        import json

        # ── parameter decisions ──────────────────────────────────────────────────
        airfoils   = sorted(p.stem for p in _datfiles.glob('*.dat'))
        Re_list    = [0.5e6, 1e6, 5e6, 10e6, 15e6, 20e6]
        cl_list    = list(np.round(np.arange(0.0, 3.0 + 0.05, 0.1), 4))
        M_list     = [0.0]
        conditions = [(9.0, 1.0,   1.0  ),   # (N_crit, xtp_u, xtp_l) clean
                      (3.0, 0.05,  0.05 )]    #                         rough
        N_panels   = 160
        nf_model   = 'xxxlarge'
        solvers    = ('xfoil',)
        # ───────────────────────────────────────────────────────────────

        run_cases = [
            {'mode': 'cl', 'cl': float(cl), 'Re': Re, 'M': M, 'N_crit': Nc,
             'xtp_u': xu, 'xtp_l': xl, 'N_panels': N_panels}
            for Re, M, (Nc, xu, xl), cl in product(Re_list, M_list, conditions, cl_list)
        ]

        for afl_name in airfoils:
            afl = get_geometry(afl_name)
            for solver in solvers:
                records = run_sweep(solver, afl, run_cases,
                                    comm=_comm, rank=_rank, size=_size,
                                    nf_model=nf_model)
                if _rank == 0 and records:
                    jpath = find_json_path(afl_name, _datfiles, _perf_dir)
                    with open(jpath) as f:
                        jdata = json.load(f)
                    merged, na, ns, nu, nc = _merge_runs(jdata['runs'], records, source_hint=jpath.name)
                    jdata['runs'] = merged
                    with open(jpath, 'w') as f:
                        json.dump(jdata, f, indent=2, cls=_NumpyEncoder)
                    parts = [f"{na} added"]
                    if ns:
                        parts.append(f"{ns} skipped (duplicate)")
                    if nu:
                        parts.append(f"{nu} null-populated")
                    if nc:
                        parts.append(f"{nc} conflict(s)")
                    print(f"  [{solver}] {', '.join(parts)} → {jpath.name}")
