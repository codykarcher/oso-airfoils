import pathlib
from oso_airfoils.core.airfoil_family import get_geometry_from_dir, get_all_geometry_from_dir
from oso_airfoils.core.colors import default_color_cycle as dcc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=dcc)

_datfiles = pathlib.Path(__file__).parent.resolve() / 'datfiles'
_perf_dir = pathlib.Path(__file__).resolve().parents[2] / 'data' / pathlib.Path(__file__).parent.name / 'performance_data'
_FAMILY = 'S-series'
_here = pathlib.Path(__file__).parent.resolve()

# Groups derived from Somers (NREL/SR-500-36340) table
_PLOT_GROUPS = [
    {
        'keys':     ['s833', 's834', 's835'],
        'title':    'NREL S-Series: 1–3 m, Variable Speed/Pitch',
        'filename': 'Family_S_1-3m_VSVP.png',
    },
    {
        'keys':     ['s822', 's823'],
        'title':    'NREL S-Series: 3–10 m, Variable Speed/Pitch, Thick',
        'filename': 'Family_S_3-10m_VSVP_Thick.png',
    },
    {
        'keys':     ['s802', 's803', 's804'],
        'title':    'NREL S-Series: 10–20 m, Variable Speed/Pitch, Thin',
        'filename': 'Family_S_10-20m_VSVP_Thin.png',
    },
    {
        'keys':     ['s805a', 's806a', 's807', 's808'],
        'title':    'NREL S-Series: 10–20 m, Stall Regulated, Thin',
        'filename': 'Family_S_10-20m_Stall_Thin.png',
    },
    {
        'keys':     ['s819', 's820', 's821'],
        'title':    'NREL S-Series: 10–20 m, Stall Regulated, Thick',
        'filename': 'Family_S_10-20m_Stall_Thick.png',
    },
    {
        'keys':     ['s809', 's810', 's811'],
        'title':    'NREL S-Series: 20–30 m, Stall Regulated, Thick (Primary/Tip/Root Set 1)',
        'filename': 'Family_S_20-30m_Stall_Thick_1.png',
    },
    {
        'keys':     ['s812', 's813', 's814', 's815'],
        'title':    'NREL S-Series: 20–30 m, Stall Regulated, Thick (Primary/Tip/Root Set 2)',
        'filename': 'Family_S_20-30m_Stall_Thick_2.png',
    },
    {
        'keys':     ['s825', 's826', 's814', 's815'],
        'title':    'NREL S-Series: 20–40 m, Variable Speed/Pitch',
        'filename': 'Family_S_20-40m_VSVP.png',
    },
    {
        'keys':     ['s816', 's817', 's818'],
        'title':    'NREL S-Series: 30–50 m, Stall Regulated, Thick',
        'filename': 'Family_S_30-50m_Stall_Thick.png',
    },
    {
        'keys':     ['s827', 's828', 's818'],
        'title':    'NREL S-Series: 30–50 m, Stall Regulated',
        'filename': 'Family_S_30-50m_Stall.png',
    },
    {
        'keys':     ['s830', 's831', 's832'],
        'title':    'NREL S-Series: 40–50 m, Variable Speed/Pitch, Thick',
        'filename': 'Family_S_40-50m_VSVP_Thick.png',
    },
]


def _normalize_s_name(n):
    """Prepend 's' to bare numeric names, e.g. '809' -> 's809'."""
    return ('s' + n) if n[0].isdigit() else n


def get_geometry(name=None):
    import warnings
    if name is None:
        return get_all_geometry_from_dir(_datfiles)
    if isinstance(name, (list, tuple)):
        result = {}
        for n in name:
            try:
                result[n] = get_geometry_from_dir(_normalize_s_name(n), _datfiles, _FAMILY)
            except ValueError:
                warnings.warn('Airfoil "{}" could not be found, returning None.'.format(n))
                result[n] = None
        return result
    return get_geometry_from_dir(_normalize_s_name(name), _datfiles, _FAMILY)


def get_coordinates(name=None):
    result = get_geometry(name)
    if isinstance(result, dict):
        return {n: (afl.xcoordinates, afl.ycoordinates) if afl is not None else None
                for n, afl in result.items()}
    return result.xcoordinates, result.ycoordinates


def _plot_s_group(group):
    """Plot one S-series group, ordered thinnest to thickest, with (xx%) labels."""
    from metafoil.core.kulfan import Kulfan
    entries = []
    for key in group['keys']:
        fl = key.lower() + '.dat'
        afl = Kulfan()
        afl.readFile(_datfiles / fl)
        tau_val = afl.tau
        if hasattr(tau_val, 'magnitude'):
            tau_val = tau_val.to('').magnitude
        entries.append((tau_val, key.upper(), afl))
    entries.sort(key=lambda e: e[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    for tau_val, name, afl in entries:
        ax.plot(afl.xcoordinates, afl.ycoordinates,
                label='{} ({}%)'.format(name, round(tau_val * 100)))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title(group['title'])
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize='small')
    fig.tight_layout()
    fig.savefig(_here / group['filename'], dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_family():
    for group in _PLOT_GROUPS:
        _plot_s_group(group)



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

