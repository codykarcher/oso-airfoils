#!/usr/bin/env python3
"""
generate_empty_jsons.py

Walk every family subfolder under this directory. For each folder that contains
a family_functions.py, find all .dat files in datfiles/ and write an empty
performance JSON to performance_data/<airfoil_name>.json.

Usage:
    python generate_empty_jsons.py                         # skip existing files
    python generate_empty_jsons.py --override              # overwrite existing files (regenerates geometry)
    python generate_empty_jsons.py --reset-runs            # wipe runs to [] but keep existing geometry
    python generate_empty_jsons.py --kulfan-order 10      # use 10th-order representation

MPI-parallel (geometry computation is the expensive step):
    mpirun -n 120 python -m mpi4py generate_empty_jsons.py --override
    mpirun -n 120 python -m mpi4py generate_empty_jsons.py --reset-runs
"""

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent.resolve()

DEFAULT_KULFAN_ORDER = 8


def _extract(val):
    """Convert a pint Quantity (scalar or array) to a plain float or list."""
    try:
        mag = val.magnitude
    except AttributeError:
        mag = val
    mag = np.asarray(mag)
    return float(mag) if mag.ndim == 0 else [float(x) for x in mag.flatten()]


def make_geometry_record(airfoil, kulfan_order):
    """Build the geometry dict for one airfoil at the requested Kulfan order."""
    from pint import UnitRegistry; units = UnitRegistry()
    airfoil.chord = 1.0 * units.m          # required for dimensioned properties

    ler = airfoil.leadingEdgeRadius()       # [upper, lower], dimensionless
    return {
        'kulfan_order':      kulfan_order,
        'upperCoefficients': _extract(airfoil.upperCoefficients),
        'lowerCoefficients': _extract(airfoil.lowerCoefficients),
        'TE_gap':            float(airfoil.constants.TE_gap),
        'area':              float(airfoil.area),
        'perimeter':         float(airfoil.perimeter),
        'Ixx':               float(airfoil.Ixx),
        'Iyy':               float(airfoil.Iyy),
        'Izz':               float(airfoil.Izz),
        'LE_radius_upper':   float(ler[0]),
        'LE_radius_lower':   float(ler[1]),
        'tau':               float(airfoil.tau),
        'x_centroid':        float(airfoil.xcentroid),
        'y_centroid':        float(airfoil.ycentroid),
        'taumax_psi':        float(airfoil.taumax_psi),
        'taumax_psi_upper':  float(airfoil.taumax_psi_upper),
        'taumax_psi_lower':  float(airfoil.taumax_psi_lower),
    }


def make_empty_record(airfoil_id: str, geometry: dict) -> dict:
    return {
        "airfoil_id": airfoil_id,
        "geometry":   geometry,
        "runs":       [],
    }


def generate(override: bool = False, kulfan_order: int = DEFAULT_KULFAN_ORDER,
             reset_runs: bool = False, rank: int = 0, size: int = 1) -> None:
    from oso_airfoils.core.airfoil_family import get_geometry_from_dir

    # ── collect full work list (identical on every rank) ────────────────
    # Each item is (family_name, airfoil_id, datfiles_dir, out_path, reset_only)
    # reset_only=True  → load existing file, wipe runs, keep geometry
    # reset_only=False → full create / overwrite including geometry
    work = []
    for family_dir in sorted(HERE.iterdir()):
        if not family_dir.is_dir():
            continue
        if not (family_dir / "family_functions.py").exists():
            continue

        datfiles_dir = family_dir / "datfiles"
        perf_dir     = HERE.parent / 'data' / family_dir.name / "performance_data"

        if not datfiles_dir.is_dir():
            if rank == 0:
                print(f"  [skip] {family_dir.name}: no datfiles/ directory")
            continue

        perf_dir.mkdir(exist_ok=True)

        dat_files = sorted(datfiles_dir.glob("*.dat"))
        if not dat_files:
            if rank == 0:
                print(f"  [skip] {family_dir.name}: no .dat files found")
            continue

        for dat in dat_files:
            airfoil_id = dat.stem
            out_path   = perf_dir / f"{airfoil_id}.json"
            if out_path.exists():
                if override:
                    reset_only = False
                elif reset_runs:
                    reset_only = True
                else:
                    if rank == 0:
                        print(f"  [exists] {family_dir.name}/{out_path.name}")
                    continue
            else:
                reset_only = False
            work.append((family_dir.name, airfoil_id, datfiles_dir, out_path, reset_only))

    # ── each rank processes its slice ───────────────────────────────────
    for family_name, airfoil_id, datfiles_dir, out_path, reset_only in work[rank::size]:
        if reset_only:
            with open(out_path) as f:
                record = json.load(f)
            record['runs'] = []
            with open(out_path, 'w') as f:
                json.dump(record, f, indent=2)
            print(f"  [reset] {family_name}/{out_path.name}")
        else:
            action = "overwritten" if out_path.exists() else "created"
            try:
                afl      = get_geometry_from_dir(airfoil_id, datfiles_dir,
                                                 kulfan_order=kulfan_order)
                geometry = make_geometry_record(afl, kulfan_order)
            except Exception as exc:
                print(f"  [warning] {family_name}/{airfoil_id}: geometry failed — {exc}")
                geometry = None

            record = make_empty_record(airfoil_id, geometry)
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  [{action}] {family_name}/{out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing JSON files (default: skip them)",
    )
    parser.add_argument(
        "--reset-runs",
        action="store_true",
        dest="reset_runs",
        help="For existing JSONs, wipe runs to [] while keeping geometry unchanged.",
    )
    parser.add_argument(
        "--kulfan-order",
        type=int,
        default=DEFAULT_KULFAN_ORDER,
        dest="kulfan_order",
        help=f"Kulfan polynomial order for geometry storage (default: {DEFAULT_KULFAN_ORDER})",
    )
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        _comm = MPI.COMM_WORLD
        _rank = _comm.Get_rank()
        _size = _comm.Get_size()
    except ImportError:
        _rank, _size = 0, 1

    generate(override=args.override, kulfan_order=args.kulfan_order,
             reset_runs=args.reset_runs, rank=_rank, size=_size)

