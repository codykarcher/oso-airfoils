#!/usr/bin/env python3
"""
generate_single_json.py

Generate (or overwrite) the performance JSON for one airfoil.

The airfoil is identified by its .dat file.  The output JSON is written to
the sibling performance_data/ directory (created if needed).

Usage:
    python generate_single_json.py du/datfiles/du_91-w2-250.dat
    python generate_single_json.py du/datfiles/du_91-w2-250.dat --override
    python generate_single_json.py du/datfiles/du_91-w2-250.dat --kulfan-order 10
    python generate_single_json.py du/datfiles/du_91-w2-250.dat --perf-dir /custom/path
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent.parent))

DEFAULT_KULFAN_ORDER = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dat_file",
        type=pathlib.Path,
        help="Path to the airfoil .dat file.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite the JSON if it already exists (default: skip).",
    )
    parser.add_argument(
        "--kulfan-order",
        type=int,
        default=DEFAULT_KULFAN_ORDER,
        dest="kulfan_order",
        help=f"Kulfan polynomial order for geometry storage (default: {DEFAULT_KULFAN_ORDER})",
    )
    parser.add_argument(
        "--perf-dir",
        type=pathlib.Path,
        default=None,
        dest="perf_dir",
        help="Output directory for the JSON (default: <dat_file parent>/../performance_data/)",
    )
    args = parser.parse_args()

    dat_file = args.dat_file.resolve()
    if not dat_file.exists():
        print(f"Error: dat file not found: {dat_file}", file=sys.stderr)
        sys.exit(1)

    datfiles_dir = dat_file.parent
    airfoil_id   = dat_file.stem
    perf_dir     = args.perf_dir or (datfiles_dir.parents[2] / 'data' / datfiles_dir.parent.name / "performance_data")
    perf_dir.mkdir(exist_ok=True)
    out_path = perf_dir / f"{airfoil_id}.json"

    if out_path.exists() and not args.override:
        print(f"[exists] {out_path}  (use --override to regenerate)")
        return

    from oso_airfoils.core.airfoil_family import get_geometry_from_dir
    from oso_airfoils.airfoils.generate_empty_jsons import make_geometry_record, make_empty_record

    action = "overwritten" if out_path.exists() else "created"
    try:
        afl      = get_geometry_from_dir(airfoil_id, datfiles_dir,
                                         kulfan_order=args.kulfan_order)
        geometry = make_geometry_record(afl, args.kulfan_order)
    except Exception as exc:
        print(f"[warning] geometry extraction failed — {exc}", file=sys.stderr)
        geometry = None

    record = make_empty_record(airfoil_id, geometry)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[{action}] {out_path}")


if __name__ == "__main__":
    main()
