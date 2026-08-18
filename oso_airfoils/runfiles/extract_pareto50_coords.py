"""
extract_pareto50_coords.py
--------------------------
For each non-bad run folder in data/cases_111_to_120/case_116, loads the
final generation population JSON, identifies the 50% point along the Pareto
front (sorted by rough LoD, same convention as oso_polar), and writes out
the airfoil as a coordinate (.dat) file in that run folder.

Naming:  <run_folder>/<run_prefix>_rf050.dat
         e.g. c116_t18_k16_n752_l13_e15_rf050.dat

Output format: upper surface (TE→LE) then lower surface (LE→TE), two columns
of floats in scientific notation – identical to Kulfan.write2file().
"""

import json
import pathlib
import sys

import natsort
import numpy as np

# ── insert project root so local imports work when run directly ───────────────
_HERE   = pathlib.Path(__file__).resolve().parent
_PROJ   = _HERE.parent.parent          # …/oso-airfoils
sys.path.insert(0, str(_PROJ))

from metafoil.core.kulfan import Kulfan

# ── configuration ─────────────────────────────────────────────────────────────
CASE_DIR       = _PROJ / 'oso_airfoils' / 'data' / 'cases_111_to_120' / 'case_116'
OUTPUT_DIR     = _PROJ / 'oso_airfoils' / 'airfoils' / 'oso_2026_ht1' / 'datfiles'
N_PARETO       = 21        # same as oso_polar: 0.00 → 1.00 in steps of 0.05
TARGET_FRAC    = 0.50      # fraction along the Pareto front to extract
N_COORD_POINTS = 200       # points per surface for the output coordinate file
# ─────────────────────────────────────────────────────────────────────────────


def _sample_pareto(pareto: list, n: int) -> list:
    """Return up to n members evenly sampled by index along the Pareto front."""
    if len(pareto) <= n:
        return pareto
    indices = np.unique(np.round(np.linspace(0, len(pareto) - 1, n)).astype(int))
    return [pareto[i] for i in indices]


def _last_json(run_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the highest-generation population JSON in run_dir, or None."""
    candidates = natsort.natsorted(
        list(run_dir.glob('population_*.json')), alg=natsort.ns.IGNORECASE
    )
    return pathlib.Path(candidates[-1]) if candidates else None


def _run_prefix(run_dir: pathlib.Path) -> str:
    """Strip trailing timestamp from folder name to make a clean file prefix."""
    # e.g. 'c116_t18_k16_n752_l13_e15__2026_06_15_14-18-2267'
    #   →  'c116_t18_k16_n752_l13_e15'
    name = run_dir.name
    # remove the datetime suffix (double-underscore onward)
    if '__' in name:
        name = name[:name.index('__')]
    return name.rstrip('_')


def process_run(run_dir: pathlib.Path) -> None:
    json_path = _last_json(run_dir)
    if json_path is None:
        print(f'  [skip] no population JSON found in {run_dir.name}')
        return

    data   = json.loads(json_path.read_text())
    params = data['input_parameters']
    te_gap = float(params.get('TE_gap', 0.0))

    # Build and sort the Pareto front by rough LoD (ascending)
    pareto_raw = [p for p in data['population'] if p['pareto_index'] == 1]
    if not pareto_raw:
        print(f'  [skip] no Pareto members in {run_dir.name}')
        return

    pareto = sorted(pareto_raw, key=lambda p: p['LoD_rough_at_design'])
    pareto = _sample_pareto(pareto, N_PARETO)

    # Pick the 50% point
    n = len(pareto)
    idx50 = int(round(TARGET_FRAC * (n - 1)))
    member = pareto[idx50]
    rf_actual = idx50 / max(n - 1, 1)

    # Build Kulfan object
    afl = Kulfan(TE_gap=te_gap, Npoints=N_COORD_POINTS)
    afl.upperCoefficients = member['K_upper']
    afl.lowerCoefficients = member['K_lower']

    # Write .dat file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tau_pct  = int(round(member['tau'] * 100))
    out_path = OUTPUT_DIR / f'OSO-2026-HT1-T{tau_pct:02d}.dat'
    afl.write2file(str(out_path))

    print(
        f'  {run_dir.name}\n'
        f'    rf_actual={rf_actual:.4f}  '
        f'LoD_rough={member["LoD_rough_at_design"]:.2f}  '
        f'LoD_clean={member["LoD_clean_at_design"]:.2f}  '
        f'tau={member["tau"]:.4f}  TE_gap={te_gap:.6f}\n'
        f'    → {out_path.name}'
    )


def main() -> None:
    if not CASE_DIR.is_dir():
        print(f'Error: case directory not found:\n  {CASE_DIR}')
        sys.exit(1)

    run_dirs = sorted(
        d for d in CASE_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('bad_') and not d.name.startswith('.')
    )

    if not run_dirs:
        print('No valid (non-bad) run directories found.')
        sys.exit(0)

    print(f'Processing {len(run_dirs)} run folder(s) in:\n  {CASE_DIR}\n')
    for run_dir in run_dirs:
        process_run(run_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
