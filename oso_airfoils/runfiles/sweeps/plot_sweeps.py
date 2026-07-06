#!/usr/bin/env python3
"""plot_sweeps.py — Plot all data from sweeps/*.json.

Generates (all saved to sweeps/):
  1. Per-airfoil polars PNG for every JSON in sweeps/
     XFoil + NeuralFoil, clean (Ncrit=9) + rough (Ncrit=3), with Cpmin.

  2. Per-thickness comparison PNG for each OSO-2026-HT1 member.
     Shows the HT1 airfoil alongside similarly-thick airfoils (±1.5 %)
     from all other families present in sweeps/.  XFoil, clean + rough.

Run:
    python plot_sweeps.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
# Script lives inside sweeps/ — all output goes to the same directory
SWEEPS_DIR = pathlib.Path(__file__).resolve().parent

# ── condition constants ────────────────────────────────────────────────────────
RE            = 1.5e6
TURB_CLEAN    = [9.0, 1.0,  1.0 ]
TURB_ROUGH    = [3.0, 0.05, 0.05]
TURB_BOTH     = [TURB_CLEAN, TURB_ROUGH]
# Design CL varies with thickness for the HT1 family (and used as default for others)
_CL_DESIGN_BY_TAU_PCT: dict[int, float] = {
    18: 1.3,
    21: 1.3,
    24: 1.3,
    27: 1.3,
    30: 1.3,
    33: 1.2,
    36: 1.1,
}
_CL_DESIGN_DEFAULT = 1.3

TAU_TOL       = 0.015          # ±1.5 % thickness tolerance for grouping
HT1_PREFIX    = 'OSO-2026-HT1'

# ── oso_airfoils imports ───────────────────────────────────────────────────────
from oso_airfoils.postprocessing.polars import polars_compare, polars_rainbow, _filter_records
from oso_airfoils.postprocessing.runners import _find_stem_in_tree, _load_kulfan
from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_sweeps() -> dict[str, dict]:
    """Return {stem: json_doc} for all JSONs in SWEEPS_DIR."""
    docs = {}
    for jf in sorted(SWEEPS_DIR.glob('*.json')):
        with open(jf) as fh:
            docs[jf.stem] = json.load(fh)
    return docs


def _get_tau(doc: dict, stem: str) -> float | None:
    """Return tau (thickness/chord) from the JSON geometry section, or estimate from stem name."""
    geom = doc.get('geometry')
    if geom and geom.get('tau') is not None:
        return float(geom['tau'])
    # Fallback: extract the last numeric group in the stem
    # e.g. 'FFA-W3-241' → 241 → 0.241, 'mhkf1-240' → 240 → 0.240, 'riso-b-23' → 23 → 0.23
    nums = re.findall(r'\d+', stem)
    if not nums:
        return None
    last = int(nums[-1])
    # Numbers ≥ 100 are interpreted as 10× percentage (e.g. 241 → 24.1 %)
    if last >= 100:
        return last / 1000.0
    # Numbers < 100 are direct percentage (e.g. 23 → 23 %)
    return last / 100.0


def _kulfan_for(stem: str):
    """Load a Kulfan from the installed oso_airfoils datfiles; return None on failure."""
    try:
        family, found_stem = _find_stem_in_tree(stem, _DEFAULT_AFL_ROOT)
        return _load_kulfan(family, found_stem, _DEFAULT_AFL_ROOT)
    except Exception:
        pass
    return None


def _has_data(records: list, re: float, turb_cases: list, tools: list) -> bool:
    """Return True only if every (re, turb_case, tool) combo has ≥1 matching record."""
    for tc in turb_cases:
        nc, xu, xl = tc[0], tc[1], tc[2]
        for tool in tools:
            if not _filter_records(records, re, nc, xu, xl, tool):
                return False
    return True


def _available_tools(records: list, re: float, turb_cases: list,
                      candidates: list[str]) -> list[str]:
    """Return the subset of candidate tools for which ALL turb_cases have ≥1 record."""
    out = []
    for tool in candidates:
        if all(
            _filter_records(records, re, tc[0], tc[1], tc[2], tool)
            for tc in turb_cases
        ):
            out.append(tool)
    return out


def _cl_design_for_tau(tau: float | None) -> float:
    """Return the appropriate design CL for a given thickness ratio."""
    if tau is None:
        return _CL_DESIGN_DEFAULT
    tau_pct = round(tau * 100)
    return _CL_DESIGN_BY_TAU_PCT.get(tau_pct, _CL_DESIGN_DEFAULT)


def _plt_close_all():
    plt.close('all')


# ── 1. individual per-airfoil plots ───────────────────────────────────────────

def plot_individual(docs: dict[str, dict]) -> None:
    """One polars_compare PNG per airfoil.  XFoil + NeuralFoil, clean + rough."""
    print(f"\n── Individual polar plots ({len(docs)} airfoils) ──────────────────")

    for stem, doc in docs.items():
        records  = doc.get('runs', [])
        out_path = SWEEPS_DIR / f'{stem}_polar.png'

        # Determine which tools actually have complete data
        tools = _available_tools(records, RE, TURB_BOTH, ['xfoil', 'neuralfoil'])
        if not tools:
            print(f'  SKIP {stem}: no complete data for any tool')
            continue

        # Load geometry (optional — skipped gracefully if not found)
        afl  = _kulfan_for(stem)
        geom = {stem: afl} if afl is not None else None

        data_dict = {stem: records}
        tau       = _get_tau(doc, stem)
        cl_design = _cl_design_for_tau(tau)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fig = polars_compare(
                    data_dict,
                    reynolds_numbers=[RE],
                    turb_cases=TURB_BOTH,
                    tools=tools,
                    figure_path=None,
                    geometry_dict=geom,
                    show_cpmin=True,
                    cl_design=cl_design,
                )
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  {stem}  →  {out_path.name}  (tools: {tools})')
        except Exception as exc:
            plt.close('all')
            print(f'  ERROR {stem}: {exc}')


# ── 2. HT1 thickness-comparison plots ─────────────────────────────────────────

def plot_ht1_comparisons(docs: dict[str, dict]) -> None:
    """One polars_compare PNG per HT1 thickness, overlaying similar-tau airfoils."""
    ht1_stems = sorted(s for s in docs if s.startswith(HT1_PREFIX))
    if not ht1_stems:
        print('\nNo HT1 airfoils found in sweeps/ — skipping comparison plots.')
        return

    print(f"\n── HT1 comparison plots ({len(ht1_stems)} thicknesses) ─────────────")

    # Build tau table for all airfoils in sweeps/
    tau_map: dict[str, float] = {}
    for stem, doc in docs.items():
        t = _get_tau(doc, stem)
        if t is not None:
            tau_map[stem] = t

    for ht1_stem in ht1_stems:
        tau_ht1 = tau_map.get(ht1_stem)
        if tau_ht1 is None:
            print(f'  SKIP {ht1_stem}: tau unknown')
            continue

        tau_pct = round(tau_ht1 * 100)

        # Find similar-tau airfoils from OTHER families (non-HT1)
        peers = [ht1_stem]   # HT1 member always first
        for stem, tau in sorted(tau_map.items()):
            if stem.startswith(HT1_PREFIX):
                continue
            if abs(tau - tau_ht1) <= TAU_TOL:
                peers.append(stem)

        # Filter to airfoils that have xfoil data for both conditions
        # (neuralfoil is included if present but not required)
        valid_peers = []
        for stem in peers:
            recs = docs[stem].get('runs', [])
            if _has_data(recs, RE, TURB_BOTH, ['xfoil']):
                valid_peers.append(stem)
            else:
                print(f'  NOTE: {stem} (tau={tau_map.get(stem, "?"):.3f}) '
                      f'missing xfoil data — excluded from T{tau_pct} comparison')

        if not valid_peers:
            print(f'  SKIP T{tau_pct}: no airfoils with complete xfoil data')
            continue

        data_dict = {s: docs[s]['runs'] for s in valid_peers}
        geom_dict = {}
        for s in valid_peers:
            afl = _kulfan_for(s)
            if afl is not None:
                geom_dict[s] = afl

        # Use whichever tools all valid_peers have complete data for
        tools = _available_tools(
            docs[valid_peers[0]].get('runs', []), RE, TURB_BOTH,
            ['xfoil', 'neuralfoil'],
        )
        for s in valid_peers[1:]:
            tools = [t for t in tools
                     if _available_tools(docs[s].get('runs', []), RE, TURB_BOTH, [t])]
        if not tools:
            tools = ['xfoil']   # fallback

        out_path = SWEEPS_DIR / f'compare_T{tau_pct:02d}.png'

        tau_strs  = ', '.join(
            f'{s} ({tau_map.get(s, 0):.3f})'
            for s in valid_peers
        )
        cl_design = _CL_DESIGN_BY_TAU_PCT.get(tau_pct, _CL_DESIGN_DEFAULT)
        print(f'  T{tau_pct}: [{tau_strs}]  tools: {tools}  cl_design: {cl_design}')

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fig = polars_compare(
                    data_dict,
                    reynolds_numbers=[RE],
                    turb_cases=TURB_BOTH,
                    tools=tools,
                    figure_path=None,
                    geometry_dict=geom_dict if geom_dict else None,
                    show_cpmin=True,
                    cl_design=cl_design,
                )
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'    →  {out_path.name}')
        except Exception as exc:
            plt.close('all')
            print(f'  ERROR T{tau_pct}: {exc}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not SWEEPS_DIR.is_dir():
        print(f'ERROR: sweeps/ directory not found at {SWEEPS_DIR}')
        sys.exit(1)

    docs = _load_sweeps()
    if not docs:
        print(f'No JSON files found in {SWEEPS_DIR}')
        sys.exit(1)

    print(f'Loaded {len(docs)} sweep JSONs from {SWEEPS_DIR}')

    plot_individual(docs)
    plot_ht1_comparisons(docs)

    print('\nDone.')


if __name__ == '__main__':
    main()
