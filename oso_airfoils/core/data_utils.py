"""oso_airfoils.core.data_utils
==============================

Utilities for loading and filtering run records from performance JSON files.

Public API
----------
load_runs(family, stem, ...)
    Return all run records from a performance JSON file.

find_record(family, stem, ...)
    Return the first run record matching the supplied filter conditions.
"""
from __future__ import annotations

import json
import pathlib

# Default root for airfoil geometry/dat files: oso_airfoils/airfoils/
_DEFAULT_AFL_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent / 'airfoils'

# Default root for performance JSON files: oso_airfoils/data/
_DEFAULT_PERF_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent / 'data'


def load_runs(
    family: str,
    stem: str,
    *,
    afl_root: pathlib.Path | str | None = None,
) -> list[dict]:
    """Return all run records from a performance JSON file.

    Parameters
    ----------
    family : str
        Airfoil family sub-directory (e.g. ``'ffa'``).
    stem : str
        File stem without extension (e.g. ``'FFA-W3-241'``).
    afl_root : path-like, optional
        Root directory containing family sub-directories.
        Defaults to ``oso_airfoils/airfoils/``.

    Returns
    -------
    list[dict]
        All run records from ``<afl_root>/<family>/performance_data/<stem>.json``.
    """
    root = pathlib.Path(afl_root) if afl_root is not None else _DEFAULT_PERF_ROOT
    jf = root / family / 'performance_data' / f'{stem}.json'
    return json.loads(jf.read_text())['runs']


def find_record(
    family: str,
    stem: str,
    *,
    alpha: float | None = None,
    Re: float | None = None,
    N_crit: float | None = None,
    xtp_top: float | None = None,
    xtp_bot: float | None = None,
    source: str | None = None,
    require_bl: bool = False,
    afl_root: pathlib.Path | str | None = None,
    tol: float = 1e-3,
) -> dict:
    """Return the first run record matching all supplied filter conditions.

    Parameters
    ----------
    family : str
        Airfoil family sub-directory.
    stem : str
        File stem without extension.
    alpha : float, optional
        Angle of attack to match (tolerance 1e-6 °).
    Re : float, optional
        Reynolds number to match (relative tolerance *tol*).
    N_crit, xtp_top, xtp_bot : float, optional
        Further filter conditions (absolute tolerance *tol*).
    source : str, optional
        Simulation source: ``'xfoil'``, ``'rfoil'``, or ``'neuralfoil'``.
    require_bl : bool
        If ``True``, skip records without ``bl_data``.
    afl_root : path-like, optional
        Root of airfoil data.  Defaults to ``oso_airfoils/airfoils/``.
    tol : float
        Tolerance used for Re (relative) and N_crit / xtp_* (absolute).

    Returns
    -------
    dict
        The first matching run record.

    Raises
    ------
    ValueError
        When no matching record is found.
    """
    runs = load_runs(family, stem, afl_root=afl_root)
    for r in runs:
        if require_bl and r.get('bl_data') is None:
            continue
        if source  is not None and r.get('source') != source:
            continue
        if alpha   is not None and abs(r.get('alpha',   999) - alpha)         > 1e-6:
            continue
        if Re      is not None and abs(r.get('Re',        0) - Re) / max(Re, 1) > tol:
            continue
        if N_crit  is not None and abs(r.get('N_crit',   0) - N_crit)          > tol:
            continue
        if xtp_top is not None and abs(r.get('xtp_top',  0) - xtp_top)         > tol:
            continue
        if xtp_bot is not None and abs(r.get('xtp_bot',  0) - xtp_bot)         > tol:
            continue
        return r

    conds = []
    if alpha   is not None: conds.append(f'alpha={alpha}')
    if Re      is not None: conds.append(f'Re={Re:.2e}')
    if N_crit  is not None: conds.append(f'N_crit={N_crit}')
    if xtp_top is not None: conds.append(f'xtp_top={xtp_top}')
    if xtp_bot is not None: conds.append(f'xtp_bot={xtp_bot}')
    if source  is not None: conds.append(f'source={source}')
    if require_bl:          conds.append('require_bl=True')
    raise ValueError(
        f'No record found in {stem} matching: {", ".join(conds)}'
    )
