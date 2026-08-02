"""oso_airfoils.postprocessing.runners
======================================

Convenience wrappers that fetch-or-compute run records and call the
corresponding plotting function.

Data-loading priority
---------------------
For every (airfoil, Re, turbulence-case, tool) combination the runner first
searches the on-disk JSON performance file.  A combination is considered
*covered* when at least one matching record is found.  Missing combinations
are computed via the requested solver, and the new records may optionally be
appended back to the JSON file (``save_data=True``).

Public API
----------
run_and_plot_polars_compare(...)
run_and_plot_polars_rainbow(...)
run_and_plot_boundary_layer_compare(...)
run_and_plot_boundary_layer_rainbow(...)
"""
from __future__ import annotations

import json
import pathlib
import re as _re
from typing import Any

import numpy as np

from oso_airfoils.core.display_names import pretty_name as _pretty_name

from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
from metafoil.core.kulfan import Kulfan
from oso_airfoils.postprocessing.polars import polars_compare, polars_rainbow
from oso_airfoils.postprocessing.boundary_layer import (
    boundary_layer_compare,
    boundary_layer_rainbow,
)


# ── path helpers ──────────────────────────────────────────────────────────────

def _root(afl_root) -> pathlib.Path:
    return pathlib.Path(afl_root) if afl_root is not None else _DEFAULT_AFL_ROOT


def _perf_root_fn(perf_root) -> pathlib.Path:
    return pathlib.Path(perf_root) if perf_root is not None else _DEFAULT_PERF_ROOT


def _json_path(family: str, stem: str, perf_root) -> pathlib.Path:
    return _perf_root_fn(perf_root) / family / 'performance_data' / f'{stem}.json'


def _dat_path(family: str, stem: str, afl_root) -> pathlib.Path:
    return _root(afl_root) / family / 'datfiles' / f'{stem}.dat'


def _load_kulfan(family: str, stem: str, afl_root) -> Kulfan:
    afl = Kulfan()
    afl.readFile(str(_dat_path(family, stem, afl_root)))
    return afl


# ── spec resolution ───────────────────────────────────────────────────────────

def _find_stem_in_tree(name: str, afl_root) -> tuple:
    """Search every family sub-directory for a datfile matching *name*.

    Matching is case-insensitive and treats hyphens, underscores, and spaces
    as equivalent, so ``'DU-91-w2-250'``, ``'du_91_w2_250'``, and
    ``'du 91 w2 250'`` all resolve to the same file.

    Returns ``(family, stem)`` where *stem* is the actual filename stem on
    disk.  Raises ``FileNotFoundError`` or ``ValueError`` (ambiguous).
    """
    def _norm(s: str) -> str:
        return _re.sub(r'[-_ ]+', '-', s.lower())

    root = _root(afl_root)
    norm_name = _norm(name)
    matches = []
    for fdir in sorted(root.iterdir()):
        if not fdir.is_dir():
            continue
        dfiles_dir = fdir / 'datfiles'
        if not dfiles_dir.is_dir():
            continue
        for dat in dfiles_dir.glob('*.dat'):
            if _norm(dat.stem) == norm_name:
                matches.append((fdir.name, dat.stem))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Airfoil name {name!r} is ambiguous — found in families: "
            f"{[m[0] for m in matches]}. Add a family= key to disambiguate."
        )
    raise FileNotFoundError(
        f"No datfile found for airfoil {name!r} under {root}."
    )


def _is_coordinate_pair(arr: np.ndarray) -> bool:
    """True when *arr* looks like x-coordinates (spans ~0 to ~1).

    Coordinate arrays always pass through x ≈ 0 and x ≈ 1 (leading/trailing
    edge), giving min ≈ 0 and max ≈ 1.  Kulfan coefficient arrays are small
    numbers (~0–1) but are short (8-12 elements) and do not reach both bounds.
    """
    return len(arr) > 10 and float(np.min(arr)) < 0.05 and float(np.max(arr)) > 0.95


def _resolve_spec(display_name: str, spec, afl_root) -> tuple:
    """Return ``(family, stem, kulfan_or_None)`` for an airfoil spec value.

    Accepted spec forms
    -------------------
    ``str``
        Unique airfoil name; auto-resolved to ``(family, stem)`` by searching
        the airfoil tree.  JSON cache and geometry are loaded from disk.
    ``Kulfan``
        Used directly.  JSON cache is bypassed; polars always computed fresh.
    ``(upper_arr, lower_arr)``  — 1-D arrays
        Kulfan coefficients.  A ``Kulfan`` object is constructed and used
        directly (no cache).
    ``(x_arr, y_arr)``  — 1-D arrays where *x* spans ~0 → 1
        Airfoil coordinates.  A ``Kulfan`` is fitted via
        ``fit2coordinates`` and used directly (no cache).
    """
    if isinstance(spec, str):
        family, stem = _find_stem_in_tree(spec, afl_root)
        return family, stem, None

    if isinstance(spec, Kulfan):
        return '', '', spec

    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        a0 = np.asarray(spec[0], dtype=float)
        a1 = np.asarray(spec[1], dtype=float)
        if a0.ndim == 1 and a1.ndim == 1:
            if _is_coordinate_pair(a0):
                afl = Kulfan()
                afl.fit2coordinates(a0, a1)
            else:
                # metafoil's Kulfan takes snake_case constructor kwargs; the
                # camelCase names survive only as attribute aliases, so passing them
                # to __init__ raises. This is the (upper_coefs, lower_coefs) entry
                # form -- the one a caller uses to plot a raw design vector.
                afl = Kulfan()
                afl.upperCoefficients = a0
                afl.lowerCoefficients = a1
            return '', '', afl

    raise TypeError(
        f"Unrecognised airfoil spec for {display_name!r}: {type(spec).__name__}. "
        "Expected: str (name), Kulfan, (upper_coefs, lower_coefs), or (x_coords, y_coords)."
    )


def _resolve_entry(entry, idx: int, afl_root) -> tuple:
    """Resolve one element of the ``airfoils`` list.

    Returns ``(display_name, family, stem, kulfan_or_None)``.

    Accepted forms
    --------------
    ``str``
        Unique airfoil name; looked up in the tree.  Display name = name.
    ``Kulfan``
        Used directly.  Display name = ``'Airfoil {idx+1}'``.
    ``(arr, arr)``
        Kulfan coefficients or coordinate pair.  Display name auto-generated.
    ``[name_str, geometry]``
        Explicit display name + geometry (str name, Kulfan, or array pair).
    """
    # [name, geometry] — list whose first element is a string
    if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str):
        display_name = entry[0]
        family, stem, kulfan = _resolve_spec(display_name, entry[1], afl_root)
        return display_name, family, stem, kulfan

    # Plain name string — apply pretty formatting to the display label
    if isinstance(entry, str):
        family, stem = _find_stem_in_tree(entry, afl_root)
        return _pretty_name(entry), family, stem, None

    # Kulfan or array pair — auto-name
    display_name = f'Airfoil {idx + 1}'
    family, stem, kulfan = _resolve_spec(display_name, entry, afl_root)
    return display_name, family, stem, kulfan


def _resolve_ref_entry(entry, idx: int, afl_root) -> tuple:
    """Resolve one element of a ``reference_airfoils`` list.

    Returns ``(display_name, family, stem, kulfan_or_None, color)``.

    Accepted forms
    --------------
    ``str``
        Name lookup, color ``'k'``.
    ``(name_str, color_str)``
        Both strings: name lookup + explicit colour.
    ``[name_str, geometry]``
        Explicit name + geometry, color ``'k'``.
    ``[name_str, geometry, color_str]``
        Explicit name + geometry + colour.
    Bare ``Kulfan`` or ``(arr, arr)``
        Geometry only, auto-name, color ``'k'``.
    """
    # Plain name string — apply pretty formatting to the display label
    if isinstance(entry, str):
        family, stem = _find_stem_in_tree(entry, afl_root)
        return _pretty_name(entry), family, stem, None, 'k'

    # (name_str, color_str) — shorthand for named airfoil + colour. This is a
    # name lookup just like the plain-string case above, so apply the same
    # pretty formatting to the legend label (osopolar -c passes (stem, 'k')).
    if (isinstance(entry, (list, tuple)) and len(entry) == 2
            and isinstance(entry[0], str) and isinstance(entry[1], str)):
        name, color = entry[0], entry[1]
        family, stem = _find_stem_in_tree(name, afl_root)
        return _pretty_name(name), family, stem, None, color

    # [name_str, geometry] or [name_str, geometry, color_str]
    if (isinstance(entry, (list, tuple)) and len(entry) in (2, 3)
            and isinstance(entry[0], str) and not isinstance(entry[1], str)):
        display_name = entry[0]
        color = entry[2] if len(entry) == 3 else 'k'
        family, stem, kulfan = _resolve_spec(display_name, entry[1], afl_root)
        return display_name, family, stem, kulfan, color

    # Bare geometry — auto-name
    display_name = f'Ref {idx + 1}'
    family, stem, kulfan = _resolve_spec(display_name, entry, afl_root)
    return display_name, family, stem, kulfan, 'k'


# ── record I/O ────────────────────────────────────────────────────────────────

def _existing_records(family: str, stem: str, perf_root) -> list[dict]:
    """Return existing records from the JSON file; [] if absent or unreadable."""
    jf = _json_path(family, stem, perf_root)
    if not jf.exists():
        return []
    try:
        return json.loads(jf.read_text()).get('runs', [])
    except Exception:
        return []


def _append_to_json(jf: pathlib.Path, new_records: list[dict]) -> None:
    """Append *new_records* to the performance JSON file, creating it if needed."""
    if jf.exists():
        data = json.loads(jf.read_text())
    else:
        jf.parent.mkdir(parents=True, exist_ok=True)
        data = {'runs': []}
    data['runs'].extend(new_records)
    jf.write_text(json.dumps(data))


# ── condition matching ────────────────────────────────────────────────────────

def _match_condition(r: dict, Re: float, N_crit: float,
                     xtp_top: float, xtp_bot: float, source: str,
                     tol: float = 1e-3) -> bool:
    if abs(r.get('Re', 0) - Re) / max(abs(Re), 1.0) > tol:
        return False
    if abs(r.get('N_crit', 0) - N_crit) > tol:
        return False
    if abs(r.get('xtp_top', 0) - xtp_top) > tol:
        return False
    if abs(r.get('xtp_bot', 0) - xtp_bot) > tol:
        return False
    if r.get('source') != source:
        return False
    return True


def _expand_sweep_values(sweep_range) -> list[float]:
    """Expand a (start, stop, step) triplet or explicit iterable to a flat list."""
    sr = list(sweep_range)
    if len(sr) == 3:
        start, stop, step = float(sr[0]), float(sr[1]), float(sr[2])
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 8) for i in range(n)]
    return [float(v) for v in sr]


def _has_sweep_value(records: list[dict], sweep_param: str, val: float,
                    Re: float, N_crit: float, xtp_top: float, xtp_bot: float,
                    source: str) -> bool:
    """True when *records* contains an entry for the given sweep value + condition."""
    key = 'alpha' if sweep_param == 'alpha' else 'cl'
    tol = 0.05 if sweep_param == 'alpha' else 0.005
    for r in records:
        if abs(r.get(key, float('nan')) - val) > tol:
            continue
        if _match_condition(r, Re, N_crit, xtp_top, xtp_bot, source):
            return True
    return False



def _find_bl_record(records: list[dict], mode: str, val: float, Re: float,
                    N_crit: float, xtp_top: float, xtp_bot: float,
                    source: str) -> dict | None:
    """Return the first record with matching condition + bl_data, or None."""
    for r in records:
        if mode == 'alpha':
            if abs(r.get('alpha', float('nan')) - val) > 1e-4:
                continue
        elif mode == 'cl':
            if abs(r.get('cl', float('nan')) - val) > 0.01:
                continue
        if not _match_condition(r, Re, N_crit, xtp_top, xtp_bot, source):
            continue
        if r.get('bl_data') is None:
            continue
        return r
    return None


# ── record construction ───────────────────────────────────────────────────────

def _iterable_to_records(raw: dict, source: str) -> list[dict]:
    """Split iterable (polar-sweep) wrapper output into per-alpha records."""
    alphas  = np.asarray(raw['alpha'])
    cdp_arr = raw.get('cdp')
    cp_data = raw.get('cp_data')   # list[dict] or None
    bl_data = raw.get('bl_data')   # list[dict] or None
    records = []
    for i, a in enumerate(alphas):
        records.append({
            'source':                source,
            'alpha':                 float(a),
            'cl':                    float(raw['cl'][i]),
            'cd':                    float(raw['cd'][i]),
            'cdp':                   (float(cdp_arr[i]) if cdp_arr is not None else None),
            'cm':                    float(raw['cm'][i]),
            'cpmin':                 float(raw['cpmin'][i]),
            'xtr_top':               float(raw['xtr_top'][i]),
            'xtr_bot':               float(raw['xtr_bot'][i]),
            'xtp_top':               float(raw['xtp_top']),
            'xtp_bot':               float(raw['xtp_bot']),
            'Re':                    float(raw['Re']),
            'M':                     float(raw.get('M', 0.0)),
            'N_crit':                float(raw['N_crit']),
            'N_panels':              raw.get('N_panels_xfoil'),
            'stagnation_index':      None,
            'bl_data':               (bl_data[i] if bl_data is not None else None),
        })
    return records


def _scalar_to_record(raw: dict, source: str) -> dict:
    """Build one record from a single-alpha wrapper output."""
    return {
        'source':                source,
        'alpha':                 float(raw['alpha']),
        'cl':                    float(raw['cl']),
        'cd':                    float(raw['cd']),
        'cdp':                   (float(raw['cdp']) if raw.get('cdp') is not None else None),
        'cm':                    float(raw['cm']),
        'cpmin':                 float(raw['cpmin']),
        'xtr_top':               float(raw['xtr_top']),
        'xtr_bot':               float(raw['xtr_bot']),
        'xtp_top':               float(raw['xtp_top']),
        'xtp_bot':               float(raw['xtp_bot']),
        'Re':                    float(raw['Re']),
        'M':                     float(raw.get('M', 0.0)),
        'N_crit':                float(raw['N_crit']),
        'N_panels':              raw.get('N_panels_xfoil'),
        'stagnation_index':      None,
        'cp_data':               raw.get('cp_data'),
        'bl_data':               raw.get('bl_data'),
    }


# ── compute helpers ───────────────────────────────────────────────────────────

def _to_aseq_triplet(sweep_range) -> tuple:
    """Convert any sweep_range to a ``(start, stop, step)`` triplet for xfoil aseq.

    If the range is already a 3-element triplet it is returned as-is.
    Otherwise the step is inferred from the explicit values:

    1. If the spacing is uniform (all gaps equal within 1 % tolerance) that
       step is used.
    2. Otherwise the first candidate from [0.1, 0.25, 0.5, 1.0, 2.0] whose
       grid covers every supplied value (within 1 % of the step) is used.
    3. Falls back to 1.0 if no candidate matches.
    """
    import numpy as _np
    sr = [float(v) for v in sweep_range]
    if len(sr) == 3 and not _np.isclose(sr[0], sr[1]):
        # Already a triplet — use as-is.
        return (sr[0], sr[1], sr[2])
    if len(sr) < 2:
        return (sr[0], sr[0], 1.0)

    start = min(sr)
    stop  = max(sr)

    # Check for uniform spacing.
    diffs = _np.diff(sorted(sr))
    if len(diffs) > 0:
        mean_step = float(_np.mean(diffs))
        if mean_step > 0:
            rel_var = (float(_np.max(diffs)) - float(_np.min(diffs))) / mean_step
            if rel_var < 0.01:
                return (start, stop, round(mean_step, 8))

    # Try candidate steps.
    for step in [0.1, 0.25, 0.5, 1.0, 2.0]:
        n = round((stop - start) / step) + 1
        grid = [start + i * step for i in range(n)]
        if all(any(abs(v - g) < step * 0.01 for g in grid) for v in sr):
            return (start, stop, step)

    return (start, stop, 1.0)

def _polar_xfoil(afl: Kulfan, sweep_param: str, sweep_range, Re: float,
                 N_crit: float, xtp_u: float, xtp_l: float,
                 force_explicit: bool = False,
                 timelimit: int = 60,
                 max_iter: int = 100,
                 stdout_log_path=None,
                 exec_script_path=None,
                 airfoil_name=None) -> list[dict]:
    try:
        from oso_airfoils.core import xfoil_wrapper
        raw = xfoil_wrapper.run(
            sweep_param, afl.upperCoefficients, afl.lowerCoefficients,
            val=list(sweep_range), Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
            TE_gap=float(afl.constants.TE_gap),
            force_list=force_explicit,
            timelimit=timelimit,
            max_iter=max_iter,
            stdout_log_path=stdout_log_path,
            exec_script_path=exec_script_path,
            airfoil_name=airfoil_name,
        )
        return _iterable_to_records(raw, 'xfoil')
    except Exception as _exc:
        print(f'  [xfoil polar] failed — skipping ({_exc})')
        return []


def _polar_qfoil(afl: Kulfan, sweep_param: str, sweep_range, Re: float,
                 N_crit: float, xtp_u: float, xtp_l: float,
                 force_explicit: bool = False,
                 timelimit: int = 60,
                 max_iter: int = 100,
                 stdout_log_path=None,
                 exec_script_path=None,
                 airfoil_name=None) -> list[dict]:
    try:
        from oso_airfoils.core import qfoil_wrapper
        raw = qfoil_wrapper.run(
            sweep_param, afl.upperCoefficients, afl.lowerCoefficients,
            val=list(sweep_range), Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
            TE_gap=float(afl.constants.TE_gap),
            force_list=force_explicit,
            timelimit=timelimit,
            max_iter=max_iter,
            stdout_log_path=stdout_log_path,
            exec_script_path=exec_script_path,
            airfoil_name=airfoil_name,
        )
        return _iterable_to_records(raw, 'qfoil')
    except Exception as _exc:
        print(f'  [qfoil polar] failed — skipping ({_exc})')
        return []


def _polar_neuralfoil(afl: Kulfan, sweep_param: str, sweep_range, Re: float,
                      N_crit: float, xtp_u: float, xtp_l: float,
                      neuralfoil_model: str = 'xxxlarge') -> list[dict]:
    from oso_airfoils.core import neuralfoil_wrapper
    raw = neuralfoil_wrapper.run(
        sweep_param, afl.upperCoefficients, afl.lowerCoefficients,
        val=list(sweep_range), Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
        TE_gap=float(afl.constants.TE_gap), model=neuralfoil_model,
    )
    return _iterable_to_records(raw, 'neuralfoil')


#: Cache of loaded surrogate nets, keyed by (backend, model). Reloading the net for
#: every airfoil in a rainbow would dominate the runtime.
_SURROGATE_CACHE: dict = {}


def _expand_sweep(sweep_range) -> list:
    """Explicit alpha list from a sweep spec.

    Follows the wrappers' convention: a THREE-element spec is
    ``(start, stop, step)``, anything else is already an explicit list of values.
    Getting this wrong is silent and ugly -- treating ``(-5, 25, 0.5)`` as three
    alphas builds a 3-point cache that the serving layer then nearest-matches a
    61-point request against, producing a staircase that still looks like a polar.
    """
    vals = list(sweep_range)
    if len(vals) == 3:
        start, stop, step = (float(v) for v in vals)
        if step > 0 and stop > start:
            n = int(round((stop - start) / step)) + 1
            return list(np.linspace(start, stop, n))
    return [float(v) for v in vals]


def _polar_surrogate(afl: Kulfan, sweep_range, Re: float, N_crit: float,
                     xtp_u: float, xtp_l: float, backend: str,
                     model: str) -> list[dict]:
    """Polar from one of metafoil's batched surrogates (``nxfoil`` or ``nqfoil``).

    Goes through the same BatchSurrogate the optimizer uses, so a polar plotted here
    is produced by exactly the code path that generated the design -- and its output
    dict matches the file-I/O wrappers' contract, so the record builder is shared.
    """
    from oso_airfoils.optimization.batch_surrogate import BatchSurrogate
    key = (backend, model)
    if key not in _SURROGATE_CACHE:
        _SURROGATE_CACHE[key] = BatchSurrogate(backend=backend, model_size=model,
                                               device='cpu')
    bs = _SURROGATE_CACHE[key]
    alphas = _expand_sweep(sweep_range)
    te_gap = float(afl.constants.TE_gap)
    upper = np.asarray(afl.upperCoefficients, float)
    lower = np.asarray(afl.lowerCoefficients, float)
    sweep = dict(name=f'{Re:g}|{N_crit:g}|{xtp_u:g}|{xtp_l:g}', Re=Re, ncrit=N_crit,
                 xtr_u=xtp_u, xtr_l=xtp_l, alphas=np.asarray(alphas, float))
    # Serve from a cache primed for the WHOLE plot if one is live (see
    # prime_surrogate_cache); otherwise fall back to a private single-airfoil build.
    if not _cache_has(bs, upper, lower, te_gap, sweep['name']):
        bs.build_population_cache([upper], [lower], te_gap, [sweep])
    raw = bs.make_cached_run()('alpha', upper, lower, val=alphas, Re=Re,
                               N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
                               TE_gap=te_gap, model=model)
    return _iterable_to_records(raw, backend)


def _cache_has(bs, upper, lower, te_gap, sweep_name) -> bool:
    from oso_airfoils.optimization.batch_surrogate import _key
    return (getattr(bs, '_cache', None) is not None
            and (_key(upper, lower, te_gap), sweep_name) in bs._cache)


def prime_surrogate_cache(kulfans, reynolds_numbers, turb_cases, sweep_range,
                          backend: str, model: str) -> None:
    """Evaluate EVERY (airfoil x condition) for a plot in ONE batched forward.

    Without this each airfoil/turbulence-case pair built its own single-airfoil cache
    -- 18 separate forwards for a 9-airfoil rainbow, which measured as ~86% of the
    figure's total cost. Batching them collapses that to one forward, the same way
    the optimizer evaluates a whole generation at once.
    """
    from oso_airfoils.optimization.batch_surrogate import BatchSurrogate
    kulfans = [k for k in kulfans if k is not None]
    if not kulfans:
        return
    key = (backend, model)
    if key not in _SURROGATE_CACHE:
        _SURROGATE_CACHE[key] = BatchSurrogate(backend=backend, model_size=model,
                                               device='cpu')
    bs = _SURROGATE_CACHE[key]
    alphas = np.asarray(_expand_sweep(sweep_range), float)
    uppers = [np.asarray(k.upperCoefficients, float) for k in kulfans]
    lowers = [np.asarray(k.lowerCoefficients, float) for k in kulfans]
    tes = np.array([float(k.constants.TE_gap) for k in kulfans], float)
    sweeps = [dict(name=f'{re:g}|{tc[0]:g}|{tc[1]:g}|{tc[2]:g}', Re=re, ncrit=tc[0],
                   xtr_u=tc[1], xtr_l=tc[2], alphas=alphas)
              for re in reynolds_numbers for tc in turb_cases]
    bs.build_population_cache(np.array(uppers), np.array(lowers), tes, sweeps)


def _bl_xfoil(afl: Kulfan, mode: str, val: float, Re: float,
              N_crit: float, xtp_u: float, xtp_l: float) -> dict | None:
    try:
        from oso_airfoils.core import xfoil_wrapper
        raw = xfoil_wrapper.run(
            mode, afl.upperCoefficients, afl.lowerCoefficients,
            val=val, Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
            save_boundary_layer_data=True,
            TE_gap=float(afl.constants.TE_gap),
        )
        return _scalar_to_record(raw, 'xfoil')
    except Exception as _exc:
        print(f'  [xfoil BL] failed — skipping ({_exc})')
        return None


def _bl_qfoil(afl: Kulfan, mode: str, val: float, Re: float,
              N_crit: float, xtp_u: float, xtp_l: float) -> dict | None:
    try:
        from oso_airfoils.core import qfoil_wrapper
        raw = qfoil_wrapper.run(
            mode, afl.upperCoefficients, afl.lowerCoefficients,
            val=val, Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
            save_boundary_layer_data=True,
            TE_gap=float(afl.constants.TE_gap),
        )
        return _scalar_to_record(raw, 'qfoil')
    except Exception as _exc:
        print(f'  [qfoil BL] failed — skipping ({_exc})')
        return None


def _bl_neuralfoil(afl: Kulfan, mode: str, val: float, Re: float,
                   N_crit: float, xtp_u: float, xtp_l: float) -> dict:
    from oso_airfoils.core import neuralfoil_wrapper
    raw = neuralfoil_wrapper.run(
        mode, afl.upperCoefficients, afl.lowerCoefficients,
        val=val, Re=Re, N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l,
        save_boundary_layer_data=True,
        TE_gap=float(afl.constants.TE_gap),
    )
    return _scalar_to_record(raw, 'neuralfoil')


# ── get-or-compute: polar ─────────────────────────────────────────────────────

def _get_polar_records(
    family: str, stem: str, afl_root,
    reynolds_numbers, turb_cases, tools,
    sweep_param: str, sweep_range, save_data: bool,
    kulfan: 'Kulfan | None' = None,
    perf_root=None,
    bypass_json: bool = False,
    run_seq: bool = False,
    timelimit: int = 60,
    max_iter: int = 100,
    stdout_log_path=None,
    exec_script_path=None,
    airfoil_name=None,
    neuralfoil_model: str = 'xxxlarge',
) -> list[dict]:
    """Return all records needed for a polar plot, computing any missing combos.

    When *kulfan* is supplied **or** *bypass_json* is ``True``:
    - JSON files are not read or written
    - neuralfoil always recomputes (fast, model-version-independent)
    - xfoil behaviour is controlled by *run_seq*:
      - ``run_seq=True``  → single aseq/cseq invocation (fast; needs triplet
        sweep_range such as ``(-5, 30, 1.0)``).
      - ``run_seq=False`` → individual per-alpha invocations (slower but
        compatible with any sweep_range, including ``np.linspace(...)``).  
    When only *kulfan* is supplied (and bypass_json is False) the same
    direct-compute path is used (geometry was provided in-process).
    """
    if kulfan is not None or bypass_json:
        _afl = kulfan
        if _afl is None:
            _afl = _load_kulfan(family, stem, afl_root)
        records: list[dict] = []
        # rfoil cannot be re-run; load from JSON cache even in bypass mode
        if 'rfoil' in tools:
            _existing_all = _existing_records(family, stem, perf_root)
            records.extend([r for r in _existing_all if r.get('source') == 'rfoil'])
        for re in reynolds_numbers:
            for tc in turb_cases:
                N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]
                for tool in tools:
                    if tool in ('nxfoil', 'nqfoil'):
                        records.extend(_polar_surrogate(_afl, sweep_range, re, N_crit,
                                                        xtp_u, xtp_l, tool, neuralfoil_model))
                    elif tool == 'neuralfoil':
                        records.extend(_polar_neuralfoil(_afl, sweep_param, sweep_range, re, N_crit, xtp_u, xtp_l, neuralfoil_model))
                    elif tool == 'rfoil':
                        continue  # already loaded above
                    elif tool == 'xfoil':
                        if run_seq:
                            aseq_range = _to_aseq_triplet(sweep_range)
                            records.extend(_polar_xfoil(_afl, sweep_param, aseq_range, re, N_crit, xtp_u, xtp_l,
                                                        timelimit=timelimit, max_iter=max_iter,
                                                        stdout_log_path=stdout_log_path,
                                                        exec_script_path=exec_script_path,
                                                        airfoil_name=airfoil_name))
                        else:
                            all_vals = _expand_sweep_values(sweep_range)
                            records.extend(_polar_xfoil(_afl, sweep_param, all_vals, re, N_crit, xtp_u, xtp_l,
                                                        force_explicit=True,
                                                        timelimit=timelimit, max_iter=max_iter,
                                                        stdout_log_path=stdout_log_path,
                                                        exec_script_path=exec_script_path,
                                                        airfoil_name=airfoil_name))
                    elif tool == 'qfoil':
                        if run_seq:
                            aseq_range = _to_aseq_triplet(sweep_range)
                            records.extend(_polar_qfoil(_afl, sweep_param, aseq_range, re, N_crit, xtp_u, xtp_l,
                                                        timelimit=timelimit, max_iter=max_iter,
                                                        stdout_log_path=stdout_log_path,
                                                        exec_script_path=exec_script_path,
                                                        airfoil_name=airfoil_name))
                        else:
                            all_vals = _expand_sweep_values(sweep_range)
                            records.extend(_polar_qfoil(_afl, sweep_param, all_vals, re, N_crit, xtp_u, xtp_l,
                                                        force_explicit=True,
                                                        timelimit=timelimit, max_iter=max_iter,
                                                        stdout_log_path=stdout_log_path,
                                                        exec_script_path=exec_script_path,
                                                        airfoil_name=airfoil_name))
                    else:
                        raise ValueError(f"Unknown tool: {tool!r}")
        return records

    existing = _existing_records(family, stem, perf_root)
    # rfoil records are pre-computed; they live in the JSON and need no gap-fill.
    new_xfoil_records: list[dict] = []
    fresh_nf_records:  list[dict] = []
    _afl = None  # lazy-load geometry only when a computation is required
    all_vals = _expand_sweep_values(sweep_range)

    for re in reynolds_numbers:
        for tc in turb_cases:
            N_crit, xtp_u, xtp_l = tc[0], tc[1], tc[2]
            for tool in tools:
                if tool in ('nxfoil', 'nqfoil'):
                    new_records.extend(
                        _polar_surrogate(_afl, all_vals, re, N_crit, xtp_u, xtp_l,
                                         tool, neuralfoil_model))
                elif tool == 'neuralfoil':
                    # Always recompute — fast, model-version-independent, never cache.
                    if _afl is None:
                        _afl = _load_kulfan(family, stem, afl_root)
                    fresh_nf_records.extend(
                        _polar_neuralfoil(_afl, sweep_param, all_vals, re, N_crit, xtp_u, xtp_l, neuralfoil_model)
                    )

                elif tool == 'rfoil':
                    continue  # pre-computed; records already in `existing`

                elif tool in ('xfoil', 'qfoil'):
                    # Per-alpha gap-fill from the JSON cache.
                    missing = [v for v in all_vals
                               if not _has_sweep_value(existing, sweep_param, v,
                                                       re, N_crit, xtp_u, xtp_l, tool)]
                    if not missing:
                        continue

                    if _afl is None:
                        _afl = _load_kulfan(family, stem, afl_root)

                    print(f'  Computing {stem}  Re={re:.2e}  Ncrit={N_crit}'
                          f'  xtp={xtp_u}/{xtp_l}  [{tool}]'
                          f'  ({len(missing)}/{len(all_vals)} values missing)...')
                    _polar_fn = _polar_qfoil if tool == 'qfoil' else _polar_xfoil
                    computed = _polar_fn(_afl, sweep_param, missing,
                                        re, N_crit, xtp_u, xtp_l,
                                        force_explicit=True,
                                        timelimit=timelimit,
                                        max_iter=max_iter,
                                        stdout_log_path=stdout_log_path,
                                        exec_script_path=exec_script_path,
                                        airfoil_name=airfoil_name)
                    if not computed:
                        print(f'  Warning: no records returned for {stem}  '
                              f'Re={re:.2e}  Ncrit={N_crit}  xtp={xtp_u}/{xtp_l}  [{tool}]'
                              f' — this condition will be missing from the plot.')
                    new_xfoil_records.extend(computed)
                    existing.extend(computed)

                else:
                    raise ValueError(f"Unknown tool: {tool!r}")

    # Save only new xfoil/qfoil records — neuralfoil is always recomputed fresh.
    if save_data and new_xfoil_records:
        jf = _json_path(family, stem, perf_root)
        _append_to_json(jf, new_xfoil_records)
        print(f'  Saved {len(new_xfoil_records)} new records -> {jf}')

    # Return cached non-neuralfoil records + freshly computed neuralfoil records.
    non_nf = [r for r in existing if r.get('source') != 'neuralfoil']
    return non_nf + fresh_nf_records


# ── get-or-compute: boundary layer ───────────────────────────────────────────

def _get_bl_record(
    family: str, stem: str, afl_root,
    mode: str, val: float, Re: float, N_crit: float,
    xtp_top: float, xtp_bot: float, source: str,
    save_data: bool,
    kulfan: 'Kulfan | None' = None,
    perf_root=None,
) -> dict:
    """Return a single BL record, computing it via the solver if not cached.

    *mode* is ``'alpha'`` or ``'cl'``; *val* is the corresponding float.
    When *kulfan* is supplied the JSON cache is bypassed.
    """
    if kulfan is not None:
        if source == 'neuralfoil':
            return _bl_neuralfoil(kulfan, mode, val, Re, N_crit, xtp_top, xtp_bot)
        elif source == 'xfoil':
            return _bl_xfoil(kulfan, mode, val, Re, N_crit, xtp_top, xtp_bot)
        elif source == 'qfoil':
            return _bl_qfoil(kulfan, mode, val, Re, N_crit, xtp_top, xtp_bot)
        else:
            raise ValueError(f"Unknown source: {source!r}")

    existing = _existing_records(family, stem, perf_root)
    rec = _find_bl_record(existing, mode, val, Re, N_crit, xtp_top, xtp_bot, source)
    if rec is not None:
        return rec

    afl = _load_kulfan(family, stem, afl_root)
    print(f'  Computing BL for {stem}  {mode}={val}  Re={Re:.2e}'
          f'  Ncrit={N_crit}  [{source}]...')
    if source == 'xfoil':
        rec = _bl_xfoil(afl, mode, val, Re, N_crit, xtp_top, xtp_bot)
    elif source == 'qfoil':
        rec = _bl_qfoil(afl, mode, val, Re, N_crit, xtp_top, xtp_bot)
    elif source == 'neuralfoil':
        rec = _bl_neuralfoil(afl, mode, val, Re, N_crit, xtp_top, xtp_bot)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    if save_data:
        jf = _json_path(family, stem, perf_root)
        _append_to_json(jf, [rec])
        print(f'  Saved 1 new BL record -> {jf}')

    return rec


# ── public API ────────────────────────────────────────────────────────────────

def run_and_plot_polars_compare(
    airfoils,
    reynolds_numbers,
    turb_cases,
    tools,
    figure_path,
    sweep_param,
    sweep_range,
    load_geometry=True,
    save_data=False,
    afl_root=None,
    perf_root=None,
    color_override=None,
    show_cpmin=True,
    cl_design=None,
    legend_ncols=None,
    style=None,
    neuralfoil_model='xxxlarge',
    bypass_json=False,
    run_seq=False,
    use_save_figure=False,
    save_figure_kwargs=None,
    timelimit=60,
    max_iter=100,
    stdout_log_path=None,
    exec_script_path=None,
):
    """Run (if needed) and plot a polars comparison.

    Parameters
    ----------
    airfoils : list
        Each element can be a unique airfoil name string (auto-resolved),
        a ``[name_str, geometry]`` pair, a bare :class:`~oso_airfoils.geometry.kulfan.Kulfan`
        object, or a ``(upper_arr, lower_arr)`` coefficient pair.
    reynolds_numbers : list[float]
        Reynolds numbers to include.
    turb_cases : list[[N_crit, xtp_u, xtp_l]]
        One or two turbulence / transition conditions.
    tools : list[str]
        Solvers to use: ``'xfoil'``, ``'neuralfoil'``, or ``'rfoil'``.
    figure_path : str or path-like or None
        Output file path; ``None`` skips saving.
    sweep_param : str
        Sweep variable passed to the solver: ``'alpha'`` (default) or ``'cl'``.
    sweep_range : (float, float, float)
        ``(start, stop, step)`` for the sweep variable, used only when a
        condition is not already in the JSON file.
        Default ``(-10, 25, 0.5)`` (degrees for alpha; use e.g.
        ``(0.0, 1.8, 0.05)`` when *sweep_param* is ``'cl'``).
    load_geometry : bool
        When ``True`` (default), load the ``.dat`` file and pass a shape panel
        to the plot.
    save_data : bool
        Append newly computed records to the JSON performance files.
    afl_root : path-like, optional
        Override the default airfoil root (``oso_airfoils/airfoils/``).
    bypass_json : bool
        When ``True``, skip reading **and** writing the JSON performance cache
        entirely.  Results are discarded after plotting.  Use this for one-off
        runs or quick design iterations where caching is unwanted.
    run_seq : bool
        When ``True`` (and *bypass_json* is ``True`` or geometry is supplied
        in-process), xfoil runs the entire sweep in a **single** aseq/cseq
        invocation.  *sweep_range* must be a ``(start, stop, step)`` triplet for
        this to produce an aseq command.  When ``False`` (default), xfoil is
        called once per alpha value (compatible with any sweep_range, including
        ``np.linspace(...)``).
    timelimit : int
        Per-invocation OS-level time limit (seconds) passed to xfoil.  The
        default is 60 s, which is generally sufficient for a full aseq sweep
        of 35 points at a single condition.  Increase this if rough-case runs
        (low N_crit) are terminating prematurely.
    max_iter : int
        Number of iterations passed to xfoil's ``iter`` command (default 100).
        Increase (e.g. to 200-400) if rough cases (low N_crit) fail to converge.
    use_save_figure : bool
        When ``True``, pass the figure through
        :func:`~oso_airfoils.postprocessing.save_figure` (generates SVG, PDF,
        PGF, and dark-mode variants) instead of a plain ``savefig`` call.
    save_figure_kwargs : dict, optional
        Extra keyword arguments forwarded to ``save_figure`` (e.g.
        ``{'dpi': 300, 'transparent': True}``).  Ignored when
        *use_save_figure* is ``False``.
    color_override, show_cpmin, cl_design, legend_ncols, style
        Passed directly to :func:`~oso_airfoils.postprocessing.polars_compare`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data_dict = {}
    geometry_dict = {} if load_geometry else None

    for idx, entry in enumerate(airfoils):
        display_name, family, stem, _kulfan = _resolve_entry(entry, idx, afl_root)
        data_dict[display_name] = _get_polar_records(
            family, stem, afl_root,
            reynolds_numbers, turb_cases, tools, sweep_param, sweep_range, save_data,
            kulfan=_kulfan,
            perf_root=perf_root,
            bypass_json=bypass_json,
            run_seq=run_seq,
            timelimit=timelimit,
            max_iter=max_iter,
            stdout_log_path=stdout_log_path,
            exec_script_path=exec_script_path,
            airfoil_name=display_name,
            neuralfoil_model=neuralfoil_model,
        )
        if load_geometry:
            geometry_dict[display_name] = _kulfan if _kulfan is not None else _load_kulfan(family, stem, afl_root)

    # When use_save_figure is True, suppress polars_compare's own savefig call
    # and apply save_figure (which generates dark/SVG/PDF/PGF variants) after.
    _pc_figure_path = None if use_save_figure else figure_path
    fig = polars_compare(
        data_dict,
        reynolds_numbers=reynolds_numbers,
        turb_cases=turb_cases,
        tools=tools,
        figure_path=_pc_figure_path,
        geometry_dict=geometry_dict,
        color_override=color_override,
        show_cpmin=show_cpmin,
        cl_design=cl_design,
        legend_ncols=legend_ncols,
        style=style,
    )

    if use_save_figure and figure_path is not None:
        from oso_airfoils.postprocessing.save_figure import save_figure
        save_figure(fig, figure_path, **(save_figure_kwargs or {}))

    return fig


def run_and_plot_polars_rainbow(
    airfoils,
    reynolds_numbers,
    turb_cases,
    tools,
    figure_path,
    sweep_param,
    sweep_range,
    load_geometry=True,
    save_data=False,
    afl_root=None,
    perf_root=None,
    reference_airfoils=None,
    reverse_plot_order=False,
    show_cpmin=True,
    cl_design=None,
    legend_ncols=None,
    style=None,
    neuralfoil_model='xxxlarge',
):
    """Run (if needed) and plot a polars rainbow.

    Parameters
    ----------
    airfoils : list
        Each element can be a unique airfoil name string (auto-resolved),
        a ``[name_str, geometry]`` pair, a bare :class:`~oso_airfoils.geometry.kulfan.Kulfan`
        object, or a ``(upper_arr, lower_arr)`` coefficient pair.
    reference_airfoils : list, optional
        Each element follows the same forms as *airfoils*, with an optional
        colour string appended: ``(name_str, color_str)`` or
        ``[name_str, geometry, color_str]``.
    All other parameters as :func:`run_and_plot_polars_compare`.
    """
    data_dict = {}
    geometry_dict = {} if load_geometry else None

    for idx, entry in enumerate(airfoils):
        display_name, family, stem, _kulfan = _resolve_entry(entry, idx, afl_root)
        data_dict[display_name] = _get_polar_records(
            family, stem, afl_root,
            reynolds_numbers, turb_cases, tools, sweep_param, sweep_range, save_data,
            kulfan=_kulfan,
            perf_root=perf_root,
            neuralfoil_model=neuralfoil_model,
        )
        if load_geometry:
            geometry_dict[display_name] = _kulfan if _kulfan is not None else _load_kulfan(family, stem, afl_root)

    ref_data_dict = None
    if reference_airfoils:
        ref_data_dict = {}
        for idx, entry in enumerate(reference_airfoils):
            display_name, family, stem, _kulfan, color = _resolve_ref_entry(entry, idx, afl_root)
            records = _get_polar_records(
                family, stem, afl_root,
                reynolds_numbers, turb_cases, tools, sweep_param, sweep_range, save_data,
                kulfan=_kulfan,
                perf_root=perf_root,
                neuralfoil_model=neuralfoil_model,
            )
            ref_entry = {'records': records, 'color': color}
            if load_geometry:
                ref_entry['kulfan'] = (_kulfan if _kulfan is not None
                                       else _load_kulfan(family, stem, afl_root))
            ref_data_dict[display_name] = ref_entry

    return polars_rainbow(
        data_dict,
        reynolds_numbers=reynolds_numbers,
        turb_cases=turb_cases,
        tools=tools,
        figure_path=figure_path,
        geometry_dict=geometry_dict,
        reference_data_dict=ref_data_dict,
        reverse_plot_order=reverse_plot_order,
        show_cpmin=show_cpmin,
        cl_design=cl_design,
        legend_ncols=legend_ncols,
        style=style,
    )


def run_and_plot_boundary_layer_compare(
    airfoils,
    figure_path,
    save_data=False,
    afl_root=None,
    perf_root=None,
    parameter=None,
    show_airfoil=True,
    labels=None,
    style=None,
):
    """Run (if needed) and plot a boundary-layer comparison.

    Parameters
    ----------
    airfoils : dict
        ``{display_name: {'airfoil': str | Kulfan,
                          'alpha': float,   # specify alpha OR cl (not both)
                          'cl': float,
                          'Re': float,
                          'N_crit': float, 'xtp_top': float,
                          'xtp_bot': float, 'tool': str}}``.  
        Each entry specifies exactly one flight condition via either
        ``'alpha'`` (degrees) or ``'cl'`` (lift coefficient).  ``N_crit``,
        ``xtp_top``, ``xtp_bot`` default to 9.0 / 1.0 / 1.0; ``tool``
        defaults to ``'xfoil'``.  The ``'airfoil'`` value may be a name
        string (auto-resolved) or a :class:`~oso_airfoils.geometry.kulfan.Kulfan`
        object (bypasses the cache).
    figure_path : str or path-like or None
    save_data : bool
        Append newly computed records to the JSON performance files.
    afl_root : path-like, optional
    parameter, show_airfoil, labels, style
        Passed to :func:`~oso_airfoils.postprocessing.boundary_layer_compare`.
    """
    data_dict = {}
    for idx, (name, spec) in enumerate(airfoils.items()):
        afl_val = spec['airfoil']
        Re      = float(spec['Re'])
        N_crit  = float(spec.get('N_crit',  9.0))
        xtp_top = float(spec.get('xtp_top', 1.0))
        xtp_bot = float(spec.get('xtp_bot', 1.0))
        tool    = spec.get('tool', 'xfoil')

        if 'alpha' in spec:
            mode, val = 'alpha', float(spec['alpha'])
        elif 'cl' in spec:
            mode, val = 'cl', float(spec['cl'])
        else:
            raise KeyError(f"Entry {name!r} must specify either 'alpha' or 'cl'.")

        _display, family, stem, _kulfan = _resolve_entry(afl_val, idx, afl_root)

        rec = _get_bl_record(family, stem, afl_root,
                             mode, val, Re, N_crit, xtp_top, xtp_bot, tool,
                             save_data, kulfan=_kulfan, perf_root=perf_root)
        data_dict[name] = [rec]

    return boundary_layer_compare(
        data_dict,
        parameter=parameter,
        show_airfoil=show_airfoil,
        figure_path=figure_path,
        labels=labels,
        style=style,
    )


def run_and_plot_boundary_layer_rainbow(
    airfoils,
    alpha=None,
    cl=None,
    Re=None,
    N_crit=9.0,
    xtp_top=1.0,
    xtp_bot=1.0,
    source='xfoil',
    figure_path=None,
    save_data=False,
    afl_root=None,
    perf_root=None,
    reference_airfoils=None,
    parameter=None,
    show_airfoil=False,
    labels=None,
    style=None,
):
    """Run (if needed) and plot a boundary-layer rainbow.

    Two modes are supported, selected automatically from the type of *alpha*
    or *cl*:

    **Multi-airfoil mode** (scalar *alpha* or *cl*)
        Each entry in *airfoils* is one rainbow series, all evaluated at the
        same condition.

    **Sweep mode** (list *alpha* or *cl*)
        *airfoils* must contain exactly one entry.  The rainbow sweeps over
        the values in the list; each value becomes one series.

    Parameters
    ----------
    airfoils : list
        Each element can be a unique airfoil name string (auto-resolved),
        a ``[name_str, geometry]`` pair, a bare :class:`~oso_airfoils.geometry.kulfan.Kulfan`
        object, or a ``(upper_arr, lower_arr)`` coefficient pair.
    alpha : float or list[float], optional
        Angle of attack [deg].  Pass a list to sweep over multiple alphas
        (sweep mode).  Mutually exclusive with *cl*.
    cl : float or list[float], optional
        Lift coefficient.  Pass a list to sweep over multiple CL values
        (sweep mode).  Mutually exclusive with *alpha*.
    Re : float
        Reynolds number.
    N_crit : float
        Transition criterion; default 9.0.
    xtp_top, xtp_bot : float
        Forced transition locations; default 1.0 (free transition).
    source : str
        Solver: ``'xfoil'`` (default) or ``'neuralfoil'``.
    reference_airfoils : list, optional
        Multi-airfoil mode only.  Each element follows the same forms as
        *airfoils*, with an optional colour string appended:
        ``(name_str, color_str)`` or ``[name_str, geometry, color_str]``.
    figure_path, save_data, afl_root
        See :func:`run_and_plot_polars_compare`.
    parameter, show_airfoil, labels, style
        Passed to :func:`~oso_airfoils.postprocessing.boundary_layer_rainbow`.
    """
    if Re is None:
        raise ValueError("'Re' must be provided.")
    if (alpha is None) == (cl is None):
        raise ValueError("Provide exactly one of 'alpha' or 'cl'.")

    # Determine flight-condition mode and whether this is a sweep
    if alpha is not None:
        _mode = 'alpha'
        _val = alpha
    else:
        _mode = 'cl'
        _val = cl

    _sweep = isinstance(_val, (list, tuple, np.ndarray))

    if _sweep:
        # ── Sweep mode ────────────────────────────────────────────────────────
        airfoils_list = list(airfoils)
        if len(airfoils_list) != 1:
            raise ValueError(
                "When a list is passed for 'alpha' or 'cl', 'airfoils' must "
                "contain exactly one entry."
            )
        display_name, family, stem, _kulfan = _resolve_entry(airfoils_list[0], 0, afl_root)

        def _fmt(v):
            v = float(v)
            if _mode == 'alpha':
                return (rf'$\alpha = {int(v):+d}^\circ$'
                        if v == int(v) else rf'$\alpha = {v:+.1f}^\circ$')
            else:
                return rf'$C_L = {v:.2f}$'

        data_dict = {}
        for v in _val:
            rec = _get_bl_record(family, stem, afl_root,
                                 _mode, float(v), Re, N_crit, xtp_top, xtp_bot,
                                 source, save_data, kulfan=_kulfan,
                                 perf_root=perf_root)
            if rec is not None:
                data_dict[_fmt(v)] = [rec]

        return boundary_layer_rainbow(
            data_dict,
            parameter=parameter,
            show_airfoil=show_airfoil,
            figure_path=figure_path,
            labels=labels,
            style=style,
            single_airfoil_name=display_name,
        )

    # ── Multi-airfoil mode ────────────────────────────────────────────────────
    data_dict = {}
    # Preserve insertion order (Pareto front order from caller) so that
    # effective_labels below can reconstruct it after _detect_sweep re-sorts.
    _name_order: list[str] = []
    for idx, entry in enumerate(airfoils):
        display_name, family, stem, _kulfan = _resolve_entry(entry, idx, afl_root)
        rec = _get_bl_record(family, stem, afl_root,
                             _mode, float(_val), Re, N_crit, xtp_top, xtp_bot,
                             source, save_data, kulfan=_kulfan,
                             perf_root=perf_root)
        if rec is None:
            print(f'  No BL record for {display_name!r} — skipping.')
            continue
        data_dict[display_name] = [rec]
        _name_order.append(display_name)

    ref_data_dict = None
    if reference_airfoils:
        ref_data_dict = {}
        for idx, entry in enumerate(reference_airfoils):
            display_name, family, stem, _kulfan, color = _resolve_ref_entry(entry, idx, afl_root)
            rec = _get_bl_record(family, stem, afl_root,
                                 _mode, float(_val), Re, N_crit, xtp_top, xtp_bot,
                                 source, save_data, kulfan=_kulfan,
                                 perf_root=perf_root)
            ref_data_dict[display_name] = {'record': rec, 'color': color}

    # In multi-airfoil mode the sweep may be detected as 'alpha' (each airfoil
    # achieves the same CL at a different alpha), which would generate alpha-
    # value labels instead of airfoil-name labels.  boundary_layer_rainbow
    # will re-sort sorted_pairs to rf order when names are 'rf=X.XX', so we
    # must pass labels in that same rf order — _name_order is already in rf
    # insertion order from oso_bl.py, so use it directly.
    effective_labels = labels if labels is not None else (_name_order if _name_order else None)

    return boundary_layer_rainbow(
        data_dict,
        parameter=parameter,
        show_airfoil=show_airfoil,
        figure_path=figure_path,
        labels=effective_labels,
        style=style,
        reference_data_dict=ref_data_dict,
    )
