#!/usr/bin/env python3
"""
ingest_data.py  —  Add aerodynamic run data to a performance JSON file.

Supported modes
---------------
'xfoil_run'      Run XFoil and ingest the result.
'neuralfoil_run' Run NeuralFoil and ingest the result.
'rfoil_data'     Parse an existing RFoil polar file.
'xfoil_data'     Parse an existing XFoil polar file.

The canonical entry point is :func:`ingest`.  Family modules should also call
:func:`find_json_path` to resolve the correct JSON path from an airfoil name.
"""

import json
import os
import pathlib
import warnings

import numpy as np

SCHEMA_VERSION = "1.0"

# Fields whose combination identifies a unique run ("inputs").
_KEY_FIELDS = (
    'version', 'source', 'model',
    'Re', 'M', 'N_crit', 'N_panels',
    'xtp_top', 'xtp_bot',
    'f0', 'chord_to_radius_ratio',
    'drag_model',
    'alpha',
)

# Fields that are outputs — checked for conflicts on a key match.
_OUTPUT_FIELDS = (
    'cl', 'cd', 'cdp', 'cm', 'cpmin',
    'xtr_top', 'xtr_bot',
    'stagnation_index', 'stagnation_x',
    # cp_data / bl_data are excluded from conflict comparison (too complex)
)


# ──────────────────────────────────────────────────────────────────────────── #
#  JSON serialisation helper
# ──────────────────────────────────────────────────────────────────────────── #

class _NumpyEncoder(json.JSONEncoder):
    """Extend the default encoder to handle numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────────────────────────────────── #
#  Low-level record builder
# ──────────────────────────────────────────────────────────────────────────── #

def _f(val):
    """Cast to Python float, or return None."""
    return float(val) if val is not None else None


def _fields_equal(a, b, rtol=1e-6):
    """Return True if two output-field values are considered equal."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) and isinstance(b, float):
        if a == b:
            return True
        denom = max(abs(a), abs(b))
        return (abs(a - b) / denom) < rtol if denom > 0 else abs(a - b) < 1e-12
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return a == b


def _merge_runs(existing_runs, new_records, source_hint=''):
    """
    Merge *new_records* into *existing_runs* with duplicate detection.

    For each new record:
    - Key not found                → append (new run).
    - Key found, all outputs match → skip silently.
    - Key found, existing null → new value for some fields
                                → populate nulls, no warning.
    - Key found, genuine conflict  → warn and keep existing.

    Returns ``(merged_list, n_added, n_skipped, n_updated, n_conflicted)``.
    """
    # Build index: key_tuple → position in merged list
    lookup = {}
    merged = list(existing_runs)
    for i, rec in enumerate(merged):
        lookup[tuple(rec.get(f) for f in _KEY_FIELDS)] = i

    n_added = n_skipped = n_updated = n_conflicted = 0

    for rec in new_records:
        k = tuple(rec.get(f) for f in _KEY_FIELDS)
        if k not in lookup:
            lookup[k] = len(merged)
            merged.append(rec)
            n_added += 1
            continue

        existing = merged[lookup[k]]
        has_conflict = False
        has_update   = False
        merged_rec   = dict(existing)

        for field in _OUTPUT_FIELDS:
            ev = existing.get(field)
            nv = rec.get(field)
            if ev is None and nv is not None:
                merged_rec[field] = nv
                has_update = True
            elif ev is not None and nv is not None:
                if not _fields_equal(ev, nv):
                    has_conflict = True
                    break

        if has_conflict:
            key_desc = ', '.join(
                f"{f}={rec.get(f)}" for f in ('source', 'Re', 'alpha')
            )
            warnings.warn(
                f"{source_hint}: conflict on record ({key_desc}). "
                "Keeping stored data; check your input files.",
                stacklevel=3,
            )
            n_conflicted += 1
        elif has_update:
            merged[lookup[k]] = merged_rec
            n_updated += 1
        else:
            n_skipped += 1

    return merged, n_added, n_skipped, n_updated, n_conflicted


def _make_run_record(
    source, alpha, cl, cd, cm,
    Re, M, N_crit, N_panels, xtp_top, xtp_bot,
    cpmin=None, xtr_top=None, xtr_bot=None,
    model=None, f0=None, chord_to_radius_ratio=None,
    drag_model=None,
    cdp=None,
    version=None,
    stagnation_index=None, stagnation_x=None,
    cp_data=None, bl_data=None,
):
    """Return a single-alpha run record dict matching the v1.0 schema."""
    return {
        "version":               version,
        "source":                source,
        "model":                 model,
        "Re":                    _f(Re),
        "M":                     _f(M),
        "N_crit":                _f(N_crit),
        "N_panels":              int(N_panels) if N_panels is not None else None,
        "xtp_top":               _f(xtp_top),
        "xtp_bot":               _f(xtp_bot),
        "f0":                    _f(f0),
        "chord_to_radius_ratio": _f(chord_to_radius_ratio),
        "drag_model":            bool(drag_model) if drag_model is not None else None,
        "alpha":                 _f(alpha),
        "cl":                    _f(cl),
        "cd":                    _f(cd),
        "cdp":                   _f(cdp),
        "cm":                    _f(cm),
        "cpmin":                 _f(cpmin),
        "xtr_top":               _f(xtr_top),
        "xtr_bot":               _f(xtr_bot),
        "stagnation_index":      int(stagnation_index) if stagnation_index is not None else None,
        "stagnation_x":          _f(stagnation_x),
        "cp_data":               cp_data,
        "bl_data":               bl_data,
    }


# ──────────────────────────────────────────────────────────────────────────── #
#  Wrapper result → records
# ──────────────────────────────────────────────────────────────────────────── #

def _wrapper_result_to_records(source, res, model=None, f0=None,
                                chord_to_radius_ratio=None, version=None):
    """
    Convert a dict returned by xfoil_wrapper.run() or neuralfoil_wrapper.run()
    into a list of single-alpha run records (one entry per alpha point).
    """
    alpha = res['alpha']
    scalar = not hasattr(alpha, '__len__')

    def _listify(key):
        v = res[key]
        if scalar:
            return [v]
        return list(np.asarray(v, dtype=float))

    alphas   = [float(alpha)] if scalar else list(np.asarray(alpha, float))
    cls      = _listify('cl')
    cds      = _listify('cd')
    cms      = _listify('cm')
    cpmins   = _listify('cpmin')
    xtr_tops = _listify('xtr_top')
    xtr_bots = _listify('xtr_bot')
    n        = len(alphas)
    _raw_cdp = res.get('cdp')
    cdps     = ([None] * n if _raw_cdp is None
                else ([float(_raw_cdp)] if scalar
                      else list(np.asarray(_raw_cdp, float))))

    cp_list = res.get('cp_data')
    bl_list = res.get('bl_data')

    if cp_list is None:
        cp_list = [None] * n
    elif scalar:
        cp_list = [cp_list]

    if bl_list is None:
        bl_list = [None] * n
    elif scalar:
        bl_list = [bl_list]

    records = []
    for i in range(n):
        records.append(_make_run_record(
            source=source,
            alpha=alphas[i],
            cl=float(cls[i]),
            cd=float(cds[i]),
            cdp=float(cdps[i]) if cdps[i] is not None else None,
            cm=float(cms[i]),
            cpmin=float(cpmins[i]) if cpmins[i] is not None else None,
            xtr_top=float(xtr_tops[i]) if xtr_tops[i] is not None else None,
            xtr_bot=float(xtr_bots[i]) if xtr_bots[i] is not None else None,
            Re=res['Re'], M=res['M'], N_crit=res['N_crit'],
            N_panels=res['N_panels'],
            xtp_top=res['xtp_top'], xtp_bot=res['xtp_bot'],
            model=model, f0=f0,
            chord_to_radius_ratio=chord_to_radius_ratio,
            version=version,
            cp_data=cp_list[i],
            bl_data=bl_list[i],
        ))
    return records


# ──────────────────────────────────────────────────────────────────────────── #
#  Polar file parsers
# ──────────────────────────────────────────────────────────────────────────── #

def _infer_drag_model(filepath):
    """Infer drag_model from filename: True for DragOn, False for DragOff, None if absent."""
    name = pathlib.Path(filepath).name
    if 'DragOn' in name:
        return True
    if 'DragOff' in name:
        return False
    return None


def _parse_rfoil_polar(filepath):
    """Return a list of single-alpha run records from an RFoil polar file.

    The ``drag_model`` field is inferred from the filename
    (``DragOn`` → ``True``, ``DragOff`` → ``False``, absent → ``null``).

    Expected header lines (RFoil v3)::

        xtrf =      1.00000000 (suction)     1.00000000 (pressure)
        Rot. Parameters:  f0 =      1.00000000 c/r =      0.00000000
        Mach =      0.00000000     Re =     13.00000000 e 6     Ncrit =      9.00000000

    Data columns: alpha CL CD Re(CL) CM S_xtr P_xtr CDp I_stag X_stag
    """
    filepath = pathlib.Path(filepath)
    with open(filepath) as fh:
        lines = fh.readlines()

    xtp_top, xtp_bot = 1.0, 1.0
    f0, ctr = None, None
    M, Re, N_crit = 0.0, 1e6, 9.0
    solver_version = None

    for line in lines:
        s = line.strip()
        if s.startswith('RFOIL') and 'Version' in s:
            # "RFOIL         Version 3.00"
            tok = s.split()
            vi = tok.index('Version')
            solver_version = tok[vi + 1]
        elif s.startswith('xtrf'):
            # xtrf =      1.00000000 (suction)     1.00000000 (pressure)
            tok = s.split()
            xtp_top = float(tok[2])
            xtp_bot = float(tok[4])
        elif s.startswith('Rot.'):
            # Rot. Parameters:  f0 =  1.0  c/r =  0.0
            tok = s.split()
            f0  = float(tok[tok.index('f0')  + 2])
            ctr = float(tok[tok.index('c/r') + 2])
        elif s.startswith('Mach'):
            # Mach =  0.0  Re =  13.0 e 6  Ncrit =  9.0
            tok = s.split()
            M   = float(tok[2])
            ri  = tok.index('Re') + 2
            Re  = float(tok[ri]) * 10 ** float(tok[ri + 2])
            N_crit = float(tok[tok.index('Ncrit') + 2])

    # Data starts on the line after the first '---' separator
    data_start = next(
        (i + 1 for i, ln in enumerate(lines) if ln.strip().startswith('---')),
        None,
    )
    if data_start is None:
        raise ValueError(f"Cannot find data section in: {filepath}")

    drag_model = _infer_drag_model(filepath)

    # Re in scientific notation from header (e.g. 13.0 e 6 → 13e6)
    Re_tol = Re * 1e-4  # 0.01 % tolerance for float comparison

    records = []
    warned_re = False
    for line in lines[data_start:]:
        tok = line.split()
        if len(tok) < 10:
            continue
        # alpha CL CD Re(CL) CM S_xtr P_xtr CDp I_stag X_stag
        re_col = float(tok[3]) * 1e6  # Re(CL) column is in units of 1e6
        if not warned_re and abs(re_col - Re) > Re_tol:
            import warnings
            warnings.warn(
                f"{filepath.name}: Re column ({re_col:.6g}) does not match "
                f"header Re ({Re:.6g}). Using header value.",
                stacklevel=3,
            )
            warned_re = True
        records.append(_make_run_record(
            source='rfoil',
            alpha=float(tok[0]), cl=float(tok[1]), cd=float(tok[2]),
            cm=float(tok[4]),
            xtr_top=float(tok[5]), xtr_bot=float(tok[6]),
            cdp=float(tok[7]),
            Re=Re, M=M, N_crit=N_crit, N_panels=None,
            xtp_top=xtp_top, xtp_bot=xtp_bot,
            f0=f0, chord_to_radius_ratio=ctr,
            drag_model=drag_model,
            version=solver_version,
            stagnation_index=int(tok[8]),
            stagnation_x=float(tok[9]),
        ))
    return records


def _parse_xfoil_polar(filepath, N_panels=None):
    """Return a list of single-alpha run records from an XFoil polar file.

    Handles both the standard 7-column format and the 8-column format produced
    when ``cinc`` is enabled (adds Cpmin between Cm and Top_Xtr).

    Expected header lines::

        xtrf =   1.000 (top)     1.000 (bottom)
        Mach =  0.000     Re = 0.600e6     Ncrit = 9.000

    Data columns (7-col):  alpha Cl Cd Cdp Cm Top_Xtr Bot_Xtr
    Data columns (8-col):  alpha Cl Cd Cdp Cm Cpmin Top_Xtr Bot_Xtr
    """
    filepath = pathlib.Path(filepath)
    with open(filepath) as fh:
        lines = fh.readlines()

    xtp_top, xtp_bot = 1.0, 1.0
    M, Re, N_crit = 0.0, 1e6, 9.0
    xfoil_version = None

    for line in lines:
        s = line.strip()
        if s.startswith('XFOIL') and 'Version' in s:
            # "XFOIL    Version 6.99"
            tok = s.split()
            vi = tok.index('Version')
            xfoil_version = tok[vi + 1]
        elif s.startswith('xtrf'):
            tok = s.split()
            xtp_top = float(tok[2])
            xtp_bot = float(tok[4])
        elif s.startswith('Mach'):
            tok = s.split()
            M      = float(tok[2])
            Re     = float(tok[tok.index('Re')    + 2])
            N_crit = float(tok[tok.index('Ncrit') + 2])

    data_start = next(
        (i + 1 for i, ln in enumerate(lines) if ln.strip().startswith('------')),
        None,
    )
    if data_start is None:
        raise ValueError(f"Cannot find data section in: {filepath}")

    records = []
    for line in lines[data_start:]:
        tok = line.split()
        if len(tok) < 7:
            continue
        if len(tok) >= 8:
            # 8-col (cinc enabled): alpha cl cd cdp cm cpmin xtr_top xtr_bot
            alpha, cl, cd, _, cm, cpmin, xtr_top, xtr_bot = (float(t) for t in tok[:8])
        else:
            # 7-col (standard):    alpha cl cd cdp cm xtr_top xtr_bot
            alpha, cl, cd, _, cm, xtr_top, xtr_bot = (float(t) for t in tok[:7])
            cpmin = None
        records.append(_make_run_record(
            source='xfoil',
            alpha=alpha, cl=cl, cd=cd, cm=cm, cpmin=cpmin,
            xtr_top=xtr_top, xtr_bot=xtr_bot,
            Re=Re, M=M, N_crit=N_crit, N_panels=N_panels,
            xtp_top=xtp_top, xtp_bot=xtp_bot,
            version=xfoil_version,
        ))
    return records


# ──────────────────────────────────────────────────────────────────────────── #
#  Path helpers
# ──────────────────────────────────────────────────────────────────────────── #

def find_json_path(name, datfiles_dir, perf_dir):
    """
    Resolve the performance JSON path for *name* inside *perf_dir*.

    Uses the same case-insensitive, underscore/hyphen-normalised matching as
    :func:`~oso_airfoils.core.airfoil_family.get_geometry_from_dir`, falling
    back to a stripped-prefix match (e.g. '91-w2-250' → 'du_91-w2-250.json').

    Parameters
    ----------
    name : str
    datfiles_dir : path-like
    perf_dir : path-like

    Returns
    -------
    pathlib.Path
    """
    datfiles_dir = pathlib.Path(datfiles_dir)
    perf_dir     = pathlib.Path(perf_dir)

    fls    = sorted(f for f in os.listdir(datfiles_dir) if f.endswith('.dat'))
    stems  = [f[:-4] for f in fls]
    norm   = [s.lower().replace('_', '-') for s in stems]
    norm_s = ['-'.join(s.split('-')[1:]) for s in norm]

    search = name.lower().replace('_', '-')

    if search in norm:
        idx = norm.index(search)
    elif search in norm_s:
        idx = norm_s.index(search)
    else:
        raise ValueError(
            f"Airfoil '{name}' not found in {datfiles_dir}. "
            f"Available: {[s.upper() for s in stems]}"
        )

    return perf_dir / f'{stems[idx]}.json'


def _get_xfoil_version():
    """Return the XFoil version string by interrogating the binary, or None."""
    import re
    import shutil
    import subprocess
    xfoil_path = shutil.which('xfoil')
    if xfoil_path is None:
        return None
    try:
        result = subprocess.run(
            xfoil_path, input='\n',
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            m = re.search(r'Version\s+([\d.]+)', line, re.IGNORECASE)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def _is_sweep_spec(alpha):
    """
    Return True if *alpha* looks like a ``[start, stop, step]`` sweep spec.

    Rules
    -----
    - Must have exactly 3 elements.
    - ``step`` must be non-zero and point from ``start`` toward ``stop``
      (i.e. ``(stop - start) * step > 0``).

    Any other iterable (e.g. a 46-element list of explicit angles) is treated
    as a collection of individual alpha values.
    """
    try:
        if len(alpha) != 3:
            return False
        start, stop, step = float(alpha[0]), float(alpha[1]), float(alpha[2])
        return step != 0 and (stop - start) * step > 0 and abs(step) <= abs(stop - start)
    except (TypeError, ValueError):
        return False


def _resolve_polar_files(kwargs):
    """Pop and return a list of Path objects from polar_file / polar_files."""
    if 'polar_files' in kwargs:
        return [pathlib.Path(p) for p in kwargs.pop('polar_files')]
    if 'polar_file' in kwargs:
        return [pathlib.Path(kwargs.pop('polar_file'))]
    raise ValueError("Provide 'polar_file' or 'polar_files'.")


def _normalize_and_match(candidate, datfiles_dir):
    """
    Try to match *candidate* against available .dat file stems in *datfiles_dir*.
    Uses case-insensitive, underscore/hyphen-normalised comparison with an
    optional stripped-prefix fallback.  Returns the matched stem or ``None``.
    """
    fls    = sorted(f for f in os.listdir(datfiles_dir) if f.endswith('.dat'))
    stems  = [f[:-4] for f in fls]
    norm   = [s.lower().replace('_', '-') for s in stems]
    norm_s = ['-'.join(s.split('-')[1:]) for s in norm]
    search = candidate.lower().replace('_', '-').strip()
    if search in norm:
        return stems[norm.index(search)]
    if search in norm_s:
        return stems[norm_s.index(search)]
    return None


def _infer_airfoil_name(filepath, datfiles_dir):
    """
    Attempt to identify the airfoil name for a polar file.

    Strategy
    --------
    1. Look for a ``Calculated polar for: <name>`` line in the file header.
    2. Try progressively shorter underscore-delimited prefixes of the filename
       stem (e.g. ``du_91-w2-250_r13_clean`` → ``du_91-w2-250_r13`` → …).

    Returns the matched dat-file stem, or ``None`` if no match is found.
    """
    filepath = pathlib.Path(filepath)

    # 1. Header scan
    try:
        with open(filepath) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith('Calculated polar for:'):
                    candidate = s.split(':', 1)[1].strip()
                    match = _normalize_and_match(candidate, datfiles_dir)
                    if match:
                        return match
                    break   # only check first occurrence
    except Exception:
        pass

    # 2. Progressive filename prefix matching
    parts = filepath.stem.split('_')
    for n in range(len(parts), 0, -1):
        candidate = '_'.join(parts[:n])
        match = _normalize_and_match(candidate, datfiles_dir)
        if match:
            return match

    return None


def _write_data_files_by_inference(mode_label, polar_files, parser_fn,
                                    datfiles_dir, perf_dir):
    """
    Parse each polar file, infer the target airfoil, and write records to the
    appropriate performance JSON.  Files that cannot be assigned are skipped
    with a printed warning.  JSONs are each written once (batched by target).
    """
    from collections import defaultdict

    if perf_dir is None:
        raise ValueError(
            "perf_dir is required when name is not provided to ingest()."
        )

    groups = defaultdict(list)
    for pf in polar_files:
        stem = _infer_airfoil_name(pf, datfiles_dir)
        if stem is None:
            print(
                f"  [{mode_label}] WARNING: could not assign "
                f"'{pf.name}' to any known airfoil — skipping."
            )
            continue
        target = find_json_path(stem, datfiles_dir, perf_dir)
        groups[target].append(pf)

    for target_json, files in sorted(groups.items()):
        if not target_json.exists():
            print(
                f"  [{mode_label}] WARNING: JSON not found at "
                f"'{target_json.name}' — skipping {len(files)} file(s)."
            )
            continue
        with open(target_json) as fh:
            data = json.load(fh)
        records = []
        for pf in files:
            try:
                records.extend(parser_fn(pf))
            except Exception as exc:
                print(
                    f"  [{mode_label}] WARNING: could not parse "
                    f"'{pf.name}' — {exc}"
                )
        merged, n_added, n_skipped, n_updated, n_conflicted = _merge_runs(
            data['runs'], records, source_hint=target_json.name
        )
        data['runs'] = merged
        with open(target_json, 'w') as fh:
            json.dump(data, fh, indent=2, cls=_NumpyEncoder)
        parts = [f"{n_added} added"]
        if n_skipped:    parts.append(f"{n_skipped} skipped (duplicate)")
        if n_updated:    parts.append(f"{n_updated} null-populated")
        if n_conflicted: parts.append(f"{n_conflicted} conflict(s)")
        print(f"  [{mode_label}] {', '.join(parts)} → {target_json.name}")


# ──────────────────────────────────────────────────────────────────────────── #
#  Public API
# ──────────────────────────────────────────────────────────────────────────── #

def ingest(mode, name, json_path, datfiles_dir, perf_dir=None, **kwargs):
    """
    Add aerodynamic data to a performance JSON file.

    Parameters
    ----------
    mode : {'xfoil_run', 'qfoil_run', 'neuralfoil_run', 'rfoil_data', 'xfoil_data', 'qfoil_data'}
        Data source / collection method.
    name : str
        Airfoil name, validated against the .dat files in *datfiles_dir*.
    json_path : path-like
        Target performance JSON file.  Must already exist (create with
        ``generate_empty_jsons.py``).
    datfiles_dir : path-like
        Directory containing the family's ``.dat`` geometry files.

    Run-mode kwargs  (xfoil_run / qfoil_run / neuralfoil_run)
    ----------------------------------------------------------
    alpha : float | [start, stop, step]
        Single point (float) or alpha sweep (3-element list).
    Re : float
        Reynolds number.
    M : float, default 0.0
    N_crit : float, default 9.0
    xtp_u : float, default 1.0   forced upper transition location
    xtp_l : float, default 1.0   forced lower transition location
    run_mode : {'alpha', 'cl'}, default 'alpha'          [xfoil_run / qfoil_run]
    N_panels : int, default 160                          [xfoil_run / qfoil_run]
    model : str, default 'xxlarge'                       [neuralfoil_run only]
    save_boundary_layer_data : bool, default False       [xfoil_run / qfoil_run]
    cases : list of dict
        Multiple run conditions.  Each dict may contain any of the kwargs
        listed above.  When provided, all top-level run kwargs are ignored.

    Data-mode kwargs  (rfoil_data / xfoil_data / qfoil_data)
    ---------------------------------------------------------
    polar_file : path-like
        Single polar file path.
    polar_files : list of path-like
        Multiple polar file paths.
    N_panels : int, optional                             [xfoil_data only]
        Panel count to record (not stored in the polar file header).
    """
    from oso_airfoils.core.airfoil_family import get_geometry_from_dir

    datfiles_dir = pathlib.Path(datfiles_dir)

    # For data modes, name and json_path may be None (inferred per-file below)
    if name is None:
        if mode not in ('rfoil_data', 'xfoil_data', 'qfoil_data'):
            raise ValueError("name is required for run modes.")
        afl  = None
        data = None
    else:
        json_path = pathlib.Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"Performance JSON not found: {json_path}\n"
                "Run generate_empty_jsons.py first."
            )
        afl = get_geometry_from_dir(name, datfiles_dir)
        with open(json_path) as fh:
            data = json.load(fh)

    new_records = []

    # ── xfoil_run ──────────────────────────────────────────────────────── #
    if mode in ('xfoil_run', 'qfoil_run'):
        _source_tag = 'xfoil' if mode == 'xfoil_run' else 'qfoil'
        if mode == 'xfoil_run':
            from oso_airfoils.core.xfoil_wrapper import run as _runner
            _version = _get_xfoil_version()
        else:
            from oso_airfoils.core.qfoil_wrapper import run as _runner
            _version = None

        cases = kwargs.pop('cases', None)
        if cases is None:
            cases = [kwargs]

        for _case in cases:
            case     = dict(_case)
            alpha    = case.pop('alpha')
            run_mode = case.pop('run_mode', 'alpha')
            Re       = case.pop('Re')
            M        = case.pop('M', 0.0)
            N_crit   = case.pop('N_crit', 9.0)
            N_panels = case.pop('N_panels', 160)
            xtp_u    = case.pop('xtp_u', 1.0)
            xtp_l    = case.pop('xtp_l', 1.0)
            save_bl  = case.pop('save_boundary_layer_data', True)

            _shared = dict(
                mode=run_mode,
                upperKulfanCoefficients=afl.upperCoefficients,
                lowerKulfanCoefficients=afl.lowerCoefficients,
                Re=Re, M=M, N_crit=N_crit, N_panels=N_panels,
                xtp_u=xtp_u, xtp_l=xtp_l,
                save_boundary_layer_data=save_bl,
            )
            _rec_kw = dict(version=_version) if _version is not None else {}
            if _is_sweep_spec(alpha):
                res = _runner(val=alpha, **_shared)
                new_records.extend(_wrapper_result_to_records(_source_tag, res, **_rec_kw))
            else:
                for a in alpha:
                    try:
                        res = _runner(val=float(a), **_shared)
                        new_records.extend(_wrapper_result_to_records(_source_tag, res, **_rec_kw))
                    except Exception:
                        pass

    # ── neuralfoil_run ─────────────────────────────────────────────────── #
    elif mode == 'neuralfoil_run':
        from oso_airfoils.core.neuralfoil_wrapper import run as _nf

        cases = kwargs.pop('cases', None)
        if cases is None:
            cases = [kwargs]

        for _case in cases:
            case   = dict(_case)
            alpha  = case.pop('alpha')
            Re     = case.pop('Re')
            M      = case.pop('M', 0.0)
            N_crit = case.pop('N_crit', 9.0)
            xtp_u  = case.pop('xtp_u', 1.0)
            xtp_l  = case.pop('xtp_l', 1.0)
            model  = case.pop('model', 'xxlarge')

            _shared = dict(
                mode='alpha',
                upperKulfanCoefficients=afl.upperCoefficients,
                lowerKulfanCoefficients=afl.lowerCoefficients,
                Re=Re, M=M, N_crit=N_crit,
                xtp_u=xtp_u, xtp_l=xtp_l,
                model=model,
            )
            if _is_sweep_spec(alpha):
                # [start, stop, step] — pass directly to wrapper
                res = _nf(val=alpha, **_shared)
                new_records.extend(_wrapper_result_to_records('neuralfoil', res, model=model))
            else:
                # explicit list of alpha values — one wrapper call per point
                for a in alpha:
                    res = _nf(val=float(a), **_shared)
                    new_records.extend(_wrapper_result_to_records('neuralfoil', res, model=model))

    # ── rfoil_data ─────────────────────────────────────────────────────── #
    elif mode == 'rfoil_data':
        polar_files = _resolve_polar_files(kwargs)
        if name is None:
            _write_data_files_by_inference(
                'rfoil_data', polar_files, _parse_rfoil_polar, datfiles_dir, perf_dir
            )
            return
        for pf in polar_files:
            try:
                new_records.extend(_parse_rfoil_polar(pf))
            except Exception as exc:
                print(f"  [rfoil_data] WARNING: could not parse '{pf.name}' — {exc}")

    # ── xfoil_data / qfoil_data ────────────────────────────────────────── #
    elif mode in ('xfoil_data', 'qfoil_data'):
        _source_tag = 'xfoil' if mode == 'xfoil_data' else 'qfoil'
        N_panels    = kwargs.pop('N_panels', None)
        polar_files = _resolve_polar_files(kwargs)
        if name is None:
            _write_data_files_by_inference(
                mode, polar_files,
                lambda pf: _parse_xfoil_polar(pf, N_panels=N_panels),
                datfiles_dir, perf_dir,
            )
            return
        for pf in polar_files:
            try:
                records = _parse_xfoil_polar(pf, N_panels=N_panels)
                # Re-tag the source if needed (xfoil polar format is identical for qfoil)
                if _source_tag == 'qfoil':
                    for r in records:
                        r['source'] = 'qfoil'
                new_records.extend(records)
            except Exception as exc:
                print(f"  [{mode}] WARNING: could not parse '{pf.name}' — {exc}")

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Must be one of: "
            "'xfoil_run', 'qfoil_run', 'neuralfoil_run', 'rfoil_data', 'xfoil_data', 'qfoil_data'."
        )

    merged, n_added, n_skipped, n_updated, n_conflicted = _merge_runs(
        data['runs'], new_records, source_hint=json_path.name
    )
    data['runs'] = merged
    with open(json_path, 'w') as fh:
        json.dump(data, fh, indent=2, cls=_NumpyEncoder)

    parts = [f"{n_added} added"]
    if n_skipped:    parts.append(f"{n_skipped} skipped (duplicate)")
    if n_updated:    parts.append(f"{n_updated} null-populated")
    if n_conflicted: parts.append(f"{n_conflicted} conflict(s)")
    print(f"  [{mode}] {', '.join(parts)} → {json_path.name}")
