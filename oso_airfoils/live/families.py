"""
families.py -- reference airfoil families for the live dashboard.

Only GEOMETRY is taken from the airfoil store. The reference's polar is then computed
fresh through the SAME surrogate, at the SAME Reynolds/transition conditions, as the
GA's own airfoils -- so the comparison is like-for-like. Using the stored polar data
instead would mix solvers and operating conditions and make the reference curve
meaningless next to the optimized ones.

A family is offered only if it has a member within ``TAU_MATCH_TOL`` of the case's
thickness; otherwise there is nothing meaningful to compare against at this t/c and
the button is greyed out.
"""

import json
import pathlib

from oso_airfoils.core.data_utils import _DEFAULT_PERF_ROOT
from oso_airfoils.postprocessing.oso_polar import _FAMILY_ALIASES, TAU_MATCH_TOL

#: Display names for the canonical family keys.
LABELS = {
    'du': 'DU', 'ffa': 'FFA', 'mhkf1': 'MHKF1',
    'risoa': 'Risø A', 'risob': 'Risø B', 'risop': 'Risø P',
    's20': 'S-series 20m', 's40': 'S-series 40m',
    'osowt1': 'OSO WT1', 'osowt2': 'OSO WT2', 'osowt2s': 'OSO WT2S',
}
ORDER = ['osowt1', 'osowt2', 'osowt2s', 'du', 'ffa', 'risoa', 'risob', 'risop',
         's20', 's40', 'mhkf1']


def _canonical():
    """One entry per distinct family, keyed by the alias used in LABELS."""
    out = {}
    for alias, ent in _FAMILY_ALIASES.items():
        if alias not in LABELS:
            continue
        out[alias] = ent
    return out


#: Geometry lookups are cached: the performance JSONs carry every recorded polar
#: run, so they are multi-megabyte, and json-parsing a whole family took ~13 s. The
#: geometry inside them never changes during a run.
_GEO_CACHE: dict = {}


def _closest(family_dir, tau, allowed_stems=None):
    """(stem, tau, geometry) of the closest member by thickness, or None."""
    ck = (str(family_dir), round(float(tau), 6),
          tuple(allowed_stems) if allowed_stems else None)
    if ck in _GEO_CACHE:
        return _GEO_CACHE[ck]
    _GEO_CACHE[ck] = _r = _closest_uncached(family_dir, tau, allowed_stems)
    return _r


def _closest_uncached(family_dir, tau, allowed_stems=None):
    pd = pathlib.Path(family_dir) / 'performance_data'
    if not pd.is_dir():
        return None
    best = None
    for jf in sorted(pd.glob('*.json')):
        if allowed_stems is not None and jf.stem not in allowed_stems:
            continue
        try:
            g = json.loads(jf.read_text()).get('geometry', {})
        except Exception:
            continue
        t = g.get('tau')
        if t is None or 'upperCoefficients' not in g:
            continue
        if best is None or abs(t - tau) < abs(best[1] - tau):
            best = (jf.stem, float(t), g)
    return best


def survey(tau, tol=TAU_MATCH_TOL, perf_root=None):
    """List every family with whether it has geometry near ``tau``.

    Returns dicts of {key, label, available, stem, tau, dtau} in display order.
    """
    root = pathlib.Path(perf_root or _DEFAULT_PERF_ROOT)
    fams = _canonical()
    out = []
    for key in ORDER:
        if key not in fams:
            continue
        ent = fams[key]
        d, stems = (ent, None) if isinstance(ent, str) else ent
        hit = _closest(root / d, tau, stems)
        if hit is None:
            out.append(dict(key=key, label=LABELS[key], available=False,
                            stem=None, tau=None, dtau=None))
            continue
        stem, t, _ = hit
        ok = abs(t - tau) <= tol
        out.append(dict(key=key, label=LABELS[key], available=bool(ok),
                        stem=stem, tau=round(t, 4), dtau=round(abs(t - tau), 4)))
    return out


def library(perf_root=None):
    """Every airfoil in the store that has usable Kulfan geometry.

    Returns ``[{stem, family, tau}]`` sorted by family then thickness. This backs the
    manual picker, so it deliberately ignores the thickness filter the family buttons
    apply -- picking an off-thickness reference by hand is a legitimate thing to want,
    it just isn't the sensible DEFAULT the buttons offer.
    """
    root = pathlib.Path(perf_root or _DEFAULT_PERF_ROOT)
    out = []
    for d in sorted(root.iterdir()):
        pd = d / 'performance_data'
        if not pd.is_dir():
            continue
        for jf in sorted(pd.glob('*.json')):
            try:
                g = json.loads(jf.read_text()).get('geometry', {})
            except Exception:
                continue
            if 'upperCoefficients' not in g or g.get('tau') is None:
                continue
            out.append(dict(stem=jf.stem, family=d.name, tau=round(float(g['tau']), 4)))
    out.sort(key=lambda r: (r['family'], r['tau']))
    return out


_STEM_CACHE: dict = {}


def kulfan_for_stem(stem, perf_root=None):
    """(stem, tau, Kulfan) for one named airfoil anywhere in the store, or None."""
    if stem in _STEM_CACHE:
        return _STEM_CACHE[stem]
    _STEM_CACHE[stem] = _r = _kulfan_for_stem_uncached(stem, perf_root)
    return _r


def _kulfan_for_stem_uncached(stem, perf_root=None):
    from metafoil.core.kulfan import Kulfan
    root = pathlib.Path(perf_root or _DEFAULT_PERF_ROOT)
    for d in sorted(root.iterdir()):
        jf = d / 'performance_data' / f'{stem}.json'
        if not jf.is_file():
            continue
        try:
            g = json.loads(jf.read_text()).get('geometry', {})
        except Exception:
            return None
        if 'upperCoefficients' not in g:
            return None
        afl = Kulfan(TE_gap=float(g.get('TE_gap', 0.0)))
        afl.upperCoefficients = g['upperCoefficients']
        afl.lowerCoefficients = g['lowerCoefficients']
        return stem, float(g.get('tau') or 0.0), afl
    return None


def kulfan_for(key, tau, perf_root=None):
    """Kulfan geometry for a family's thickness-matched member, or None.

    Geometry only -- the caller runs it through the surrogate itself.
    """
    from metafoil.core.kulfan import Kulfan
    root = pathlib.Path(perf_root or _DEFAULT_PERF_ROOT)
    ent = _canonical().get(key)
    if ent is None:
        return None
    d, stems = (ent, None) if isinstance(ent, str) else ent
    hit = _closest(root / d, tau, stems)
    if hit is None:
        return None
    stem, t, g = hit
    afl = Kulfan(TE_gap=float(g.get('TE_gap', 0.0)))
    afl.upperCoefficients = g['upperCoefficients']
    afl.lowerCoefficients = g['lowerCoefficients']
    return stem, t, afl
