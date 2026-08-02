"""Pin solver results to oso's historical key schema.

metafoil's backends now return ``metafoil.core.result.AeroResult``, which
stores one canonical vocabulary and resolves the other spellings on read:
``r['M']``, ``r['N_crit']`` and ``r['xtp_top']`` all still work even though the
keys actually stored are ``mach``, ``n_crit``, ``xtp_u`` and ``xtp_l``.

That aliasing is invisible right up until the result is serialized. ``json.dumps``
iterates real keys, so a record written straight from an AeroResult lands in the
archive spelled the canonical way, and every reader afterwards is a plain dict
with no alias magic left -- ``record['M']`` raises KeyError.

This matters here because ``postprocessing.runners._append_to_json`` APPENDS to
the existing per-airfoil performance JSONs. Left alone it would interleave
records in two vocabularies inside one file, and only the new ones would break
readers -- a silent, progressive corruption of the archive rather than an error
anyone would notice on the run that caused it.

oso's stored schema is the legacy spelling (verified against
``data/*/performance_data/*.json``: M, N_crit, xtp_top, xtp_bot), and the
neuralfoil wrapper never went through metafoil's ``attach`` so it still emits
that schema. Normalizing the two Fortran wrappers here keeps all three tools
writing one vocabulary, and keeps the archive readable by code that predates the
metafoil API unification.
"""
from __future__ import annotations

# canonical (metafoil) -> the spelling oso stores and reads
_TO_OSO = {
    "mach": "M",
    "n_crit": "N_crit",
    "xtp_u": "xtp_top",
    "xtp_l": "xtp_bot",
}


def to_oso_schema(res):
    """Return a plain dict keyed the way oso stores results.

    Plain dict, deliberately: the point is that the result survives a JSON
    round-trip with the same names it had going in, so it must not depend on
    AeroResult's read-time aliasing.
    """
    out = {}
    for k, v in dict.items(res) if isinstance(res, dict) else res.items():
        out[_TO_OSO.get(k, k)] = v
    return out
