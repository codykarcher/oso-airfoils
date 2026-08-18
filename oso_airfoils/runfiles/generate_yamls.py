import io
import json
import re
from oso_airfoils.postprocessing.runners import _find_stem_in_tree
from oso_airfoils.core.data_utils import _DEFAULT_AFL_ROOT, _DEFAULT_PERF_ROOT
import pathlib
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as Q
_here = pathlib.Path(__file__).parent
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.representer.add_representer(type(None), lambda dumper, _: dumper.represent_scalar('tag:yaml.org,2002:null', 'null'))

def _column_format(yaml_str):
    """Reformat top-level key: value lines to column-aligned style."""
    lines = yaml_str.splitlines()
    _KEY_PAT = re.compile(r'^([A-Za-z_]\w*)\s*:\s*(.*)')
    matches = [_KEY_PAT.match(l) for l in lines]
    col = max((len(m.group(1)) for m in matches if m), default=0) + 1
    out = []
    for line, m in zip(lines, matches):
        if m:
            out.append(f'{m.group(1):<{col}}: {m.group(2)}')
        else:
            out.append(line)
    return '\n'.join(out) + '\n'

def load_geometry_params(airfoil_name):
    """Load geometry constraint parameters directly from the cached performance JSON.

    The family directory is resolved automatically from the airfoil name.

    Parameters
    ----------
    airfoil_name : str
        Name of the airfoil (e.g. 'mhkf1-240', 'FFA-W3-241').

    Returns
    -------
    dict with keys: tau, TE_gap, Ixx_con, Iyy_con, Izz_con, A_con,
                    ler_con_upper, ler_con_lower
    """

    family, stem = _find_stem_in_tree(airfoil_name, None)
    jpath = _DEFAULT_PERF_ROOT / family / 'performance_data' / f'{stem}.json'
    with open(jpath) as f:
        d = json.load(f)
    g = d['geometry']
    return {
        'tau'          : g['tau'],
        'TE_gap'       : g['TE_gap'],
        'Ixx_con'      : g['Ixx'],
        'Iyy_con'      : g['Iyy'],
        'Izz_con'      : g['Izz'],
        'A_con'        : g['area'],
        'ler_con_upper': g['LE_radius_upper'],
        'ler_con_lower': g['LE_radius_lower'],
    }

def linear_query(tau, x1, x2, y1, y2):
    """Simple linear interpolation/extrapolation."""
    return y1 + (y2 - y1) * (tau - x1) / (x2 - x1)

# Per-thickness values taken directly from the commented "OSO Values" block at
# the top of base1.yaml (taus / TE_gaps / Ixx_con / Iyy_con / Izz_con / A_con /
# ler_con / cone_ang / CLs / Re). ler_con is a single value applied to both the
# upper and lower LE-radius constraints. Everything else stays at the base1.yaml
# default for each generated case.
tau_data = {
    0.15: {'CL': 1.5, 'Re': 12e6, 'TE_gap': 0.00196, 'Ixx_con': 0.00011000, 'Iyy_con': 0.00397999, 'Izz_con': 0.00408809, 'A_con': 0.08700496, 'ler_con_upper': 0.007, 'ler_con_lower': 0.007, 'cone_angle': 10.0},
    0.18: {'CL': 1.5, 'Re': 12e6, 'TE_gap': 0.00230, 'Ixx_con': 0.00017438, 'Iyy_con': 0.00436351, 'Izz_con': 0.00454606, 'A_con': 0.09995900, 'ler_con_upper': 0.008, 'ler_con_lower': 0.008, 'cone_angle': 10.0},
    0.21: {'CL': 1.5, 'Re': 12e6, 'TE_gap': 0.00262, 'Ixx_con': 0.00027518, 'Iyy_con': 0.00493714, 'Izz_con': 0.00521632, 'A_con': 0.11477620, 'ler_con_upper': 0.010, 'ler_con_lower': 0.010, 'cone_angle':  5.0},
    0.24: {'CL': 1.4, 'Re': 13e6, 'TE_gap': 0.00751, 'Ixx_con': 0.00041096, 'Iyy_con': 0.00561409, 'Izz_con': 0.00602287, 'A_con': 0.13051205, 'ler_con_upper': 0.025, 'ler_con_lower': 0.025, 'cone_angle':  5.0},
    0.27: {'CL': 1.3, 'Re': 16e6, 'TE_gap': 0.01012, 'Ixx_con': 0.00058321, 'Iyy_con': 0.00633417, 'Izz_con': 0.00691323, 'A_con': 0.14660942, 'ler_con_upper': 0.030, 'ler_con_lower': 0.030, 'cone_angle':  5.0},
    0.30: {'CL': 1.2, 'Re': 18e6, 'TE_gap': 0.01140, 'Ixx_con': 0.00079640, 'Iyy_con': 0.00706380, 'Izz_con': 0.00785849, 'A_con': 0.16289864, 'ler_con_upper': 0.040, 'ler_con_lower': 0.040, 'cone_angle':  0.0},
    0.33: {'CL': 1.2, 'Re': 16e6, 'TE_gap': 0.01140, 'Ixx_con': 0.00105795, 'Iyy_con': 0.00779600, 'Izz_con': 0.00885328, 'A_con': 0.17959744, 'ler_con_upper': 0.060, 'ler_con_lower': 0.060, 'cone_angle':  0.0},
    0.36: {'CL': 1.2, 'Re': 13e6, 'TE_gap': 0.01140, 'Ixx_con': 0.00137822, 'Iyy_con': 0.00855043, 'Izz_con': 0.00991577, 'A_con': 0.19731100, 'ler_con_upper': 0.080, 'ler_con_lower': 0.080, 'cone_angle':  0.0},
}

global_changes = {
    # 'cp_min_design'                                : -2.0,
    # 'cp_min_prestall'                              : -7.0,
    'N_generations'                                : 500,
}

tool_cases = {
    'neuralfoil': {
        # base1.yaml already sets tool: neuralfoil
        'target_cl'                                    : 1.5,
        'target_alpha'                                 : 10.0,
    },
    'xfoil': {
        'tool'                                         : Q("xfoil"),
        'target_cl'                                    : 1.5,
        'target_alpha'                                 : 10.0,
    },
    'qfoil': {
        'tool'                                         : Q("qfoil"),
        # qfoil intentionally leaves target_cl / target_alpha at the
        # base1.yaml default (null).
    },
}

# MPI process count per tool for run_all.sh. xfoil and qfoil both run
# in-memory through metafoil (comparable cost); neuralfoil is lighter.
proc_counts = {'neuralfoil': 96, 'xfoil': 188, 'qfoil': 188}

generated = {}   # tool name -> list of written yaml filenames (in tau order)

for nm, case_data in tool_cases.items():

    list_keys = [k for k, v in case_data.items() if isinstance(v, list)]

    for tau_idx, (tau, td) in enumerate(tau_data.items()):
        # Reload base fresh each iteration to get a clean ruamel round-trip object
        with open(_here / 'base1.yaml') as f:
            params = _yaml.load(f)

        # Apply all thickness-specific values (keys match YAML names directly)
        for k, v in td.items():
            params[k] = v
        params['tau'] = tau

        # Global changes
        for k, v in global_changes.items():
            params[k] = v

        # Apply tool/case-specific overrides; cycle through list values by tau index
        for k, v in case_data.items():
            params[k] = v[tau_idx % len(v)] if isinstance(v, list) else v

        fname = _here / "".join([f't{int(tau * 100):02d}_', nm, '.yaml'])
        buf = io.StringIO()
        _yaml.dump(params, buf)
        fname.write_text(_column_format(buf.getvalue()))
        generated.setdefault(nm, []).append(fname.name)
        print(f'Wrote {fname.name}  (tau={tau})')

# ---------------------------------------------------------------------------
# Emit run_all.sh: one `mpirun ... runner <case>.yaml` line per generated case,
# grouped by tool, with the tool's MPI process count.
# ---------------------------------------------------------------------------
sh_lines = [
    '#!/usr/bin/env bash',
    '# Run all generated YAML cases through the OSO airfoil optimizer.',
    '# (auto-generated by generate_yamls.py -- edit that script, not this file)',
]
for nm in generated:
    sh_lines.append(f'#   - {nm} cases : {proc_counts.get(nm, 96)} MPI processes')
sh_lines += ['', 'cd "$(dirname "${BASH_SOURCE[0]}")"', '']

for nm, files in generated.items():
    n = proc_counts.get(nm, 96)
    sh_lines.append(f'# --- {nm} ---')
    for fn in files:
        sh_lines.append(f'mpirun -n {n:<3} python -m oso_airfoils.optimization.runner {fn}')
    sh_lines.append('')

sh_path = _here / 'run_all.sh'
sh_path.write_text('\n'.join(sh_lines))
sh_path.chmod(0o755)
print(f'Wrote {sh_path.name}  ({sum(len(v) for v in generated.values())} cases)')

