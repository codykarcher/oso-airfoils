"""
sync_pulse_case.py  —  per-case GA state for the sync-and-pulse fleet runner.

NEW FILE. One SyncCase owns everything runner.py holds for a single input file: parsed params,
labels, output folder, the current population, the surrogate cache container, the constraint-
geometry config, and the generation counter. The fleet orchestrator (runner_sync_pulse.py)
advances every SyncCase one generation in lockstep, sharing a single GPU pulse.

The param parsing, label list, filecode/folder layout, and save_json format are copied from
runner.py so the JSON output is byte-compatible with the stock runner's (same schema, same
population reconstruction path). Continuation files are not handled in this v1 — pass fresh
configs; a continuation extension can reuse runner.py's reconstruction block later.
"""

import os
import copy
import json
import time
import math
import platform
import pathlib
import shutil

import numpy as np
from metafoil.core.kulfan import Kulfan

from oso_airfoils.geometry.newMember import newMember
from oso_airfoils.optimization.geometry_functions import TE_gap_function
from oso_airfoils.optimization.batched_new_generation import _build_sweeps


def read_input_file(input_file):
    _ext = os.path.splitext(input_file)[1].lower()
    with open(input_file, 'r') as f:
        if _ext in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(f)
        elif _ext == '.json':
            return json.load(f)
        raise ValueError(f"Unsupported input extension '{_ext}'. Use .json/.yaml/.yml.")


def build_labels(N_k):
    labels = []
    for i in range(0, int(N_k / 2)):
        labels += ['U%d' % (i + 1)]
    for i in range(0, int(N_k / 2)):
        labels += ['L%d' % (i + 1)]
    labels += [
        'obj1', 'obj2', 'con_tag', 'alpha_design', 'LoD_clean_at_design', 'LoD_rough_at_design',
        'stall_margin_clean', 'stall_margin_rough', 'lift_margin_clean', 'delta_cl_from_roughness',
        'LoD_c_1d_left', 'LoD_c_1d_right', 'tau', 'ler_upper', 'ler_lower', 'Ixx', 'Iyy', 'Izz',
        'A', 'cpmin', 'con_sm_clean', 'con_sm_rough', 'con_clmax_clean', 'con_clmax_rough',
        'con_ixx', 'con_iyy', 'con_izz', 'con_a', 'con_leru', 'con_lerl', 'con_te_cone',
        'con_max_tau', 'con_max_tau_u', 'con_max_tau_l', 'con_ler_skew', 'con_tau', 'con_concave',
        'con_aftcurve', 'con_lower_flips', 'con_10deg', 'con_mom_c', 'con_mom_r',
        'con_cpmin_design_clean', 'con_cpmin_design_rough', 'con_cpmin_offset_clean',
        'con_cpmin_offset_rough', 'con_cpmin_prestall_clean', 'con_cpmin_prestall_rough',
        'con_min_rad_loc_upper', 'con_min_rad_loc_lower', 'con_toothpick', 'pareto_index',
    ]
    return labels


def save_json(fname, pop, params, labels, datestr, current_generation, path_to_here):
    save_dict = {}
    save_dict['input_parameters'] = params
    save_dict['input_parameters']['start_time'] = datestr
    save_dict['input_parameters']['current_generation'] = current_generation
    save_dict['input_parameters']['write_time'] = (
        time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime()) + f"{int((time.time() % 1) * 100):02d}")
    save_dict['input_parameters']['path_to_here'] = str(path_to_here)
    save_dict['input_parameters']['operating_system'] = platform.system()
    pop_arr = np.array(pop)

    n_half = sum(1 for lb in labels if lb.startswith('U') and lb[1:].isdigit())
    rest_labels = labels[2 * n_half:]
    population = []
    for row in pop_arr:
        row = row.tolist()
        entry = {'K_upper': row[:n_half], 'K_lower': row[n_half:2 * n_half]}
        entry.update(dict(zip(rest_labels, row[2 * n_half:])))
        population.append(entry)
    save_dict['population'] = population
    with open(fname, 'w') as f:
        json.dump(save_dict, f, indent=4)
    return save_dict


_UID = [0]


class SyncCase:
    def __init__(self, input_file, mcs, model_override=None, path_to_here=None):
        input_file = str(pathlib.Path(input_file).resolve())
        self.input_file = input_file
        params = read_input_file(input_file)

        if params.get('continuation_file'):
            raise NotImplementedError(
                f"{input_file}: continuation_file is not supported by the sync-pulse runner yet; "
                "use a fresh config.")
        if model_override:
            params['neuralfoil_model'] = model_override
        params.setdefault('tool', 'neuralfoil')

        self.params = params
        self.uid = _UID[0]; _UID[0] += 1
        self._path_to_here = str(path_to_here or pathlib.Path(input_file).parent.resolve())

        self.case_number = params.get('case_number')
        self.tau = params.get('tau')
        self.N_k = int(params.get('N_k'))
        self.N_pop = int(params.get('N_pop'))
        self.CL = params.get('CL')
        self.Re = params.get('Re')
        self.file_system = params.get('file_system')
        self.N_generations = int(params.get('N_generations'))
        self.te_gap = params['TE_gap'] if params.get('TE_gap') is not None else TE_gap_function(self.tau)
        params['TE_gap'] = self.te_gap

        if self.N_k < 4:
            raise ValueError(f"{input_file}: N_k must be >= 4")

        # filecode + output folder (identical scheme to runner.py)
        filecode = f"c{self.case_number}_t{int(self.tau*100)}_k{self.N_k}_n{self.N_pop}"
        if self.CL is not None:
            filecode += f"_l{int(self.CL*10)}"
        if self.Re is not None:
            filecode += f"_e{int(self.Re/1e5)}"
        if self.file_system is not None:
            filecode += f"_s{self.file_system}"
        self.filecode = filecode

        ldr = params.get('outfile_leader') or ('.' + os.sep)
        self.ldr = str((pathlib.Path(input_file).parent / ldr).resolve()) + os.sep
        self.datestr = (time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
                        + f"{int((time.time() % 1) * 100):02d}")
        self.folderstr = filecode + '__' + self.datestr
        self.outdir = self.ldr + self.folderstr
        os.makedirs(self.outdir, exist_ok=True)
        _bn = os.path.basename(input_file)
        shutil.copy(input_file, os.path.join(self.outdir, self.datestr + '.' + _bn.split('.')[-1]))
        shutil.copy(input_file, os.path.join(self.outdir, _bn))

        self.labels = build_labels(self.N_k)
        self.sweeps = _build_sweeps(params)

        # constraint-geometry config (same probe the batched path uses)
        _probe = Kulfan(TE_gap=self.te_gap)
        self.n_pts = int(_probe.n_pts)
        self.spacing = _probe.spacing
        self.tooth = params.get('toothpick_location', None)

        # per-case surrogate cache container (shares the fleet's GPU net)
        self.surr = mcs.new_case_cache()

        self.pop = None
        self.sortedData = None
        self.children = None
        self.counter = 0
        self.done = False

    # ---- population helpers ----
    def init_population(self):
        self.pop = newMember(int(self.N_k / 2), self.tau, self.N_pop, te_gap=self.te_gap)

    def _coeffs(self, rows):
        arr = np.asarray(rows, float)
        dv = arr[:, :self.N_k]
        half = self.N_k // 2
        return dv[:, :half], dv[:, half:self.N_k]

    def aero_item(self, rows):
        U, L = self._coeffs(rows)
        return {'surr': self.surr, 'uppers': U, 'lowers': L, 'tes': self.te_gap,
                'sweeps': self.sweeps}

    def geo_item(self, rows):
        U, L = self._coeffs(rows)
        return {'id': self.uid, 'uppers': U, 'lowers': L, 'tes': self.te_gap,
                'n_pts': self.n_pts, 'spacing': self.spacing, 'tooth': self.tooth}

    def save(self, gen):
        width = max(1, math.ceil(np.log10(max(self.N_generations, 2))))
        fname = os.path.join(self.outdir,
                             'population_%s_g%s.json' % (self.filecode, str(gen).zfill(width)))
        return save_json(fname, np.array(self.pop), self.params, self.labels, self.datestr,
                         gen, self._path_to_here)
