"""
case.py  --  everything one optimization case owns.

A :class:`Case` holds the parsed parameters, the output folder and naming, the column
labels, the current population, and the generation counter for a single input file.
It knows nothing about how its population is evaluated -- that is the evaluator's job
-- which is what lets one case be advanced by any of the three execution modes, and
lets a whole fleet of cases be advanced in lockstep by the same driver loop.

The output format (folder name, ``population_*_g*.json`` schema, the copy of the input
file dropped in the run folder) is unchanged from the original runner, so existing
post-processing and continuation files keep working.
"""

import copy
import json
import os
import pathlib
import platform
import re
import shutil
import time

import numpy as np
from metafoil.core.kulfan import Kulfan

from oso_airfoils.geometry.newMember import newMember
from oso_airfoils.optimization.config import (
    build_labels, generation_filename_width, read_input_file, resolve_path,
)
from oso_airfoils.optimization.solvers import build_sweeps, resolve_te_gap

#: matches the generation number in ``population_<filecode>_g<NNN>.json``
_GEN_RE = re.compile(r'_g(\d+)\.json$')

_UID = [0]


def _timestamp():
    return (time.strftime("%Y_%m_%d_%H-%M-%S", time.localtime())
            + f"{int((time.time() % 1) * 100):02d}")


def latest_snapshot(path):
    """Resolve a ``continuation_file`` to an actual population JSON file.

    Accepts either a file or a run directory. For a directory, the snapshot with the
    highest generation NUMBER is returned.

    The original runner sorted the filenames as strings, which picks the wrong file
    whenever the generation number overflows its zero-padding width (with the usual
    ``N_generations: 1000``, ``g999`` sorts after ``g1000``, so a run continued past
    generation 999 silently resumed from 999). It also read the directory path itself
    rather than the file it had just resolved, so the directory form raised on the
    extension check before it ever got that far.
    """
    path = str(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise ValueError(f"continuation_file does not exist: {path}")
    snaps = [f for f in os.listdir(path) if _GEN_RE.search(f)]
    if not snaps:
        raise ValueError(f"No population_*_g*.json snapshots found in {path}")
    snaps.sort(key=lambda f: int(_GEN_RE.search(f).group(1)))
    return os.path.join(path, snaps[-1])


def save_json(fname, pop, params, labels, datestr, current_generation, path_to_here):
    """Write one population snapshot. Schema is unchanged from the original runner."""
    save_dict = {}
    save_dict['input_parameters'] = params
    save_dict['input_parameters']['start_time'] = datestr
    save_dict['input_parameters']['current_generation'] = current_generation
    save_dict['input_parameters']['write_time'] = _timestamp()
    save_dict['input_parameters']['path_to_here'] = str(path_to_here)
    save_dict['input_parameters']['operating_system'] = platform.system()
    pop_arr = np.array(pop)

    # Split the per-row label list into upper-coeff, lower-coeff, and the rest.
    n_half = sum(1 for lb in labels if lb.startswith('U') and lb[1:].isdigit())
    rest_labels = labels[2 * n_half:]

    population = []
    for row in pop_arr:
        row = row.tolist()
        entry = {
            'K_upper': row[:n_half],
            'K_lower': row[n_half:2 * n_half],
        }
        entry.update(dict(zip(rest_labels, row[2 * n_half:])))
        population.append(entry)

    save_dict['population'] = population
    with open(fname, 'w') as f:
        json.dump(save_dict, f, indent=4)
    return save_dict


class Case:
    """One optimization case: its parameters, its output folder, and its population."""

    def __init__(self, input_file, model_override=None, path_to_here=None,
                 create_output=True):
        """``create_output=False`` builds the case without touching the filesystem.

        Under MPI every rank constructs the Case so that it has the parameters, but
        only the root writes: without this guard each rank would stamp its own
        timestamp and create its own run folder, scattering one run across N
        directories.
        """
        input_file = str(pathlib.Path(input_file).resolve())
        self.input_file = input_file
        self.uid = _UID[0]
        _UID[0] += 1

        params = read_input_file(input_file)
        params.setdefault('continuation_file_overwrite', False)
        if params['continuation_file_overwrite'] is None:
            params['continuation_file_overwrite'] = False

        # ---- continuation ----------------------------------------------------
        # When continuing and not overwriting, the SAVED parameters win and the new
        # file's N_generations is read as ADDITIONAL generations on top of those
        # already run.
        self.is_continuation = False
        self._snapshot = None
        self.previous_generations = 0
        if params.get('continuation_file') and not params['continuation_file_overwrite']:
            snapshot = latest_snapshot(params['continuation_file'])
            params_original = copy.deepcopy(params)
            self._snapshot = read_input_file(snapshot)
            params = self._snapshot['input_parameters']
            self.is_continuation = True
            self.previous_generations = int(params.get('current_generation'))
            params['N_generations'] = (self.previous_generations
                                       + int(params_original.get('N_generations')))
            self.outdir = str(pathlib.Path(snapshot).parent.resolve())
        elif params.get('continuation_file'):
            snapshot = latest_snapshot(params['continuation_file'])
            self._snapshot = read_input_file(snapshot)
            self.is_continuation = True
            self.previous_generations = int(
                self._snapshot['input_parameters'].get('current_generation'))
            self.outdir = str(pathlib.Path(snapshot).parent.resolve())

        if model_override:
            params['neuralfoil_model'] = model_override
        self.params = params
        self._path_to_here = str(path_to_here
                                 or pathlib.Path(input_file).parent.resolve())

        # ---- core parameters -------------------------------------------------
        self.case_number = params.get('case_number')
        self.tau = params.get('tau')
        self.N_k = int(params.get('N_k'))
        self.N_pop = int(params.get('N_pop'))
        self.CL = params.get('CL')
        self.Re = params.get('Re')
        self.file_system = params.get('file_system')
        self.N_generations = int(params.get('N_generations'))
        self.tool = params.get('tool')

        if self.N_k < 4:
            raise ValueError(
                f"{input_file}: must use at least 2 design variables top and bottom "
                f"(N_k >= 4), got {self.N_k}")
        if self.file_system is not None and self.file_system not in (0, 1, 2, 3):
            raise ValueError(
                f"{input_file}: file_system flag must be 0 (default), 1 (gpfs), "
                "2 (pscratch), or 3 (tscratch)")

        self.te_gap = resolve_te_gap(params)
        params['TE_gap'] = self.te_gap

        # ---- naming and output folder ---------------------------------------
        filecode = f"c{self.case_number}_t{int(self.tau*100)}_k{self.N_k}_n{self.N_pop}"
        if self.CL is not None:
            filecode += f"_l{int(self.CL*10)}"
        if self.Re is not None:
            filecode += f"_e{int(self.Re/1e5)}"
        if self.file_system is not None:
            filecode += f"_s{self.file_system}"
        self.filecode = filecode

        if self.is_continuation:
            self.datestr = params['start_time']
            self.folderstr = os.path.basename(self.outdir)
        else:
            self.datestr = _timestamp()
            self.folderstr = filecode + '__' + self.datestr
            ldr = resolve_path(input_file, params.get('outfile_leader'))
            self.outdir = ldr + self.folderstr
            if create_output:
                os.makedirs(self.outdir, exist_ok=True)
                # Keep both a timestamped and a plain copy of the input file with the run.
                basename = os.path.basename(input_file)
                shutil.copy(input_file, os.path.join(
                    self.outdir, self.datestr + '.' + basename.split('.')[-1]))
                shutil.copy(input_file, os.path.join(self.outdir, basename))

        self.labels = build_labels(self.N_k)

        # ---- batched-evaluation config (ignored by the serial/MPI evaluators) --
        self.sweeps = build_sweeps(params) if self.tool == 'neuralfoil' else None
        probe = Kulfan(TE_gap=self.te_gap)
        self.n_pts = int(probe.n_pts)
        self.spacing = probe.spacing
        self.toothpick_location = params.get('toothpick_location', None)
        self.surrogate_cache = None   # set by the GPU-batched evaluator

        # ---- population state ------------------------------------------------
        self.pop = None
        self.sortedData = None
        self.children = None
        self.counter = self.previous_generations
        self.done = self.counter >= self.N_generations

    # -- population --------------------------------------------------------------

    def init_population(self):
        """Seed generation 0, either fresh or reconstructed from a continuation file."""
        if self.is_continuation:
            self.pop = self._restore_population()
        elif self.params.get('seed_population_file'):
            # WARM-START (not a continuation): load only the DESIGN VECTORS from a prior
            # run's snapshot and hand them back as a fresh gen-0 population, so the driver
            # RE-EVALUATES them under THIS run's params. Used for two-phase constraint
            # activation -- phase 1 (cap off) explores to the high corner, phase 2 seeds
            # from it and re-scores every member with the cap on, avoiding stale fitness.
            self.pop = self._seed_from_file(self.params['seed_population_file'])
        else:
            self.pop = newMember(int(self.N_k / 2), self.tau, self.N_pop,
                                 te_gap=self.te_gap)
        return self.pop

    def _seed_from_file(self, path):
        """Design-vectors-only population from a snapshot (K_upper+K_lower per member),
        matched to N_pop (truncated, or topped up with fresh random members)."""
        snap = latest_snapshot(path)
        data = read_input_file(snap)
        pop = [list(e['K_upper']) + list(e['K_lower']) for e in data['population']]
        pop = np.array(pop, float)
        if len(pop) > self.N_pop:
            pop = pop[:self.N_pop]
        elif len(pop) < self.N_pop:
            extra = newMember(int(self.N_k / 2), self.tau, self.N_pop - len(pop),
                              te_gap=self.te_gap)
            pop = np.vstack([pop, np.array(extra, float)])
        return pop

    def _restore_population(self):
        """Rebuild the population array from a saved snapshot.

        ``save_json`` writes each row as ``K_upper + K_lower + [entry[l] for l in rest]``,
        so reading it back is the same walk in reverse.
        """
        rest_labels = self.labels[self.N_k:]
        pop = []
        for entry in self._snapshot['population']:
            row = list(entry['K_upper']) + list(entry['K_lower'])
            row += [entry[lbl] for lbl in rest_labels]
            pop.append(row)
        return np.array(pop)

    def design_vectors(self, rows):
        """Split population rows into (upper, lower) Kulfan coefficient blocks."""
        arr = np.asarray(rows, float)
        dv = arr[:, :self.N_k]
        half = self.N_k // 2
        return dv[:, :half], dv[:, half:self.N_k]

    # -- batched-evaluation descriptors -------------------------------------------

    def aero_item(self, rows):
        """Descriptor for one case's slice of a batched aerodynamic forward."""
        U, L = self.design_vectors(rows)
        return {'surr': self.surrogate_cache, 'uppers': U, 'lowers': L,
                'tes': self.te_gap, 'sweeps': self.sweeps}

    def geometry_item(self, rows):
        """Descriptor for one case's slice of a batched geometry precompute."""
        U, L = self.design_vectors(rows)
        return {'id': self.uid, 'uppers': U, 'lowers': L, 'tes': self.te_gap,
                'n_pts': self.n_pts, 'spacing': self.spacing,
                'tooth': self.toothpick_location}

    # -- output --------------------------------------------------------------------

    def snapshot_path(self, gen):
        width = generation_filename_width(self.N_generations)
        return os.path.join(
            self.outdir,
            'population_%s_g%s.json' % (self.filecode, str(gen).zfill(width)))

    def save(self, gen):
        """Write the current population as generation ``gen``; returns the saved dict."""
        return save_json(self.snapshot_path(gen), np.array(self.pop), self.params,
                         self.labels, self.datestr, gen, self._path_to_here)

    def __repr__(self):
        return (f"<Case {self.filecode} tool={self.tool} "
                f"gen={self.counter}/{self.N_generations}>")
