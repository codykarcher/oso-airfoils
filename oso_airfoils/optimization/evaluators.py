"""
evaluators.py  --  the three ways of turning design vectors into fitness rows.

All three implement the same small protocol, so the driver loop is written once::

    ev.evaluate(items)   items = [(case, rows), ...]  ->  {case.uid: [fitness_row, ...]}
    ev.broadcast(obj)    keep every process's copy of a Python object in sync
    ev.is_root           True on the process that owns population state and file output

``items`` carries the WHOLE fleet's work for one generation, not one case's. That is
what makes the GPU-batched evaluator possible -- it can fire a single forward across
every case at once -- while the serial and MPI evaluators simply loop over the items.
A single-case run is just a fleet of one.

  SerialEvaluator      one process, one airfoil at a time. The debugging baseline.
  MPIEvaluator         the classic fan-out: every rank evaluates its stride of the
                       fleet's rows and results are gathered on rank 0.
  GPUBatchEvaluator    one batched surrogate forward plus one batched geometry
                       precompute for the entire fleet, then per-row constraint
                       arithmetic served from those caches.

Note what is NOT here any more: none of these patch module globals. The solver and the
geometry class are passed into ``airfoil_fitness`` as arguments, so two cases can be in
flight in one process without stepping on each other, and there is no ordering
requirement between importing a module and patching a name on it.
"""

import sys

import numpy as np

from oso_airfoils.optimization.objective_function import airfoil_fitness
from oso_airfoils.optimization.solvers import make_solver


def cprint(x):
    sys.stdout.flush()
    print(x, flush=True)


def _chunks(seq, n):
    """Split ``seq`` into chunks of at most ``n`` (``n<=0`` means one chunk)."""
    if not n or n <= 0 or n >= len(seq):
        return [list(seq)]
    return [list(seq[i:i + n]) for i in range(0, len(seq), n)]


class Evaluator:
    """Interface + the single-process defaults."""

    name = 'evaluator'
    is_root = True
    n_rows_last = None

    def broadcast(self, obj):
        """Return ``obj`` as seen by the root process. A no-op unless distributed."""
        return obj

    def barrier(self):
        pass

    def evaluate(self, items):
        """``[(case, rows), ...]`` -> ``{case.uid: [fitness_row, ...]}`` in row order.

        Returns None on non-root processes of a distributed evaluator.
        """
        raise NotImplementedError

    def describe(self):
        return self.name

    def close(self):
        pass


def _eval_rows(case, rows, solver, kulfan=None):
    """Evaluate one case's rows with an already-built solver."""
    return [airfoil_fitness({'pid': i, 'individual': row, 'params': case.params},
                            solver=solver, kulfan=kulfan)
            for i, row in enumerate(rows)]


class SerialEvaluator(Evaluator):
    """One process, one airfoil at a time."""

    name = 'serial'

    def evaluate(self, items):
        out = {}
        n = 0
        for case, rows in items:
            solver = make_solver(case.params)
            out[case.uid] = _eval_rows(case, rows, solver)
            n += len(rows)
        self.n_rows_last = n
        return out


class MPIEvaluator(Evaluator):
    """MPI fan-out: each rank evaluates its stride of the fleet's rows.

    The stride is taken over the fleet's rows CONCATENATED across cases, so a fleet of
    small cases balances across ranks just as well as one big case. For a single case
    this is exactly the original ``i % size == rank`` distribution.
    """

    name = 'mpi'

    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    @property
    def is_root(self):
        return self.rank == 0

    def broadcast(self, obj):
        return self.comm.bcast(obj, root=0)

    def barrier(self):
        self.comm.Barrier()

    def describe(self):
        return f"mpi ({self.size} ranks)"

    def evaluate(self, items):
        # Every rank must see identical `items`; the driver broadcasts rows first.
        tasks = [(ci, i) for ci, (_, rows) in enumerate(items) for i in range(len(rows))]
        solvers = {}
        local = []
        for t, (ci, i) in enumerate(tasks):
            if t % self.size != self.rank:
                continue
            case, rows = items[ci]
            if ci not in solvers:
                solvers[ci] = make_solver(case.params)
            res = airfoil_fitness({'pid': i, 'individual': rows[i], 'params': case.params},
                                  solver=solvers[ci])
            local.append((ci, i, res))

        gathered = self.comm.gather(local, root=0)
        self.n_rows_last = len(tasks)
        if self.rank != 0:
            return None

        out = {case.uid: [None] * len(rows) for case, rows in items}
        for chunk in gathered:
            for ci, i, res in chunk:
                out[items[ci][0].uid][i] = res
        return out


class GPUBatchEvaluator(Evaluator):
    """One batched forward across the whole fleet, then per-row constraint arithmetic.

    Per generation this does:
      1. ``pulse_aero``     -- every case's (airfoil x sweep x alpha) rows in a single
                               GPU forward, chunked by ``max_pulse`` if the fleet is
                               larger than device memory allows;
      2. ``pulse_geometry`` -- one batched Kulfan geometry precompute, shared between
                               cases with identical geometry config;
      3. per case, evaluate its rows with a solver that serves polars from its slice of
         (1) and a Kulfan factory bound to its registry from (2).

    Only surrogate tools can be batched, which ``config.resolve_execution`` enforces
    before this class is ever constructed.
    """

    name = 'gpu-batched'

    def __init__(self, backend='nxfoil', model_size='xxxlarge', device='cuda',
                 use_cuda_graph=False, max_pulse=0):
        from oso_airfoils.optimization.multicase_surrogate import MultiCaseSurrogate
        self.mcs = MultiCaseSurrogate(backend=backend, model_size=model_size,
                                      device=device, use_cuda_graph=use_cuda_graph)
        self.backend = backend
        self.model_size = model_size
        self.device = self.mcs.device
        self.max_pulse = int(max_pulse or 0)
        self._fallback_run = None

    def describe(self):
        dev = f"device={self.device}"
        if self.mcs.geometry_device != self.device:
            # e.g. MPS runs the float32 net but has no float64 for the geometry
            dev += f" geometry={self.mcs.geometry_device}"
        return (f"gpu-batched (backend={self.backend} model={self.model_size} "
                f"{dev} max_pulse={self.max_pulse or 'all'})")

    def _fallback(self):
        """The real NeuralFoil wrapper, used only on a cache miss."""
        if self._fallback_run is None:
            from oso_airfoils.core.neuralfoil_wrapper import run
            self._fallback_run = run
        return self._fallback_run

    def evaluate(self, items):
        from oso_airfoils.optimization.batch_geometry import TorchKulfanFactory

        for case, _ in items:
            if case.surrogate_cache is None:
                case.surrogate_cache = self.mcs.new_case_cache()

        # 1. the pulse: one aero forward across the fleet (chunked if requested)
        n = 0
        for chunk in _chunks(items, self.max_pulse):
            n += self.mcs.pulse_aero([case.aero_item(rows) for case, rows in chunk])
        self.n_rows_last = n

        # 2. batched geometry, shared across cases with matching geometry config
        geo = self.mcs.pulse_geometry([case.geometry_item(rows) for case, rows in items])

        # 3. per-case evaluation out of those two caches
        out = {}
        for case, rows in items:
            registry, psi, tooth = geo[case.uid]
            kulfan = TorchKulfanFactory(registry, psi, tooth)
            cached_run = case.surrogate_cache.make_cached_run(fallback=self._fallback())
            solver = make_solver(case.params, neuralfoil_run=cached_run)
            out[case.uid] = _eval_rows(case, rows, solver, kulfan=kulfan)
        return out


def make_evaluator(mode, settings=None, max_pulse=0):
    """Construct the evaluator for a resolved execution mode.

    ``mode`` must already have been validated by ``config.resolve_execution``.
    """
    if mode == 'serial':
        return SerialEvaluator()
    if mode == 'mpi':
        return MPIEvaluator()
    if mode == 'gpu-batched':
        return GPUBatchEvaluator(max_pulse=max_pulse, **(settings or {}))
    raise ValueError(f"unknown execution mode {mode!r}")
