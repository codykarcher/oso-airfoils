"""
batched_new_generation.py  —  drop-in `newGeneration` that evaluates a whole GA generation
with ONE batched GPU surrogate forward instead of the MPI, one-airfoil-at-a-time fan-out.

NEW FILE — does not modify new_generation.py or objective_function.py. It REUSES the original
newGeneration (all of its selection / crossover / mutation / NSGA_sort / crowding / front-cap /
dedup logic is untouched) and changes only the evaluation dispatch, via two runtime shims:

  1. `objective_function.run_neuralfoil` is monkeypatched to a cache-backed drop-in that serves
     each individual's polar from a single batched nxfoil/nqfoil forward (batch_surrogate.py).
     It is bit-for-bit equivalent to the real NeuralFoil wrapper (nxfoil is a float-exact port),
     with a fallback to the real wrapper on any cache miss or 'cl'-mode call.
  2. A single-rank comm shim runs the whole generation in ONE process (no MPI). Its `bcast`
     hook is where the CHILDREN population becomes known (new_generation.py bcasts resultantPop
     right before its eval loop), so that is where the batch cache for the children is built.

The metafoil `Kulfan` geometry/constraint path in core_fitness_function is left AS-IS, so all 31
constraints are computed exactly as before (geometry batching is provided separately in
batch_geometry.py). Net effect: identical results, but the per-generation surrogate cost drops
from N_pop separate NeuralFoil calls to one batched forward.

Speedups active here: one forward/generation, genome dedup, persistent on-GPU net, TF32 +
inference_mode, optional CUDA graphs — all inside batch_surrogate.BatchSurrogate. Extra CPU win:
the MPI fan-out (process spawn + serialization) is gone; the per-individual work left on the
loop is pure constraint arithmetic (microseconds).

Use via runner_batched.py, or call this newGeneration exactly like the original.
"""
import numpy as np

from oso_airfoils.optimization import new_generation as _ng
from oso_airfoils.optimization import objective_function as _objf
from oso_airfoils.optimization.geometry_functions import TE_gap_function
from oso_airfoils.optimization.batch_surrogate import BatchSurrogate
from oso_airfoils.optimization import batch_geometry as _bg
from metafoil.core.kulfan import Kulfan as _RealKulfan

_ORIG = _ng.newGeneration       # true original, captured before any launcher patch
_SURR = {}                      # (backend, model_size, device) -> persistent BatchSurrogate


def _get_surrogate(params):
    backend = params.get("surrogate_backend", "nxfoil")
    model = params.get("neuralfoil_model", "xxxlarge")
    device = params.get("surrogate_device", "cuda")
    use_graph = bool(params.get("surrogate_cuda_graph", False))
    key = (backend, model, str(device))
    if key not in _SURR:
        _SURR[key] = BatchSurrogate(backend=backend, model_size=model, device=device,
                                    use_cuda_graph=use_graph)
    return _SURR[key]


def _build_sweeps(params):
    """Clean + rough sweeps with SUPERSET alpha grids (covering the downward extension), so the
    cache can serve any alpha sub-range core_fitness_function requests."""
    Re = float(params["Re"])
    a_ext = float(params.get("alpha_min_extend", -10.0))
    step_c = float(params["alpha_step_clean"]); step_r = float(params["alpha_step_rough"])
    a_lo_c = min(float(params["alpha_min_clean"]), a_ext); a_hi_c = float(params["alpha_max_clean"])
    a_lo_r = min(float(params["alpha_min_rough"]), a_ext); a_hi_r = float(params["alpha_max_rough"])
    grid = lambda lo, hi, st: np.arange(lo, hi + 0.5 * st, st)
    return [
        dict(name="clean", Re=Re, ncrit=float(params["N_crit_clean"]),
             xtr_u=float(params["xtp_u_clean"]), xtr_l=float(params["xtp_l_clean"]),
             alphas=grid(a_lo_c, a_hi_c, step_c)),
        dict(name="rough", Re=Re, ncrit=float(params["N_crit_rough"]),
             xtr_u=float(params["xtp_u_rough"]), xtr_l=float(params["xtp_l_rough"]),
             alphas=grid(a_lo_r, a_hi_r, step_r)),
    ]


def _pop_coeffs(pop, Nk):
    pop = np.asarray(pop, float)
    dv = pop[:, :Nk]                 # design vector = [K_upper | K_lower]
    half = Nk // 2
    return dv[:, :half], dv[:, half:Nk]


class _BatchComm:
    """Single-process comm shim (size=1, rank=0). `on_bcast` is called with any 2-D population
    array passed to bcast — that's the hook where the children population becomes known."""
    def __init__(self, Nk, on_bcast):
        self._Nk = Nk; self._on_bcast = on_bcast
    def Get_size(self): return 1
    def Get_rank(self): return 0
    def Barrier(self): pass
    def gather(self, x, root=0): return [x]
    def bcast(self, x, root=0):
        if self._on_bcast is not None and isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= self._Nk and len(x):
            self._on_bcast(x)
        return x


def newGeneration(fitnessFunction, population, normalizationVector, encodingTypes,
                  lowerBounds, upperBounds, initalize=False, comm=None, params=None):
    if params is None or params.get("tool") != "neuralfoil":
        # not a NN-surrogate run: defer entirely to the original (no batching applicable)
        return _ORIG(fitnessFunction, population, normalizationVector, encodingTypes,
                     lowerBounds, upperBounds, initalize=initalize, comm=comm, params=params)

    Nk = len(normalizationVector)
    te_gap = params["TE_gap"] if "TE_gap" in params and params["TE_gap"] is not None \
        else TE_gap_function(params["tau"])
    sweeps = _build_sweeps(params)
    surr = _get_surrogate(params)
    device = params.get("surrogate_device", "cuda")
    tooth = params.get("toothpick_location", None)
    # probe a real Kulfan to get the EXACT internal grid the constraints use (n_pts/spacing)
    _probe = _RealKulfan(TE_gap=te_gap)
    n_pts, spacing = int(_probe.n_pts), _probe.spacing

    def build_cache(pop):
        U, L = _pop_coeffs(pop, Nk)
        surr.build_population_cache(U, L, te_gap, sweeps)                 # batched aero
        reg, psi = _bg.precompute_population_geometry(                    # batched geometry
            U, L, te_gap, n_pts=n_pts, spacing=spacing,
            toothpick_location=tooth, device=device)
        _bg.TorchKulfan.install_registry(reg, psi, tooth)

    orig_run = _objf.run_neuralfoil
    orig_kulfan = _objf.Kulfan
    _objf.run_neuralfoil = surr.make_cached_run(fallback=orig_run)
    _objf.Kulfan = _bg.TorchKulfan                        # batched geometry shim (constraint path)
    try:
        if initalize:
            build_cache(population)                       # population is known up front
            bc = _BatchComm(Nk, on_bcast=None)
        else:
            bc = _BatchComm(Nk, on_bcast=build_cache)     # children cache built at bcast()
        return _ORIG(fitnessFunction, population, normalizationVector, encodingTypes,
                     lowerBounds, upperBounds, initalize=initalize, comm=bc, params=params)
    finally:
        _objf.run_neuralfoil = orig_run                   # always restore
        _objf.Kulfan = orig_kulfan
