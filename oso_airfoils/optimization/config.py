"""
config.py  --  input-file loading, output labels, and execution-mode resolution.

This module owns the two INDEPENDENT axes of an optimization run, which used to be
tangled together across three separate launcher scripts:

  * ``tool``      -- WHAT PHYSICS is evaluated:  'xfoil' | 'qfoil' | 'neuralfoil'
  * ``execution`` -- HOW it is executed:         'serial' | 'mpi' | 'gpu-batched'

``tool`` picks the aerodynamic solver (see ``solvers.py``); ``execution`` picks the
evaluator that drives it (see ``evaluators.py``). Not every combination is legal --
the GPU-batched evaluator only exists for surrogate tools -- so
:func:`resolve_execution` validates the pair loudly at startup instead of silently
degrading to something the user did not ask for.
"""

import json
import math
import os
import pathlib

#: Execution backends. See ``evaluators.py`` for the implementations.
EXECUTION_MODES = ('serial', 'mpi', 'gpu-batched')

#: Aerodynamic solvers. See ``solvers.py``.
TOOLS = ('xfoil', 'qfoil', 'neuralfoil')

#: Tools that are neural-network surrogates, and therefore batchable on a GPU.
SURROGATE_TOOLS = ('neuralfoil',)

#: Kulfan coefficients per surface expected by the surrogate nets. A case whose N_k
#: differs is refit to this order on the way into the forward (see
#: ``batch_surrogate.to_net_order``), exactly as the per-airfoil NeuralFoil wrapper
#: does with ``afl.changeOrder(8)``, so any N_k is batchable.
BATCHED_N_K = 16

#: Environment variables set by the common MPI launchers, used to auto-detect that
#: we are running under ``mpirun``/``srun`` without paying the mpi4py import cost
#: on a run that turns out to be single-process.
_MPI_SIZE_ENV_VARS = (
    'OMPI_COMM_WORLD_SIZE',     # Open MPI
    'PMI_SIZE',                 # MPICH / Intel MPI
    'MV2_COMM_WORLD_SIZE',      # MVAPICH2
    'MSMPI_RANK_COUNT',         # MS-MPI
)


# ---------------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------------

def read_input_file(input_file):
    """Load a case file. Accepts ``.json``, ``.yaml`` or ``.yml``."""
    ext = os.path.splitext(str(input_file))[1].lower()
    with open(input_file, 'r') as f:
        if ext in ('.yaml', '.yml'):
            import yaml
            return yaml.safe_load(f)
        if ext == '.json':
            return json.load(f)
    raise ValueError(
        f"Unsupported input file extension '{ext}' for {input_file}. "
        "Use .json, .yaml, or .yml.")


# ---------------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------------

#: Per-member columns written after the ``N_k`` design variables, in order. These
#: must stay aligned with what ``objective_function.core_fitness_function`` returns
#: (minus its leading ``pid``), plus the trailing ``pareto_index`` appended by the
#: NSGA sort.
#:
#: NOTE on ``alpha_design``: for a REJECTED member that column holds a rejection code
#: rather than an angle (see ``objective_function.REJECTION_CODES``). Mask on
#: ``obj1 == inf`` or ``con_tag`` before doing statistics on it, or you will average
#: status codes in with real angles.
REPORTED_LABELS = [
    'obj1', 'obj2', 'con_tag', 'alpha_design', 'LoD_clean_at_design',
    'LoD_rough_at_design', 'stall_margin_clean', 'stall_margin_rough',
    'lift_margin_clean', 'delta_cl_from_roughness', 'LoD_c_1d_left', 'LoD_c_1d_right',
    'tau', 'ler_upper', 'ler_lower', 'Ixx', 'Iyy', 'Izz', 'A', 'cpmin',
    'con_sm_clean', 'con_sm_rough', 'con_clmax_clean', 'con_clmax_rough', 'con_ixx',
    'con_iyy', 'con_izz', 'con_a', 'con_leru', 'con_lerl', 'con_te_cone',
    'con_max_tau', 'con_max_tau_u', 'con_max_tau_l', 'con_ler_skew', 'con_tau',
    'con_concave', 'con_aftcurve', 'con_lower_flips', 'con_10deg', 'con_mom_c',
    'con_mom_r', 'con_cpmin_design_clean', 'con_cpmin_design_rough',
    'con_cpmin_offset_clean', 'con_cpmin_offset_rough', 'con_cpmin_prestall_clean',
    'con_cpmin_prestall_rough', 'con_min_rad_loc_upper', 'con_min_rad_loc_lower',
    'con_toothpick', 'con_curvature_accel', 'con_bulge', 'viol', 'pareto_index',
]


def build_labels(N_k):
    """Column labels for one population row: ``U1..Un, L1..Ln`` then REPORTED_LABELS."""
    half = int(N_k / 2)
    labels = ['U%d' % (i + 1) for i in range(half)]
    labels += ['L%d' % (i + 1) for i in range(half)]
    labels += list(REPORTED_LABELS)
    return labels


def generation_filename_width(N_generations):
    """Zero-pad width for the generation number in ``population_*_g*.json``.

    Reproduces the stock runner's formula exactly so output filenames are unchanged
    for every existing case. Note this can under-pad (N_generations=1000 gives width
    3, so generation 1000 is written as ``g1000``); the continuation-file picker in
    ``case.py`` therefore sorts snapshots NUMERICALLY rather than lexicographically,
    which is correct regardless of padding.
    """
    return math.ceil(math.log10(N_generations))


# ---------------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------------

def detect_mpi_size():
    """Number of ranks we were launched with, or 1 if not under an MPI launcher.

    Reads the launcher's environment rather than importing ``mpi4py``, so a serial or
    GPU run never pays for an MPI import it will not use. If ``mpi4py`` has already
    been imported by someone else, its world size is authoritative.
    """
    import sys
    if 'mpi4py.MPI' in sys.modules:
        return int(sys.modules['mpi4py.MPI'].COMM_WORLD.Get_size())
    for var in _MPI_SIZE_ENV_VARS:
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return 1


def cuda_available():
    """True if torch reports a usable CUDA device. Never raises."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_execution(params, requested=None, mpi_size=None):
    """Decide which evaluator to use, and reject impossible combinations.

    Precedence: ``requested`` (CLI ``--execution``) > ``params['execution']`` > auto.

    Auto resolves to ``mpi`` when launched under an MPI launcher with more than one
    rank; otherwise to ``gpu-batched`` when the tool is a surrogate and CUDA is
    present; otherwise to ``serial``.

    Returns the resolved mode. Raises ValueError on an unusable combination -- the
    old launchers silently did something else instead (``runner_batched.py`` ran
    unbatched for a non-surrogate tool), which made a mis-specified run look like a
    slow one rather than a mistake.
    """
    tool = params.get('tool')
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {list(TOOLS)}, got {tool!r}")

    if mpi_size is None:
        mpi_size = detect_mpi_size()

    mode = requested or params.get('execution') or 'auto'
    mode = str(mode).lower().replace('_', '-')

    if mode == 'auto':
        if mpi_size > 1:
            mode = 'mpi'
        elif tool in SURROGATE_TOOLS and cuda_available():
            mode = 'gpu-batched'
        else:
            mode = 'serial'

    if mode not in EXECUTION_MODES:
        raise ValueError(
            f"execution must be one of {list(EXECUTION_MODES)} (or 'auto'), got {mode!r}")

    if mode == 'gpu-batched':
        if tool not in SURROGATE_TOOLS:
            raise ValueError(
                f"execution 'gpu-batched' requires a surrogate tool "
                f"{list(SURROGATE_TOOLS)}, but tool is {tool!r}. "
                f"{tool} runs an external solver per airfoil and cannot be batched; "
                "use execution 'mpi' (or 'serial') for it.")
        if mpi_size > 1:
            raise ValueError(
                f"execution 'gpu-batched' is single-process by design but was launched "
                f"with {mpi_size} MPI ranks. Run it without mpirun, or use "
                "execution 'mpi'.")

    if mode == 'serial' and mpi_size > 1:
        raise ValueError(
            f"execution 'serial' was requested but the job was launched with {mpi_size} "
            "MPI ranks, which would run the whole optimization redundantly on every "
            "rank. Use execution 'mpi', or launch without mpirun.")

    return mode


def surrogate_settings(params, model=None, backend=None, device=None, cuda_graph=None):
    """Collect the surrogate knobs, with CLI overrides taking precedence over the
    case file. Only consulted by the GPU-batched evaluator."""
    return dict(
        backend=backend or params.get('surrogate_backend', 'nxfoil'),
        model_size=model or params.get('neuralfoil_model', 'xxxlarge'),
        device=device or params.get('surrogate_device', 'cuda'),
        use_cuda_graph=(params.get('surrogate_cuda_graph', False)
                        if cuda_graph is None else bool(cuda_graph)),
    )


def resolve_path(input_file, leader):
    """Resolve an ``outfile_leader`` relative to the case file's own directory, so a
    run behaves the same regardless of the caller's working directory."""
    leader = leader or ('.' + os.sep)
    return str((pathlib.Path(input_file).parent / leader).resolve()) + os.sep
