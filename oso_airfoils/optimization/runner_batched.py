"""
runner_batched.py  —  launch the EXISTING oso-airfoils runner with GPU-batched surrogate
evaluation, in a single process (no MPI fan-out).

NEW FILE — overwrites nothing. It patches `new_generation.newGeneration` to the batched drop-in
(batched_new_generation.newGeneration) BEFORE the stock runner binds that name, then runs the
stock runner unchanged. All of the runner's config parsing, continuation_file handling, snapshot
writing and printing are reused verbatim.

    # single process, GPU-batched (only meaningful when the YAML has tool: neuralfoil)
    python -m oso_airfoils.optimization.runner_batched  <case.yaml>

Optional YAML keys picked up by the batched path (all have sane defaults):
    surrogate_backend : "nxfoil" (default, float-exact NeuralFoil replica) | "nqfoil"
    surrogate_device  : "cuda" (default) | "cpu"
    surrogate_cuda_graph : false (default) | true   # capture/replay for fixed-shape gens
    neuralfoil_model  : model size (already used by the stock neuralfoil path)

For tool != "neuralfoil" this simply behaves like the stock runner (no batching applicable).
Do NOT launch this under mpirun — it is single-process by design; the whole generation is one
batched forward. Use the stock `runner.py` under mpirun for the non-surrogate solvers.
"""
import runpy

from oso_airfoils.optimization import new_generation as _ng
from oso_airfoils.optimization import batched_new_generation as _bng

# Swap the symbol the stock runner will import. batched_new_generation captured the true
# original (_ORIG) at its own import, so this patch cannot cause recursion.
_ng.newGeneration = _bng.newGeneration

# Run the stock runner as __main__ (argv passes through: the case file is sys.argv[1]).
runpy.run_module("oso_airfoils.optimization.runner", run_name="__main__")
