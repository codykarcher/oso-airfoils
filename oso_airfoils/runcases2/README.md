# runcases2 — isolated neuralfoil GA case using ALL the optional GA changes

Purpose: exercise every optional improvement developed in the 2026-07 session on a
fast NeuralFoil case, fully isolated from the production run path. **No production
`.py` file is modified** — this case runs a *copied* runner that imports the
constraint-domination module; it only reads/executes the installed `oso_airfoils`
package.

## What this case exercises
1. Crowding-distance sort **within each front** (extremes protected every generation)
2. **Constraint-domination** (Deb 2002: feasible dominates infeasible; infeasibles
   ranked by total violation) via `new_generation_constrained`
3. Per-generation `[diversity]` logging (mean gene std, unique genomes, feasible
   front-1 L/D span)
4. `maximum_parent_fraction = 0.7` (stronger, diversity-preserving elitism)

## Files
- `runner_constrained.py` — copy of `optimization/runner.py`, ONLY change: imports
  `newGeneration` from `new_generation_constrained` (activates constraint-domination).
- `t21_neuralfoil_crowded_constrained.yaml` — scaled config: t21, NeuralFoil xxxlarge,
  N_pop=200, N_generations=200, maximum_parent_fraction=0.7.
- `_code_snapshot/` — frozen copies of the exact source used (provenance).

## Run
    cd oso_airfoils/runcases2
    mpirun -n 96 python -u runner_constrained.py t21_neuralfoil_crowded_constrained.yaml

Output lands in `runcases2/c120_t21_k16_n200_l15_e120__<timestamp>/` (snapshots +
input copy). Verify constraint-domination is active: every snapshot's front-1
(pareto_index==1) is 100% feasible.
