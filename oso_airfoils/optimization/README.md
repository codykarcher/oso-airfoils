Optimization Package
====================

This package runs the airfoil GA. There is **one entry point**:

```bash
python -m oso_airfoils.optimization case.yaml
```

Everything about a run is described by two independent axes in the case file:

| Axis | Key | Values | Meaning |
| :--- | :-- | :----- | :------ |
| Physics | `tool` | `xfoil`, `qfoil`, `neuralfoil` | which aerodynamic solver evaluates an airfoil |
| Execution | `execution` | `auto` (default), `serial`, `mpi`, `gpu-batched` | how those evaluations are distributed |

`tool` decides *what* is computed; `execution` decides *how fast and on what hardware*.
Changing one never requires changing the other.


Execution modes
---------------

**`serial`** — one process, one airfoil at a time. The debugging baseline, and the
right choice for a small neuralfoil case on a laptop.

```bash
python -m oso_airfoils.optimization -x serial case.yaml
```

**`mpi`** — the classic fan-out. Every rank evaluates its stride of the generation's
airfoils and results are gathered on rank 0. This is the only mode that works for
`xfoil` and `qfoil`, which shell out to an external solver per airfoil.

```bash
mpirun -n 188 python -m mpi4py -m oso_airfoils.optimization case.yaml
```

**`gpu-batched`** — the whole generation becomes a *single* batched forward through
the surrogate net, plus one batched geometry precompute. Requires a surrogate tool
(`neuralfoil`). Single-process by design; do not launch it under `mpirun`.

Any `N_k` works. The surrogate nets take a fixed 8 Kulfan coefficients per surface, so
a design vector of a different order is refit to order 8 on the way into the forward
(`batch_surrogate.to_net_order`, batched via `metafoil.core.kulfan_torch.batch_change_order`)
— the same `afl.changeOrder(8)` the per-airfoil NeuralFoil wrapper does, just for the
whole population in one solve. Raising the order is exact, since a lower-degree
Bernstein polynomial lies exactly in the higher-degree basis. The *constraint* geometry
still uses the original order, and the polar cache is still keyed on the original
coefficients, so the objective function's lookups hit unchanged.

```bash
python -m oso_airfoils.optimization -x gpu-batched case.yaml
```

Pass **several case files** and they advance in lockstep, sharing one batched forward
per generation across the whole fleet. This feeds a large GPU one saturating matmul
instead of N small ones, which is what lets a fat card run many concurrent cases
rather than a handful:

```bash
python -m oso_airfoils.optimization t21.yaml t24.yaml t27.yaml --max-pulse 40
```

A fleet and a single case run through the same driver — a single case is simply a
fleet of one.

### Automatic selection

With `execution: auto` (the default), the mode resolves as:

1. launched under an MPI launcher with more than one rank → `mpi`
2. else, surrogate tool + CUDA available → `gpu-batched`
3. else → `serial`

Illegal combinations are **rejected at startup**, not silently downgraded: batching a
non-surrogate tool, running `gpu-batched` under `mpirun`, or running `serial` under
`mpirun` (which would redundantly repeat the whole optimization on every rank).


Command-line options
--------------------

| Flag | Description |
| :--- | :---------- |
| `-x`, `--execution` | pin the execution backend (overrides the case file) |
| `--model` | surrogate model size override (e.g. `xxxlarge`, `medium`) |
| `--backend` | surrogate backend: `nxfoil` (default) or `nqfoil` |
| `--device` | `cuda` (default), `mps` (Apple GPU), or `cpu`; falls back to `cpu` when unavailable |
| `--cuda-graph` | enable CUDA-graph capture in the surrogate |
| `--max-pulse` | max cases per GPU forward chunk (0 = whole fleet in one) |
| `--verbose` / `--quiet` | force the full Pareto table or the compact one-line form |

Surrogate settings can also be set per case: `surrogate_backend`, `surrogate_device`,
`surrogate_cuda_graph`, `neuralfoil_model`.


### Devices

`--device` picks where the surrogate net runs. An unavailable device falls back to CPU
rather than failing, and the resolved choice is printed at startup.

Apple's **MPS** backend has no float64, and the batched geometry computes in float64 for
the exactness the constraint set needs (~1e-14 against the `Kulfan` class). On MPS the
aero net therefore runs on the GPU while the geometry stays on the CPU; `describe()`
reports `device=mps geometry=cpu` when that split is in effect.

Note that MPS is useful for exercising the batched *plumbing*, not for predicting CUDA
performance: CUDA-graph capture and TF32 are CUDA-only, so neither is active there.


Architecture
------------

```
                         ┌──────────────┐
   case.yaml  ──────────>│   Case       │  params, output folder, population state
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │   driver     │  one lockstep loop over N cases
                         └──────┬───────┘
                                │  Phase A: produce_children
                                │  Phase B: evaluator.evaluate(whole fleet)
                                │  Phase C: finish_generation
                  ┌─────────────┼─────────────┐
                  │             │             │
            ┌─────▼────┐  ┌─────▼────┐  ┌─────▼──────┐
            │ Serial   │  │  MPI     │  │ GPUBatch   │   evaluators
            └─────┬────┘  └─────┬────┘  └─────┬──────┘
                  └─────────────┼─────────────┘
                                │
                     ┌──────────▼──────────┐
                     │ objective_function  │  objectives + 31 constraints
                     └──────────┬──────────┘
                                │ solver=, kulfan=   (injected, never patched)
                     ┌──────────▼──────────┐
                     │ solvers.make_solver │  xfoil / qfoil / neuralfoil / cached batch
                     └─────────────────────┘
```

| Module | Responsibility |
| :----- | :------------- |
| `config.py` | input-file loading, output labels, execution-mode resolution and validation |
| `case.py` | one case: params, filecode, output folder, population state, continuation |
| `generation.py` | the NSGA-II step, split into `produce_children` / `finish_generation` |
| `solvers.py` | `make_solver(params)` → one uniform AeroSolver callable |
| `evaluators.py` | the three execution backends behind one protocol |
| `driver.py` | the lockstep generation loop, reporting, crash recovery |
| `__main__.py` | the CLI |
| `objective_function.py` | objectives and the 31 constraints for one design vector |
| `ga_functions.py` | encoding, crossover, mutation, non-dominated sort, crowding |
| `batch_surrogate.py`, `multicase_surrogate.py`, `batch_geometry.py` | the batched-GPU machinery used by `GPUBatchEvaluator` |

### Why evaluation is lifted out of the GA

`generation.py` exposes the NSGA-II step as three call-points with a hole in the
middle:

```python
sortedData, children = produce_children(pop, ...)   # Phase A: selection, crossover, mutation
result                = <<< evaluated by the caller >>>   # Phase B
pop                   = finish_generation(pop, sortedData, children, result, ...)  # Phase C
```

The algorithm therefore owns no MPI comm and no per-individual fitness callback — it
hands out children and takes back results. That is the entire reason the three
execution modes can share one implementation: Phase B is a fan-out, a serial loop, or
a single fleet-wide GPU forward, and Phase A and C do not know which.

Likewise, `core_fitness_function` takes its aerodynamic solver and geometry class as
**arguments**. Nothing monkeypatches module globals, so two cases can be in flight in
one process without stepping on each other, and there is no requirement to patch a
name before some other module imports it.

### GA variants

Both are ordinary case-file parameters, not separate modules or launchers:

| Key | Values | Meaning |
| :-- | :----- | :------ |
| `mutation_mode` | `legacy` (default), `corrected` | `legacy` reproduces every published OSO run, including the for/else defect in `mutateChromosome` that only ever sets bits. Use `corrected` for new work. |
| `nsga_sort` | `penalty` (default), `constrained` | `penalty` folds constraint violations into the objectives; `constrained` uses Deb (2002) constraint-domination. |


Batched geometry
----------------

Batching the aerodynamics exposed a bigger cost: the GA's own geometry bookkeeping.
Two per-airfoil loops dominated a generation, and both ran on a single rank in *every*
execution mode, so neither MPI nor a GPU helped them.

| Operation | Was | Now |
| :-------- | :-- | :-- |
| Thickness rescale of children (`produce_children`) | one `Kulfan` + scipy Newton per child | `kulfan_torch.batch_scale_to_thickness`, one vectorized secant for the population |
| Seeding a population (`newMember`) | one `Kulfan` + scale + order refit per sample | `batch_scale_to_thickness` + `batch_change_order` for the whole draw |

Measured on a 752-member population (N_k=16, `xxxlarge`):

| Phase | before | after |
| :---- | -----: | ----: |
| A — breeding + thickness rescale | 39.0 s | 1.0 s |
| B — batched evaluation | 2.1 s | 2.1 s |
| C — dedup / NSGA sort / packing | 0.8 s | 0.8 s |
| **per generation** | **41.8 s** | **3.9 s** |
| 1000 generations | 11.6 h | **1.1 h** |

Seeding the initial population dropped from ~40 s to 0.4 s on the same measurement.

These are numerically equivalent, not approximations: scaled coefficients agree with
the per-airfoil path to ~6e-16 and the achieved thickness hits its target to ~2e-16.
Both call sites fall back to the original per-airfoil loop if the batched path raises,
so a failure costs speed rather than a generation.

Supporting this required an exact batched max-thickness. `kulfan_geometry`'s default
`max_thickness` is a fine-grid argmax that sits ~1e-6 off `Kulfan.thickness_ratio` —
fine as a reported quantity, not fine as the residual of a solve targeting `tau`.
`kulfan_torch.batch_max_thickness` mirrors `Kulfan._locate_max_thickness` instead
(coarse scan to bracket, then bisection on the analytic derivative, both vectorized)
and agrees to ~4e-16.


Deprecated entry points
-----------------------

These still work and forward to the unified CLI, but emit a `DeprecationWarning`:

| Old | New |
| :-- | :-- |
| `python -m mpi4py runner.py case.yaml` | `python -m oso_airfoils.optimization case.yaml` |
| `python -m ...runner_batched case.yaml` | `python -m oso_airfoils.optimization -x gpu-batched case.yaml` |
| `python -m ...runner_sync_pulse a.yaml b.yaml` | `python -m oso_airfoils.optimization a.yaml b.yaml` |

`new_generation.newGeneration` also keeps its original signature and is now a thin
wrapper that wires the `generation.py` phases back together around an MPI comm.
