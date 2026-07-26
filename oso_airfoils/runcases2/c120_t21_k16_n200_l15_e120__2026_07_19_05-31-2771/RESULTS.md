# runcases2 t21 NeuralFoil — results (crowding + constraint-domination)

**Case:** t21 (tau=0.21), NeuralFoil xxxlarge, N_pop=200, N_generations=200 (g0–g200),
`maximum_parent_fraction=0.7`. Run via `runner_constrained.py` (constraint-domination +
crowding). ~31 s/gen, ~1.7 h wall. NO production `.py` modified.

## What was validated (all optional changes, together)
- **Crowding-distance sort within fronts** — front-1 kept a broad clean-L/D span the whole run.
- **Constraint-domination (Deb 2002)** — front-1 was **100% feasible every single generation**
  (`front1 == feas` in every diversity line). By construction (runner imports
  `new_generation_constrained`) and by population check.
- **Diversity logging** — per-gen `[diversity]` line (see run.log).
- **maximum_parent_fraction=0.7**.

## Headline result: NO population collapse
| gen | mean gene std | front-1 (all feasible) | clean-L/D span |
|----:|----:|----:|----:|
| 25  | 0.054 | 8   | 78.7 |
| 50  | 0.029 | 35  | 89.9 |
| 100 | 0.025 | 99  | 89.7 |
| 150 | 0.023 | 100 | 87.7 |
| 200 | 0.0135| 100 | 73.9 |

The Pareto front grew to the 100-member cap (0.5·N_pop) and held a wide, fully-feasible
spread (clean L/D ~140–232 peak; 139–213 at g200) — the crowding sort preserving spread
rather than collapsing to a point.

## Final front (g200)
- 100 front-1 members, **all feasible**; total feasible 184/200.
- Front-1 clean L/D range: **139.3 – 213.2**.

## Note on constraint-domination vs plain sort
This case reaches ~95% feasibility by ~gen 15, so high-penalty infeasibles are easily
dominated and plain Pareto ≈ constraint-domination in OUTPUT here. The constraint-domination
*guarantee* (front-1 always feasible) held throughout; its advantage would show on
harder-constrained problems where near-feasible infeasibles compete for the front.

## Artifacts
- `diversity_trend.png` — front size/feasibility + gene std/spread vs generation
- `pareto_final.png` — final Pareto front (clean vs rough L/D)
- `population_..._g000.json` … `_g200.json` — per-generation snapshots
- `../_code_snapshot/` — exact source used (provenance)
