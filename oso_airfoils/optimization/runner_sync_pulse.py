"""
runner_sync_pulse.py  —  fleet runner: advance MANY oso-airfoils GA cases in LOCKSTEP with ONE
shared GPU surrogate pulse per generation.

NEW FILE — overwrites nothing. Takes a LIST of input yaml/json files (each a stock oso-airfoils
case, tool: neuralfoil) and runs them together: every generation it produces all cases' children
(Phase A, cheap CPU), fires ONE batched forward across the whole fleet (the "pulse",
multicase_surrogate), then evaluates + selects each case (Phase B/C) from its slice of that
pulse. This feeds a big idle GPU one saturating matmul instead of N small ones — the ~3x
utilization win from the sizing analysis, which is what lets a fat GPU (e.g. the 48 GB Ada) run
~30 concurrent 1000-airfoil cases instead of the ~8-16 an independent-launch fleet saturates at.

Usage:
    python -m oso_airfoils.optimization.runner_sync_pulse case1.yaml case2.yaml ... \
        [--model xxxlarge] [--backend nxfoil] [--device cuda] [--max-pulse 40] [--cuda-graph]

Each case writes its own population_*.json into its own output folder exactly as runner.py does.
Only tool: neuralfoil cases are supported (the surrogate is the whole point); a non-surrogate
case should use the stock mpirun runner.

--max-pulse chunks the fleet's forward if you launch more cases than fit in GPU memory at once
(cases are split into chunks of that size for the forward; they still advance in lockstep).
"""

import sys
import argparse
import contextlib

import numpy as np

from oso_airfoils.optimization import objective_function as _objf
from oso_airfoils.optimization import batch_geometry as _bg
from oso_airfoils.optimization.objective_function import airfoil_fitness
from oso_airfoils.optimization.multicase_surrogate import MultiCaseSurrogate
from oso_airfoils.optimization.sync_pulse_case import SyncCase
from oso_airfoils.optimization.new_generation_synced import (
    produce_children, finish_generation, initialize_sort,
)

# capture the true, unpatched wrapper entry points ONCE, before any monkeypatch
_ORIG_RUN = _objf.run_neuralfoil
_ORIG_KULFAN = _objf.Kulfan


def cprint(x):
    sys.stdout.flush()
    print(x, flush=True)


def _chunks(seq, n):
    if not n or n <= 0 or n >= len(seq):
        return [list(seq)]
    return [list(seq[i:i + n]) for i in range(0, len(seq), n)]


def _pulse(mcs, cases, rows_of, max_pulse):
    """rows_of(case) -> the design-vector rows to evaluate for this case (pop or children).
    Fires the aero forward (chunked by max_pulse) and the batched geometry, returns the
    geometry registry map {uid: (registry, psi, tooth)}."""
    for chunk in _chunks(cases, max_pulse):
        mcs.pulse_aero([c.aero_item(rows_of(c)) for c in chunk])
    return mcs.pulse_geometry([c.geo_item(rows_of(c)) for c in cases])


@contextlib.contextmanager
def _patched(case, geo):
    """Point objective_function at THIS case's aero cache slice + its geometry registry."""
    registry, psi, tooth = geo[case.uid]
    _bg.TorchKulfan.install_registry(registry, psi, tooth)
    _objf.run_neuralfoil = case.surr.make_cached_run(fallback=_ORIG_RUN)
    _objf.Kulfan = _bg.TorchKulfan
    try:
        yield
    finally:
        _objf.run_neuralfoil = _ORIG_RUN
        _objf.Kulfan = _ORIG_KULFAN


def _eval_rows(case, rows):
    """Phase B for one case: evaluate its rows via airfoil_fitness, served from its cache slice.
    Returns the per-row fitnessFunction outputs in row order."""
    return [airfoil_fitness({'pid': i, 'individual': r, 'params': case.params})
            for i, r in enumerate(rows)]


def _log_generation(gen, cases):
    for c in cases:
        try:
            pop = np.array(c.pop)
            K = pop[:, :c.N_k]
            gene_std = float(K.std(axis=0).mean())
            uniq = len({tuple(np.round(row, 6)) for row in K})
            # pareto_index is the last label column; feasibility tag is 'con_tag' (col N_k+2)
            front = pop[:, -1]
            f1 = int((front == 1).sum())
            cprint(f"  [{c.filecode}] gen {c.counter}/{c.N_generations} | "
                   f"uniq={uniq}/{len(pop)} gene_std={gene_std:.4f} front1={f1}"
                   + ("  DONE" if c.done else ""))
        except Exception as e:
            cprint(f"  [{c.filecode}] gen {c.counter} | (log error: {e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync-and-pulse fleet GA runner for oso-airfoils")
    ap.add_argument("input_files", nargs="+", help="one or more .yaml/.yml/.json case files")
    ap.add_argument("--model", default=None, help="surrogate model size override (e.g. xxxlarge, medium)")
    ap.add_argument("--backend", default=None, help="nxfoil (default) or nqfoil")
    ap.add_argument("--device", default=None, help="cuda (default) or cpu")
    ap.add_argument("--cuda-graph", action="store_true", help="enable CUDA-graph capture in the surrogate")
    ap.add_argument("--max-pulse", type=int, default=0,
                    help="max cases per GPU forward chunk (0 = all in one pulse)")
    args = ap.parse_args(argv)

    # surrogate config: CLI > first case's params > defaults
    from oso_airfoils.optimization.sync_pulse_case import read_input_file
    p0 = read_input_file(args.input_files[0])
    backend = args.backend or p0.get("surrogate_backend", "nxfoil")
    model = args.model or p0.get("neuralfoil_model", "xxxlarge")
    device = args.device or p0.get("surrogate_device", "cuda")

    cprint(f"[sync-pulse] fleet of {len(args.input_files)} cases | backend={backend} "
           f"model={model} device={device} max_pulse={args.max_pulse or 'all'}")
    mcs = MultiCaseSurrogate(backend=backend, model_size=model, device=device,
                             use_cuda_graph=args.cuda_graph)

    cases = [SyncCase(f, mcs, model_override=args.model) for f in args.input_files]
    for c in cases:
        c.init_population()

    # ---- generation 0: evaluate + sort each initial population ----
    cprint("[sync-pulse] generation 0 (initial populations)")
    geo = _pulse(mcs, cases, lambda c: c.pop, args.max_pulse)
    for c in cases:
        with _patched(c, geo):
            result = _eval_rows(c, c.pop)
        c.pop = initialize_sort(c.pop, result, c.N_k)
        c.save(0)
    _log_generation(0, cases)

    Nk_args = lambda c: ([1] * c.N_k, [float] * c.N_k, [-2.0] * c.N_k, [2.0] * c.N_k)

    # ---- lockstep generations ----
    while any(not c.done for c in cases):
        active = [c for c in cases if not c.done]

        # Phase A: produce every case's children (no surrogate needed)
        for c in active:
            nv, et, lb, ub = Nk_args(c)
            c.sortedData, c.children = produce_children(c.pop, nv, et, lb, ub, c.params)

        # THE PULSE: one batched forward across the whole active fleet, + batched geometry
        geo = _pulse(mcs, active, lambda c: c.children, args.max_pulse)

        # Phase B + C: evaluate children from the pulse, then select the next population
        for c in active:
            with _patched(c, geo):
                result = _eval_rows(c, c.children)
            nv, et, lb, ub = Nk_args(c)
            c.pop = finish_generation(c.pop, c.sortedData, c.children, result, nv, c.params)
            next_gen = c.counter + 1
            c.save(next_gen)
            c.counter = next_gen
            if c.counter >= c.N_generations:
                c.done = True

        done_n = sum(c.done for c in cases)
        cprint(f"[sync-pulse] pulse done | active={len(active)} finished={done_n}/{len(cases)} "
               f"| last forward rows={getattr(mcs, 'n_rows_last', '?')}")
        _log_generation(None, active)

    cprint(f"[sync-pulse] all {len(cases)} cases complete.")


if __name__ == "__main__":
    main()
