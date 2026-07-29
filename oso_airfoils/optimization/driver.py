"""
driver.py  --  the one GA loop, for one case or a whole fleet.

The driver advances every case in LOCKSTEP: each generation it produces all cases'
children (Phase A), hands the whole fleet's children to the evaluator in ONE call
(Phase B), then selects each case's next population (Phase C) and writes its snapshot.

A single-case run is a fleet of one, so there is exactly one loop to maintain rather
than one per execution mode. What differs between modes is entirely inside the
evaluator: whether that one Phase-B call fans out over MPI ranks, runs serially, or
becomes a single batched GPU forward spanning every case at once.

Phase A and Phase C run only on the root process and their results are broadcast, so
the random draws that drive selection and breeding happen exactly once no matter how
many processes are running -- same as the original runner.
"""

import sys

import numpy as np

from oso_airfoils.optimization.generation import (
    finish_generation, initialize_sort, produce_children,
)


def cprint(x):
    sys.stdout.flush()
    print(x, flush=True)


def _ga_bounds(case):
    """The per-variable GA arguments: normalization, encoding, and box bounds."""
    n = case.N_k
    return [1] * n, [float] * n, [-2.0] * n, [2.0] * n


def _pareto_table(save_dict, max_rows=15):
    """The per-generation Pareto-front summary table, as printed by the stock runner."""
    pareto = [e for e in save_dict['population'] if e['pareto_index'] == 1]
    pareto.sort(key=lambda e: e['LoD_clean_at_design'], reverse=True)
    tbl = {'Index': [], 'Clean L/D': [], 'Rough L/D': [], 'Feasible': []}
    for i, e in enumerate(pareto):
        tbl['Index'].append(i)
        tbl['Clean L/D'].append(e['LoD_clean_at_design'])
        tbl['Rough L/D'].append(e['LoD_rough_at_design'])
        tbl['Feasible'].append(e['con_tag'])

    out = ''.join(k.ljust(15) for k in tbl) + '\n'
    n = len(pareto)
    if n <= max_rows:
        rows = list(range(n))
    else:
        rows = [int(round(j * (n - 1) / (max_rows - 1))) for j in range(max_rows)]
    for i in rows:
        for key in tbl:
            v = tbl[key][i]
            out += (f"{v:.2f}" if isinstance(v, float) else str(v)).ljust(15)
        out += '\n'
    return out


def _diversity_line(save_dict, gen, prefix=''):
    """Diversity / collapse monitor.

    Mean per-gene std over the Kulfan design vector, unique-genome count, and the
    feasible Pareto-front clean-L/D spread. A steadily shrinking gene std or a
    collapsing L/D span is the early signature of a population collapse; logging it
    every generation makes that visible in real time instead of only in hindsight.
    """
    pop = save_dict['population']
    K = np.array([list(p['K_upper']) + list(p['K_lower']) for p in pop])
    mean_gene_std = float(K.std(axis=0).mean())
    uniq = len({tuple(np.round(row, 6)) for row in K})
    f1 = [p for p in pop if p['pareto_index'] == 1]
    f1_feas = [p for p in f1 if p['con_tag'] >= 1.0]
    src = f1_feas if f1_feas else f1
    if src:
        cl = [p['LoD_clean_at_design'] for p in src]
        lo, hi = float(np.nanmin(cl)), float(np.nanmax(cl))
    else:
        lo = hi = float('nan')
    return ('[diversity] %sgen %d | mean_gene_std=%.4f | uniq_genomes=%d/%d | '
            'front1=%d (feas=%d) | feas_cleanLoD=[%.1f..%.1f] span=%.1f'
            % (prefix, gen, mean_gene_std, uniq, len(pop), len(f1), len(f1_feas),
               lo, hi, hi - lo))


def _report(case, save_dict, gen, verbose):
    if verbose:
        cprint('Generation %d' % gen)
        cprint(_pareto_table(save_dict))
        cprint(_diversity_line(save_dict, gen))
    else:
        cprint(_diversity_line(save_dict, gen, prefix=f"[{case.filecode}] "))


def run(cases, evaluator, verbose=None):
    """Advance every case to its ``N_generations``.

    ``verbose`` defaults to the full Pareto table for a single case and the compact
    one-line-per-case form for a fleet.
    """
    root = evaluator.is_root
    if verbose is None:
        verbose = len(cases) == 1

    # ---- generation 0: seed (or restore) and evaluate the initial populations ----
    if root:
        for c in cases:
            c.init_population()
    state = evaluator.broadcast([c.pop for c in cases] if root else None)
    for c, pop in zip(cases, state):
        c.pop = pop

    fresh = [c for c in cases if not c.is_continuation]
    if fresh:
        cprint(f"[oso] evaluating initial population for {len(fresh)} case(s)")
        result = evaluator.evaluate([(c, c.pop) for c in fresh])
        if root:
            for c in fresh:
                c.pop = initialize_sort(c.pop, result[c.uid], c.N_k, c.params)
                save_dict = c.save(0)
                _report(c, save_dict, 0, verbose)
        state = evaluator.broadcast([c.pop for c in fresh] if root else None)
        for c, pop in zip(fresh, state):
            c.pop = pop

    for c in cases:
        if c.is_continuation:
            cprint(f"[oso] {c.filecode}: continuing from generation "
                   f"{c.previous_generations} in {c.outdir}")

    # ---- lockstep generations ----
    # The iteration budget is 20% above N_generations, matching the stock runner: a
    # generation that raises is retried from the cached previous population without
    # consuming a generation number, and the surplus bounds how often that can happen
    # before the run gives up.
    max_iters = int(1.2 * max(c.N_generations for c in cases)) + 1
    for _ in range(max_iters):
        active = [c for c in cases if not c.done]
        if not active:
            break

        pop_cache = {c.uid: c.pop for c in active}
        try:
            # Phase A -- selection / crossover / mutation, root only.
            if root:
                for c in active:
                    nv, et, lb, ub = _ga_bounds(c)
                    c.sortedData, c.children = produce_children(c.pop, nv, et, lb, ub,
                                                               c.params)
            children = evaluator.broadcast([c.children for c in active] if root else None)
            for c, ch in zip(active, children):
                c.children = ch

            # Phase B -- the whole fleet's children, in one evaluator call.
            result = evaluator.evaluate([(c, c.children) for c in active])

            # Phase C -- selection into the next population, root only.
            if root:
                for c in active:
                    nv, _, _, _ = _ga_bounds(c)
                    c.pop = finish_generation(c.pop, c.sortedData, c.children,
                                              result[c.uid], nv, c.params)
                    gen = c.counter + 1
                    save_dict = c.save(gen)
                    c.counter = gen
                    c.done = c.counter >= c.N_generations
                    _report(c, save_dict, gen, verbose)
        except Exception as e:
            cprint(f"[oso] error during generation ({type(e).__name__}: {e}); "
                   "reverting to the previous generation and retrying")
            if root:
                for c in active:
                    c.pop = pop_cache[c.uid]

        state = evaluator.broadcast(
            [(c.pop, c.counter, c.done) for c in active] if root else None)
        for c, (pop, counter, done) in zip(active, state):
            c.pop, c.counter, c.done = pop, counter, done

        if len(cases) > 1:
            n_done = sum(c.done for c in cases)
            cprint(f"[oso] pulse complete | active={len(active)} "
                   f"finished={n_done}/{len(cases)} "
                   f"| last evaluation rows={evaluator.n_rows_last}")

    remaining = [c for c in cases if not c.done]
    if remaining:
        cprint(f"[oso] WARNING: iteration budget exhausted with {len(remaining)} "
               f"case(s) unfinished: "
               + ', '.join(f"{c.filecode} at {c.counter}/{c.N_generations}"
                           for c in remaining))
    else:
        cprint(f"[oso] all {len(cases)} case(s) complete.")
    return cases
