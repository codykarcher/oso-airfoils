"""
generation.py  --  the NSGA-II generation step, split into phases around evaluation.

This is the ONE implementation of the GA step. It is expressed as three call-points
so that the fitness evaluation sits OUTSIDE the algorithm::

    initialize_sort(population, result, Nvars)                  -> sortedData   [gen 0]
    produce_children(population, ..., params)  -> (sortedData, children)        [Phase A]
        <<< the caller evaluates the children -- serially, over MPI, or in one
            batched GPU forward across a whole fleet of cases >>>               [Phase B]
    finish_generation(population, sortedData, children, result, ..., params)    [Phase C]

Lifting evaluation out is what makes the three execution modes possible without any
monkeypatching: the algorithm no longer owns an MPI comm or a per-individual fitness
callback, it just hands out children and takes back results. Run one case through
{produce_children -> evaluate -> finish_generation} and you reproduce the original
monolithic ``newGeneration`` exactly.

``result`` is the per-individual fitness output list in child order:
``result[i] = [pid, obj1, obj2, extra3, ...]``.

Two fixes versus textbook NSGA-II are applied here, both inherited from the original
implementation:

  Fix #1 -- Design-vector de-duplication of the combined (parent + child) pool BEFORE
            non-dominated sorting. Identical genomes, which the sort cannot tell apart
            because it only reads objective columns, otherwise pile up in front 1.

  Fix #2 -- A hard cap on front-1 occupancy in the next population. Once the front
            grows past ``front1_cap_fraction * N_pop`` the multi-front structure that
            drives selection collapses and the algorithm degenerates into crowding-
            distance selection on a single front.
"""

import copy
import math
import random
import sys

import numpy as np
from metafoil.core.kulfan import Kulfan

from oso_airfoils.optimization.geometry_functions import TE_gap_function
from oso_airfoils.geometry.newMember import newMember
from oso_airfoils.optimization.ga_functions import crowding_sort


def cprint(x):
    sys.stdout.flush()
    print(x)


#: Non-dominated sort variants, selected by ``params['nsga_sort']``.
#:   'penalty'     -- the standard sort; constraint violations are already folded into
#:                    the objectives as weighted penalties by the objective function.
#:   'constrained' -- Deb (2002) constraint-domination: feasible beats infeasible, and
#:                    infeasible solutions are ranked by total violation.
#: This replaces the old ``new_generation_constrained`` module, which selected the
#: variant by rebinding ``NSGA_sort`` on another module at import time -- a
#: process-wide side effect that made the choice invisible at the call site and
#: impossible to vary between two cases in one process.
SORT_MODES = ('penalty', 'constrained')


def get_nsga_sort(params):
    """Resolve the non-dominated sort for this case."""
    mode = (params or {}).get('nsga_sort', 'penalty') or 'penalty'
    if mode not in SORT_MODES:
        raise ValueError(f"nsga_sort must be one of {list(SORT_MODES)}, got {mode!r}")
    if mode == 'constrained':
        from oso_airfoils.optimization.ga_functions_constrained import NSGA_sort
    else:
        from oso_airfoils.optimization.ga_functions import NSGA_sort
    return NSGA_sort


def _dedup_by_design_vector(data, Nvars, decimals=12):
    """Keep only the FIRST occurrence of each unique design vector.

    Order is preserved, so if ``data`` was already pareto-sorted the representative
    kept for each duplicate group is the best-ranked one.
    """
    if len(data) == 0:
        return data
    arr = np.asarray(data, dtype=float)
    seen = set()
    keep = []
    for i in range(arr.shape[0]):
        key = tuple(np.round(arr[i, 0:Nvars], decimals).tolist())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return arr[keep]


def parse_params(params):
    """Pull the GA hyper-parameters out of ``params``, applying the documented defaults."""
    tau = params['tau']
    TE_gap = params['TE_gap'] if params.get('TE_gap') is not None else TE_gap_function(tau)

    N_crossovers = max([int(params['N_crossovers']), 1]) if 'N_crossovers' in params else 3
    if 'probability_of_mutation' in params:
        probability_of_mutation = max([min([1.0, params['probability_of_mutation']]), 0.0])
    else:
        probability_of_mutation = 0.3
    if 'maximum_parent_fraction' in params:
        maximum_parent_fraction = max([min([1.0, params['maximum_parent_fraction']]), 0.0])
    else:
        maximum_parent_fraction = 0.7
    if params.get('front1_cap_fraction') is not None:
        front1_cap_fraction = max([min([1.0, float(params['front1_cap_fraction'])]), 1e-6])
    else:
        front1_cap_fraction = 0.8
    if params.get('N_mutations') is not None:
        N_mutations = max([int(params['N_mutations']), 1])
    else:
        N_mutations = 1

    # Mutation operator selection.
    #   'corrected' -- a true bit flip (DEFAULT; the right thing for all work).
    #   'legacy'    -- the old operator used for previously published OSO runs. It
    #                  contains a for/else defect that makes it set bits rather than
    #                  flip them (see ga_functions.mutateChromosome). Kept ONLY so those
    #                  historical runs stay reproducible; select it explicitly to use it.
    # Defaults to 'corrected' -- new runs get true bit-flip mutation without opting in.
    mutation_mode = params.get('mutation_mode', 'corrected') or 'corrected'
    if mutation_mode not in ('legacy', 'corrected'):
        raise ValueError(f"mutation_mode must be 'legacy' or 'corrected', got {mutation_mode!r}")

    return dict(tau=tau, TE_gap=TE_gap, N_crossovers=N_crossovers,
                probability_of_mutation=probability_of_mutation,
                maximum_parent_fraction=maximum_parent_fraction,
                front1_cap_fraction=front1_cap_fraction,
                N_mutations=N_mutations, mutation_mode=mutation_mode)


#: Absolute tolerance on the achieved max thickness. The scalar path relied on scipy's
#: Newton raising to signal failure; a vectorized secant can't raise per row, so
#: convergence is checked on the result instead.
_TAU_TOL = 1e-9


def _rescale_to_thickness_scalar(children, tau, TE_gap):
    """Original per-airfoil rescale: one metafoil Kulfan and one scipy solve each.
    Kept as the fallback for :func:`rescale_to_thickness`."""
    out = list(children)
    for i in range(0, len(out)):
        afl = Kulfan(TE_gap=TE_gap)
        K = out[i]
        if any([math.isnan(kv) for kv in K]):
            N_k = len(K)
            K = newMember(int(N_k / 2), tau, 1)[0]
        Ku = K[0:int(len(K) / 2)]
        Kl = K[int(len(K) / 2):]
        afl.upperCoefficients = Ku
        afl.lowerCoefficients = Kl
        try:
            afl.scaleThickness(tau)
            out[i] = (afl.upperCoefficients.magnitude.tolist()
                      + afl.lowerCoefficients.magnitude.tolist())
            if any([math.isnan(rpv) for rpv in out[i]]):
                N_k = len(out[i])
                out[i] = newMember(int(N_k / 2), tau, 1)[0]
        except Exception:
            N_k = len(K)
            out[i] = newMember(int(N_k / 2), tau, 1)[0]
    return out


def rescale_to_thickness(children, tau, TE_gap):
    """Scale every child so its max thickness equals ``tau``, batched.

    This replaces a per-airfoil loop that constructed one metafoil ``Kulfan`` and ran
    a scipy Newton solve for each child. At production population size that loop cost
    ~40 s per generation -- an order of magnitude more than evaluating the entire
    population's aerodynamics -- and it ran on a single rank in every execution mode,
    so no amount of MPI or GPU helped it.

    The batched solve is numerically equivalent: coefficients agree with the scalar
    version to ~6e-16 and the achieved thickness hits the target to ~2e-16.

    Children arriving with NaN coefficients, and children the solve cannot bring to
    the target, are replaced by a fresh random member exactly as before. Any failure
    of the batched path falls back to the original loop rather than losing a
    generation.
    """
    try:
        from metafoil.core.kulfan_torch import (
            batch_max_thickness, batch_scale_to_thickness,
        )
    except Exception:
        return _rescale_to_thickness_scalar(children, tau, TE_gap)

    try:
        arr = np.asarray(children, dtype=float)
        if arr.ndim != 2:
            return _rescale_to_thickness_scalar(children, tau, TE_gap)
        half = arr.shape[1] // 2

        # NaN children are re-seeded BEFORE scaling, as in the scalar path.
        for i in np.flatnonzero(~np.isfinite(arr).all(axis=1)):
            arr[i] = newMember(half, tau, 1)[0]

        su, sl = batch_scale_to_thickness(arr[:, :half], arr[:, half:], tau,
                                          te_gap=TE_gap)
        out = np.concatenate([su, sl], axis=1)

        achieved = batch_max_thickness(su, sl, te_gap=TE_gap)[0]
        achieved = achieved.detach().cpu().numpy()
        failed = (~np.isfinite(out).all(axis=1)) | (np.abs(achieved - tau) > _TAU_TOL)
        n_failed = int(failed.sum())
        if n_failed:
            cprint(f"[GA] thickness rescale: re-seeding {n_failed} child(ren) that "
                   f"did not reach tau={tau}")
            for i in np.flatnonzero(failed):
                out[i] = newMember(half, tau, 1)[0]
        return [row.tolist() for row in out]
    except Exception as e:
        cprint(f"[GA] batched thickness rescale failed ({type(e).__name__}: {e}); "
               "falling back to the per-airfoil path")
        return _rescale_to_thickness_scalar(children, tau, TE_gap)


_mutation_mode_announced = False


def _announce_mutation_mode(mutation_mode):
    global _mutation_mode_announced
    if _mutation_mode_announced:
        return
    cprint(f"[GA] mutation operator: {mutation_mode}"
           + ("  (NOTE: legacy operator only sets bits, never clears them --"
              " see ga_functions.mutateChromosome)" if mutation_mode == 'legacy' else ""))
    _mutation_mode_announced = True


def _build_data(members, result, Nvars):
    """Stack design vectors + per-member fitness outputs into the ``data`` matrix:
    ``[design | obj1 | obj2 | extra3 | extra4 | ...]``."""
    members = np.asarray(members, float)
    fitness = np.array([result[i][1] for i in range(len(result))])
    data = np.append(members, np.array([fitness]).T, axis=1)
    for ii in range(2, len(result[0])):
        da = np.array([result[i][ii] for i in range(len(result))]).astype(float)
        data = np.append(data, np.array([da]).T, axis=1)
    return data


def initialize_sort(population, result, Nvars, params=None):
    """Generation-0 sort: build ``data`` from the evaluated initial population and sort it."""
    NSGA_sort = get_nsga_sort(params)
    data = _build_data(population, result, Nvars)
    return np.array(NSGA_sort(data, Nvars, 2))


def produce_children(population, normalizationVector, encodingTypes, lowerBounds,
                     upperBounds, params):
    """Phase A: tournament selection, crossover, mutation, and child thickness-rescale.

    Returns ``(sortedData, children)``. No fitness evaluation happens here -- that is
    the caller's job, and is where the three execution modes differ.
    """
    NSGA_sort = get_nsga_sort(params)
    cfg = parse_params(params)
    tau, TE_gap = cfg['tau'], cfg['TE_gap']
    N_crossovers = cfg['N_crossovers']
    probability_of_mutation = cfg['probability_of_mutation']
    N_mutations = cfg['N_mutations']
    mutation_mode = cfg['mutation_mode']
    _announce_mutation_mode(mutation_mode)

    population = np.array(population)
    if len(population) % 4 != 0:
        raise ValueError('Population length must be evenly divisible by 4')
    Nvars = len(normalizationVector)

    # -------------------------------
    # Tournament Selection, no elites
    # -------------------------------
    sortedData = np.array(NSGA_sort(population, Nvars, 2))
    remainingMembers = list(range(0, len(sortedData)))
    breedingPop = []
    for i in range(0, int(len(remainingMembers) / 2)):
        ix1 = random.randrange(0, len(remainingMembers))
        ix2 = ix1 - random.randrange(1, len(remainingMembers))
        if ix2 < 0:
            ix2 += len(remainingMembers)
        pi1 = remainingMembers[ix1]
        pi2 = remainingMembers[ix2]
        parent1 = sortedData[pi1]
        parent2 = sortedData[pi2]
        if ix1 <= ix2:
            breedingPop.append(parent1)
        else:
            breedingPop.append(parent2)
        remainingMembers.remove(pi1)
        remainingMembers.remove(pi2)

    ipList = []
    resultantPop = []
    for ii in [0, 1]:  # each parent must produce 4 offspring
        remainingMembers = list(range(0, len(breedingPop)))
        for i in range(0, int(len(remainingMembers) / 2)):
            ins = {}
            ins['pid'] = (ii + 1) * i
            ix1 = random.randrange(0, len(remainingMembers))
            ix2 = ix1 - random.randrange(1, len(remainingMembers))
            pi1 = remainingMembers[ix1]
            pi2 = remainingMembers[ix2]
            parent1 = breedingPop[pi1][0:Nvars]
            parent2 = breedingPop[pi2][0:Nvars]
            ins['parent1'] = copy.deepcopy(parent1)
            ins['parent2'] = copy.deepcopy(parent2)
            ins['normalizationVector'] = normalizationVector
            ins['encodingTypes'] = encodingTypes
            ins['upperBounds'] = upperBounds
            ins['lowerBounds'] = lowerBounds
            ins['Ncrossovers'] = N_crossovers
            ins['probabilityOfMutation'] = probability_of_mutation
            ins['N_mutations'] = N_mutations
            ins['mutation_mode'] = mutation_mode
            ipList.append(ins)
            remainingMembers.remove(pi1)
            remainingMembers.remove(pi2)

    from oso_airfoils.optimization.ga_functions import breedDesignVectorsParallel
    childrenList = []
    for i in range(0, len(ipList)):
        children = breedDesignVectorsParallel(ipList[i])
        childrenList.append(children)

    for i in range(0, len(childrenList)):
        resultantPop.append(childrenList[i][0])
        resultantPop.append(childrenList[i][1])

    ############
    # Thickness re-scaling of children
    ############
    resultantPop = rescale_to_thickness(resultantPop, tau, TE_gap)
    for row in resultantPop:
        assert (not any([math.isnan(v) for v in row]))

    return sortedData, resultantPop


def finish_generation(population, sortedData, resultantPop, result, normalizationVector,
                      params):
    """Phase C: combine top parents with the evaluated children, dedup, sort, apply the
    front-1 cap, and pack the next population. Returns the next population as a list."""
    NSGA_sort = get_nsga_sort(params)
    cfg = parse_params(params)
    maximum_parent_fraction = cfg['maximum_parent_fraction']
    front1_cap_fraction = cfg['front1_cap_fraction']
    Nvars = len(normalizationVector)
    population = np.array(population)

    data = _build_data(resultantPop, result, Nvars)

    # Combine top fraction of parents with all children.
    last_parent_index = int(maximum_parent_fraction * len(population))
    data_combined = np.append(sortedData[0:last_parent_index, 0:-2], data, axis=0)

    # -------------------------------------------------------------
    # Fix #1: dedup by design vector before non-dominated sorting.
    # sortedData (parents) is already pareto-ordered and children follow, so keeping
    # the first occurrence of each genome retains the best-ranked copy.
    # -------------------------------------------------------------
    n_before = len(data_combined)
    data_combined = _dedup_by_design_vector(data_combined, Nvars)
    n_after = len(data_combined)
    if n_after < n_before:
        cprint(f"[GA] dedup removed {n_before - n_after} duplicate genomes "
               f"({n_after} unique going into NSGA_sort)")

    resultantPop = np.array(NSGA_sort(data_combined, Nvars, 2))
    rpop_length = len(population)

    # Front-number bookkeeping
    front_numbers = resultantPop[:, -1]
    unique, counts = np.unique(front_numbers, return_counts=True)
    front_count_dict = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    max_front = int(front_numbers.max())

    # -------------------------------------------------------------
    # Fix #2: cap front-1 occupancy at front1_cap_fraction * N_pop. Excess front-1
    # members are displaced by subsequent fronts; if those don't supply enough rows,
    # the worst (by crowding) front-1 members refill the remainder so the population
    # size is preserved.
    # -------------------------------------------------------------
    front1_cap = max(1, int(math.ceil(front1_cap_fraction * rpop_length)))
    front1_mask = (front_numbers == 1)
    front1_inds = resultantPop[front1_mask]

    finalPop = []
    front1_excess_sorted = None  # crowding-sorted leftovers if the cap fires

    if len(front1_inds) > front1_cap:
        cprint(f"[GA] front-1 cap engaged: |F1|={len(front1_inds)} > cap={front1_cap}")
        f1_sorted = np.array(crowding_sort(front1_inds, Nvars, 2))
        # crowding_sort appends Nobj (=2) columns; drop them to restore the
        # [.., front_number] layout used by the rest of the loop.
        f1_kept = f1_sorted[:front1_cap, 0:-2]
        front1_excess_sorted = f1_sorted[front1_cap:, 0:-2]
        for row in f1_kept:
            finalPop.append(row.tolist())
        total_num = front1_cap
    else:
        for row in front1_inds:
            finalPop.append(row.tolist())
        total_num = len(front1_inds)
    next_front_start = 2

    # Pack subsequent fronts whole until adding the next one would overflow.
    last_front = None
    for i in range(next_front_start, max_front + 1):
        cnt = front_count_dict.get(i, 0)
        if cnt == 0:
            continue
        if total_num + cnt >= rpop_length:
            last_front = i
            break
        this_front = resultantPop[front_numbers == i]
        for tf in this_front:
            finalPop.append(tf.tolist())
        total_num += cnt

    remaining_slots = rpop_length - total_num
    if remaining_slots > 0:
        if last_front is not None:
            # Standard NSGA-II: crowding-sort the splitting front.
            last_front_inds = resultantPop[front_numbers == last_front]
            last_front_inds = np.array(crowding_sort(last_front_inds, Nvars, 2))
            for i in range(0, remaining_slots):
                finalPop.append(last_front_inds[i][0:-2].tolist())
        elif front1_excess_sorted is not None:
            # All other fronts were absorbed but the population is not full because
            # the front-1 cap fired. Refill from the displaced front-1 excess
            # (already crowding-sorted, best first).
            take = min(remaining_slots, len(front1_excess_sorted))
            for i in range(take):
                finalPop.append(front1_excess_sorted[i].tolist())
            if take < remaining_slots:
                # Defensive: should not occur, because the total excess is at least
                # |F1| - cap and we only reach here if the combined pool was entirely
                # front 1, meaning |F1| >= rpop_length.
                cprint(f"[GA] WARNING: short by {remaining_slots - take} rows after "
                       f"refill; padding with front-1 best.")
                pad_src = np.array(crowding_sort(front1_inds, Nvars, 2))[:, 0:-2]
                for i in range(remaining_slots - take):
                    finalPop.append(pad_src[i % len(pad_src)].tolist())
        else:
            # No splitting front and no excess: the combined pool was smaller than
            # the population, only possible if dedup removed enough rows.
            cprint(f"[GA] WARNING: combined pool smaller than N_pop after dedup; "
                   f"padding {remaining_slots} rows from best ranked.")
            for i in range(remaining_slots):
                finalPop.append(resultantPop[i % len(resultantPop)].tolist())

    assert len(finalPop) == rpop_length, (
        f"finalPop size {len(finalPop)} != N_pop {rpop_length}")

    return finalPop
