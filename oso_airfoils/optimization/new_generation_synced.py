"""
new_generation_synced.py  —  the NSGA-II generation step of ``new_generation.py``, FACTORED
into three call-points so a fleet of GA cases can be advanced in LOCKSTEP with a single shared
GPU surrogate pulse per generation:

    initialize_sort(population, result, Nvars)               -> sortedData        (generation 0)
    produce_children(population, ..., params)                -> (sortedData, children)   [Phase A]
      <<< caller evaluates ALL cases' children in ONE batched forward here >>>   [Phase B, external]
    finish_generation(population, sortedData, children, result, ..., params) -> finalPop  [Phase C]

NEW FILE — does not modify new_generation.py. The Phase-A (selection / crossover / mutation /
thickness-rescale) and Phase-C (parent-combine / dedup / NSGA_sort / front-1 cap) blocks are
copied VERBATIM from new_generation.newGeneration so behaviour is identical; the only change is
that the fitness evaluation of the children (the single ``for i in range(len(resultantPop))``
loop that sat between them, plus its MPI gather) is lifted OUT, so the caller can batch that
evaluation across every case at once. Run one case through
{produce_children -> evaluate -> finish_generation} and you reproduce newGeneration exactly.

``result`` (both here and in newGeneration) is the per-individual fitnessFunction output list,
already re-ordered into population/children order: result[i] = [pid, obj1, obj2, extra3, ...].
"""

import copy
import math
import random
import sys

import numpy as np
from metafoil.core.kulfan import Kulfan

from oso_airfoils.optimization.geometry_functions import TE_gap_function
from oso_airfoils.geometry.newMember import newMember
from oso_airfoils.optimization.ga_functions import (
    breedDesignVectorsParallel,
    NSGA_sort,
    crowding_sort,
)


def cprint(x):
    sys.stdout.flush()
    print(x)


def _dedup_by_design_vector(data, Nvars, decimals=12):
    """Keep only the FIRST occurrence of each unique design vector (order preserved, so the
    best-ranked copy of a duplicate group survives). Identical to new_generation._dedup_*."""
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


def _parse_params(params):
    """Pull the GA hyper-parameters out of params exactly as newGeneration does (same defaults)."""
    tau = params['tau']
    TE_gap = params['TE_gap'] if 'TE_gap' in params else TE_gap_function(tau)

    N_crossovers = max([int(params['N_crossovers']), 1]) if 'N_crossovers' in params else 3
    if 'probability_of_mutation' in params:
        probability_of_mutation = max([min([1.0, params['probability_of_mutation']]), 0.0])
    else:
        probability_of_mutation = 0.3
    if 'maximum_parent_fraction' in params:
        maximum_parent_fraction = max([min([1.0, params['maximum_parent_fraction']]), 0.0])
    else:
        maximum_parent_fraction = 0.7
    if 'front1_cap_fraction' in params and params['front1_cap_fraction'] is not None:
        front1_cap_fraction = max([min([1.0, float(params['front1_cap_fraction'])]), 1e-6])
    else:
        front1_cap_fraction = 0.5
    if 'N_mutations' in params and params['N_mutations'] is not None:
        N_mutations = max([int(params['N_mutations']), 1])
    else:
        N_mutations = 1
    mutation_mode = params.get('mutation_mode', 'legacy') or 'legacy'
    if mutation_mode not in ('legacy', 'corrected'):
        raise ValueError(f"mutation_mode must be 'legacy' or 'corrected', got {mutation_mode!r}")
    return dict(tau=tau, TE_gap=TE_gap, N_crossovers=N_crossovers,
                probability_of_mutation=probability_of_mutation,
                maximum_parent_fraction=maximum_parent_fraction,
                front1_cap_fraction=front1_cap_fraction,
                N_mutations=N_mutations, mutation_mode=mutation_mode)


_mutation_mode_announced = False


def _build_data(members, result, Nvars):
    """Stack member design vectors + the per-member fitnessFunction outputs into the ``data``
    matrix newGeneration builds: [design | obj1 | obj2 | extra3 | extra4 | ...]."""
    members = np.asarray(members, float)
    fitness = np.array([result[i][1] for i in range(len(result))])
    data = np.append(members, np.array([fitness]).T, axis=1)
    for ii in range(2, len(result[0])):
        da = np.array([result[i][ii] for i in range(len(result))]).astype(float)
        data = np.append(data, np.array([da]).T, axis=1)
    return data


def initialize_sort(population, result, Nvars):
    """Generation-0 sort: given the evaluated initial population, build ``data`` and NSGA_sort it.
    Mirrors newGeneration's ``initalize=True`` rank-0 branch."""
    data = _build_data(population, result, Nvars)
    return np.array(NSGA_sort(data, Nvars, 2))


def produce_children(population, normalizationVector, encodingTypes, lowerBounds, upperBounds,
                     params):
    """Phase A: tournament selection + crossover + mutation + child thickness-rescale.
    Returns (sortedData, resultantPop) where resultantPop is the list of child design vectors
    to be evaluated. No surrogate/fitness call happens here."""
    global _mutation_mode_announced
    cfg = _parse_params(params)
    tau, TE_gap = cfg['tau'], cfg['TE_gap']
    N_crossovers = cfg['N_crossovers']
    probability_of_mutation = cfg['probability_of_mutation']
    N_mutations = cfg['N_mutations']
    mutation_mode = cfg['mutation_mode']

    if not _mutation_mode_announced:
        cprint(f"[GA-synced] mutation operator: {mutation_mode}"
               + ("  (NOTE: legacy operator only sets bits, never clears them)"
                  if mutation_mode == 'legacy' else ""))
        _mutation_mode_announced = True

    population = np.array(population)
    if len(population) % 4 != 0:
        raise ValueError('Population length must be evenly divisible by 4')
    Nvars = len(normalizationVector)

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

    childrenList = []
    for i in range(0, len(ipList)):
        children = breedDesignVectorsParallel(ipList[i])
        childrenList.append(children)

    for i in range(0, len(childrenList)):
        resultantPop.append(childrenList[i][0])
        resultantPop.append(childrenList[i][1])

    # Thickness re-scaling of children (identical to newGeneration).
    for i in range(0, len(resultantPop)):
        afl = Kulfan(TE_gap=TE_gap)
        K = resultantPop[i]
        if any([math.isnan(kv) for kv in K]):
            N_k = len(K)
            K = newMember(int(N_k / 2), tau, 1)[0]
        Ku = K[0:int(len(K) / 2)]
        Kl = K[int(len(K) / 2):]
        afl.upperCoefficients = Ku
        afl.lowerCoefficients = Kl
        try:
            afl.scaleThickness(tau)
            resultantPop[i] = afl.upperCoefficients.magnitude.tolist() + afl.lowerCoefficients.magnitude.tolist()
            if any([math.isnan(rpv) for rpv in resultantPop[i]]):
                N_k = len(resultantPop[i])
                resultantPop[i] = newMember(int(N_k / 2), tau, 1)[0]
        except Exception:
            N_k = len(K)
            resultantPop[i] = newMember(int(N_k / 2), tau, 1)[0]
        assert (not any([math.isnan(rpv) for rpv in resultantPop[i]]))

    return sortedData, resultantPop


def finish_generation(population, sortedData, resultantPop, result, normalizationVector, params):
    """Phase C: combine top parents with evaluated children, dedup, NSGA_sort, apply the front-1
    cap, and pack the next population. ``result`` is the children's fitnessFunction outputs in
    child order. Identical to newGeneration's rank-0 post-eval block. Returns finalPop (list)."""
    cfg = _parse_params(params)
    maximum_parent_fraction = cfg['maximum_parent_fraction']
    front1_cap_fraction = cfg['front1_cap_fraction']
    Nvars = len(normalizationVector)
    population = np.array(population)

    data = _build_data(resultantPop, result, Nvars)

    # Combine top fraction of parents with all children.
    last_parent_index = int(maximum_parent_fraction * len(population))
    data_combined = np.append(sortedData[0:last_parent_index, 0:-2], data, axis=0)

    # Fix #1: dedup by design vector before non-dominated sorting.
    n_before = len(data_combined)
    data_combined = _dedup_by_design_vector(data_combined, Nvars)
    n_after = len(data_combined)
    if n_after < n_before:
        cprint(f"[synced] dedup removed {n_before - n_after} duplicate genomes "
               f"({n_after} unique going into NSGA_sort)")

    resultantPop = np.array(NSGA_sort(data_combined, Nvars, 2))
    rpop_length = len(population)

    front_numbers = resultantPop[:, -1]
    unique, counts = np.unique(front_numbers, return_counts=True)
    front_count_dict = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))
    max_front = int(front_numbers.max())

    # Fix #2: cap front-1 occupancy.
    front1_cap = max(1, int(math.ceil(front1_cap_fraction * rpop_length)))
    front1_mask = (front_numbers == 1)
    front1_inds = resultantPop[front1_mask]

    finalPop = []
    front1_excess_sorted = None

    if len(front1_inds) > front1_cap:
        cprint(f"[synced] front-1 cap engaged: |F1|={len(front1_inds)} > cap={front1_cap}")
        f1_sorted = np.array(crowding_sort(front1_inds, Nvars, 2))
        f1_kept = f1_sorted[:front1_cap, 0:-2]
        front1_excess_sorted = f1_sorted[front1_cap:, 0:-2]
        for row in f1_kept:
            finalPop.append(row.tolist())
        total_num = front1_cap
        next_front_start = 2
    else:
        for row in front1_inds:
            finalPop.append(row.tolist())
        total_num = len(front1_inds)
        next_front_start = 2

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
            last_front_inds = resultantPop[front_numbers == last_front]
            last_front_inds = np.array(crowding_sort(last_front_inds, Nvars, 2))
            for i in range(0, remaining_slots):
                finalPop.append(last_front_inds[i][0:-2].tolist())
        elif front1_excess_sorted is not None:
            take = min(remaining_slots, len(front1_excess_sorted))
            for i in range(take):
                finalPop.append(front1_excess_sorted[i].tolist())
            if take < remaining_slots:
                cprint(f"[synced] WARNING: short by {remaining_slots - take} rows after refill; "
                       f"padding with front-1 best.")
                pad_src = np.array(crowding_sort(front1_inds, Nvars, 2))[:, 0:-2]
                for i in range(remaining_slots - take):
                    finalPop.append(pad_src[i % len(pad_src)].tolist())
        else:
            cprint(f"[synced] WARNING: combined pool smaller than N_pop after dedup; "
                   f"padding {remaining_slots} rows from best ranked.")
            for i in range(remaining_slots):
                finalPop.append(resultantPop[i % len(resultantPop)].tolist())

    assert len(finalPop) == rpop_length, f"finalPop size {len(finalPop)} != N_pop {rpop_length}"
    return finalPop
