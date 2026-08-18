"""
new_generation.py  --  the original monolithic ``newGeneration`` entry point.

The algorithm itself now lives in :mod:`oso_airfoils.optimization.generation`, split
into ``produce_children`` / evaluate / ``finish_generation`` so that the fitness
evaluation sits outside it. This module keeps the old call signature working by
wiring those phases back together around an MPI comm.

New code should use the driver instead::

    python -m oso_airfoils.optimization case.yaml

which selects a serial, MPI or batched-GPU evaluator from the case's ``execution``
key. This wrapper exists so that existing scripts holding a ``newGeneration``
reference -- and the constrained variant in ``new_generation_constrained`` -- keep
running unchanged.
"""

import numpy as np

from oso_airfoils.optimization.generation import (  # noqa: F401  (re-exported)
    _build_data, _dedup_by_design_vector, cprint, finish_generation, initialize_sort,
    produce_children,
)
from oso_airfoils.optimization.objective_function import airfoil_fitness  # noqa: F401


def newGeneration(fitnessFunction,
                  population,
                  normalizationVector,
                  encodingTypes,
                  lowerBounds,
                  upperBounds,
                  initalize=False,
                  comm=None,
                  params=None,
                  ):
    """One NSGA-II generation, evaluated across ``comm``.

    Returns the next population on rank 0 and ``None`` elsewhere, as before.
    """
    Nvars = len(normalizationVector)
    size = comm.Get_size()
    rank = comm.Get_rank()

    def evaluate(members):
        """Fan the members out over the comm and gather them back in member order."""
        local = []
        for i in range(0, len(members)):
            if i % size == rank:
                local.append(fitnessFunction({'pid': i, 'individual': members[i],
                                              'params': params}))
        gathered = comm.gather(local, root=0)
        if rank != 0:
            return None
        result = [None] * len(members)
        for chunk in gathered:
            for row in chunk:
                result[int(row[0])] = row
        return result

    if initalize:
        result = evaluate(population)
        if rank != 0:
            return None
        return initialize_sort(population, result, Nvars, params)

    if rank == 0:
        sortedData, children = produce_children(population, normalizationVector,
                                                encodingTypes, lowerBounds, upperBounds,
                                                params)
    else:
        sortedData, children = None, None

    children = comm.bcast(children, root=0)
    result = evaluate(children)

    if rank != 0:
        return None
    return finish_generation(population, sortedData, children, result,
                             normalizationVector, params)
