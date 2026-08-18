"""
Constraint-domination (Deb 2002) variant of ``NSGA_sort`` for the OSO airfoil GA.

This is a DROP-IN alternative to
``oso_airfoils.optimization.ga_functions.NSGA_sort``. Instead of folding the
constraint penalty into the objectives and doing pure Pareto domination, it
ranks the population by *constrained* non-domination:

  1. A feasible solution always dominates an infeasible one.
  2. Between two infeasible solutions, the one with the smaller total
     constraint violation dominates.
  3. Between two feasible solutions, standard Pareto domination on the
     objectives applies.

Within each front, members are ordered by crowding distance (descending) so the
objective-space extremes (infinite crowding) rank first -- the same
crowding-aware ordering used by the baseline ``NSGA_sort``.

WHY THIS HELPS
--------------
``objective_function.airfoil_fitness`` folds the weighted constraint penalty
``conpen`` into BOTH objectives::

    obj1 = -LoD_clean + conpen
    obj2 = -LoD_rough + conpen

Near the constraint boundary -- exactly where the best airfoils live -- that
shared penalty shifts points diagonally in objective space and distorts the
crowding metric, weakening diversity preservation. Constraint-domination keeps
the objective space clean (a raw L/D tradeoff among feasible members) and
handles feasibility separately, which tends to preserve the high-performance
front edge better.

INTEGRATION (do this later, when ready)
---------------------------------------
In ``oso_airfoils/optimization/runner.py`` change::

    from oso_airfoils.optimization.new_generation import newGeneration

to::

    from oso_airfoils.optimization.new_generation_constrained import newGeneration

``new_generation_constrained`` rebinds ``NSGA_sort`` to the version in THIS
module; nothing else changes.

COLUMN LAYOUT ASSUMPTIONS
-------------------------
Rows are ``[Nvars design vars] + r_list[1:]`` where ``r_list`` is the return of
``airfoil_fitness`` (pid dropped). Relative to ``Nvars``::

    obj1       -> col Nvars + 0
    obj2       -> col Nvars + 1
    con_tag    -> col Nvars + 2     (>= 1.0 / True  == feasible)
    LoD_clean  -> col Nvars + 4
    LoD_rough  -> col Nvars + 5

Total violation is recovered as
``conpen = (obj1 + obj2)/2 + (LoD_clean + LoD_rough)/2``, which is robust to how
many trailing constraint columns (or a ``pareto_index`` column) are present.
Override the offsets via the ``*_off`` kwargs if the objective_function layout
changes.
"""

import numpy as np

# Reuse the crowding utilities from the baseline module so there is a single
# source of truth for crowding distance.
from oso_airfoils.optimization.ga_functions import _crowding_distance, crowding_sort  # noqa: F401
from oso_airfoils.optimization import config as _cfg

# Column offset (from the first reported column, i.e. relative to Nvars) of the
# total-constraint-violation `viol` written by objective_function. Derived from the
# canonical label order so it stays correct if columns are reordered.
_VIOL_OFF = _cfg.REPORTED_LABELS.index('viol')
_CONTAG_OFF = _cfg.REPORTED_LABELS.index('con_tag')

# Threshold below which a member is treated as feasible (0 == no violation; the
# tolerance absorbs float round-off in the summed constraint magnitudes).
_FEAS_TOL = 1e-9


def _feasible_and_violation(dta, Nvars, eps=0.0, viol_off=_VIOL_OFF, **_ignored):
    """Return (feasible_mask, total_violation) arrays aligned to ``dta`` rows.

    Reads objective_function's dedicated ``viol`` column: the sum of the raw
    (non-negative) geometric constraint magnitudes plus any aero constraint
    promoted into the feasibility set. A member is feasible iff ``viol`` is ~0.
    This replaces the earlier reconstruction of the *weighted* penalty from the
    objective columns -- Deb's rules need the true unweighted violation.
    """
    arr = np.asarray(dta, dtype=float)
    viol = np.maximum(arr[:, Nvars + viol_off], 0.0)
    # Feasibility is keyed off the objective's OWN ``con_tag`` (col Nvars+2), so the
    # constrained sort's feasible set EXACTLY matches the con_tag front filter used at
    # collection -- including the per-constraint 1e-3 tolerance and the promoted aero/cap
    # groups, which the raw ``viol`` sum does not reproduce. ``viol`` is retained only to
    # RANK infeasible members (Deb: smaller total violation dominates).
    feas = arr[:, Nvars + _CONTAG_OFF] >= 0.5
    # eps-constrained relaxation (Takahama & Sakai): also admit near-feasible members whose
    # total violation is within the current tolerance eps. With eps shrinking to 0 the
    # population refines onto the boundary; eps=0 (default) is strict con_tag feasibility.
    if eps > 0.0:
        feas = feas | (viol <= eps)
    return feas, viol


def NSGA_sort(dta, Nvars, Nobj, eps=0.0, **col_kwargs):
    """Constrained non-dominated sort. Same I/O contract as the baseline:
    input rows ``[design vars, objectives, ...]``; output rows are the same with
    a ``front_number`` column appended (last), ordered by front then crowding.

    ``eps`` is the current feasibility tolerance for the eps-constrained relaxation
    (0 => strict Deb constraint-domination).
    """
    dta = np.array(dta, dtype=float)
    n = len(dta)
    if n == 0:
        return dta

    feas, viol = _feasible_and_violation(dta, Nvars, eps=eps, **col_kwargs)

    domination_count = np.zeros(n, dtype=int)
    dominated_solutions = [[] for _ in range(n)]

    def constrained_dominates(i, j):
        """True iff i constrained-dominates j (Deb 2002 rules)."""
        if feas[i] and not feas[j]:
            return True
        if feas[j] and not feas[i]:
            return False
        if not feas[i] and not feas[j]:
            if viol[i] < viol[j]:
                return True
            if viol[i] > viol[j]:
                return False
            # equal violation -> fall through to Pareto on objectives
        # both feasible (or equal violation): standard Pareto domination
        le_all = True
        lt_any = False
        for k in range(Nobj):
            c = Nvars + k
            if dta[i][c] > dta[j][c]:
                le_all = False
            elif dta[i][c] < dta[j][c]:
                lt_any = True
        return le_all and lt_any

    for i in range(n):
        for j in range(i + 1, n):
            if constrained_dominates(i, j):
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif constrained_dominates(j, i):
                dominated_solutions[j].append(i)
                domination_count[i] += 1

    front_number = np.ones(n, dtype=int)
    current_front = [i for i in range(n) if domination_count[i] == 0]
    front = 1
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    front_number[j] = front + 1
                    next_front.append(j)
        current_front = next_front
        front += 1

    # Order by front (ascending), and within each front by crowding distance
    # (descending) -- identical to the crowding-aware baseline NSGA_sort.
    result = np.hstack([dta, front_number.reshape(-1, 1)])
    max_front = int(front_number.max())
    ordered_blocks = []
    for f in range(1, max_front + 1):
        block = result[result[:, -1] == f]
        if len(block) == 0:
            continue
        if len(block) > 2:
            cd = _crowding_distance(block, Nvars, Nobj)
            block = block[np.argsort(cd, kind='stable')[::-1]]
        ordered_blocks.append(block)
    result = np.vstack(ordered_blocks) if ordered_blocks else result

    return result.tolist()
