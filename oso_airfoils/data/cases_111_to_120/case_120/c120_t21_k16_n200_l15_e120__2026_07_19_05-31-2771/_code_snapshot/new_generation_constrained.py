"""
Constraint-domination variant of ``newGeneration``.

Behaviourally identical to ``oso_airfoils.optimization.new_generation`` EXCEPT
that the non-dominated sort is the constraint-domination version from
``oso_airfoils.optimization.ga_functions_constrained`` (Deb 2002 feasibility
rules) instead of the penalty-in-objective Pareto sort.

USAGE (move-in-later)
---------------------
Point the runner at this module::

    # in oso_airfoils/optimization/runner.py
    from oso_airfoils.optimization.new_generation_constrained import newGeneration

Everything else (breeding, mutation, front packing, the front-1 cap, dedup)
is inherited unchanged from ``new_generation``.

IMPLEMENTATION NOTE
-------------------
``new_generation.newGeneration`` resolves ``NSGA_sort`` from its own module
globals at call time, so we rebind that name on the base module to the
constrained implementation. This is a deliberate, process-wide rebind: import
THIS module (not the plain ``new_generation``) in the runner, and do not rely on
the un-patched ``new_generation.NSGA_sort`` elsewhere in the same process. If
you prefer full isolation, copy ``new_generation.py`` to this file verbatim and
change only its ``NSGA_sort`` import line instead of using the rebind below.
"""

from oso_airfoils.optimization import new_generation as _base
from oso_airfoils.optimization.ga_functions_constrained import (
    NSGA_sort as _NSGA_sort_constrained,
)

# Rebind the name that _base.newGeneration looks up at call time so the existing
# generation logic runs unchanged but with constrained non-domination.
_base.NSGA_sort = _NSGA_sort_constrained

newGeneration = _base.newGeneration
