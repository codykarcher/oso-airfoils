"""
Constraint-domination variant of ``newGeneration``.

Behaviourally identical to :mod:`oso_airfoils.optimization.new_generation` except that
the non-dominated sort is the constraint-domination version (Deb 2002 feasibility
rules) from :mod:`oso_airfoils.optimization.ga_functions_constrained` instead of the
penalty-in-objective Pareto sort.

This module used to select that variant by rebinding ``NSGA_sort`` on the base module
at import time -- a process-wide side effect that could not be undone and could not be
varied between two cases running in the same process. The sort is now an ordinary case
parameter, so the preferred way to get this behaviour is to set it in the case file::

    nsga_sort : constrained

and run the normal driver. This wrapper remains for scripts that import
``newGeneration`` from here.
"""

from oso_airfoils.optimization import new_generation as _base


def newGeneration(*args, params=None, **kwargs):
    """``new_generation.newGeneration`` with the constrained non-dominated sort."""
    params = dict(params or {})
    params['nsga_sort'] = 'constrained'
    return _base.newGeneration(*args, params=params, **kwargs)
