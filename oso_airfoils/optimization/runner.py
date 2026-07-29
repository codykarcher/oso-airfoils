"""
runner.py  --  DEPRECATED launcher, kept so existing commands keep working.

Superseded by the unified entry point, which picks the execution backend itself::

    python -m oso_airfoils.optimization  case.yaml
    mpirun -n 8 python -m mpi4py -m oso_airfoils.optimization  case.yaml

This shim forwards to it unchanged. Under ``mpirun`` the backend auto-resolves to
``mpi``, so::

    mpirun -n 8 python -m mpi4py -m oso_airfoils.optimization.runner  case.yaml

still behaves exactly as it always did.
"""

import sys
import warnings

from oso_airfoils.optimization.__main__ import main


def _run(argv=None):
    warnings.warn(
        "oso_airfoils.optimization.runner is deprecated; use "
        "'python -m oso_airfoils.optimization <case files>' instead.",
        DeprecationWarning, stacklevel=2)
    return main(argv)


if __name__ == '__main__':
    sys.exit(_run())
