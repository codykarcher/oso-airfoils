"""
runner_batched.py  --  DEPRECATED launcher, kept so existing commands keep working.

Superseded by the unified entry point::

    python -m oso_airfoils.optimization --execution gpu-batched  case.yaml

or simply ``python -m oso_airfoils.optimization case.yaml``, which resolves to the
batched GPU path on its own when the tool is a surrogate and CUDA is available.

This shim pins ``--execution gpu-batched`` and forwards. Note the new entry point
REJECTS a non-surrogate tool in this mode instead of quietly running unbatched, which
is what this launcher used to do.
"""

import sys
import warnings

from oso_airfoils.optimization.__main__ import main


def _run(argv=None):
    warnings.warn(
        "oso_airfoils.optimization.runner_batched is deprecated; use "
        "'python -m oso_airfoils.optimization --execution gpu-batched <case files>'.",
        DeprecationWarning, stacklevel=2)
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--execution' not in argv and '-x' not in argv:
        argv = ['--execution', 'gpu-batched'] + argv
    return main(argv)


if __name__ == '__main__':
    sys.exit(_run())
