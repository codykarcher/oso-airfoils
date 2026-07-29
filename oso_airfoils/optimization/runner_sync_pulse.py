"""
runner_sync_pulse.py  --  DEPRECATED launcher, kept so existing commands keep working.

The "sync and pulse" behaviour -- advancing many cases in lockstep so one batched GPU
forward per generation spans the whole fleet -- is now simply what you get by passing
several case files to the unified entry point::

    python -m oso_airfoils.optimization case1.yaml case2.yaml ... --max-pulse 40

A fleet and a single case run through the same driver; the fleet just makes the batch
bigger. All of this launcher's flags (``--model``, ``--backend``, ``--device``,
``--cuda-graph``, ``--max-pulse``) are accepted there unchanged.
"""

import sys
import warnings

from oso_airfoils.optimization.__main__ import main


def _run(argv=None):
    warnings.warn(
        "oso_airfoils.optimization.runner_sync_pulse is deprecated; pass multiple case "
        "files to 'python -m oso_airfoils.optimization' instead.",
        DeprecationWarning, stacklevel=2)
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--execution' not in argv and '-x' not in argv:
        argv = ['--execution', 'gpu-batched'] + argv
    return main(argv)


if __name__ == '__main__':
    sys.exit(_run())
