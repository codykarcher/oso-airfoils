"""
Command-line entry point for the airfoil optimization.

    python -m oso_airfoils.optimization  case.yaml                 # one case
    python -m oso_airfoils.optimization  t21.yaml t24.yaml ...     # a fleet, in lockstep
    mpirun -n 8 python -m mpi4py -m oso_airfoils.optimization case.yaml

The execution backend is chosen automatically -- MPI when launched under an MPI
launcher, the batched GPU path when the tool is a surrogate and CUDA is available,
otherwise serial -- and can be pinned with ``--execution``. Illegal combinations
(batching a non-surrogate tool, running the single-process GPU path under mpirun)
are rejected at startup rather than silently degraded.
"""

import argparse
import sys

from oso_airfoils.optimization.case import Case
from oso_airfoils.optimization.config import (
    EXECUTION_MODES, detect_mpi_size, read_input_file, resolve_execution,
    surrogate_settings,
)
from oso_airfoils.optimization.driver import run
from oso_airfoils.optimization.evaluators import make_evaluator


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m oso_airfoils.optimization',
        description='Run one or more airfoil GA optimization cases.')
    p.add_argument('input_files', nargs='+',
                   help='one or more .yaml/.yml/.json case files')
    p.add_argument('-x', '--execution', default=None,
                   choices=list(EXECUTION_MODES) + ['auto'],
                   help='execution backend (default: auto, or the case file\'s '
                        '"execution" key)')
    p.add_argument('--model', default=None,
                   help='surrogate model size override (e.g. xxxlarge, medium)')
    p.add_argument('--backend', default=None,
                   help='surrogate backend: nxfoil (default) or nqfoil')
    p.add_argument('--device', default=None,
                   help='cuda (default), mps (Apple GPU), or cpu; falls back to cpu if unavailable')
    p.add_argument('--cuda-graph', action='store_true', default=None,
                   help='enable CUDA-graph capture in the surrogate')
    p.add_argument('--max-pulse', type=int, default=0,
                   help='max cases per GPU forward chunk (0 = whole fleet in one)')
    p.add_argument('--verbose', dest='verbose', action='store_true', default=None,
                   help='force the full per-generation Pareto table (default: on for '
                        'a single case, off for a fleet)')
    p.add_argument('--quiet', dest='verbose', action='store_false',
                   help='force the compact one-line-per-case output')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # The execution mode is a property of the run, not of an individual case, so it is
    # resolved once from the first case file and every case is checked against it.
    mpi_size = detect_mpi_size()
    first = read_input_file(args.input_files[0])
    mode = resolve_execution(first, requested=args.execution, mpi_size=mpi_size)

    for path in args.input_files[1:]:
        other = read_input_file(path)
        resolve_execution(other, requested=mode, mpi_size=mpi_size)
        if other.get('tool') != first.get('tool'):
            raise ValueError(
                f"all cases in one run must use the same tool; "
                f"{args.input_files[0]} uses {first.get('tool')!r} but {path} uses "
                f"{other.get('tool')!r}")

    settings = surrogate_settings(first, model=args.model, backend=args.backend,
                                  device=args.device, cuda_graph=args.cuda_graph)
    evaluator = make_evaluator(mode, settings=settings, max_pulse=args.max_pulse)

    if evaluator.is_root:
        print(f"[oso] {len(args.input_files)} case(s) | tool={first.get('tool')} "
              f"| execution={evaluator.describe()}", flush=True)

    # Only the root process writes; the others build the same Case purely for its
    # parameters (see Case.create_output).
    cases = [Case(f, model_override=args.model, create_output=evaluator.is_root)
             for f in args.input_files]
    try:
        run(cases, evaluator, verbose=args.verbose)
    finally:
        evaluator.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
