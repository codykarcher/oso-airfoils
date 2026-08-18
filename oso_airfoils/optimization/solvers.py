"""
solvers.py  --  one uniform aerodynamic-solver interface for the objective function.

An **AeroSolver** is a callable::

    solver(K_upper, K_lower, [alpha_min, alpha_max, alpha_step],
           N_crit=..., xtp_u=..., xtp_l=...)  ->  {'cl', 'cd', 'cm', 'cpmin', 'alpha', ...}

:func:`make_solver` builds one from a case's ``params``, closing over everything that
is tool-specific (executable paths, time limits, temp-file prefixes, surrogate model
size) and everything that is constant for the case (Re, TE gap). The objective
function therefore never branches on ``tool`` and never reaches for a module-global
solver -- it just calls whatever solver it was handed.

That indirection is what lets the GPU-batched evaluator work without monkeypatching:
it passes ``neuralfoil_run=<cache-backed drop-in>`` and gets a solver that serves the
whole generation from a single batched forward, while the objective function is none
the wiser.
"""

from oso_airfoils.optimization.geometry_functions import TE_gap_function


def resolve_te_gap(params):
    """The trailing-edge gap for this case: explicit if set, else the tau-fit default.
    Matches what ``core_fitness_function`` computes, so the solver and the geometry
    path always agree."""
    if params.get('TE_gap') is None:
        return TE_gap_function(params['tau'])
    return params['TE_gap']


def make_solver(params, neuralfoil_run=None):
    """Build the AeroSolver for ``params['tool']``.

    ``neuralfoil_run`` optionally replaces the NeuralFoil wrapper with any callable
    of the same signature -- used by the GPU-batched evaluator to serve polars out of
    a batched cache. It is ignored for the external-solver tools.
    """
    tool = params.get('tool')
    Re = params.get('Re')
    te_gap = resolve_te_gap(params)

    if tool == 'xfoil':
        from oso_airfoils.core.xfoil_wrapper import run as run_xfoil
        path = params.get('xfoil_path', None)
        tfpre = params.get('xfoil_tempfile_path_leader', 't_')
        timelimit = params['xfoil_timelimit']

        def solver(K_upper, K_lower, alpha_range, N_crit, xtp_u, xtp_l):
            return run_xfoil('alfa', K_upper, K_lower, alpha_range, Re=Re,
                             N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l, TE_gap=te_gap,
                             timelimit=timelimit, path_to_XFOIL=path, tfpre=tfpre)

    elif tool == 'qfoil':
        from oso_airfoils.core.qfoil_wrapper import run as run_qfoil
        path = params.get('qfoil_path', None)
        tfpre = params.get('qfoil_tempfile_path_leader', 't_')
        timelimit = params.get('qfoil_timelimit', 10)

        def solver(K_upper, K_lower, alpha_range, N_crit, xtp_u, xtp_l):
            return run_qfoil('alfa', K_upper, K_lower, alpha_range, Re=Re,
                             N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l, TE_gap=te_gap,
                             timelimit=timelimit, path_to_QFOIL=path, tfpre=tfpre)

    elif tool == 'neuralfoil':
        run = neuralfoil_run
        if run is None:
            from oso_airfoils.core.neuralfoil_wrapper import run as run
        model = params['neuralfoil_model']

        def solver(K_upper, K_lower, alpha_range, N_crit, xtp_u, xtp_l):
            return run('alfa', K_upper, K_lower, alpha_range, Re=Re,
                       N_crit=N_crit, xtp_u=xtp_u, xtp_l=xtp_l, TE_gap=te_gap,
                       model=model)

    else:
        raise ValueError(
            f"Invalid tool selection {tool!r}; expected 'xfoil', 'qfoil' or 'neuralfoil'")

    solver.tool = tool
    solver.TE_gap = te_gap
    return solver


def build_sweeps(params):
    """The clean and rough alpha sweeps for a case, as batched-surrogate sweep specs.

    Each grid is a SUPERSET of what the objective function will ask for: it extends
    down to ``alpha_min_extend`` so that the on-demand downward polar extension in
    ``core_fitness_function`` is served from the same cached forward rather than
    falling through to a one-off call.
    """
    import numpy as np
    Re = float(params['Re'])
    a_ext = float(params.get('alpha_min_extend', -10.0))
    step_c = float(params['alpha_step_clean'])
    step_r = float(params['alpha_step_rough'])
    lo_c = min(float(params['alpha_min_clean']), a_ext)
    lo_r = min(float(params['alpha_min_rough']), a_ext)

    def grid(lo, hi, st):
        return np.arange(lo, hi + 0.5 * st, st)

    return [
        dict(name='clean', Re=Re, ncrit=float(params['N_crit_clean']),
             xtr_u=float(params['xtp_u_clean']), xtr_l=float(params['xtp_l_clean']),
             alphas=grid(lo_c, float(params['alpha_max_clean']), step_c)),
        dict(name='rough', Re=Re, ncrit=float(params['N_crit_rough']),
             xtr_u=float(params['xtp_u_rough']), xtr_l=float(params['xtp_l_rough']),
             alphas=grid(lo_r, float(params['alpha_max_rough']), step_r)),
    ]
