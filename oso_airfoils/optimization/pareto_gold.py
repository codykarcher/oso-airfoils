#!/usr/bin/env python3
"""
pareto_gold.py — clean/rough L/D Pareto front for the gradient-driven
oso-airfoils optimization (epsilon-constraint method, Ipopt + ma27).

PORTED from metafoil/examples/pareto_mpi.py, which is the process referred to as
"gold". The algorithm below is unchanged; what differs is packaging:

  * the objective is vendored into this package as `gradient_objective` (metafoil's
    examples/ is not an importable package path);
  * `nqfoil` is a first-class tool and the default, evaluated through
    `nqfoil_torch` (differentiable, batched, device-aware);
  * design configs are read from oso_airfoils/runfiles/ (the original pointed at a
    runfiles1/ that does not exist in this repo);
  * mpi4py is replaced by a serial-COMM_WORLD shim, so every `if size > 1` branch
    in main() takes its already-present serial path.

PERFORMANCE, measured 2026-07-26 (T21, nqfoil xxlarge, 4-point front, 486 s):
one evaluate() costs 146 ms, of which `geometry_duals` is 138 ms (94.6%) and BOTH
aero sweeps are 10.3 ms (7.1%). The cost is scalar bisection root-finding in
Kulfan (652 `_dzeta_dpsi` calls per airfoil), NOT the surrogate. Accelerating the
network alone therefore caps the speedup at 1.08x by Amdahl; the batched-torch
geometry is where real speed lives.

The front is built in two phases:

  Phase 1 (LOCATORS, multi-start): find the achievable rough range only. Both
    single-objective optima (max clean, max rough) are seed-sensitive on this
    non-convex problem -- e.g. max rough from the family seed sticks at a local
    optimum while seeding from the clean optimum finds a 4%-higher rough_max --
    so each is multi-started (family seed + perturbations + cross-seeding) and
    the best kept. The max-rough AIRFOIL is discarded (the rough optimum is a
    flat, near-degenerate ridge); only rough_max and the clean-optimal airfoil
    are carried forward. Tolerance is tightened for NeuralFoil truths only.

  Phase 2 (uniform sweep, per-point multi-start): the ENTIRE front -- both
    endpoints and all interiors -- as one uniform epsilon-constraint sweep,
    maximize clean L/D subject to rough L/D >= epsilon, epsilon uniform on
    [r_lo, rough_max]. epsilon=r_lo recovers the clean corner; epsilon=rough_max
    is the rough corner (exact). No special endpoint objectives, so the front is evenly
    spaced. The rough tail is ill-conditioned (different seeds reach different
    feasible clean at the same floor), so each point is solved from K seeds and
    the best FEASIBLE clean kept. All M*K sub-solves are independent and
    distribute round-robin across ranks.

Knobs: --locator-starts, --sweep-starts (drop to 1-2 for expensive qfoil/cxfoil
truths; tight NeuralFoil tolerance is auto-gated by --tool).

Run:
    python -m oso_airfoils.optimization.pareto_gold --thickness 21
    python -m oso_airfoils.optimization.pareto_gold --n 20 --thickness 21 \
           --tool nqfoil --out pareto.json

--model defaults per tool: xxlarge for nqfoil (its ladder stops there), xxxlarge
for neuralfoil. Passing an nqfoil-invalid size is rejected at parse time rather
than failing inside the solve.

Requires: cyipopt linked against an ma27-enabled Ipopt (verified working on this
machine; ma57 and mumps are NOT available here), the metafoil package, and torch.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import yaml
import multiprocessing as mp

# ----------------------------------------------------------------------
# MPI shim. The original ran under mpirun; on the Mac we drive parallelism with
# a multiprocessing Pool instead (see _parallel_solve). This object reproduces
# COMM_WORLD's interface at size==1, which makes every `if size > 1` branch in
# main() take its already-present serial path -- so the algorithm is unchanged
# and the collectives below are never actually exercised.
# ----------------------------------------------------------------------
class _SerialComm:
    def Get_rank(self):  return 0
    def Get_size(self):  return 1
    def allgather(self, x): return [x]
    def gather(self, x, root=0): return [x]
    def bcast(self, x, root=0): return x


class MPI:                       # namespace stand-in: MPI.COMM_WORLD
    COMM_WORLD = _SerialComm()


# ----------------------------------------------------------------------
# process-pool parallelism for the two independent sub-solve loops
#
# Every (point, seed) sub-solve is independent -- the original distributed them
# round-robin across MPI ranks. Measured cost is ~67 s per sub-solve, so serial
# a 20-point front at the default 4 seeds is 80 sub-solves ~= 98 minutes; across
# 8 cores it is ~12. The solver closure captures ctx/params and is not
# picklable, so each worker rebuilds it once in an initializer and thereafter
# receives only plain arrays.
# ----------------------------------------------------------------------
_W: dict = {}


def _init_worker(cfg):
    """Build this worker's solver once, and pin it to a single compute thread.

    Each sub-solve is a 17-variable NLP whose linear algebra is far too small to
    thread usefully, but importing torch and numpy spins up their default pools
    (~4 threads each here). With --workers 8 that is ~32 threads contending for 8
    cores, and it gets worse if a live GA is running alongside. live_ga's render
    worker caps the same variables for the same reason -- its docstring records
    renders coming out 9x slower when left uncapped.
    """
    import os as _os
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        _os.environ[_v] = '1'
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass
    import numpy as _np
    ctx = og.make_context(tool=cfg["tool"], model_size=cfg["model"],
                          params=cfg["params"])
    ctx["n_aux"] = 1
    bounds = list(zip(_np.full(16, -2.0), _np.full(16, 2.0))) + [(0.10, 0.98)]
    _W["solve"] = make_solver(ctx, cfg["params"], bounds, cfg["max_iter"],
                              proj_max_iter=cfg["proj_max_iter"])


def _run_task(task):
    """task = (tag, which, z0, rough_floor). Returns (tag, z, diag).

    A single degenerate probe (most often DesignPointError -- the airfoil's CL never
    reaches the design value in a usable range) must NOT crash the whole front: the
    exception would propagate out of pool.map and abort the entire sweep. Catch it and
    return a max-violation record; downstream keeps the best-FEASIBLE / least-violating
    per point, so a failed sub-solve simply loses to any real solve and the sweep finishes.
    """
    tag, which, z0, floor = task
    z0 = np.asarray(z0, float)
    try:
        z, d = _W["solve"](which, z0, rough_lod_min=floor)
        return tag, z, d
    except Exception as e:
        print(f"[sub-solve {tag} FAILED: {type(e).__name__}: {str(e)[:110]}]", flush=True)
        return tag, z0, {"lod_clean": 0.0, "lod_rough": 0.0, "violation": 1e9}


def _parallel_solve(tasks, cfg, workers=None, log=None):
    """Run sub-solves across a process pool; fall back to serial for 1 worker."""
    workers = workers or min(len(tasks), os.cpu_count() or 1)
    if workers <= 1 or len(tasks) == 1:
        _init_worker(cfg)
        return [_run_task(t) for t in tasks]
    if log:
        log(f"dispatching {len(tasks)} sub-solves across {workers} workers")
    ctxmp = mp.get_context("spawn")     # fork + torch/Ipopt in one process is unsafe
    with ctxmp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        return pool.map(_run_task, tasks, chunksize=1)


def _default_oso_root():
    _e = os.environ.get("OSO_ROOT")
    if _e:
        return _e
    # auto-detect: oso-airfoils is a sibling of the metafoil repo under .../research/
    _sib = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oso-airfoils"))
    if os.path.isdir(_sib):
        return _sib
    return "/Users/codykarcher/Dropbox/research/oso-airfoils"  # mac fallback
DEFAULT_OSO_ROOT = _default_oso_root()

warnings.filterwarnings("ignore")                       # benign 0*inf at the LE/TE grid ends
# objective is vendored into this package (metafoil/examples is not importable)

from oso_airfoils.optimization.airfoil_io import load_airfoil_dat        # noqa: E402
from metafoil.core.kulfan import Kulfan                                  # noqa: E402
from metafoil.core.kulfan_geometry import (fit_kulfan_to_coordinates,    # noqa: E402
                                           kulfan_to_coordinates)
from cyipopt import minimize_ipopt                                       # noqa: E402
from oso_airfoils.optimization import gradient_objective as og           # noqa: E402


# ----------------------------------------------------------------------
# Which named requirements are enforced as hard constraints (see
# og.CONSTRAINT_GROUPS). Same convention as the notebook's §2 block; edit here.
# ----------------------------------------------------------------------
CONSTRAINTS = {
    'non_intersection': True, 'reach_design_cl': True, 'thickness': True,
    'stall_margin': True, 'moments_of_inertia': True, 'area': True,
    'leading_edge_radius': True, 'radii_skew': False, 'max_thickness_location': True,
    'te_cone': True, 'curvature': True, 'lower_signflip': True,
    'min_radius_location': True, 'rough_xtr_cap': True, 'roughness_delta_cl': True, 'lod_falloff': True,
    'lod_falloff_2': True, 'transition_slope': True, 'cl_linearity': False, 'moment': True,
    'cpmin': False, 'cl_target': False, 'slope_ratio': True,
}


# ----------------------------------------------------------------------
# OSO-2026-WT2S 30%-thick design point (CL=1.2, Re=18e6 from the family's
# design matrix); structural targets are the t30 values.
# ----------------------------------------------------------------------
def default_params():
    return dict(
        CL=1.2, Re=18.0e6, tau=0.30, TE_gap=0.0328,
        Ixx_con=0.0007964, Iyy_con=0.0070638, Izz_con=0.00785849, A_con=0.16289864,
        ler_con_upper=0.03170504, ler_con_lower=0.04430395, cone_angle=0.0,
        target_cl=1.5, target_alpha=None, CMc_min=None, CMr_min=None,
        cp_min_design=None, cp_min_prestall=None, cp_min_at_alpha_offset=None,
        cp_min_alpha_offset=None,
        percent_delta_cl_from_roughness_threshold=0.1, percent_LoD_falloff_threshold=0.15,
        percent_LoD_falloff_threshold_down=0.15, percent_LoD_falloff_threshold_up=0.30,
        xtr_slope_threshold=0.05, xtr_slope_offset=1.0,
        cl_linearity_tol_clean=0.01, cl_linearity_tol_rough=0.01, cl_linearity_offset=2.0,
        alpha_falloff_offset_2=2.0, percent_LoD_falloff_threshold_2_down=0.15, percent_LoD_falloff_threshold_2_up=0.30,
        rough_xtr_max=0.05,
        rough_slope_ratio_min=0.70, slope_ratio_h=0.5,
        max_thickness_loc=0.275, max_thickness_loc_upper=0.275, max_thickness_loc_lower=0.275,
        ler_skew_factor=1.9, curvature_bound=-750, ec_cutoff=0.9, te_frac=0.95,
        min_radius_location_upper=None, min_radius_location_lower=None,
        min_radius_location_cutoff=0.08,
        target_stall_margin=4.0, alpha_falloff_offset=2.0, cm_alpha_band=5.0,
        cl_max_limit_clean=None, cl_max_limit_rough=None,
        toothpick_height=0.0, toothpick_location=0.85,
        stall_margin_clean_weighting=1e2, stall_margin_rough_weighting=1e2,
        lift_margin_clean_weighting=0.5, cl_max_limit_clean_weighting=1e4,
        cl_max_limit_rough_weighting=1e4, delta_cl_from_roughness_weighting=1e4,
        LoD_falloff_weighting=2000.0, ixx_weighting=1e6, iyy_weighting=1e4,
        izz_weighting=1e4, a_weighting=1e4, leading_edge_radius_upper_weighting=1e3,
        leading_edge_radius_lower_weighting=1e3, min_radius_location_upper_weighting=1e4,
        min_radius_location_lower_weighting=1e4, max_thickness_weighting=1e4,
        max_thickness_upper_weighting=1e4, max_thickness_lower_weighting=1e4,
        radii_skew_weighting=1e3, curvature_weighting=100,
        lower_surface_curvature_weighting=1e2, te_cone_violation_weighting=1e5,
        CL_target_weighting=1e4, clean_moment_weighting=1e4, rough_moment_weighting=1e4,
        cp_min_design_weighting=1e4, cp_min_at_alpha_offset_weighting=1e4,
        cp_min_prestall_weighting=1e4, toothpick_weighting=0.0, infeasibility_penalty=1e4,
        N_crit_clean=9.0, xtp_u_clean=1.0, xtp_l_clean=1.0,
        alpha_min_clean=-3, alpha_max_clean=22, alpha_step_clean=1.0,
        N_crit_rough=3.0, xtp_u_rough=0.05, xtp_l_rough=0.05,
        alpha_min_rough=-3, alpha_max_rough=18, alpha_step_rough=1.0,
        alpha_min_extend=-10.0,   # extend clean+rough sweeps down to here when design CL is below the sweep
    )


def fill_tau_defaults(params, oso_root=DEFAULT_OSO_ROOT):
    """Fill None geometry-constraint params with the OSO tau-based defaults, exactly
    as the original objective_function does. Without this a `null` in the yaml (e.g.
    min_radius_location_upper/lower in the WT2 configs) silently DISABLES the
    constraint in the gradient objective, whereas the GA fills it from tau -> the
    constraint stays active. Mutates and returns `params`."""
    if oso_root not in sys.path:
        sys.path.insert(0, oso_root)
    from oso_airfoils.optimization.geometry_functions import (
        TE_gap_function, cone_angle_function, Ixx_function, Iyy_function,
        Izz_function, area_function, ler_function,
        min_radius_location_upper_function, min_radius_location_lower_function)
    tau = params["tau"]
    tau_fns = {
        "TE_gap": TE_gap_function, "cone_angle": cone_angle_function,
        "Ixx_con": Ixx_function, "Iyy_con": Iyy_function, "Izz_con": Izz_function,
        "A_con": area_function, "ler_con_upper": ler_function,
        "ler_con_lower": ler_function,
        "min_radius_location_upper": min_radius_location_upper_function,
        "min_radius_location_lower": min_radius_location_lower_function,
    }
    for key, fn in tau_fns.items():
        if params.get(key) is None:
            params[key] = float(fn(tau))
    if params.get("min_radius_location_cutoff") is None:
        params["min_radius_location_cutoff"] = 0.08
    return params


def load_family_params(thickness, oso_root):
    """Overlay the OSO-2025-WT2 family design config for this thickness onto the
    template. Only template keys are taken (the yaml also carries GA/penalty-weight
    keys this hard-constraint objective doesn't use). None values are filled with the
    tau-based defaults (as the GA does), so e.g. min_radius_location stays active.
    Returns (params, yaml_path)."""
    yml = os.path.join(oso_root, "oso_airfoils", "runfiles",
                       f"t{thickness:02d}_neuralfoil.yaml")
    if not os.path.isfile(yml):
        raise FileNotFoundError(f"family config not found: {yml}")
    with open(yml) as f:
        y = yaml.safe_load(f)
    p = default_params()
    p.update({k: v for k, v in y.items() if k in p})
    fill_tau_defaults(p, oso_root)
    return p, yml


def family_dat(thickness, oso_root):
    return os.path.join(oso_root, "oso_airfoils", "airfoils", "oso_2025_wt2",
                        "datfiles", f"OSO-2025-WT2-T{thickness:02d}.dat")


def _afl_entry(label, z, te_gap, cl=None, rl=None, v=None, e=None):
    """One airfoil as a JSON-serializable dict: Kulfan coefficients + coordinates
    (+ aero/violation if given). z = [8 upper, 8 lower, psi*]."""
    xy = np.asarray(kulfan_to_coordinates(z[:8], z[8:16], n_pts=200, te_gap=te_gap))[:, :2]
    d = dict(label=label,
             upper_coefficients=[float(c) for c in z[:8]],
             lower_coefficients=[float(c) for c in z[8:16]],
             psi_star=float(z[16]),
             x=[float(a) for a in xy[:, 0]], y=[float(a) for a in xy[:, 1]])
    if cl is not None:
        d.update(clean_LD=float(cl), rough_LD=float(rl), violation=float(v),
                 rough_floor_eps=(None if e is None else float(e)))
    return d


def write_front_json(out_json, meta, labels, coeffs, clean_lod, rough_lod, viol, eps_list,
                     te_gap, nf_seed=None):
    """Write the final optimized front (coordinates + coefficients + aero) to JSON.
    Optionally include the NeuralFoil seed airfoils under 'nf_seed_airfoils'."""
    doc = dict(meta=meta,
               airfoils=[_afl_entry(l, z, te_gap, cl, rl, v, e)
                         for l, z, cl, rl, v, e in
                         zip(labels, coeffs, clean_lod, rough_lod, viol, eps_list)])
    if nf_seed is not None:
        doc["nf_seed_airfoils"] = [_afl_entry(f"nf_seed_{i:02d}", z, te_gap)
                                   for i, z in enumerate(nf_seed)]
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=2)
    return len(doc["airfoils"])


# ----------------------------------------------------------------------
# One Ipopt solve (mode 'clean'/'rough', optional rough-L/D floor)
# ----------------------------------------------------------------------
_OPTS = dict(linear_solver="ma27", hessian_approximation="limited-memory",
             limited_memory_max_history=25,   # >= n_vars => effectively FULL dense BFGS (converges cleanly)
             tol=1e-4, constr_viol_tol=1e-3, acceptable_tol=1e-3, acceptable_iter=10,
             mu_strategy="adaptive", print_level=0, sb="yes")


def make_solver(ctx, params, bounds, max_iter, feasibility_first=True,
                proj_max_iter=200):
    """proj_max_iter caps EACH of the two feasibility projections. Those two
    projections dominate the per-point budget (2*200 vs max_iter=100 for the
    optimize step), so a point that cannot converge burns ~500 Ipopt iterations
    before giving up. With a slow model (NF-xxxlarge is ~6.4 s per evaluate, 21x
    medium) and several function evaluations per line search, that is hours for a
    single hard point. Lower this to bound the worst case."""
    def solve(which, z_start, rough_lod_min=None, clean_lod_min=None):
        cache = {}
        last_good_grad = {}          # fallback direction when evaluate() fails

        nvar = len(np.asarray(z_start, float))

        def R(z):
            z = np.asarray(z, float); key = z.tobytes()
            if key not in cache:
                cache.clear()
                try:
                    cache[key] = og.evaluate(z, ctx)
                except og.DesignPointError as e:
                    cache[key] = e          # remember the failure; don't re-evaluate
            r = cache[key]
            if isinstance(r, og.DesignPointError):
                raise r
            return r

        def cons(z):
            return og.constraint_list(R(z), params, rough_lod_min=rough_lod_min,
                                      clean_lod_min=clean_lod_min, enabled=CONSTRAINTS)

        # constraint dimensions from the (valid) start point, for penalty sentinels
        _c0 = cons(np.asarray(z_start, float))
        n_ineq = sum(1 for n, k, d in _c0 if k == "ineq")
        n_eq = sum(1 for n, k, d in _c0 if k == "eq")
        BIG = 1e6   # penalty when a trial airfoil can't reach the design CL

        # Ipopt line-search steps can overshoot into geometries where the design
        # point is unreachable (evaluate raises DesignPointError). Return a heavy
        # penalty / deep infeasibility there so the step is rejected — never crash.
        def obj(z):
            try:
                return og.objective(R(z), which).v
            except og.DesignPointError:
                return BIG
        FEAS_TOL = 1e-3
        # Best feasible iterate seen anywhere during the solve. The optimiser
        # passes through good feasible points on its way; capturing them means a
        # phase-1 run that later wanders off cannot lose the progress it made.
        best_feasible = {"z": None, "obj": np.inf}

        def note_if_best(z):
            """Record z if it is feasible and the best objective seen so far.
            Cheap: R(z) is cached from the evaluation that just preceded this
            call, so only the (aero-free) constraint assembly is recomputed."""
            try:
                if violation(z) <= FEAS_TOL:
                    v = obj(z)
                    if v < best_feasible["obj"]:
                        best_feasible["obj"] = v
                        best_feasible["z"] = np.array(z, float)
            except Exception:
                pass

        def obj_grad(z):
            try:
                g = og.objective(R(z), which).g
                last_good_grad['obj'] = g
                note_if_best(z)          # capture feasible iterates as they pass
                return g
            except og.DesignPointError:
                # Returning zeros here tells Ipopt the objective is FLAT, which
                # leaves it with no descent direction: it cannot retreat from the
                # unreachable region and instead burns its whole iteration budget
                # thrashing the line search. Re-use the last valid gradient so
                # there is still a direction to back out along.
                return last_good_grad.get('obj', np.zeros(nvar))
        def c_ineq(z):
            try:
                return np.array([d.v for n, k, d in cons(z) if k == "ineq"])
            except og.DesignPointError:
                return np.full(n_ineq, -BIG)
        def j_ineq(z):
            try:
                return np.array([d.g for n, k, d in cons(z) if k == "ineq"])
            except og.DesignPointError:
                return np.zeros((n_ineq, nvar))
        def c_eq(z):
            try:
                return np.array([d.v for n, k, d in cons(z) if k == "eq"])
            except og.DesignPointError:
                return np.full(n_eq, BIG)
        def j_eq(z):
            try:
                return np.array([d.g for n, k, d in cons(z) if k == "eq"])
            except og.DesignPointError:
                return np.zeros((n_eq, nvar))

        def constraints_at(_z=None):
            cl = []
            if n_ineq: cl.append({"type": "ineq", "fun": c_ineq, "jac": j_ineq})
            if n_eq:   cl.append({"type": "eq", "fun": c_eq, "jac": j_eq})
            return cl

        def violation(z):
            ci = c_ineq(z); ce = c_eq(z)
            return max(0.0, float(-ci.min()) if ci.size else 0.0,
                       float(np.abs(ce).max()) if ce.size else 0.0)

        def project_feasible(anchor):
            """Nearest feasible point to anchor (min ||z-anchor||^2 s.t. all constraints)."""
            a = np.asarray(anchor, float)
            fr = minimize_ipopt(
                lambda z: 0.5 * float(np.sum((np.asarray(z, float) - a) ** 2)),
                a, jac=lambda z: np.asarray(z, float) - a,
                bounds=bounds, constraints=constraints_at(),
                options=dict(_OPTS, max_iter=proj_max_iter, tol=1e-6,
                             constr_viol_tol=1e-6, acceptable_iter=15))
            return np.asarray(fr.x, float)

        # phase 0: project the seed onto the feasible set so phase 1 starts feasible.
        z_feas = project_feasible(z_start) if feasibility_first else np.asarray(z_start, float)
        note_if_best(z_feas)
        # phase 1: optimize the real objective from the feasible point.
        # NOTE: cyipopt's minimize_ipopt ACCEPTS a `callback` kwarg but raises
        # NotImplementedError("`callback` is not yet supported by Ipopt.") the
        # moment one is passed -- the parameter exists only to reject it. The
        # signature is not evidence of support. cyipopt.Problem has no
        # `intermediate` attribute in this build either, so there is no live
        # per-iteration hook available at all -- do not re-attempt without first
        # running ONE solve to confirm the mechanism actually fires.
        z_opt = np.asarray(minimize_ipopt(
            obj, z_feas, jac=obj_grad,
            bounds=bounds, constraints=constraints_at(),
            options=dict(_OPTS, max_iter=max_iter)).x, float)
        # phase 2: L/D-maximization is unbounded & non-convex — Ipopt often stops
        # (max_iter) still infeasible. If so, project the result back onto the
        # feasible set so the returned airfoil is ALWAYS feasible.
        if violation(z_opt) > FEAS_TOL:
            z_opt = project_feasible(z_opt)
        note_if_best(z_opt)

        # Candidate final points. Previously only (z_opt, z_feas) were compared, so
        # when phase 1 wandered to an infeasible point and the re-projection could
        # not recover, the solver silently returned the projected seed z_feas --
        # reporting a perfect violation while having done no optimization at all
        # (observed on T21 NF-large max-clean: returned 228.5, the projected seed,
        # while feasible iterates of L/D ~315 had been passed and discarded). The
        # best feasible iterate seen anywhere is now a candidate, so real progress
        # can no longer be thrown away.
        cands = [z for z in (z_opt, z_feas) if violation(z) <= FEAS_TOL]
        if best_feasible["z"] is not None:
            cands.append(best_feasible["z"])
        z_final = min(cands, key=obj) if cands else z_feas

        d = og.evaluate(z_final, ctx).diag
        d["violation"] = violation(z_final)
        # Observability: whether the optimisation actually beat the projected seed,
        # so a silent fallback shows up in the log instead of looking converged.
        obj_seed, obj_final = obj(z_feas), obj(z_final)
        d["obj_seed"] = float(obj_seed)
        d["obj_final"] = float(obj_final)
        d["improved_on_seed"] = bool(obj_final < obj_seed - 1e-9)
        d["from_best_iterate"] = bool(
            best_feasible["z"] is not None
            and np.array_equal(z_final, best_feasible["z"]))
        return z_final, d
    return solve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16, help="number of intermediate Pareto points (default 16; +2 endpoints = 18)")
    ap.add_argument("--tool", default="nqfoil",
                    choices=["nqfoil", "neuralfoil", "cxfoil", "cqfoil"])
    ap.add_argument("--device", default="cpu",
                    help="torch device for nqfoil (cpu|mps). One Ipopt subproblem is only "
                         "~80 network rows, where MPS measured 0.76x CPU (i.e. SLOWER) -- "
                         "leave this on cpu and use the batched driver for GPU work.")
    ap.add_argument("--model", default=None,
                    help="surrogate model size. Default depends on --tool: 'xxxlarge' for "
                         "neuralfoil (matches the oso polar path's hardcoded model and the "
                         "family's design model, so constraints read consistently in the "
                         "rendered polar) and 'xxlarge' for nqfoil, whose ladder stops there "
                         "-- nqfoil has no xxxlarge, so the neuralfoil default would fail.")
    ap.add_argument("--thickness", type=int, default=None,
                    help="OSO-2025-WT2 thickness (21,24,...,36): load that family design config "
                         "+ its designed airfoil as the seed. If omitted, use the built-in t30 "
                         "params and --seed.")
    ap.add_argument("--oso-root", default=DEFAULT_OSO_ROOT, help="oso-airfoils repo root")
    ap.add_argument("--warm", default="interp", choices=["interp", "endpoint", "seed"],
                    help="starting guess for each interior point: 'interp' = linear blend of the "
                         "two endpoint designs at that point's position along the front (guesses "
                         "distributed across the front); 'endpoint' = the nearer endpoint; "
                         "'seed' = the cold seed. Default interp.")
    ap.add_argument("--seed", default="data/FFA-W3-301.dat", help="seed airfoil .dat (relative to this dir)")
    ap.add_argument("--plot-tool", default=None, choices=["neuralfoil", "xfoil", "qfoil"],
                    help="solver for the final rainbow polar (switchable hook). "
                         "Default: same as --tool when that is a real solver, else neuralfoil.")
    ap.add_argument("--plot-case", default="both", choices=["clean", "rough", "both"],
                    help="turbulence case(s) on the rainbow polar (default both)")
    ap.add_argument("--no-plot", action="store_true", help="skip the rainbow polar (still saves .npz + scatter)")
    ap.add_argument("--proj-max-iter", type=int, default=200,
                    help="cap on EACH feasibility projection (there are two per point). "
                         "These dominate the per-point budget; lower to bound hard points.")
    ap.add_argument("--max-iter", type=int, default=100,
                    help="Ipopt max iterations per solve (a modest cap avoids the "
                         "restoration phase wandering on this non-convex problem)")
    ap.add_argument("--sweep-starts", type=int, default=4,
                    help="per-point multi-start count in the front sweep. The rough "
                         "tail is multi-modal (multiple feasible local optima at one "
                         "rough floor), so each point is solved from this many seeds "
                         "(interp + perturbations) and the best FEASIBLE clean kept. "
                         "M*K sub-solves run fully in parallel.")
    ap.add_argument("--locator-starts", type=int, default=4,
                    help="multi-start count per endpoint locator (phase 1). The "
                         "endpoints are non-convex single-objective optima and are "
                         "seed-sensitive; this many starts (family seed + "
                         "perturbations, plus cross-seeding) are tried and the best "
                         "kept. Phase 1 has spare threads, so this is ~free.")
    ap.add_argument("--xtr-rough", type=float, default=None,
                    help="override the ROUGH forced-transition x/c for BOTH surfaces "
                         "(e.g. 0 = trip at the leading edge). Default: use the config's "
                         "xtp_u_rough / xtp_l_rough.")
    ap.add_argument("--xtr-clean", type=float, default=None,
                    help="override the CLEAN forced-transition x/c for both surfaces "
                         "(default: use the config, normally 1.0 = free transition)")
    ap.add_argument("--workers", type=int, default=None,
                    help="process-pool size for the independent sub-solves "
                         "(default: all cores). Measured ~67 s per sub-solve, so a "
                         "20-point front at 4 seeds is 80 sub-solves = ~98 min "
                         "serial, ~12 min on 8 cores. Use 1 to force serial.")
    ap.add_argument("--out", default="pareto_front.json",
                    help="output JSON basename (rank 0 writes it; the rainbow pdf/scatter png "
                         "are named from the same stem)")
    ap.add_argument("--le-cutoff", type=float, default=0.02,
                    help="lower_signflip_le_cutoff: x/c from which the lower-surface single-flip "
                         "curvature constraint is enforced. Was 0.10, which left the 0.05-0.10 "
                         "chord band (rough BL is turbulent from xtr=0.05) unconstrained for the "
                         "optimizer to park a curvature flip that games the rough polar. 0.02 "
                         "closes it (the LE curvature-singularity issue only bites below ~0.005).")
    ap.add_argument("--cross-seed", type=int, default=0,
                    help="after the initial solve, run this many rounds of NEIGHBOR PROPAGATION: "
                         "re-seed each eps from the best-found designs of itself and its two "
                         "neighbors, then re-solve, keeping the best. Walks a good basin along "
                         "the front, reaching moderate-laminar solutions the corner seeds can't.")
    ap.add_argument("--corner-seed", action="store_true",
                    help="seed EVERY sweep point from both front corners (z_clean = the "
                         "laminar-mode max-clean airfoil, z_rmax) in addition to the interp "
                         "blend. The blend alone rolls into the early-transition basin, so most "
                         "points miss the laminar-bucket optimum; corner-seeding gives each eps a "
                         "genuine laminar start and traces the true (non-stair-stepped) front.")
    ap.add_argument("--cm-min", type=float, default=None,
                    help="clean CM limit: clean CM must stay >= -|cm-min| across +-cm_alpha_band of "
                         "design (sets CMc_min). Default: off (no moment constraint).")
    ap.add_argument("--cl-lin-tol", type=float, default=None,
                    help="override cl_linearity tolerance (both clean and rough) -- the "
                         "max fractional lift-curve slope mismatch across a_des-2->a_des-1->a_des. "
                         "Default: the params value (0.01).")
    args = ap.parse_args()

    # Per-tool default model size. nqfoil's ladder stops at xxlarge; neuralfoil's
    # xxxlarge default would raise a bare "no weights" error under --tool nqfoil.
    if args.model is None:
        args.model = "xxlarge" if args.tool == "nqfoil" else "xxxlarge"
    if args.tool == "nqfoil":
        from metafoil.nqfoil import full_bl as _fb
        _sizes = _fb.available_sizes()
        if args.model not in _sizes:
            ap.error(f"--model {args.model!r} is not an nqfoil size; available: {_sizes}")

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()

    def log(msg, tag=None):
        pre = f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:6.0f}s]" + (f"[{tag}]" if tag else "")
        print(f"{pre} {msg}", flush=True)

    if args.thickness is not None:
        params, yml = load_family_params(args.thickness, args.oso_root)
        seed_path = family_dat(args.thickness, args.oso_root)
    else:
        params, yml = default_params(), None
        seed_path = os.path.join(here, args.seed)
    params['lower_signflip_le_cutoff'] = args.le_cutoff   # enforce lower-surface curvature inboard
    if args.cl_lin_tol is not None:
        params['cl_linearity_tol_clean'] = args.cl_lin_tol
        params['cl_linearity_tol_rough'] = args.cl_lin_tol
    if args.cm_min is not None:
        params['CMc_min'] = args.cm_min
    # The family yaml hard-sets alpha_min_clean/rough=0, which clamps the a_des-2deg
    # falloff/linearity samples for low-a_des airfoils (and lets the optimizer hide
    # falloff by pushing a_des below the grid). Force the grid below design-2deg.
    params['alpha_min_clean'] = min(params['alpha_min_clean'], -3.0)
    # rough grid extended down to -8 (was -3) so the sweep brackets the rough zero-lift
    # angle (~-4..-6 for these cambered CLd sections) that the slope_ratio constraint needs.
    params['alpha_min_rough'] = min(params['alpha_min_rough'], -8.0)
    # min-radius-of-curvature LOCATION: flatten the LOWER threshold to a constant.
    # The tau-based min_radius_location_lower_function over-inflates for thick sections
    # (0.05 @ t33, 0.085 @ t36), letting lower-nose weirdness (aft curvature spike at
    # ~3-4% chord) slip through. The signal is bimodal (~0 normal vs 0.02-0.05 weird),
    # so a flat 0.008 sits in the clean gap for every thickness (verified via grey-out:
    # catches the T27/T33/T36 max-rough noses, zero false positives). thrU kept tau-based.
    params['min_radius_location_lower'] = 0.008
    # rough lift-slope-ratio (surrogate-trust): FLAT threshold on the nqfoil rough curve.
    # On the real (nqfoil) model the clean-biased thin sections round hard (r~0.15-0.45) while
    # healthy thick sections stay high (>=~0.5), so a single flat bar isolates them cleanly --
    # no per-family taper needed (that was an artifact of validating against NeuralFoil).
    params['rough_slope_ratio_min'] = 0.50
    if args.xtr_rough is not None:
        params["xtp_u_rough"] = params["xtp_l_rough"] = args.xtr_rough
    if args.xtr_clean is not None:
        params["xtp_u_clean"] = params["xtp_l_clean"] = args.xtr_clean
    ctx = og.make_context(tool=args.tool, model_size=args.model, params=params)
    ctx['n_aux'] = 1                                          # psi*: free lower-curvature sign-flip split
    bounds = list(zip(np.full(16, -2.0), np.full(16, 2.0))) + [(0.10, 0.98)]  # 16 coeffs + psi*
    solve = make_solver(ctx, params, bounds, args.max_iter, proj_max_iter=args.proj_max_iter)
    # picklable recipe so pool workers can rebuild `solve` (closures cannot cross
    # a process boundary); the in-process `solve` above still serves phase 1's
    # cheap cross-seeding and any --workers 1 run.
    wcfg = dict(tool=args.tool, model=args.model, params=params,
                max_iter=args.max_iter, proj_max_iter=args.proj_max_iter)

    fit = fit_kulfan_to_coordinates(*load_airfoil_dat(seed_path).T[:2],
                                    fit_order=8, n_pts=160)
    seed = Kulfan(upper_coefficients=np.asarray(fit["upper_coefficients"], float),
                  lower_coefficients=np.asarray(fit["lower_coefficients"], float),
                  te_gap=params["TE_gap"])
    z0 = np.concatenate([seed.upper_coefficients, seed.lower_coefficients, [0.35]])  # +psi* (curvature-flip split)

    if rank == 0:
        cfg = os.path.basename(yml) if yml else f"built-in t30 + {os.path.basename(args.seed)}"
        log(f"START {time.strftime('%a %b %d %H:%M:%S %Y')}  tool={args.tool} model={args.model} "
            f"ranks={size} interior={args.n} warm={args.warm}")
        log(f"config={cfg}  CL={params['CL']} Re={params['Re']:.1e} "
            f"tau={params['tau']} TE_gap={params['TE_gap']}")
        log(f"clean: N={params['N_crit_clean']} xtr=({params['xtp_u_clean']},{params['xtp_l_clean']})  "
            f"rough: N={params['N_crit_rough']} xtr=({params['xtp_u_rough']},{params['xtp_l_rough']})")

    def _done(name, d):
        return (f"{name} done  L/D=({d['lod_clean']:.1f},{d['lod_rough']:.1f})  "
                f"viol={d['violation']:.1e}")

    # ---- Phase 1: LOCATORS (multi-start). Find the achievable rough range. ---
    # Both endpoints are single-objective optima of a NON-CONVEX problem, so they
    # are seed-sensitive: measured on T21 medium, max-rough from the family seed
    # sticks at a local optimum of 133 (unchanged even at tol 1e-7 / 600 iters),
    # while seeding from the optimized CLEAN airfoil finds 139 -- a 4% higher, and
    # correct, rough_max. So each locator is multi-started and we keep the best.
    # The single highest-value start is cross-seeding (rough from the clean
    # optimum, clean from the rough optimum); perturbed family seeds fill any
    # spare phase-1 threads. Tolerance is tightened only for NeuralFoil truths
    # (cheap, differentiable); qfoil/cxfoil keep the default (tight tol there is
    # neither affordable nor numerically reliable).
    n_start = max(2, getattr(args, "locator_starts", 4))
    rng = np.random.default_rng(0)
    lo_b = np.array([b[0] for b in bounds]); hi_b = np.array([b[1] for b in bounds])

    def perturb(z, scale=0.05):
        zc = np.array(z, float)
        zc[:16] += rng.normal(0.0, scale, 16)         # jitter coeffs, leave psi*
        return np.clip(zc, lo_b, hi_b)

    # tighten locator tolerance for NeuralFoil only
    tight = (args.tool == "neuralfoil")
    _saved = dict(_OPTS)
    if tight:
        _OPTS.update(dict(tol=1e-7, acceptable_iter=99999, constr_viol_tol=1e-6))

    def best_of(results, which):
        key = "lod_" + which
        cand = [r for r in results if r[0] == which]
        return max(cand, key=lambda r: r[3][key]) if cand else None

    # Round 1: multi-start both objectives from the family seed + perturbations.
    seeds = [z0] + [perturb(z0) for _ in range(n_start - 1)]
    tasks = [("clean", s) for s in seeds] + [("rough", s) for s in seeds]
    ptasks = [((i, which), which, np.asarray(zs, float), None)
              for i, (which, zs) in enumerate(tasks)]
    res = _parallel_solve(ptasks, wcfg, workers=args.workers, log=log)
    by_tag = {tag: (z, d) for tag, z, d in res}
    r1 = [(which, zs, *by_tag[(i, which)]) for i, (which, zs) in enumerate(tasks)]
    bc, br = best_of(r1, "clean"), best_of(r1, "rough")
    zc1, dc1 = bc[2], bc[3]
    zr1, dr1 = br[2], br[3]

    # Round 2: cross-seed (the big win) -- clean from the rough optimum, rough
    # from the clean optimum. Two independent tasks, on ranks 0/1 when parallel.
    cross = None
    # The two cross-seeds are independent, so they go out as one 2-task batch
    # rather than running back to back (the MPI original put them on ranks 0/1
    # for exactly this reason).
    xres = _parallel_solve([("x_clean", "clean", np.asarray(zr1, float), None),
                            ("x_rough", "rough", np.asarray(zc1, float), None)],
                           wcfg, workers=args.workers, log=log)
    xd = {tag: (z, d) for tag, z, d in xres}
    zc2, dc2 = xd["x_clean"]
    zr2, dr2 = xd["x_rough"]

    if tight:                                          # restore default tolerance
        _OPTS.clear(); _OPTS.update(_saved)

    # keep the best over both rounds
    z_clean, d_clean = (zc1, dc1) if dc1["lod_clean"] >= dc2["lod_clean"] else (zc2, dc2)
    z_rmax, d_rmax = (zr1, dr1) if dr1["lod_rough"] >= dr2["lod_rough"] else (zr2, dr2)
    rough_max = d_rmax["lod_rough"]
    r_lo = d_clean["lod_rough"]                        # rough at the clean-optimal airfoil
    r_hi = rough_max                                   # top of the epsilon range (exact rough_max)
    if rank == 0:
        log(f"phase 1: multi-start ({n_start}x) locators, tight={tight}")
        log(f"phase 1: clean_max={d_clean['lod_clean']:.1f} (rough {r_lo:.1f})  "
            f"rough_max={rough_max:.2f}  [round1 rough {dr1['lod_rough']:.1f}, "
            f"cross-seed rough {dr2['lod_rough']:.1f}]")
        log(f"phase 1 complete: rough range [{r_lo:.1f}, {r_hi:.1f}]")

    # ---- Phase 2: the ENTIRE front as one uniform epsilon-constraint sweep, --
    #      with PER-POINT MULTI-START. Every point maximises clean s.t. rough >=
    #      eps, eps uniform on [r_lo, r_hi]; eps=r_lo recovers the clean corner,
    #      eps=r_hi is the rough corner (exact rough_max). The rough tail is genuinely MULTI-MODAL:
    #      at a fixed rough floor the design space has two real basins of
    #      attraction (measured at rough 131: a sharp clean-226 basin and a flat
    #      clean-207 basin, both feasible). It is not under-convergence -- from the
    #      clean-optimal seed the solve reaches 207.5 and stays there across
    #      max_iter 100->2000, tol 1e-4->1e-8, and monotone mu, so a gradient
    #      method cannot cross the barrier. The only reliable remedy is to solve
    #      each point from K seeds (interp guess + perturbations) and keep the best
    #      FEASIBLE clean. All M*K solves are independent, so they distribute
    #      round-robin across ranks; most restarts hit the good basin.
    M = args.n + 2
    eps_levels = np.linspace(r_lo, r_hi, M)
    K = max(1, getattr(args, "sweep_starts", 4))

    def interp_seed(idx):
        t = idx / (M - 1)
        return (1.0 - t) * z_clean + t * z_rmax

    # build (idx, seed) tasks: seed 0 is the interp guess, 1..K-1 are perturbations
    tasks = []
    for idx in range(M):
        base = interp_seed(idx)
        if getattr(args, "corner_seed", False):
            # anchor every point at BOTH corners so it can reach either basin (z_clean is
            # laminar-mode; the interp blend alone lands in the early-transition basin), then
            # fill the rest of K with perturbations of the local blend.
            seedlist = [z_clean, z_rmax, base]
            while len(seedlist) < max(K, 3):
                seedlist.append(perturb(base, 0.04))
            for kk, s_ in enumerate(seedlist):
                tasks.append((idx, kk, s_))
        else:
            tasks.append((idx, 0, base))
            for kk in range(1, K):
                tasks.append((idx, kk, perturb(base, 0.04)))
    log(f"phase 2: {len(tasks)} (M={M} x K={K}) sub-solves")
    ptasks = [((idx, kk), "clean", np.asarray(zs, float), float(eps_levels[idx]))
              for (idx, kk, zs) in tasks]
    res = _parallel_solve(ptasks, wcfg, workers=args.workers, log=log)
    allsub = [(tag[0], float(eps_levels[tag[0]]), d["lod_clean"], d["lod_rough"],
               d["violation"], z) for tag, z, d in res]

    # rank 0: keep the best FEASIBLE clean per idx
    if rank == 0:
        FEAS = 1e-3
        by_idx = {}
        for idx, eps, cl, rg, v, z in allsub:
            feasible = (v <= FEAS) and (rg >= eps - 0.5)
            key = idx
            cur = by_idx.get(key)
            # prefer feasible; among feasible, highest clean; else least-violating
            cand = (feasible, cl if feasible else -v, (idx, eps, cl, rg, v, z))
            if cur is None or (cand[0], cand[1]) > (cur[0], cur[1]):
                by_idx[key] = cand

        # ---- cross-seed: walk each point's best design into its neighbors ----
        for rnd in range(getattr(args, "cross_seed", 0)):
            xtasks = []
            for idx in range(M):
                for j in (idx - 1, idx, idx + 1):
                    if 0 <= j < M:
                        zj = by_idx[j][2][5]
                        xtasks.append(((idx, "x%d_%d" % (rnd, j)), "clean",
                                       np.asarray(zj, float), float(eps_levels[idx])))
            xres = _parallel_solve(xtasks, wcfg, workers=args.workers, log=log)
            nimp = 0
            for tag, z, dd in xres:
                idx = tag[0]; eps = float(eps_levels[idx])
                feas = (dd["violation"] <= FEAS) and (dd["lod_rough"] >= eps - 0.5)
                cand = (feas, dd["lod_clean"] if feas else -dd["violation"],
                        (idx, eps, dd["lod_clean"], dd["lod_rough"], dd["violation"], z))
                if (cand[0], cand[1]) > (by_idx[idx][0], by_idx[idx][1]):
                    by_idx[idx] = cand; nimp += 1
            log("cross-seed round %d: %d points improved" % (rnd + 1, nimp))

        pts = [by_idx[i][2] for i in range(M)]        # best feasible per idx, clean->rough
        for p in pts:
            log(f"phase 2: point {p[0]:2d}  eps={p[1]:6.1f}  "
                f"L/D=({p[2]:.1f},{p[3]:.1f})  viol={p[4]:.1e}")

        # ---- assemble + save ------------------------------------------------
        clean_lod = [p[2] for p in pts]
        rough_lod = [p[3] for p in pts]
        coeffs = np.array([p[5] for p in pts])
        viol = [p[4] for p in pts]
        eps_list = [float(p[1]) for p in pts]
        log(f"phase 2 complete: {len(pts)}-point uniform front")
        print("\n  clean L/D   rough L/D   (rough floor)   max-viol")
        labels = (["max-clean"] + [f"eps={p[1]:.1f}" for p in pts[1:-1]] + ["max-rough"])
        for lab, cl, rl, v in zip(labels, clean_lod, rough_lod, viol):
            print(f"  {cl:8.1f}   {rl:8.1f}     {lab:15s}   {v:.1e}")
        out = os.path.join(here, args.out) if not os.path.isabs(args.out) else args.out
        out_json = os.path.splitext(out)[0] + ".json"
        meta = dict(tool=args.tool, model=args.model, warm=args.warm, thickness=args.thickness,
                    config=(os.path.basename(yml) if yml else None),
                    design_CL=float(params["CL"]), Re=float(params["Re"]),
                    tau=float(params["tau"]), TE_gap=float(params["TE_gap"]),
                    delta_cl_threshold=float(params["percent_delta_cl_from_roughness_threshold"]),
                    n_points=len(coeffs), n_interior=args.n)
        n = write_front_json(out_json, meta, labels, coeffs, clean_lod, rough_lod, viol,
                             eps_list, params["TE_gap"])
        log(f"wrote {out_json}  ({n} airfoils: coords+coeffs+aero)")
        try:
            from pareto_shape_plot import make_pareto_shape_plot
            spng = os.path.splitext(out)[0] + "_shapes.png"
            make_pareto_shape_plot(spng, coeffs, clean_lod, rough_lod,
                                   params["TE_gap"], args.thickness)
            log(f"wrote {spng}")
        except Exception as e:
            log(f"shape plot skipped ({type(e).__name__}: {e})")
        log(f"DONE {time.strftime('%a %b %d %H:%M:%S %Y')}  total {time.time()-t0:.0f}s")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # clean-vs-rough Pareto scatter (quick front-shape check) --------------
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(clean_lod, rough_lod, "o-", color="#333")
            ax.scatter([clean_lod[0]], [rough_lod[0]], c="C0", zorder=5, label="max clean")
            ax.scatter([clean_lod[-1]], [rough_lod[-1]], c="C1", zorder=5, label="max rough")
            ax.set_xlabel("clean  L/D"); ax.set_ylabel("rough  L/D"); ax.grid(alpha=.3); ax.legend()
            ax.set_title(f"Clean vs rough L/D Pareto front ({args.tool}, {args.n} intermediates)")
            png = os.path.splitext(out)[0] + "_front.png"
            fig.tight_layout(); fig.savefig(png, dpi=130)
            print(f"[pareto_mpi] wrote {png}", flush=True)
        except Exception as e:
            print(f"[pareto_mpi] scatter skipped: {e}", flush=True)

        # rainbow polar of the whole front (+ family reference airfoil) -------
        if not args.no_plot:
            plot_tool = args.plot_tool or (args.tool if args.tool in
                                           ("neuralfoil", "xfoil", "qfoil") else "neuralfoil")
            try:
                if args.oso_root not in sys.path:
                    sys.path.insert(0, args.oso_root)
                from oso_airfoils.postprocessing.runners import run_and_plot_polars_rainbow

                def coord_pair(z):
                    xy = np.asarray(kulfan_to_coordinates(z[:8], z[8:16], n_pts=200,
                                                          te_gap=params["TE_gap"]))[:, :2]
                    return (xy[:, 0], xy[:, 1])

                airfoils = [[lab, coord_pair(z)] for lab, z in zip(labels, coeffs)]
                refs = None
                if args.thickness is not None:
                    rxy = np.asarray(load_airfoil_dat(seed_path))[:, :2]
                    refs = [[f"OSO-2025-WT2-T{args.thickness:02d}",
                             (rxy[:, 0], rxy[:, 1]), "k"]]
                turb = {"clean": [[9.0, 1.0, 1.0]],
                        "rough": [[3.0, 0.05, 0.05]],
                        "both": [[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]]}[args.plot_case]
                rpng = os.path.splitext(out)[0] + f"_rainbow_{plot_tool}.pdf"
                print(f"[pareto_mpi] rendering rainbow polar with {plot_tool} "
                      f"({args.plot_case}) -> {rpng}", flush=True)
                run_and_plot_polars_rainbow(
                    airfoils,
                    reynolds_numbers=[params["Re"]],
                    turb_cases=turb,
                    tools=[plot_tool],
                    figure_path=rpng,
                    sweep_param="alpha",
                    sweep_range=(-6.0, 24.0, 0.5),
                    reference_airfoils=refs,
                    load_geometry=True,
                    show_cpmin=True,
                    cl_design=params["CL"],
                    neuralfoil_model=args.model,   # plot NF at the SAME model the optimizer used
                )
                print(f"[pareto_mpi] wrote {rpng}", flush=True)
            except Exception as e:
                print(f"[pareto_mpi] rainbow polar skipped ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
