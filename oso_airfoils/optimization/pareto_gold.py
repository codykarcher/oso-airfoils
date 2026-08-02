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
    # propagate the opt-in curvature-accel flag to this (spawned) worker's module
    # global so its make_solver -> constraint_list(enabled=CONSTRAINTS) enforces it.
    if "curvature_accel" in cfg:
        CONSTRAINTS["curvature_accel"] = bool(cfg["curvature_accel"])
    if "bulge" in cfg:
        CONSTRAINTS["bulge"] = bool(cfg["bulge"])
    ctx = og.make_context(tool=cfg["tool"], model_size=cfg["model"],
                          params=cfg["params"])
    ctx["n_aux"] = 1
    ctx["curvature_accel"] = bool(CONSTRAINTS.get("curvature_accel", False))
    ctx["bulge"] = bool(CONSTRAINTS.get("bulge", False))
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
    tag, which, z0, floor = task[:4]
    explore = task[4] if len(task) > 4 else False   # loose-explore flag (5-tuple robust tasks)
    z0 = np.asarray(z0, float)
    # A 'clean'-objective task floors ROUGH L/D; a 'rough'-objective task floors CLEAN L/D
    # (the two-sided sweep). floor=None (phase-1 locators) -> no floor either way.
    kw = {"clean_lod_min": floor} if which == "rough" else {"rough_lod_min": floor}
    try:
        z, d = _W["solve"](which, z0, explore=explore, **kw)
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
from numpy.polynomial import polynomial as _npoly                        # noqa: E402
from metafoil.core.kulfan import _binom as _kbinom                       # noqa: E402


# ----------------------------------------------------------------------
# FIX 2(a): exact lower-surface inflection count (analytic polynomial roots).
# zeta_lower = psi^0.5 (1-psi) S(psi) + linear; the linear term drops out of
# zeta'', and sign(zeta'') = sign(P) with P = -0.25 g + psi g' + psi^2 g'',
# g = (1-psi) S. Counting sign-flip roots of P in (le_cutoff, 1) is the exact
# inflection count -- the same method verified to match constraint_list's
# signflip terms to 0.0. Used as a HARD reject: any candidate with count > 1 is
# treated as infeasible so it can never become a reported front point (the smooth
# psi*-gated surrogate leaks and certifies 2-3 inflection airfoils as feasible).
# ----------------------------------------------------------------------
def _exact_lower_inflections(lower, le_cutoff=0.02):
    lo = np.asarray(lower, float); M = len(lo); n = M - 1
    one_minus = np.array([1.0, -1.0])
    S = np.array([0.0])
    for kk in range(M):
        psi_k = np.zeros(kk + 1); psi_k[kk] = 1.0
        opp = _npoly.polypow(one_minus, n - kk)
        S = _npoly.polyadd(S, _kbinom(n, kk) * lo[kk] * _npoly.polymul(psi_k, opp))
    g = _npoly.polymul(one_minus, S)
    gp = _npoly.polyder(g, 1); gpp = _npoly.polyder(g, 2)
    P = _npoly.polyadd(-0.25 * g, _npoly.polymul(np.array([0.0, 1.0]), gp))
    P = _npoly.polyadd(P, _npoly.polymul(np.array([0.0, 0.0, 1.0]), gpp))
    cnt = 0
    for r in _npoly.polyroots(P):
        if abs(r.imag) < 1e-8 and le_cutoff < r.real < 1.0:
            e = 1e-6
            if _npoly.polyval(r.real - e, P) * _npoly.polyval(r.real + e, P) < 0:
                cnt += 1
    return cnt


# ----------------------------------------------------------------------
# INFLECTION REPAIR: project a lower surface onto <=1 curvature sign-change with the
# SMALLEST coefficient change, instead of rejecting it (which drains the guarantee-pass
# pool and collapses thin fronts). d2zeta_lower is LINEAR in the CST coeffs -- the constant
# Jacobian J2 (d2 = J2 @ lower) is probed once by unit vectors -- so for a fixed split psi*
# the projection is a convex QP:  min ||delta||^2  s.t.  tgt_j*(d2_j + J2_j@delta) >= margin,
# tgt_j = +/-s0 (match the near-LE sign before psi*, flip after). We try each existing
# sign-change of d2 as psi* and keep the smallest-norm feasible delta, then VERIFY with the
# exact polynomial count (grid-sign enforcement can miss a sub-grid wiggle; margin catches it).
# ----------------------------------------------------------------------
_J2_LOWER_CACHE = {}

def _lower_curvature_jacobian(psi_int, te_gap, n_coeff):
    key = (n_coeff, round(float(te_gap), 12), len(psi_int), round(float(psi_int[-1]), 10))
    J = _J2_LOWER_CACHE.get(key)
    if J is None:
        z = np.zeros(n_coeff)
        J = np.empty((len(psi_int), n_coeff))
        for k in range(n_coeff):
            e = z.copy(); e[k] = 1.0
            kf = Kulfan(upper_coefficients=z, lower_coefficients=e, te_gap=te_gap)
            J[:, k] = np.asarray(kf.d2zeta_dpsi2(psi_int, 'lower'), float)
        _J2_LOWER_CACHE[key] = J
    return J


def _ldp(G, h):
    """Least Distance Programming (Lawson-Hanson): min ||x|| s.t. G@x >= h, via NNLS.
    Robust where SLSQP fails ('positive directional derivative'). Returns x, or None if
    the constraints are inconsistent (infeasible)."""
    from scipy.optimize import nnls
    G = np.asarray(G, float); h = np.asarray(h, float); n = G.shape[1]
    E = np.vstack([G.T, h.reshape(1, -1)])          # (n+1, m)
    f = np.zeros(n + 1); f[-1] = 1.0
    u, _ = nnls(E, f)
    r = E @ u - f
    if abs(r[-1]) < 1e-12:
        return None
    return -r[:n] / r[-1]


def repair_lower_inflection(lower, te_gap=0.0, le_cutoff=0.02, margin=1e-5, max_delta=5e-3, verify=True):
    """Return the min-coefficient-change lower coeffs with <=1 exact inflection, or None if the
    projection can't be verified OR would require a change larger than ``max_delta`` (max abs
    coeff move) -- i.e. only a genuinely SHALLOW wiggle is repaired; a deep 2nd inflection is
    left for the caller to reject/rebuild (it's a real feature, not a surrogate leak). Unchanged
    if already <=1. Projects onto convex-then-concave (sign s0 before a split psi*, -s0 after)
    with the SMALLEST ||delta|| via LDP; tries each dense-grid sign-change of d2 as psi* and keeps
    the smallest-norm delta that passes the EXACT count. Fine grid catches narrow near-LE lobes."""
    lo = np.asarray(lower, float); nC = len(lo)
    if _exact_lower_inflections(lo, le_cutoff) <= 1:
        return lo
    psi = np.linspace(le_cutoff, 0.9995, 2500)        # fine: catches ~0.002-wide near-LE lobes
    J2 = _lower_curvature_jacobian(psi, te_gap, nC)
    d2 = J2 @ lo
    s0 = 1.0 if d2[0] >= 0 else -1.0
    sc = np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0]
    if len(sc) == 0:
        return lo
    best = None
    for jc in sc:
        tgt = np.where(psi <= psi[jc], s0, -s0)
        x = _ldp(tgt[:, None] * J2, margin - tgt * d2)   # min ||x|| s.t. tgt*(d2 + J2@x) >= margin
        if x is None:
            continue
        rep = lo + x
        if _exact_lower_inflections(rep, le_cutoff) <= 1:
            nrm = float(x @ x)
            if best is None or nrm < best[1]:
                best = (rep, nrm, float(np.max(np.abs(x))))
    if best is None or best[2] > max_delta:      # no verified projection, or not a SHALLOW one
        return None
    return best[0]


# ----------------------------------------------------------------------
# FIX 1: diverse multistart seeds for the endpoint locators. Seed 0 is the family
# .dat fit unchanged; the rest are a ball of restarts around it with the per-restart
# jitter sigma ramped sig_lo..sig_hi (near-seed -> far-basin) plus a random psi*
# start. The old seeding was a sigma=0.05 ball (spread ~0.2 in coeff L2) while the
# distinct basins sit ~0.8-1.3 apart, so it was effectively single-start and which
# basin a run reached was luck; this spread reliably reaches the clean-biased basin.
# ----------------------------------------------------------------------
def _diverse_seeds(z0, n, rng, lo_b, hi_b, sig_lo=0.05, sig_hi=0.5):
    z0 = np.asarray(z0, float)
    seeds = [np.array(z0, float)]
    m = max(0, int(n) - 1)
    if m == 0:
        return seeds
    for sig in np.linspace(sig_lo, sig_hi, m):
        zc = np.array(z0, float)
        zc[:16] += rng.normal(0.0, sig, 16)
        zc[16] = rng.uniform(0.15, 0.90)                 # random psi* (curvature-split) start
        seeds.append(np.clip(zc, lo_b, hi_b))
    return seeds


# ----------------------------------------------------------------------
# Which named requirements are enforced as hard constraints (see
# og.CONSTRAINT_GROUPS). Same convention as the notebook's §2 block; edit here.
# ----------------------------------------------------------------------
CONSTRAINTS = {
    'non_intersection': True, 'reach_design_cl': True, 'thickness': True,
    'stall_margin': True, 'moments_of_inertia': True, 'area': True,
    'leading_edge_radius': True, 'radii_skew': False, 'max_thickness_location': True,
    'te_cone': True, 'curvature': True, 'lower_signflip': True,
    'min_radius_location': True, 'rough_xtr_cap': False, 'roughness_delta_cl': True, 'lod_falloff': True,  # V7: cap OFF
    'lod_falloff_2': True, 'transition_slope': True, 'cl_linearity': False, 'moment': True,
    'cpmin': False, 'cl_target': False, 'slope_ratio': True,
    'curvature_accel': False,   # |d2kappa/ds2| <= frozen envelope; OFF by default (opt-in)
    'bulge': True,              # lower-surface no-rising-mid-chord-hump guard; ON (relative, inert unless bulging)
}


def eps_grid(r_lo, r_hi, M, params=None):
    """The epsilon-constraint sweep levels: a linear blend of UNIFORM and
    ROUGH-DENSE COSINE spacing, weight w in [0, 1] from ``eps_cosine_blend``.

    Cosine = sin(pi/2 * t), which flattens as t->1 so levels bunch near r_hi (the
    hard rough end); uniform is even. w=0 pure uniform, w=1 pure cosine, w=0.5 a
    50/50 mix. Both share t, so every blend hits r_lo and r_hi exactly -- the
    clean and rough corners are preserved for any w. Legacy
    ``eps_cosine_rough`` == w=1.

    Lives here, and is imported by live.live_gradient, because both the batch
    sweep and the live dashboard have to lay out the SAME front. They each had
    their own ``np.linspace`` before, so the blend landed in one and not the
    other and the two silently produced differently-spaced fronts from identical
    settings.
    """
    params = params or {}
    t = np.linspace(0.0, 1.0, M)
    uni = r_lo + (r_hi - r_lo) * t
    cos = r_lo + (r_hi - r_lo) * np.sin(0.5 * np.pi * t)
    w = params.get('eps_cosine_blend', None)
    if w is None:
        w = 1.0 if params.get('eps_cosine_rough', False) else 0.0
    w = float(np.clip(w, 0.0, 1.0))
    return (1.0 - w) * uni + w * cos


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
             lower_inflections=int(_exact_lower_inflections(z[8:16])),
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
    def solve(which, z_start, rough_lod_min=None, clean_lod_min=None, explore=False):
        # loose-explore: multi-start seeds only need to FIND the basin, so run them at
        # reduced iterations + looser tol; the winner is later re-solved with explore=False
        # (tight) to polish. Cuts exploration cost without losing basins.
        _mit = max(30, max_iter // 2) if explore else max_iter
        _pit = max(60, proj_max_iter // 2) if explore else proj_max_iter
        _otol = 2e-3 if explore else _OPTS['tol']
        _ptol = 3e-3 if explore else 1e-6
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
    ap.add_argument("--locator-starts", type=int, default=10,
                    help="DIVERSE multi-start count per endpoint locator (phase 1), "
                         "default 10. The endpoints are non-convex single-objective "
                         "optima and are seed-sensitive/multi-modal; this many starts "
                         "(family seed + a ball of restarts with jitter sigma ramped "
                         "0.05..0.30 and random psi*, plus cross-seeding) are tried and "
                         "the best FEASIBLE, <=1-inflection one kept. A sigma=0.05-only "
                         "ball was effectively single-start and landed in the wrong basin "
                         "run-to-run; the spread reliably reaches the clean-biased basin. "
                         "~10-12 is reliable without being runaway on the 6-thickness fleet.")
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
    ap.add_argument("--two-sided", action="store_true",
                    help="two-sided epsilon sweep: the clean half maximizes CLEAN L/D s.t. rough>=eps "
                         "(rough floors), the rough half maximizes ROUGH L/D s.t. clean>=eps (clean floors), "
                         "then merge nondominated. The rough end becomes a genuine rough-optimum instead of "
                         "'cleanest airfoil that still hits max rough'. Default: one-sided (max clean throughout).")
    ap.add_argument("--eps-cosine-rough", action="store_true",
                    help="space the epsilon sweep with a cosine cluster whose DENSE end is on the "
                         "rough side (near rough_max), instead of uniform -- more front resolution "
                         "in the hard rough band. Endpoints (both corners) preserved. (== --eps-cosine-blend 1.0)")
    ap.add_argument("--eps-cosine-blend", type=float, default=None, metavar="W",
                    help="linear blend of uniform and rough-dense-cosine eps spacing: "
                         "eps = (1-W)*uniform + W*cosine, W in [0,1]. W=0 uniform, W=1 full cosine, "
                         "W=0.5 a 50/50 mix. Overrides --eps-cosine-rough when given.")
    ap.add_argument("--eps-equality", action="store_true",
                    help="pin each epsilon point to its floor with an EQUALITY (LoD_rough==eps) instead of the "
                         "default inequality (LoD_rough>=eps). Forces even spacing along the rough axis and stops "
                         "several eps collapsing onto the rough corner; a point becomes infeasible (honest gap) "
                         "where no airfoil sits at exactly that rough.")
    ap.add_argument("--repair-inflection", action="store_true",
                    help="REPAIR points whose raw solve has a shallow 2nd lower-surface inflection "
                         "(surrogate leak) by projecting onto the exact <=1-inflection manifold with a "
                         "tiny coeff nudge (L/D preserved), instead of collapsing them onto the sparse "
                         "guarantee pool. Deeper inflections still fall back to the rebuild.")
    ap.add_argument("--no-inflection-guarantee", action="store_true",
                    help="DIAGNOSTIC: skip the exact <=1-lower-inflection guarantee-pass rebuild and report the "
                         "RAW epsilon-solves (which may include a small 2nd inflection the smooth signflip "
                         "surrogate admits). Shows every distinct point the solver found.")
    ap.add_argument("--robust", action="store_true",
                    help="robust-solver bundle for the multi-modal rough tail: adaptive seeding (few seeds "
                         "on the unimodal clean half, full K on the rough tail), relax-to-consistency + "
                         "dominated-repair (re-seed each point from its neighbours' best until the front is "
                         "neighbour-consistent), and loose-explore/tight-polish (cheap exploration, then one "
                         "tight polish per point). Finds the true front without merging separate runs.")
    ap.add_argument("--max-relax", type=int, default=8,
                    help="max relax-to-consistency rounds under --robust (stops early when no point improves).")
    ap.add_argument("--curvature-accel", action="store_true",
                    help="enable the frozen |d2kappa/ds2| <= E(x/c) envelope constraint (OFF by default): "
                         "keeps the blocky gradient airfoils from poking curvature-acceleration spikes "
                         "through the V13-GA-derived envelope (curvature_envelope.json).")
    ap.add_argument("--curvature-accel-stride", type=int, default=2,
                    help="station stride for the curvature-accel constraint (1=every frozen station; "
                         "higher = fewer Ipopt inequalities). Default 2.")
    ap.add_argument("--no-bulge-guard", dest="bulge_guard", action="store_false",
                    help="disable the lower-surface secondary-bulge guard (ON by default): forbids a "
                         "rising mid-chord curvature hump (|kappa(x_band)| <= |kappa(2%%)| + margin). "
                         "Relative/thickness-agnostic and inert unless the section is bulging.")
    ap.set_defaults(bulge_guard=True)
    ap.add_argument("--bulge-margin", type=float, default=None,
                    help="allowed kappa rise above the 2%% shoulder before the bulge guard trips "
                         "(default curvature_envelope.BULGE_MARGIN = 0.5).")
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
    params['eps_floor_equality'] = bool(getattr(args, 'eps_equality', False))  # rough==eps (vs >=)
    params['eps_cosine_rough'] = bool(getattr(args, 'eps_cosine_rough', False))  # cosine eps, dense on rough side
    if getattr(args, 'eps_cosine_blend', None) is not None:
        params['eps_cosine_blend'] = float(args.eps_cosine_blend)                 # linear uniform<->cosine blend weight
    params['skip_inflection_guarantee'] = bool(getattr(args, 'no_inflection_guarantee', False))  # raw solves
    params['repair_inflection'] = bool(getattr(args, 'repair_inflection', False))  # repair shallow 2nd inflection
    CONSTRAINTS['curvature_accel'] = bool(args.curvature_accel)          # opt-in envelope constraint
    params['curvature_accel_stride'] = int(args.curvature_accel_stride)
    CONSTRAINTS['bulge'] = bool(args.bulge_guard)                        # lower no-mid-hump guard (ON by default)
    if args.bulge_margin is not None:
        params['bulge_margin'] = float(args.bulge_margin)
    if args.cl_lin_tol is not None:
        params['cl_linearity_tol_clean'] = args.cl_lin_tol
        params['cl_linearity_tol_rough'] = args.cl_lin_tol
    if args.cm_min is not None:
        params['CMc_min'] = args.cm_min
    # The family yaml hard-sets alpha_min_clean/rough=0, which clamps the a_des-2deg
    # falloff/linearity samples for low-a_des airfoils (and lets the optimizer hide
    # falloff by pushing a_des below the grid). Force the grid below design-2deg.
    params['alpha_min_clean'] = min(params['alpha_min_clean'], -3.0)
    # rough grid extended down to -14 so the sweep brackets the rough zero-lift angle AND
    # leaves >=~2deg of margin below it for the slope_ratio's finite-difference stencil.
    # High-camber (clean-biased, high-CLd) airfoils have zero-lift near -8deg; a -8 floor put
    # zero-lift AT the grid edge, so the m0 finite-diff clamped and HALVED the zero-lift slope
    # -> slope-ratio read ~2x too high, letting rounded clean-biased airfoils pass. -14 fixes it.
    params['alpha_min_rough'] = min(params['alpha_min_rough'], -14.0)
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
    ctx['curvature_accel'] = bool(CONSTRAINTS.get('curvature_accel', False))  # |d2kappa/ds2| envelope (opt-in)
    ctx['bulge'] = bool(CONSTRAINTS.get('bulge', False))                      # lower no-mid-hump guard
    bounds = list(zip(np.full(16, -2.0), np.full(16, 2.0))) + [(0.10, 0.98)]  # 16 coeffs + psi*
    solve = make_solver(ctx, params, bounds, args.max_iter, proj_max_iter=args.proj_max_iter)
    # picklable recipe so pool workers can rebuild `solve` (closures cannot cross
    # a process boundary); the in-process `solve` above still serves phase 1's
    # cheap cross-seeding and any --workers 1 run.
    wcfg = dict(tool=args.tool, model=args.model, params=params,
                max_iter=args.max_iter, proj_max_iter=args.proj_max_iter,
                curvature_accel=bool(args.curvature_accel),
                bulge=bool(args.bulge_guard))

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

    _lec = params.get('lower_signflip_le_cutoff', 0.02)

    def _corner_ok(z, d):
        # a usable corner is feasible AND genuinely <=1 lower-surface inflection
        # (the smooth signflip surrogate leaks; enforce the exact count here).
        return (d.get("violation", 1e9) <= 1e-3
                and _exact_lower_inflections(z[8:16], _lec) <= 1)

    def best_of(results, which):
        key = "lod_" + which
        cand = [r for r in results if r[0] == which]
        if not cand:
            return None
        # prefer feasible & <=1-inflection, then max L/D
        return max(cand, key=lambda r: (1 if _corner_ok(r[2], r[3]) else 0, r[3][key]))

    def _pick_corner(cands, key):
        return max(cands, key=lambda zd: (1 if _corner_ok(*zd) else 0, zd[1][key]))

    # Round 1: DIVERSE multi-start both objectives from a ball of restarts around
    # the family seed (FIX 1): far enough to escape the family basin into the
    # clean-biased / true-rough-max basins.
    seeds = _diverse_seeds(z0, n_start, rng, lo_b, hi_b)
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

    # keep the best over both rounds (prefer feasible & <=1-inflection, then L/D)
    z_clean, d_clean = _pick_corner([(zc1, dc1), (zc2, dc2)], "lod_clean")
    z_rmax, d_rmax = _pick_corner([(zr1, dr1), (zr2, dr2)], "lod_rough")
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
    K = max(1, getattr(args, "sweep_starts", 4))
    FEAS = 1e-3
    c_hi = d_clean["lod_clean"]        # max clean L/D (the clean corner)
    c_lo = d_rmax["lod_clean"]         # clean L/D at the rough corner (bottom of the clean range)

    def _seed_tasks(M_, seed_lo, seed_hi, robust=False):
        # seed 0 is the interp guess between the two corners; 1..K-1 are perturbations.
        # BOTH ends now get the full corner-seed + K wide-sigma restart set (the old --robust
        # adaptive "clean half is unimodal -> 1-2 seeds" reduction was starving the clean end).
        tasks = []
        for idx in range(M_):
            t = idx / (M_ - 1)
            base = (1.0 - t) * seed_lo + t * seed_hi
            # SYMMETRIC treatment for BOTH ends. Previously --robust gave the clean-biased
            # half (t<0.45) only 1-2 "easy/unimodal" seeds while the rough tail got the full
            # corner-seed + K wide restarts -- that starved the clean end and it started
            # missing points. The clean end is just as basin-sensitive as the rough end, so it
            # now gets the SAME number of calls and the SAME wide-sigma diversity fills.
            if getattr(args, "corner_seed", False):
                # anchor every point at BOTH corners so it can reach either basin, then
                # fill K with WIDE, MULTI-ANCHOR, ramped-sigma restarts (sigma 0.15..0.55 off
                # base / rough-corner / clean-corner, random psi*), so restarts reach the
                # sparse basins at EITHER end instead of clustering next to the anchor.
                seedlist = [z_clean, z_rmax, base]
                n_fill = max(0, max(K, 3) - len(seedlist))
                anchors = (base, z_rmax, z_clean)
                for j, sig in enumerate(np.linspace(0.15, 0.5, n_fill) if n_fill else []):
                    zc = np.array(anchors[j % len(anchors)], float)
                    zc[:16] += rng.normal(0.0, sig, 16)
                    zc[16] = rng.uniform(0.15, 0.90)          # random psi* (curvature split)
                    seedlist.append(np.clip(zc, lo_b, hi_b))
                for kk, s_ in enumerate(seedlist):
                    tasks.append((idx, kk, s_))
            else:
                tasks.append((idx, 0, base))
                for kk in range(1, K):
                    tasks.append((idx, kk, perturb(base, 0.04)))
        return tasks

    def run_sweep(which, floor_levels, seed_lo, seed_hi, tag, extra_pool=None):
        """One epsilon sweep. which='clean' -> max clean s.t. rough>=eps (rough floors);
        which='rough' -> max rough s.t. clean>=eps (clean floors). Returns best-feasible
        (max objective) per index: [(idx, eps, lod_clean, lod_rough, viol, z), ...].
        extra_pool seeds the <=1-inflection guarantee pool with airfoils found earlier
        (e.g. the phase-1 locators), which span the whole rough range for free and keep
        the guarantee pass from collapsing sparse regions onto one airfoil."""
        M_ = len(floor_levels)
        robust = getattr(args, "robust", False)
        okey = "lod_clean" if which == "clean" else "lod_rough"   # objective maximized
        fkey = "lod_rough" if which == "clean" else "lod_clean"   # floored quantity
        _lec = params.get('lower_signflip_le_cutoff', 0.02)

        # FIX 2(a) guarantee pool: every feasible, genuinely <=1-inflection airfoil
        # seen ANYWHERE in this sweep (explore/relax/polish/cross-seed). The smooth
        # signflip surrogate leaks -- the L/D-maximizer drifts into gentle multi-wiggle
        # (2-3 exact inflection) basins that the surrogate certifies AND that no gate
        # width can forbid (the extra inflections sit below constr_viol_tol). So instead
        # of trusting any single solve, we rebuild the front from the <=1-inflection
        # subset of everything found: each point = max-objective 1-inflection airfoil
        # meeting its floor. Guarantees every reported point is exactly <=1 inflection.
        pool = list(extra_pool or [])

        def _record(z, d):
            z = np.asarray(z, float)
            if d.get("violation", 1e9) <= FEAS and _exact_lower_inflections(z[8:16], _lec) <= 1:
                pool.append((d["lod_clean"], d["lod_rough"], d["violation"], z))

        def _cand(idx, z, d):
            eps = float(floor_levels[idx])
            # FIX 2(a): HARD exact-inflection reject -- a >1-inflection airfoil is
            # infeasible regardless of the (leaky) smooth surrogate, so it can never
            # win a front point. Guarantees every reported point is genuinely <=1.
            infl = _exact_lower_inflections(z[8:16], params.get('lower_signflip_le_cutoff', 0.02))
            feas = (d["violation"] <= FEAS) and (d[fkey] >= eps - 0.5) and (infl <= 1)
            return (feas, d[okey] if feas else -d["violation"],
                    (idx, eps, d["lod_clean"], d["lod_rough"], d["violation"], z))

        tasks = _seed_tasks(M_, seed_lo, seed_hi, robust=robust)
        log(f"phase 2 [{tag}]: {len(tasks)} explore sub-solves (max {which} s.t. {fkey.split('_')[1]}>=eps)")
        ptasks = [((idx, kk), which, np.asarray(zs, float), float(floor_levels[idx]), robust)
                  for (idx, kk, zs) in tasks]
        res = _parallel_solve(ptasks, wcfg, workers=args.workers, log=log)
        by_idx = {}
        for t_, z, d in res:
            _record(z, d)
            c = _cand(t_[0], z, d); cur = by_idx.get(t_[0])
            if cur is None or (c[0], c[1]) > (cur[0], cur[1]):
                by_idx[t_[0]] = c

        if robust:
            # #1 relax-to-consistency + #4 dominated-repair: re-seed each point from BOTH
            # neighbours' best (warm start = cheap, loose-explore) until no point improves.
            # A dominated point inherits its dominating neighbour's design, so the fixed point
            # is neighbour-consistent -- this is the in-solver version of "merge best basins".
            for rnd in range(getattr(args, "max_relax", 8)):
                xtasks = []
                for idx in range(M_):
                    for j in (idx - 1, idx + 1):
                        if 0 <= j < M_:
                            xtasks.append(((idx, "r%d_%d" % (rnd, j)), which,
                                           np.asarray(by_idx[j][2][5], float), float(floor_levels[idx]), True))
                nimp = 0
                for t_, z, dd in _parallel_solve(xtasks, wcfg, workers=args.workers, log=log):
                    _record(z, dd)
                    c = _cand(t_[0], z, dd)
                    if (c[0], c[1]) > (by_idx[t_[0]][0], by_idx[t_[0]][1]):
                        by_idx[t_[0]] = c; nimp += 1
                log(f"phase 2 [{tag}] relax round {rnd+1}: {nimp} improved")
                if nimp == 0:
                    break
            # tight polish: re-solve each point's best at full tol (explore=False)
            ptp = [((idx, "P"), which, np.asarray(by_idx[idx][2][5], float), float(floor_levels[idx]), False)
                   for idx in range(M_)]
            for t_, z, dd in _parallel_solve(ptp, wcfg, workers=args.workers, log=log):
                _record(z, dd)
                c = _cand(t_[0], z, dd)
                if (c[0], c[1]) > (by_idx[t_[0]][0], by_idx[t_[0]][1]):
                    by_idx[t_[0]] = c
            log(f"phase 2 [{tag}] tight polish done")
        else:
            # existing fixed cross-seed (unchanged v4 behaviour)
            for rnd in range(getattr(args, "cross_seed", 0)):
                xtasks = []
                for idx in range(M_):
                    for j in (idx - 1, idx, idx + 1):
                        if 0 <= j < M_:
                            xtasks.append(((idx, "x%d_%d" % (rnd, j)), which,
                                           np.asarray(by_idx[j][2][5], float), float(floor_levels[idx]), False))
                nimp = 0
                for t_, z, dd in _parallel_solve(xtasks, wcfg, workers=args.workers, log=log):
                    _record(z, dd)
                    c = _cand(t_[0], z, dd)
                    if (c[0], c[1]) > (by_idx[t_[0]][0], by_idx[t_[0]][1]):
                        by_idx[t_[0]] = c; nimp += 1
                log(f"phase 2 [{tag}] cross-seed round {rnd+1}: {nimp} improved")

        # DIAGNOSTIC: skip the <=1-inflection guarantee-pass rebuild entirely and report the
        # RAW epsilon-solves (--no-inflection-guarantee). Lets you see every distinct point the
        # solver actually found, including ones with a small 2nd lower-surface inflection that
        # the smooth signflip surrogate let through. Still logs how many are >1 exact inflection.
        if params.get('skip_inflection_guarantee', False):
            n_multi = sum(1 for i in range(M_)
                          if _exact_lower_inflections(by_idx[i][2][5][8:16], _lec) > 1)
            log(f"phase 2 [{tag}] guarantee pass SKIPPED (--no-inflection-guarantee): raw solves "
                f"reported; {n_multi}/{M_} points have >1 exact lower inflection")
            return [by_idx[i][2] for i in range(M_)]

        # REPAIR PASS (--repair-inflection): before falling back to the pool rebuild, try to
        # REPAIR each point whose raw solve has a SHALLOW 2nd lower-surface inflection (the smooth
        # signflip surrogate's leak). repair_lower_inflection nudges the lower coeffs onto the
        # exact <=1-inflection manifold with the SMALLEST change (~1e-4 coeff, <0.02 L/D -- aero
        # preserved), so the high-performance airfoil is KEPT instead of collapsing onto the sparse
        # pool. Only genuinely shallow wiggles repair (max_delta gate); deeper ones return None and
        # fall through to the rebuild below. The raw L/D is kept (the repair shifts it <0.02).
        if params.get('repair_inflection', False):
            nrep = 0
            for idx in range(M_):
                cur = by_idx[idx]; z = cur[2][5]
                if _exact_lower_inflections(z[8:16], _lec) > 1:
                    rep = repair_lower_inflection(z[8:16], te_gap=params['TE_gap'], le_cutoff=_lec)
                    if rep is not None:
                        zn = np.array(z, float); zn[8:16] = rep
                        # `feas` (line ~1057) bundles the inflection check, so the raw solve was
                        # feas=False ONLY because of the >1 inflection we just removed. Recompute
                        # it (viol/floor unchanged; infl now <=1) and refresh the objective, else
                        # the guarantee pass's `not cur_ok` clause rebuilds the repaired point.
                        _viol = cur[2][4]
                        _fq = cur[2][3] if which == 'clean' else cur[2][2]   # floored quantity
                        _ok = (_viol <= FEAS) and (_fq >= float(floor_levels[idx]) - 0.5)
                        _obj = (cur[2][2] if which == 'clean' else cur[2][3]) if _ok else -_viol
                        by_idx[idx] = (_ok, _obj,
                                       (idx, cur[2][1], cur[2][2], cur[2][3], cur[2][4], zn))
                        nrep += 1
            log(f"phase 2 [{tag}] inflection repair: {nrep} point(s) repaired in place "
                f"(shallow wiggle removed, L/D preserved); rest fall through to rebuild")

        # FIX 2(a) GUARANTEE PASS: rebuild every point from the <=1-inflection feasible
        # pool -- the max-objective 1-inflection airfoil meeting that point's floor.
        # This is the airtight enforcement: it never lets a >1-inflection leak artifact
        # be reported, and (because it maximizes over ALL 1-inflection airfoils found,
        # not one solve) it also recovers the best genuine 1-inflection value per floor.
        oi = 0 if which == "clean" else 1     # pool tuple index of the maximized objective (clean=0, rough=1)
        fi = 1 if which == "clean" else 0     # pool tuple index of the floored quantity
        nfix = 0
        for idx in range(M_):
            eps = float(floor_levels[idx])
            cur = by_idx[idx]
            cur_z = cur[2][5]
            cur_ok = bool(cur[0]) and _exact_lower_inflections(cur_z[8:16], _lec) <= 1
            elig = [p for p in pool if p[fi] >= eps - 0.5]
            if not elig:
                continue
            best = max(elig, key=lambda p: p[oi])
            # (not cur_ok): >1-inflection point MUST be rebuilt from the pool. The 2nd clause
            # ("pool has a higher-objective 1-inflection airfoil at this floor") normally also
            # recovers the best genuine value per eps -- but with a sparse pool it collapses every
            # eps onto the corner, and when --repair-inflection has already put a valid, distinct
            # <=1-inflection airfoil at each point that collapse UNDOES the repair. So skip the 2nd
            # clause when repairing: keep the distinct repaired solves, only rebuild the leftovers.
            if (not cur_ok) or (best[oi] > cur[2][2 + oi] + 1e-9
                                and not params.get('repair_inflection', False)):
                z = best[3]
                by_idx[idx] = (True, best[oi],
                               (idx, eps, best[0], best[1], best[2], z))
                nfix += 1
        # any point that STILL is >1-inflection (pool had nothing meeting its floor)
        # inherits the highest-floor 1-inflection pool airfoil (dominated but valid).
        if pool:
            hi = max(pool, key=lambda p: p[fi])
            for idx in range(M_):
                z = by_idx[idx][2][5]
                if _exact_lower_inflections(z[8:16], _lec) > 1:
                    by_idx[idx] = (True, hi[oi],
                                   (idx, float(floor_levels[idx]), hi[0], hi[1], hi[2], hi[3]))
                    nfix += 1
        log(f"phase 2 [{tag}] 1-inflection guarantee pass: {nfix} point(s) rebuilt from "
            f"pool of {len(pool)} feasible <=1-inflection airfoils")
        return [by_idx[i][2] for i in range(M_)]

    if rank == 0:
        # Seed the <=1-inflection guarantee pool with the phase-1 locator airfoils:
        # the DIVERSE multistart already produced many feasible 1-inflection airfoils
        # spanning the whole rough range, for free -- this keeps sparse rough bands from
        # collapsing onto a single airfoil in the guarantee pass.
        _lecm = params.get('lower_signflip_le_cutoff', 0.02)
        _p1 = []
        for _r in r1:
            _zz, _dd = _r[2], _r[3]
            if _dd.get('violation', 1e9) <= FEAS and _exact_lower_inflections(_zz[8:16], _lecm) <= 1:
                _p1.append((_dd['lod_clean'], _dd['lod_rough'], _dd['violation'], np.asarray(_zz, float)))
        for _zz, _dd in xd.values():
            if _dd.get('violation', 1e9) <= FEAS and _exact_lower_inflections(_zz[8:16], _lecm) <= 1:
                _p1.append((_dd['lod_clean'], _dd['lod_rough'], _dd['violation'], np.asarray(_zz, float)))
        log(f"phase 1 seeded {len(_p1)} feasible <=1-inflection airfoils into the guarantee pool")
        if getattr(args, "two_sided", False):
            # clean half: max CLEAN vs rough floor; rough half: max ROUGH vs clean floor; merge.
            Mh = M // 2 + 1
            cpts = run_sweep("clean", np.linspace(r_lo, r_hi, Mh), z_clean, z_rmax, "clean-half", extra_pool=_p1)
            rpts = run_sweep("rough", np.linspace(c_lo, c_hi, Mh), z_rmax, z_clean, "rough-half", extra_pool=_p1)
            merged = [p for p in (cpts + rpts) if p[4] <= FEAS]
            _dom = lambda p, q: (q[3] >= p[3] and q[2] >= p[2] and (q[3] > p[3] or q[2] > p[2]))
            nondom = [p for i, p in enumerate(merged)
                      if not any(_dom(p, q) for j, q in enumerate(merged) if j != i)]
            spts = sorted(nondom, key=lambda p: p[3])            # rough L/D ascending
            pts = [(i, p[1], p[2], p[3], p[4], p[5]) for i, p in enumerate(spts)]
            log(f"phase 2 TWO-SIDED: clean {len(cpts)} + rough {len(rpts)} -> {len(pts)} nondominated")
        else:
            eps_levels = eps_grid(r_lo, r_hi, M, params)
            pts = run_sweep("clean", eps_levels, z_clean, z_rmax, "front", extra_pool=_p1)
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
