"""
oso_gradient.py — a faithful, **gradient-capable** port of the oso-airfoils
``core_fitness_function`` for gradient-based (Ipopt) optimization.

The production fitness function (oso_airfoils/optimization/objective_function.py)
runs a full alpha sweep (clean + rough), detects the stall peak, interpolates to
a design CL, evaluates clean/rough L/D there, and folds ~20 geometric/aero
constraints into a single penalty ``conpen``. It returns two objectives
(``obj1 = -LoD_clean + conpen``, ``obj2 = -LoD_rough + conpen``) for an NSGA-style
genetic algorithm.

This module reproduces that computation term-for-term but makes it
differentiable, so it can be minimized with a gradient method. It combines the
two objectives with a tuning weight ``w``:

    f(z) = w*obj1 + (1-w)*obj2
         = -(w*LoD_clean + (1-w)*LoD_rough) + conpen(z)

Design vector ``z = [upper_coeffs(M), lower_coeffs(M)]`` (nx = 2M). ``alpha`` is
NOT a design variable — ``alpha_design`` is *derived* from the clean sweep at
the target CL, exactly as in the fitness function.

Gradients are exact via forward-mode AD (a light ``Dual`` number carrying
d/dz alongside every value):

* aero sweep values (cl, cd, cm, cpmin per alpha) carry the tool's analytic
  ``kulfan_gradient`` (CasADi AD for NeuralFoil, complex-step for cxfoil/cqfoil);
* ``np.interp`` is replaced by a differentiable ``dinterp``;
* the discrete stall peak uses the value at the peak index (a subgradient — the
  same discreteness the fitness function itself has);
* ``min``/``max``/``abs`` penalty terms use subgradients (``dmin``/``dmax``/``dabs``);
* geometry (tau, area, Ixx/Iyy/Izz, LE radius) uses metafoil's analytic Kulfan
  gradients; the curvature/located geometry scalars use the closed-form Kulfan
  first/second zeta-derivatives (dzeta/dpsi, d2zeta/dpsi2), which are linear in the
  coefficients -> constant Jacobians, no finite differences of the coordinate grid.

Tool-swappable: ``tool in {'neuralfoil', 'cxfoil', 'cqfoil'}``.

================================================================================
Numerical deviations from the original ``core_fitness_function``
================================================================================
The original folds every requirement into one additive penalty evaluated by a
genetic algorithm, which tolerates non-smooth, discontinuous and redundant
terms. A gradient method (Ipopt) instead needs smooth, well-conditioned,
full-rank constraints. The reformulations below were made for that reason; each
preserves the original's INTENT and feasible set -- only the numerical
representation differs. (History: the earlier symptom was Ipopt diverging to a
wildly-infeasible point within ~5 iterations; items 1-3 fixed that, items 4-5
then let it reach a smooth, near-stationary optimum.)

1. Curvature / geometry constraints imposed PER STATION, not as relu-sum
   aggregates.  The GA penalizes ``sum_j relu(g_j(z))`` for the upper-concavity,
   aft-curvature, lower-convexity and TE-cone requirements. As a single scalar
   constraint that sum is 0 AND has gradient 0 once satisfied, so its Jacobian
   row vanishes: the active set loses rank, the constraint qualification (LICQ)
   fails, the KKT system goes singular and Ipopt's step blows up. Each is instead
   imposed as one inequality per interior station, built from the curvature Duals
   (or cone clearances), which are LINEAR in the coefficients -> constant,
   never-vanishing Jacobian rows. Same feasible set: aggregate == 0  <=>  every
   station individually satisfied. The psi=1 TE apex row (identically 0 >= 0) is
   skipped. See ``constraint_list`` and ``*_terms`` in ``geometry_duals``. The
   scalar ``concave_sum`` / ``aftcurve_sum`` / ``te_cone``
   aggregates are still computed, but only for diagnostics.

2. Analytic second derivative for curvature.  The GA measures curvature with a
   finite 2nd-difference on the coordinate grid -- a grid artifact ~1000x too
   large near the TE, and non-smooth. Here curvature is metafoil's closed-form
   ``Kulfan.d2zeta_dpsi2`` (exact, and linear in the coefficients).

3. Polar-moment (Izz) redundancy dropped when implied.  Izz == Ixx + Iyy exactly
   (perpendicular-axis theorem for a planar section), so grad(Izz) ==
   grad(Ixx) + grad(Iyy). Imposing all three makes the active Jacobian
   rank-deficient whenever they bind -> non-unique KKT multipliers -> dual
   infeasibility cannot converge (and the bad multipliers corrupt the search
   direction, steering Ipopt to a worse local optimum). The Izz row is added only
   when NOT already implied, i.e. Izz_con > Ixx_con + Iyy_con.

4. Lower-surface curvature "<= 1 sign change", exact, via a free split psi*
   (``lower_signflip``).  Continuous/differentiable replacement for the GA's
   discrete flip count. A function has <=1 inflection iff there is a split psi*
   with curvature convex (d2zeta_lower >= 0) before it and concave (<= 0) after.
   psi* is a genuine auxiliary optimization variable (design vector ->
   [coeffs, psi*], set ctx['n_aux']=1 and bound psi* to [~0.10, ~0.98] -- a real
   airfoil's crossover never sits forward of ~10% chord, which also skips the LE
   curvature singularity). The per-station constraint is the PRODUCT form
   t_j * d2zeta_lower(psi_j) >= 0  with a smooth target sign
   t_j = 2*sigmoid((psi* - psi_j)/w) - 1 (~+1 before psi*, ~-1 after): no big-M,
   and d2zeta is linear in the coeffs so it is smooth with a full-rank Jacobian.
   Because psi* is FREE, Ipopt slides it to the crossover; a surface needing two
   sign changes has no feasible psi* and is rejected. This is the exact (necessary
   & sufficient) version; it replaced two earlier surrogates -- a fixed-split
   convexity constraint and a smooth sign-change-count penalty -- both now removed.

5. C1 design-point interpolation.  ``dinterp`` (the differentiable ``np.interp``
   replacement, item above) is a cubic Hermite spline with Catmull-Rom node
   slopes, not piecewise-linear. Piecewise-linear interpolation of the alpha
   sweep gives a piecewise-CONSTANT gradient that jumps every time the (moving)
   design angle crosses a sweep node; that step-function objective gradient
   chatters and Ipopt can never reach dual stationarity. Cubic Hermite is C1 in
   the query point and linear in the samples, so the objective and every
   ``at()``-derived constraint (L/D, falloff, delta_cl, cl_target) are smooth in
   z. Verified: objective analytic-vs-FD relative error dropped from ~10-20%
   (drifting) to ~1e-8, and the KKT dual residual at the optimum from ~30% of
   ||grad f|| to ~2%.

Known residual: the stall peak is still a discrete argmax over the sweep (as in
the original) -- a subgradient at the peak index. Everything else is smooth.
(The dual-infeasibility plateau that once stopped Ipopt from certifying
optimality was NOT inherent -- it was the default 6-pair limited-memory Hessian.
The design space is tiny (~17 vars), so setting ``limited_memory_max_history``
>= n makes L-BFGS equivalent to a full dense BFGS and Ipopt then converges
cleanly, ``Solve_Succeeded`` in ~30 iterations. Only gradients are available, so
a true exact Hessian is not; full BFGS is as good as it needs to be at this size.)
"""
from __future__ import annotations
import math
import numpy as np


class DesignPointError(ValueError):
    """Raised when the target design CL cannot be located in the swept clean-CL
    range (design CL below the CL at alpha_min, or above CL_max). The old code
    silently CLAMPED the design alpha to the sweep end, which let the optimizer
    game the objective; we refuse to guess and raise instead."""

from metafoil.core.kulfan import Kulfan, _binom


# ======================================================================
# Forward-mode AD: a scalar Dual number over the nx-dim coefficient space
# ======================================================================
class Dual:
    __slots__ = ('v', 'g')

    def __init__(self, v, g):
        self.v = float(v)
        self.g = np.asarray(g, dtype=np.float64)

    # -- arithmetic (other may be Dual or float) -----------------------
    def __add__(s, o):
        return Dual(s.v + o.v, s.g + o.g) if isinstance(o, Dual) else Dual(s.v + o, s.g)
    __radd__ = __add__

    def __sub__(s, o):
        return Dual(s.v - o.v, s.g - o.g) if isinstance(o, Dual) else Dual(s.v - o, s.g)

    def __rsub__(s, o):
        return Dual(o - s.v, -s.g)

    def __mul__(s, o):
        if isinstance(o, Dual):
            return Dual(s.v * o.v, s.v * o.g + o.v * s.g)
        return Dual(s.v * o, s.g * o)
    __rmul__ = __mul__

    def __truediv__(s, o):
        if isinstance(o, Dual):
            return Dual(s.v / o.v, (s.g * o.v - s.v * o.g) / o.v**2)
        return Dual(s.v / o, s.g / o)

    def __rtruediv__(s, o):                 # o / self, o constant
        return Dual(o / s.v, -o * s.g / s.v**2)

    def __neg__(s):
        return Dual(-s.v, -s.g)

    def __pow__(s, p):                      # p constant
        return Dual(s.v**p, p * s.v**(p - 1) * s.g)

    # -- comparisons by value ------------------------------------------
    def __lt__(s, o): return s.v < _val(o)
    def __le__(s, o): return s.v <= _val(o)
    def __gt__(s, o): return s.v > _val(o)
    def __ge__(s, o): return s.v >= _val(o)
    def __float__(s): return s.v


def _val(x):
    return x.v if isinstance(x, Dual) else float(x)


def const(v, nx):
    return Dual(v, np.zeros(nx))


def _cg(upper_g, lower_g, nx):
    """Assemble a coefficient-space gradient of length nx from the per-surface
    upper/lower parts, zero-padding any trailing auxiliary-variable slots (e.g.
    the free curvature-split location psi*). When there are no auxiliaries
    (nx == 2*ncoef_per_surface) this is exactly the old concatenate."""
    u = np.ravel(upper_g); l = np.ravel(lower_g)
    pad = nx - len(u) - len(l)
    return np.concatenate([u, l, np.zeros(pad)]) if pad else np.concatenate([u, l])


def _onehot(i, nx):
    g = np.zeros(nx); g[i] = 1.0
    return g


def dabs(x):
    if not isinstance(x, Dual):
        return abs(x)
    return Dual(abs(x.v), math.copysign(1.0, x.v) * x.g)


def dsqrt(x):
    if not isinstance(x, Dual):
        return math.sqrt(x)
    return Dual(math.sqrt(x.v), x.g / (2.0 * math.sqrt(x.v)))


def dsigmoid(x):
    """Logistic 1/(1+exp(-x)), differentiable. sigmoid'(x) = s(1-s)."""
    if not isinstance(x, Dual):
        return 1.0 / (1.0 + math.exp(-x))
    s = 1.0 / (1.0 + math.exp(-x.v))
    return Dual(s, s * (1.0 - s) * x.g)


def dlog(x):
    if not isinstance(x, Dual):
        return math.log(x)
    return Dual(math.log(x.v), x.g / x.v)


def dmax(a, b):
    return a if _val(a) >= _val(b) else b


def dmin(a, b):
    return a if _val(a) <= _val(b) else b


def dsum(items, nx):
    s = const(0.0, nx)
    for x in items:
        s = s + x
    return s


def dtan(x):
    if not isinstance(x, Dual):
        return math.tan(x)
    return Dual(math.tan(x.v), x.g / math.cos(x.v)**2)


def dinterp(x0, xp, yp):
    """Differentiable **C1** (continuously differentiable) interpolation, clamped
    at the endpoints like numpy.interp. xp must be increasing in value. x0/xp/yp
    entries may each be Dual or float; gradients flow through whichever are Dual.
    Always returns a Dual when any input is a Dual, so a clamped return of a
    plain-float grid (e.g. when the design CL is out of range) still carries a
    zero gradient.

    Uses a cubic Hermite spline whose node slopes are the central (Catmull-Rom)
    differences of the samples. This was piecewise-LINEAR: fine for values, but
    its gradient is piecewise-CONSTANT and jumps every time the query point x0
    crosses a grid node. When the objective L/D is interpolated at the (moving)
    design angle, that step-function gradient chatters and Ipopt can never drive
    the dual infeasibility to zero. Cubic Hermite is C1 across nodes (it matches
    both the value AND the node slope from either side), so d(interp)/dx0 is
    continuous; and because the Catmull-Rom slopes are LINEAR in the samples yp,
    the interpolant stays smooth w.r.t. the design vector z too. Reduces to the
    linear interpolant when there are only two nodes."""
    nx = None
    for s in (x0, *xp, *yp):
        if isinstance(s, Dual):
            nx = s.g.shape[0]
            break
    def prom(y):
        return y if isinstance(y, Dual) or nx is None else Dual(y, np.zeros(nx))
    xv = [_val(x) for x in xp]
    x0v = _val(x0)
    n = len(xv)
    if n == 1 or x0v <= xv[0]:
        return prom(yp[0])
    if x0v >= xv[-1]:
        return prom(yp[-1])
    i = 0
    while i < n - 1 and xv[i + 1] < x0v:
        i += 1

    def slope(k):
        # node slope dy/dx: central difference in the interior, one-sided at the
        # ends (linear in yp -> smooth w.r.t. z; C1 continuity across nodes holds
        # for any consistent per-node slope).
        if k == 0:
            return (yp[1] - yp[0]) / (xp[1] - xp[0])
        if k == n - 1:
            return (yp[n - 1] - yp[n - 2]) / (xp[n - 1] - xp[n - 2])
        return (yp[k + 1] - yp[k - 1]) / (xp[k + 1] - xp[k - 1])

    xa, xb, ya, yb = xp[i], xp[i + 1], yp[i], yp[i + 1]
    if n == 2:                                   # nothing to build slopes from
        return ya + ((x0 - xa) / (xb - xa)) * (yb - ya)
    h = xb - xa
    t = (x0 - xa) / h
    t2 = t * t; t3 = t2 * t
    h00 = t3 * 2 - t2 * 3 + 1                     # cubic Hermite basis
    h10 = t3 - t2 * 2 + t
    h01 = t3 * (-2) + t2 * 3
    h11 = t3 - t2
    ma = slope(i); mb = slope(i + 1)
    return h00 * ya + h10 * (h * ma) + h01 * yb + h11 * (h * mb)


def dinterp_linear(x0, xp, yp):
    """Differentiable PIECEWISE-LINEAR interpolation, clamped at the endpoints
    exactly like numpy.interp (xp increasing in value; x0/xp/yp entries each Dual
    or float). This is the linear counterpart of ``dinterp`` and is used ONLY for
    the rough slope_ratio: the GA's feasibility reference computes that ratio with
    np.interp (linear), while ``dinterp`` is cubic-Hermite and reads the design-alpha
    slope ~2-3% high where the rough curve is rounding toward stall -- so Ipopt could
    park the ratio at ~0.50 while the GA scored ~0.47-0.48. Matching the GA's linear
    interpolant here makes the gradient optimise against the SAME slope_ratio the GA
    judges feasibility with. (Only slope_ratio uses this; every other quantity keeps
    the C1 cubic dinterp, which the objective needs for smoothness.)"""
    nx = None
    for s in (x0, *xp, *yp):
        if isinstance(s, Dual):
            nx = s.g.shape[0]
            break
    def prom(y):
        return y if isinstance(y, Dual) or nx is None else Dual(y, np.zeros(nx))
    xv = [_val(x) for x in xp]
    x0v = _val(x0)
    n = len(xv)
    if n == 1 or x0v <= xv[0]:
        return prom(yp[0])
    if x0v >= xv[-1]:
        return prom(yp[-1])
    i = 0
    while i < n - 1 and xv[i + 1] < x0v:
        i += 1
    xa, xb, ya, yb = xp[i], xp[i + 1], yp[i], yp[i + 1]
    return ya + ((x0 - xa) / (xb - xa)) * (yb - ya)


# ======================================================================
# Aero sweep (each swept quantity becomes a list of Duals over z)
# ======================================================================
def _nqfoil_sweep(up, lo, alphas, cond, Re, model, nx=None):
    """Vectorized nqfoil sweep -> the same dict-of-lists-of-Duals as _nf_sweep.

    Uses nqfoil_torch.aero_jac, which hand-propagates forward-mode tangents
    through the MLP: 3.1 ms vs 7.3 ms for torch.func.jacfwd on a 41-alpha sweep
    (warm), agreeing to 8.7e-07. A full clean+rough evaluate costs ~6.7 ms.

    Cpmin is the model's directly trained output (qfoil reports it), unlike the
    NeuralFoil path which derives 1 - max(ue/vinf)^2 from the BL channels.
    """
    from oso_airfoils.optimization import nqfoil_torch as nqt
    up = np.asarray(up, float); lo = np.asarray(lo, float); M = len(up)
    if nx is None:
        nx = 2 * M
    vals, jac = nqt.aero_jac(up, lo, np.asarray(alphas, float),
                             te=cond['te_gap'], Re=Re, n_crit=cond['N_crit'],
                             xtr_u=cond['xtp_u'], xtr_l=cond['xtp_l'],
                             model_size=model, device=cond.get('device', 'cpu'))
    res = {k: [] for k in ('cl', 'cd', 'cm', 'cpmin', 'xtr_top', 'xtr_bot', 'lod')}
    for key, name in (('cl', 'CL'), ('cd', 'CD'), ('cm', 'CM'), ('cpmin', 'Cpmin'),
                      ('xtr_top', 'Top_Xtr'), ('xtr_bot', 'Bot_Xtr')):
        v, g = vals[name], jac[name]
        for j in range(len(v)):
            res[key].append(Dual(float(v[j]), _cg(g[j, :M], g[j, M:], nx)))
    for j in range(len(vals['CL'])):
        res['lod'].append(_safe_lod(res['cl'][j], res['cd'][j]))
    return res


def _tool_grad_fn(tool):
    if tool == 'neuralfoil':
        from metafoil.neuralfoil import kulfan_gradient as f
        def call(up, lo, a, cond, Re, model):
            return f(up, lo, a, Re=Re, N_crit=cond['N_crit'], xtp_u=cond['xtp_u'],
                     xtp_l=cond['xtp_l'], model_size=model, te_gap=cond['te_gap'],
                     grads=('upper', 'lower'))
        return call
    if tool in ('cxfoil', 'cqfoil'):
        if tool == 'cxfoil':
            from metafoil.cxfoil.wrappers import cxfoil_inmem_wrapper as m
        else:
            from metafoil.cqfoil.wrappers import cqfoil_inmem_wrapper as m
        def call(up, lo, a, cond, Re, model):
            return m.kulfan_gradient(up, lo, a, Re=Re, Ncrit=cond['N_crit'],
                                     xtp_u=cond['xtp_u'], xtp_l=cond['xtp_l'],
                                     n_pts=140, te_gap=cond['te_gap'], which=('upper', 'lower'))
        return call
    raise ValueError(f'unknown tool {tool!r}')


def _nf_sweep(up, lo, alphas, cond, Re, model, nx=None):
    """Vectorized NeuralFoil sweep: ONE CasADi graph evaluates cl/cd/cm/cpmin
    and their coefficient-jacobians for the whole alpha grid at once (much
    faster than a per-alpha loop). Returns the same dict-of-lists-of-Duals."""
    from metafoil.neuralfoil.wrappers.neuralfoil_wrapper import (
        _import_real_neuralfoil, _N_BL_STATIONS)
    import casadi as cas
    real_nf = _import_real_neuralfoil()
    up = np.asarray(up, float); lo = np.asarray(lo, float); M = len(up)
    usym = cas.MX.sym('u', M); lsym = cas.MX.sym('l', M)
    kp = dict(upper_weights=usym, lower_weights=lsym,
              leading_edge_weight=0.0, TE_thickness=cond['te_gap'])
    aero = real_nf.get_aero_from_kulfan_parameters(
        kp, alpha=np.asarray(alphas, float), Re=Re, n_crit=cond['N_crit'],
        xtr_upper=cond['xtp_u'], xtr_lower=cond['xtp_l'], model_size=model)
    ue2 = [aero[f'{s}_bl_ue/vinf_{i}'] ** 2
           for s in ('upper', 'lower') for i in range(_N_BL_STATIONS)]
    umax2 = ue2[0]
    for u in ue2[1:]:
        umax2 = cas.fmax(umax2, u)
    outs = {'cl': aero['CL'], 'cd': aero['CD'], 'cm': aero['CM'], 'cpmin': 1.0 - umax2,
            'xtr_top': aero['Top_Xtr'], 'xtr_bot': aero['Bot_Xtr']}
    keys = ('cl', 'cd', 'cm', 'cpmin', 'xtr_top', 'xtr_bot')
    exprs = []
    for k in keys:
        exprs += [outs[k], cas.jacobian(outs[k], usym), cas.jacobian(outs[k], lsym)]
    vals = cas.Function('F', [usym, lsym], exprs)(up, lo)
    na = len(alphas)
    if nx is None: nx = len(up) + len(lo)
    res = {k: [] for k in ('cl', 'cd', 'cm', 'cpmin', 'xtr_top', 'xtr_bot', 'lod')}
    for ki, k in enumerate(keys):
        v = np.asarray(vals[3 * ki]).reshape(-1)
        ju = np.asarray(vals[3 * ki + 1]); jl = np.asarray(vals[3 * ki + 2])
        for j in range(na):
            res[k].append(Dual(float(v[j]), _cg(ju[j, :], jl[j, :], nx)))
    for j in range(na):
        res['lod'].append(res['cl'][j] / res['cd'][j])
    return res


def _safe_lod(cl, cd):
    """cl/cd, but robust to a FAILED aero solve returning cd=0 (cxfoil/cqfoil can do
    this at non-converged post-stall alphas). Those alphas are clamped out of the
    objective anyway; returning lod=0 (worst L/D) marks the point without crashing —
    and if it ever lands pre-stall, the 0 just steers the optimizer away from it."""
    if abs(cd.v) < 1e-12:
        return Dual(0.0, np.zeros_like(cl.g))
    return cl / cd


def aero_sweep(tool, up, lo, alphas, cond, Re, model, nx=None):
    """Return dict of lists-of-Duals: cl, cd, cm, cpmin, lod (= cl/cd), one
    entry per alpha, each carrying d/d[coeffs]. `alphas` is a fixed grid."""
    if nx is None: nx = len(up) + len(lo)
    if tool == 'neuralfoil':
        return _nf_sweep(up, lo, alphas, cond, Re, model, nx)
    if tool == 'nqfoil':
        return _nqfoil_sweep(up, lo, alphas, cond, Re, model, nx)
    call = _tool_grad_fn(tool)
    out = {k: [] for k in ('cl', 'cd', 'cm', 'cpmin', 'lod')}
    for a in alphas:
        g = call(up, lo, float(a), cond, Re, model)
        def D(key):
            return Dual(g[key], _cg(g[f'd{key}_dupper'], g[f'd{key}_dlower'], nx))
        cl, cd, cm, cp = D('cl'), D('cd'), D('cm'), D('cpmin')
        out['cl'].append(cl); out['cd'].append(cd); out['cm'].append(cm)
        out['cpmin'].append(cp); out['lod'].append(_safe_lod(cl, cd))
    return out


# ----------------------------------------------------------------------
# cxfoil/cqfoil ONLY: the complex-step gradient costs one full BL solve per
# coefficient (17 solves/alpha), so computing it at every alpha is the bottleneck.
# But the objective/constraints only DIFFERENTIATE the sweep at a handful of alphas
# (the design point, +/- the L/D-falloff offset, and the stall peak). So: sweep
# VALUES cheaply everywhere (to locate those), then fill the expensive gradient in
# only at the active alphas. NeuralFoil is unaffected (its one CasADi graph already
# gives values + the full 16-wide gradient at every alpha for free).
# ----------------------------------------------------------------------
def _tool_value_fn(tool):
    """Value-only aero call (one forward BL solve, no gradient) for cxfoil/cqfoil."""
    if tool == 'cxfoil':
        from metafoil.cxfoil.wrappers import cxfoil_inmem_wrapper as m
    elif tool == 'cqfoil':
        from metafoil.cqfoil.wrappers import cqfoil_inmem_wrapper as m
    else:
        raise ValueError(f'no value-only path for tool {tool!r}')
    from metafoil.core.kulfan_geometry import kulfan_to_coordinates
    def call(up, lo, a, cond, Re):
        coords = kulfan_to_coordinates(up, lo, n_pts=140, te_gap=cond['te_gap'])
        r = m.run(coords, a, Re=Re, Ncrit=cond['N_crit'], xtp_u=cond['xtp_u'],
                  xtp_l=cond['xtp_l'], itmax=100)
        return {k: r[k] for k in ('cl', 'cd', 'cm', 'cpmin')}
    return call


def _value_sweep(tool, up, lo, alphas, cond, Re, nx=None):
    """Sweep VALUES only (zero-gradient Duals); gradients get filled in later at the
    active alphas by _fill_grads. Fast: one forward solve per alpha, no complex step."""
    if nx is None: nx = 2 * len(up)
    callv = _tool_value_fn(tool); Z = np.zeros(nx)
    out = {k: [] for k in ('cl', 'cd', 'cm', 'cpmin', 'lod')}
    for a in alphas:
        v = callv(up, lo, float(a), cond, Re)
        cl, cd = Dual(v['cl'], Z), Dual(v['cd'], Z)
        out['cl'].append(cl); out['cd'].append(cd)
        out['cm'].append(Dual(v['cm'], Z)); out['cpmin'].append(Dual(v['cpmin'], Z))
        out['lod'].append(_safe_lod(cl, cd))
    return out


def _active_alpha_indices(alphas, pk, targets, include_stall=True):
    """Indices whose GRADIENT the objective/constraints actually use: the two grid
    points bracketing each interpolation target (clamped to <= the stall peak, as the
    `at()` interpolations are), plus (when include_stall) the stall peak neighbours
    pk-1, pk, pk+1 (used by the parabolic stall-alpha, CL_max and lift margin).
    include_stall=False drops the peak block when no enabled constraint needs it
    (stall_margin / reach_design_cl / cl_max) — see evaluate()'s grad-group gating."""
    alphas = np.asarray(alphas, float); n = len(alphas)
    idx = set()
    for t in targets:
        t = float(np.clip(t, alphas[0], alphas[pk]))     # interpolation clamps to pre-stall
        j = int(np.clip(np.searchsorted(alphas, t) - 1, 0, max(pk - 1, 0)))
        idx.add(j); idx.add(min(j + 1, pk))
    if include_stall:
        for i in (pk - 1, pk, pk + 1):
            if 0 <= i < n: idx.add(i)
    return sorted(idx)


def _fill_grads(tool, up, lo, alphas, cond, Re, sweep, idx, nx=None):
    """Overwrite the value-only Duals at alpha indices `idx` with FULL-gradient Duals
    (complex-step). Only these alphas pay the ~17-solve gradient cost."""
    if nx is None: nx = 2 * len(up)
    call = _tool_grad_fn(tool)
    for i in idx:
        g = call(up, lo, float(alphas[i]), cond, Re, None)
        for key in ('cl', 'cd', 'cm', 'cpmin'):
            sweep[key][i] = Dual(g[key], _cg(g[f'd{key}_dupper'], g[f'd{key}_dlower'], nx))
        sweep['lod'][i] = _safe_lod(sweep['cl'][i], sweep['cd'][i])


def _positive_peak_index(cl, mid):
    """Index of the first CL maximum at or above `mid` (the stall peak), as in
    the fitness function's march but without its alpha=0 wrap-around: scan
    forward and return the last index before CL first decreases. If CL never
    decreases in range (no stall captured), return the last index."""
    for i in range(mid + 1, len(cl)):
        if cl[i].v <= cl[i - 1].v:
            return i - 1
    return len(cl) - 1


# ======================================================================
# Geometry: analytic metafoil gradients throughout. The curvature/located
# scalars use closed-form Kulfan zeta-derivatives (linear in the coeffs ->
# constant Jacobians); no finite differences of the coordinate grid.
# ======================================================================
def _kulfan_gradient_dual(k, quantity, nx):
    """Wrap metafoil's exact analytic Kulfan gradient of a scalar property as a
    Dual (value + d/d[upper,lower])."""
    g = k.gradient(quantity)
    val = float(getattr(k, quantity if quantity != 'thickness_ratio' else 'thickness_ratio'))
    return Dual(val, _cg(g['upper'], g['lower'], nx))


def _parabolic_vertex(x0, x1, x2, y0, y1, y2):
    """x-location of the vertex of the parabola through the three points, as a
    Dual (y* may be Dual). Analytic — a smooth stand-in for the discrete argmax
    that carries the exact coefficient gradient of the extremum location."""
    d0 = x1 - x0; d2 = x1 - x2
    num = d0 * d0 * (y1 - y2) - d2 * d2 * (y1 - y0)
    den = d0 * (y1 - y2) - d2 * (y1 - y0)
    return x1 - 0.5 * num / den


def geometry_duals(up, lo, p, nx, psi_star=None):
    """Every geometry scalar the constraints need, as a Dual carrying its EXACT
    analytic gradient. No finite difference: CST zeta is linear in the
    coefficients (dzeta_u/dA_u = C(psi)*B(psi), a constant matrix), so building
    zeta on the psi grid as Dual arrays and evaluating the fitness function's
    geometry math with Dual arithmetic propagates the exact gradient. Integral
    quantities (area, Ixx/Iyy/Izz) and the max-thickness value use metafoil's
    closed-form Kulfan gradients directly; the located extrema (max-thickness x/c,
    per-surface extremum x/c) are exact — psi* is the bisection root of the
    closed-form dzeta/dpsi=0 and dpsi*/dA is the implicit-function-theorem
    gradient (Kulfan.psi_extremum), not a grid parabola."""
    up = np.asarray(up, float); lo = np.asarray(lo, float); M = len(up)
    k = Kulfan(upper_coefficients=up, lower_coefficients=lo, te_gap=p['TE_gap'])
    psi = k.psi
    C = psi ** k.N1 * (1.0 - psi) ** k.N2
    B = np.stack([_binom(M - 1, kk) * psi ** kk * (1.0 - psi) ** (M - 1 - kk)
                  for kk in range(M)], axis=-1)      # (npsi, M) — dzeta/dcoeff basis
    te_u = k.te_shift + k.te_gap / 2.0; te_l = k.te_shift - k.te_gap / 2.0
    zu_v = C * (B @ up) + psi * te_u
    zl_v = C * (B @ lo) + psi * te_l
    npsi = len(psi)
    # zeta as Dual arrays: dzeta_u depends only on A_u, dzeta_l only on A_l
    ZU = [Dual(zu_v[j], _cg(C[j] * B[j], np.zeros(M), nx)) for j in range(npsi)]
    ZL = [Dual(zl_v[j], _cg(np.zeros(M), C[j] * B[j], nx)) for j in range(npsi)]
    T = [ZU[j] - ZL[j] for j in range(npsi)]         # thickness Duals
    G = {}

    # self-intersection guard: min thickness over the INTERIOR grid (the LE and
    # TE endpoints are pinned to ~0 / the TE gap, so exclude them). Envelope
    # gradient (the thickness at the argmin station).
    interior = [j for j in range(npsi) if 1e-6 < psi[j] < 1.0 - 1e-6]
    G['hmin'] = T[interior[int(np.argmin([T[j].v for j in interior]))]]
    G['toothpick_h'] = dinterp(p['toothpick_location'], list(psi), T)

    # tau (max thickness) value + gradient: metafoil's exact closed form. The
    # extremum LOCATIONS are exact too: psi* is the bisection root of the
    # closed-form surface derivative dzeta/dpsi = 0 (not a grid argmax / parabola)
    # and dpsi*/dA is the implicit-function-theorem gradient (all analytic).
    def _loc_dual(kind):
        ps, du, dl = k.psi_extremum(kind)
        return Dual(ps, _cg(du, dl, nx))
    G['tau'] = _kulfan_gradient_dual(k, 'thickness_ratio', nx)
    G['taumax_psi'] = _loc_dual('thickness')                    # overall max-thickness x/c
    G['taumax_psi_upper'] = _loc_dual('upper')                  # upper-surface max-height x/c
    G['taumax_psi_lower'] = _loc_dual('lower')                  # lower-surface min-height x/c

    # curvature: the ANALYTIC second derivative d2(zeta)/dpsi2 (Kulfan.curvature
    # measure), NOT a finite difference of zeta on the grid. Second-differencing
    # the cosine grid blows up ~1e5 at the leading edge (dpsi -> 0); the closed
    # form is smooth (dzeta''/dcoeff = C''B + 2C'B' + C B'', a constant matrix).
    Z0 = const(0.0, nx)
    N1, N2 = k.N1, k.N2
    with np.errstate(divide='ignore', invalid='ignore'):
        Cp = N1 * psi ** (N1 - 1) * (1 - psi) ** N2 - N2 * psi ** N1 * (1 - psi) ** (N2 - 1)
        Cpp = (N1 * (N1 - 1) * psi ** (N1 - 2) * (1 - psi) ** N2
               - 2 * N1 * N2 * psi ** (N1 - 1) * (1 - psi) ** (N2 - 1)
               + N2 * (N2 - 1) * psi ** N1 * (1 - psi) ** (N2 - 2))
    n = M - 1
    Bp = np.zeros((npsi, M)); Bpp = np.zeros((npsi, M))
    for kk in range(M):
        b = _binom(n, kk); t = np.zeros(npsi); t2 = np.zeros(npsi)
        if kk >= 1:            t += kk * psi ** (kk - 1) * (1 - psi) ** (n - kk)
        if kk <= n - 1:        t -= (n - kk) * psi ** kk * (1 - psi) ** (n - kk - 1)
        if kk >= 2:            t2 += kk * (kk - 1) * psi ** (kk - 2) * (1 - psi) ** (n - kk)
        if 1 <= kk <= n - 1:   t2 -= 2 * kk * (n - kk) * psi ** (kk - 1) * (1 - psi) ** (n - kk - 1)
        if kk <= n - 2:        t2 += (n - kk) * (n - kk - 1) * psi ** kk * (1 - psi) ** (n - kk - 2)
        Bp[:, kk] = b * t; Bpp[:, kk] = b * t2
    J2 = Cpp[:, None] * B + 2.0 * Cp[:, None] * Bp + C[:, None] * Bpp   # d(zeta'')/dcoeff
    # d2zeta VALUES straight from the analytic Jacobian (zeta'' is linear in the coeffs, so
    # zeta'' = J2 @ coeffs exactly -- verified == scalar k.d2zeta_dpsi2 to 3e-11). This drops
    # the per-station scalar Kulfan curvature calls (a chunk of geometry_duals' 94.6% cost).
    d2u_v = J2 @ up; d2l_v = J2 @ lo
    ec = p['ec_cutoff']; cb = const(p['curvature_bound'], nx)
    concave, aft = [], []
    # Per-station LINEAR curvature Duals (d2zeta is linear in the coeffs -> constant,
    # never-vanishing Jacobian rows). These are imposed INDIVIDUALLY in constraint_list
    # instead of as one relu-sum: an aggregate sum(relu(.)) collapses to value 0 AND
    # gradient 0 when satisfied, which drops its Jacobian row to zero -> LICQ/constraint-
    # qualification failure -> singular KKT system -> Ipopt divergence. The per-station
    # linear form keeps the active-set Jacobian full-rank and well-conditioned.
    concave_terms, aft_terms = [], []
    for j in interior:                    # interior only (psi=0 is the LE singularity)
        d2u = Dual(d2u_v[j], _cg(J2[j], np.zeros(M), nx))
        concave.append(dmax(d2u, Z0))     # positive curvature = upper-surface concavity
        concave_terms.append(Z0 - d2u)    # -d2u >= 0  <=>  no upper-surface concavity
        if psi[j] >= ec:
            # upper-surface aft (trailing-edge) over-curvature: how far d2zeta_upper
            # dips BELOW the lower bound curvature_bound. relu(cb - d2u), summed over
            # the aft region; =0 iff d2zeta_upper >= curvature_bound everywhere aft.
            # NOTE the analytic d2zeta is O(1-10) here, NOT the ~O(1e2-1e4) of the GA's
            # grid 2nd-difference, so curvature_bound must be re-set accordingly.
            aft.append(dmax(cb - d2u, Z0))
            aft_terms.append(d2u - cb)    # d2u - curvature_bound >= 0 (per aft station)
    G['concave_sum'] = dsum(concave, nx)
    G['aftcurve_sum'] = dsum(aft, nx) if aft else Z0
    G['concave_terms'] = concave_terms
    G['aft_terms'] = aft_terms
    # lower-surface curvature Duals d2(zeta_lower)/dpsi2 on the interior grid,
    # linear in the coefficients; consumed by the lower_signflip constraint below.
    D2L = [Dual(d2l_v[j], _cg(np.zeros(M), J2[j], nx)) for j in interior]

    # Lower-surface "curvature changes sign AT MOST ONCE", exact version, via a FREE
    # split location psi* (an auxiliary optimization variable, not a coefficient).
    # A function has <=1 inflection iff there is a split psi* with the curvature
    # convex (d2zeta_lower >= 0) before it and concave (<= 0) after it. We impose
    # that convex-then-concave shape per station, gated by a sigmoid at psi*:
    #     g_j = sigmoid((psi* - psi_j)/w)   (~1 before psi*, ~0 after)
    #     convex-before :   d2l_j + M(1 - g_j) >= 0   (active where psi_j < psi*)
    #     concave-after : (-d2l_j) + M(g_j)     >= 0   (active where psi_j > psi*)
    # d2l is linear in the coeffs and psi* enters only through the smooth gate, so
    # both families are smooth with never-vanishing gradients (full-rank Jacobian).
    # Because psi* is FREE, Ipopt slides it to wherever a single sign change can sit;
    # a surface needing TWO sign changes has no feasible psi* -> correctly rejected.
    if psi_star is not None:
        # Product form (no big-M): a smooth target sign t_j = 2*sigmoid((psi*-psi_j)/w)-1
        # is ~+1 before psi* and ~-1 after. The single per-station constraint
        #     t_j * d2zeta_lower(psi_j) >= 0
        # says "curvature sign agrees with the target": convex (d2l>=0) before psi*,
        # concave (d2l<=0) after. At the split (t_j~0) the station is free. A surface
        # needing two sign changes has a station whose curvature fights the monotone
        # target for EVERY psi* -> infeasible. This avoids a big-M (whose magnitude
        # would have to exceed the near-LE curvature singularity ~psi^-1.5, ruining
        # the conditioning). The LE region is excluded: d2l there is huge and always
        # convex (the class function forces it), so it can only carry the leading '+'
        # sign, never an extra flip.
        inv_ws = 1.0 / p.get('lower_signflip_width', 0.03)   # FIX 2(b): sharper gate (was 0.05) reduces the surrogate leak
        le_cut = p.get('lower_signflip_le_cutoff', 0.10)   # crossover never sits fwd of ~10% chord
        sf_terms = []
        for d2l, j in zip(D2L, interior):
            if psi[j] < le_cut:
                continue
            tj = dsigmoid((psi_star - psi[j]) * inv_ws) * 2.0 - 1.0    # ~+1 before psi*, ~-1 after
            sf_terms.append(tj * d2l)                                  # >= 0  <=>  sign matches target
        G['lower_signflip_terms'] = sf_terms
    else:
        G['lower_signflip_terms'] = []

    # TE cone violation (all Dual ops; only active-side clearances accumulate)
    tf = p['te_frac']; tan = math.tan(p['cone_angle'] / 2 / 180 * math.pi)
    mid = (dinterp(tf, list(psi), ZU) + dinterp(tf, list(psi), ZL)) * 0.5
    up_cone = mid + tan * (1 - tf); lo_cone = mid - tan * (1 - tf)
    tev = Z0; te_cone_terms = []
    for j, pv in enumerate(psi):
        if pv >= tf:
            hu = up_cone - up_cone * ((pv - tf) / (1 - tf)) + ZU[-1]
            hl = lo_cone - lo_cone * ((pv - tf) / (1 - tf)) + ZL[-1]
            if ZU[j].v < hu.v: tev = tev + (hu - ZU[j])
            if ZL[j].v > hl.v: tev = tev + (ZL[j] - hl)
            # per-station cone clearances (smooth, linear in the coeffs): upper surface
            # stays above the cone's lower edge, lower surface below the upper edge.
            # Skip the ψ=1 apex: there hu==ZU[-1]==ZU[j] identically -> a 0==0 row with
            # zero gradient (LICQ-breaking); the TE point is trivially on the cone.
            if pv < 1.0 - 1e-9:
                te_cone_terms.append(ZU[j] - hu)
                te_cone_terms.append(hl - ZL[j])
    G['te_cone'] = tev
    G['te_cone_terms'] = te_cone_terms

    # structural integrals + LE radius: metafoil closed-form gradients
    for name in ('area', 'Ixx', 'Iyy', 'Izz'):
        G[name] = _kulfan_gradient_dual(k, name, nx)
    lu, ll = k.leading_edge_radius()
    du = np.zeros(nx); du[0] = up[0]
    dl = np.zeros(nx); dl[M] = lo[0]
    G['ler_u'] = Dual(float(lu), du); G['ler_l'] = Dual(float(ll), dl)

    # min radius-of-curvature location (only if the params request it; skipped
    # for t30 where both targets are null). Parabolic vertex of the ROC minimum.
    # ROC = (1 + zeta'^2)^1.5 / |zeta''| is built from the ANALYTIC first and second
    # derivatives of zeta (both linear in the coefficients, so their per-coefficient
    # Jacobians J1/J2 are constant matrices), NOT finite differences of the coordinate
    # grid. This matches the concave/aft/signflip curvature terms above and the (now
    # analytic) original OSO objective, and makes the gradient exact rather than FD.
    if (p.get('min_radius_location_upper') is not None
            or p.get('min_radius_location_lower') is not None):
        J1 = Cp[:, None] * B + C[:, None] * Bp            # d(dzeta/dpsi)/dcoeff (constant)
        d1u_v = J1 @ up + te_u; d1l_v = J1 @ lo + te_l    # dzeta/dpsi values (+ TE slope)
        pc = psi[1:-1]
        for side, d1v, d2v, is_u in (('u', d1u_v, d2u_v, True),
                                     ('l', d1l_v, d2l_v, False)):
            roc = []
            for j in range(1, npsi - 1):                 # interior stations = pc grid
                if is_u:
                    d1 = Dual(d1v[j], _cg(J1[j], np.zeros(M), nx))
                    d2 = Dual(d2v[j], _cg(J2[j], np.zeros(M), nx))
                else:
                    d1 = Dual(d1v[j], _cg(np.zeros(M), J1[j], nx))
                    d2 = Dual(d2v[j], _cg(np.zeros(M), J2[j], nx))
                roc.append((1 + d1 * d1) ** 1.5 / dabs(d2))
            cand = [j for j in range(len(roc)) if pc[j] <= p['min_radius_location_cutoff']]
            jm = min(cand, key=lambda j: roc[j].v)
            if cand[0] < jm < cand[-1]:
                G[f'minrad_loc_{side}'] = _parabolic_vertex(
                    pc[jm - 1], pc[jm], pc[jm + 1], roc[jm - 1], roc[jm], roc[jm + 1])
            else:
                G[f'minrad_loc_{side}'] = const(pc[jm], nx)
    else:
        G['minrad_loc_u'] = const(0.0, nx); G['minrad_loc_l'] = const(0.0, nx)
    return G


# ======================================================================
# Curvature-acceleration envelope constraint (differentiable). |d2kappa/ds2(x_i)|
# <= E_frozen(x_i) at fixed stations x_i>=XSTART, both surfaces. d2kappa/ds2 is the
# EXACT analytic closed form (grid-independent) built from zeta'..zeta'''' (each
# LINEAR in the coeffs), assembled here as Duals -> full-rank analytic Jacobian.
# Returns per-station two-sided term pairs (E - d2k) and (E + d2k) so |d2k|<=E is a
# pair of smooth inequalities (no non-smooth abs). Shares the frozen envelope and
# the Jk basis with the GA via curvature_envelope, so the two match exactly.
# ======================================================================
def curvature_accel_terms(up, lo, p, nx):
    from oso_airfoils.optimization import curvature_envelope as ce
    T = ce.tau_to_T(p['tau'])
    Etab = ce.E_TAB.get(T)
    if Etab is None:
        return [], []
    te = p.get('TE_gap', 0.0)
    stride = int(p.get('curvature_accel_stride', 2) or 1)
    sidx = ce.STATION_IDX[::stride]
    J1, J2, J3, J4 = ce.J1G, ce.J2G, ce.J3G, ce.J4G
    hi, lo_t = [], []
    for coeffs, te_off, is_up, Earr in ((up, +te / 2.0, True, Etab['upper']),
                                        (lo, -te / 2.0, False, Etab['lower'])):
        A = np.asarray(coeffs, float); M = len(A)
        z1v = J1 @ A + te_off; z2v = J2 @ A; z3v = J3 @ A; z4v = J4 @ A
        for i in sidx:
            def _d(val, row):
                return Dual(float(val), _cg(row, np.zeros(M), nx) if is_up
                            else _cg(np.zeros(M), row, nx))
            z1 = _d(z1v[i], J1[i]); z2 = _d(z2v[i], J2[i])
            z3 = _d(z3v[i], J3[i]); z4 = _d(z4v[i], J4[i])
            P = 1.0 + z1 * z1
            kp = z3 * P ** -1.5 - 3.0 * z1 * z2 * z2 * P ** -2.5
            kpp = (z4 * P ** -1.5
                   - (9.0 * z1 * z2 * z3 + 3.0 * (z2 * z2 * z2)) * P ** -2.5
                   + 15.0 * z1 * z1 * (z2 * z2 * z2) * P ** -3.5)
            d2 = kpp / P - kp * z1 * z2 / (P * P)
            E = float(Earr[i])
            hi.append(E - d2)      # E - d2kappa >= 0
            lo_t.append(E + d2)    # E + d2kappa >= 0  (two-sided |d2k| <= E)
    return hi, lo_t


def bulge_terms(lo, p, nx):
    """Lower-surface secondary-bulge guard as Duals. For each mid-band station j:
        (|kappa(2%)| + margin) - |kappa(x_j)| >= 0
    i.e. lower-surface curvature may not climb back into a rising hump aft of the 2%
    shoulder. Relative -> thickness-agnostic; slack unless the section is bulging. Shares
    the exact reference/band/margin with curvature_envelope.bulge_violation (the GA path)."""
    from oso_airfoils.optimization import curvature_envelope as ce
    te = p.get('TE_gap', 0.0); te_off = -te / 2.0
    margin = float(p.get('bulge_margin', ce.BULGE_MARGIN))
    J1, J2 = ce._BJ1, ce._BJ2
    A = np.asarray(lo, float); M = len(A)
    z1v = J1 @ A + te_off; z2v = J2 @ A

    def kap(i):                              # |kappa| = |zeta''| / (1+zeta'^2)^1.5 (lower surface)
        z1 = Dual(float(z1v[i]), _cg(np.zeros(M), J1[i], nx))
        z2 = Dual(float(z2v[i]), _cg(np.zeros(M), J2[i], nx))
        P = 1.0 + z1 * z1
        return dabs(z2) * P ** -1.5

    kref = kap(0)                            # reference at the 2% shoulder
    return [(kref + margin) - kap(i) for i in range(1, len(z1v))]   # each >= 0 feasible


# ======================================================================
# The objective (faithful transcription of core_fitness_function)
# ======================================================================
def make_context(tool='neuralfoil', model_size='small', params=None, enabled=None):
    """Bundle the fixed problem data (params dict + swept alpha grids). `enabled`
    is the optional {constraint_group: bool} dict (same one passed to
    constraint_list); when given, cxfoil/cqfoil fill the complex-step gradient only
    at the alphas the ENABLED constraints need (see evaluate's grad-group gating).
    Leave it None to differentiate every group (the original behaviour)."""
    p = dict(params)
    p.setdefault('TE_gap', 0.0)
    clean = dict(N_crit=p['N_crit_clean'], xtp_u=p['xtp_u_clean'], xtp_l=p['xtp_l_clean'],
                 te_gap=p['TE_gap'])
    rough = dict(N_crit=p['N_crit_rough'], xtp_u=p['xtp_u_rough'], xtp_l=p['xtp_l_rough'],
                 te_gap=p['TE_gap'])
    a_clean = np.arange(p['alpha_min_clean'], p['alpha_max_clean'] + p['alpha_step_clean'],
                        p['alpha_step_clean'], dtype=float)
    a_rough = np.arange(p['alpha_min_rough'], p['alpha_max_rough'] + p['alpha_step_rough'],
                        p['alpha_step_rough'], dtype=float)
    return dict(tool=tool, model=model_size, p=p, clean=clean, rough=rough,
                a_clean=a_clean, a_rough=a_rough, enabled=enabled)

# ======================================================================
# Evaluate: one cached sweep + geometry pass -> every objective/constraint
# quantity as a Dual. The gradient NLP uses HARD CONSTRAINTS (not penalties):
# the fitness function's penalty terms become Ipopt inequality/equality
# constraints, plus feasibility guards (non-intersection, reach design CL).
# ======================================================================
class Result:
    """Container of Dual quantities from one evaluate() call, plus the scalar
    diagnostics. Attribute access returns Duals."""
    def __init__(self, R, diag):
        self.__dict__.update(R)
        self.diag = diag


def evaluate(z, ctx):
    """Run the clean+rough sweeps and geometry once; return a Result of Duals
    for the objectives and every constraint quantity."""
    p = ctx['p']
    # The design vector is [upper_coeffs(M), lower_coeffs(M), aux(n_aux)]. n_aux>0
    # appends auxiliary optimization variables that are NOT coefficients (currently
    # just psi_star, the free curvature-sign-flip split location for the lower
    # surface). nx is the FULL gradient dimension; every Dual carries length nx,
    # with the trailing aux slots zero for coefficient-only quantities.
    n_aux = ctx.get('n_aux', 0)
    nx = len(z); ncoef = nx - n_aux; M = ncoef // 2
    up = np.asarray(z[:M], float); lo = np.asarray(z[M:ncoef], float)
    # aux: free split location psi_star (a genuine decision variable -> identity
    # gradient in its own slot; the objective does not depend on it, only the
    # lower-surface sign-flip constraints do).
    psi_star = None
    if n_aux >= 1:
        psi_star = Dual(float(z[ncoef]), _onehot(ncoef, nx))
    Re = p['Re']; CLd = p['CL']

    tool = ctx['tool']
    full_grad = (tool in ('neuralfoil', 'nqfoil') or p.get('full_aero_grad', False))
    ac, ar = list(ctx['a_clean']), list(ctx['a_rough'])
    # NeuralFoil: one vectorized graph gives values + full gradient cheaply.
    # cxfoil/cqfoil: gradient is ~17 BL solves/alpha, so (unless full_grad is
    # forced) sweep VALUES only, then fill the complex-step gradient in ONLY at the
    # alphas the objective/constraints differentiate (done AFTER any extension).
    def _sweep(cond, alphas):
        return (aero_sweep(tool, up, lo, alphas, cond, Re, ctx['model'], nx) if full_grad
                else _value_sweep(tool, up, lo, alphas, cond, Re, nx))
    _mid = lambda a: int(np.argmin(np.abs(a)))

    S = {'c': _sweep(ctx['clean'], ac), 'r': _sweep(ctx['rough'], ar)}
    midc, midr = _mid(ac), _mid(ar)
    pkc = _positive_peak_index(S['c']['cl'], midc)
    pkr = _positive_peak_index(S['r']['cl'], midr)

    # On-demand DOWNWARD extension. If the design CL is below the clean sweep's
    # starting CL, the design point sits at NEGATIVE alpha (off the low end of the
    # sweep). Sweep back from the current alpha_min down to alpha_min_extend (using
    # the clean/rough step) and PREPEND to BOTH the clean and rough sweeps, so every
    # downstream polar quantity (a_des, L/D, falloff, delta_cl, cpmin, ...) uses the
    # extended data. Only triggered when needed; in-range designs skip it entirely.
    a_min_ext = p.get('alpha_min_extend', -10.0)
    if CLd < S['c']['cl'][0].v and ac[0] > a_min_ext + 1e-9:
        e_c = list(np.arange(a_min_ext, ac[0] - 1e-9, p['alpha_step_clean']))
        e_r = list(np.arange(a_min_ext, ar[0] - 1e-9, p['alpha_step_rough']))
        Ec, Er = _sweep(ctx['clean'], e_c), _sweep(ctx['rough'], e_r)
        for k in S['c']: S['c'][k] = Ec[k] + S['c'][k]
        for k in S['r']: S['r'][k] = Er[k] + S['r'][k]
        ac, ar = e_c + ac, e_r + ar
        midc, midr = _mid(ac), _mid(ar)
        pkc = _positive_peak_index(S['c']['cl'], midc)
        pkr = _positive_peak_index(S['r']['cl'], midr)

    # design CL must be bracketed by the (possibly extended) clean pre-stall sweep;
    # otherwise refuse to guess (dinterp would silently clamp) -- see DesignPointError.
    cl_pre = S['c']['cl'][:pkc + 1]
    cl_lo, cl_hi = cl_pre[0].v, cl_pre[pkc].v
    if not (cl_lo <= CLd <= cl_hi):
        raise DesignPointError(
            f"design CL={CLd:.3f} is outside the swept clean-CL range "
            f"[{cl_lo:.3f}, {cl_hi:.3f}] over alpha={ac[0]:.2f}..{ac[pkc]:.2f} deg "
            f"(extended down to alpha_min_extend={a_min_ext:.1f} deg if it helped). "
            f"The design point cannot be located: lower alpha_min_extend, or the "
            f"airfoil's CL never reaches {CLd:.3f} in a usable range.")

    # cxfoil/cqfoil: fill the complex-step gradient at the active alphas of the
    # final (possibly extended) grid.
    # value_only: skip the expensive complex-step gradient fill (cxfoil/cqfoil) and
    # return VALUES only, with zero-gradient aero Duals. Used by the multifidelity
    # trust-region driver (notebook 09), which needs cheap high-fidelity VALUES for
    # the correction/ratio and never differentiates the high-fidelity solver.
    if not full_grad and not ctx.get('value_only', False):
        # Constraint-aware gradient fill: only pay the ~16-solve complex-step gradient
        # at the alphas an ENABLED objective/constraint actually differentiates. With
        # no `enabled` set every group is on (== the original fixed behaviour); passing
        # ctx['enabled'] (e.g. the TRMM's CONSTRAINTS dict) prunes the groups whose
        # constraint is off — e.g. cpmin off => skip its 16-solve argmin fill entirely.
        en = ctx.get('enabled') or {}
        def _on(label):
            return en.get(label, True)
        a_des_v = float(np.interp(CLd, [d.v for d in cl_pre], ac[:pkc + 1]))
        off = p['alpha_falloff_offset']
        targets = [a_des_v]                                          # design point (objective, roughness_delta_cl, rough floor)
        if _on('lod_falloff'):
            targets += [a_des_v - off, a_des_v + off]                # L/D falloff +/- offset
        if _on('cl_target') and p.get('target_alpha') is not None:
            targets.append(float(p['target_alpha']))
        if _on('slope_ratio'):
            # rough lift-slope ratio needs the rough zero-lift & rough design alphas (and
            # their +/-h finite-difference neighbours) gradient-filled. Add them as targets
            # so _active_alpha_indices brackets them on the rough sweep.
            _rclv = [d.v for d in S['r']['cl'][:pkr + 1]]
            _h0 = p.get('slope_ratio_h', 0.5)
            _a0v = float(np.interp(0.0, _rclv, ar[:pkr + 1]))
            _adv = float(np.interp(CLd, _rclv, ar[:pkr + 1]))
            targets += [_a0v - _h0, _a0v, _a0v + _h0, _adv - _h0, _adv, _adv + _h0]
        inc_stall = _on('stall_margin') or _on('reach_design_cl')    # stall-peak block (parabolic stall, CL_max, lift margin)
        idx_c = set(_active_alpha_indices(ac, pkc, targets, inc_stall))
        idx_r = set(_active_alpha_indices(ar, pkr, targets, inc_stall))
        if _on('cpmin'):
            idx_c.add(int(np.argmin([d.v for d in S['c']['cpmin'][:pkc + 1]])))  # pre-stall cpmin argmin
            idx_r.add(int(np.argmin([d.v for d in S['r']['cpmin'][:pkr + 1]])))
        _fill_grads(tool, up, lo, ac, ctx['clean'], Re, S['c'], sorted(idx_c), nx)
        _fill_grads(tool, up, lo, ar, ctx['rough'], Re, S['r'], sorted(idx_r), nx)

    G = geometry_duals(up, lo, p, nx, psi_star=psi_star)
    R = {}

    # design point: alpha where clean CL = CL_design (bracketing already ensured
    # above, after any on-demand downward sweep extension).
    a_des = dinterp(CLd, S['c']['cl'][:pkc + 1], ac[:pkc + 1])
    R['alpha_design'] = a_des

    def at(tag, key, alpha, pk):
        alphas = ac if tag == 'c' else ar
        return dinterp(alpha, alphas[:pk + 1], S[tag][key][:pk + 1])

    # ---- objectives -------------------------------------------------------
    R['lod_clean'] = at('c', 'lod', a_des, pkc)
    R['lod_rough'] = at('r', 'lod', a_des, pkr)
    # drag at the design point. Clean CL is pinned to CL_design at a_des, so
    # minimizing cd_clean is exactly maximizing clean L/D but with a BOUNDED
    # objective (cd >= 0) — no runaway into NeuralFoil's unphysical low-drag
    # regions, which makes Ipopt dramatically more robust than maximizing L/D.
    R['cd_clean'] = at('c', 'cd', a_des, pkc)
    R['cd_rough'] = at('r', 'cd', a_des, pkr)

    # ---- stall peak location: parabolic-refine so the stall-margin constraint
    # has a SMOOTH gradient (the raw grid argmax location is discrete -> zero
    # gradient, which stalls the constraint). CL_max keeps the exact grid-point
    # gradient (the peak sits within a grid step of the maximum).
    def a_peak(tag, pk):
        a = ac if tag == 'c' else ar; cl = S[tag]['cl']
        if 0 < pk < len(cl) - 1:
            return _parabolic_vertex(a[pk - 1], a[pk], a[pk + 1], cl[pk - 1], cl[pk], cl[pk + 1])
        return const(a[pk], nx)

    # ---- aero constraint quantities --------------------------------------
    R['cl_max_clean'] = S['c']['cl'][pkc]
    R['cl_max_rough'] = S['r']['cl'][pkr]
    R['stall_margin_clean'] = a_peak('c', pkc) - a_des
    R['stall_margin_rough'] = a_peak('r', pkr) - a_des
    R['lift_margin_clean'] = S['c']['cl'][pkc] - CLd
    # delta_cl reconciled to the GA's LINEAR-interp value: the GA reads the rough CL at
    # design with np.interp, while cubic dinterp here sits ~1e-3 lower, so the gradient
    # parked its high-clean tip ~2e-3 past the GA's roughness_delta_cl bound (GA-infeasible).
    # Linear interp (matching the GA) closes that gap -- same reconciliation as slope_ratio.
    R['delta_cl_pct'] = (CLd - dinterp_linear(a_des, ar[:pkr + 1], S['r']['cl'][:pkr + 1])) / CLd
    # rough lift-slope ratio (surrogate-trust): slope(dCL/dalpha) @ design / @ zero-lift on
    # the ROUGH polar. A curve that has rounded off (lost attached slope) before design =>
    # low ratio => the tool is extrapolating through incipient separation (a real test would
    # likely stall earlier). Reference-free: anchors are the two unique points CL=0 and CL=CLd,
    # found by the same dinterp inverse-interp as a_des; slopes are differentiable finite
    # differences on the same interpolant. Needs the rough sweep to bracket zero-lift
    # (alpha_min_rough ~ -8; set in the driver).
    _srh = p.get('slope_ratio_h', 0.5)
    # slope_ratio is reconciled to the GA's discrete-polar value: use LINEAR interp
    # (matching the GA's np.interp) for BOTH the CL=0 / CL=design anchors AND the
    # +/-h slope samples, so the gradient can't park ~2-3% optimistic against a
    # cubic-Hermite slope the GA never sees. Every other quantity keeps cubic dinterp.
    _rcl = S['r']['cl'][:pkr + 1]; _rar = ar[:pkr + 1]
    _at_lin = lambda a: dinterp_linear(a, _rar, _rcl)          # rough CL(alpha), linear
    _a0r = dinterp_linear(0.0, _rcl, _rar)                     # rough zero-lift alpha (linear inverse-interp)
    _adr = dinterp_linear(CLd, _rcl, _rar)                     # rough design alpha   (linear inverse-interp)
    _m0 = (_at_lin(_a0r + _srh) - _at_lin(_a0r - _srh)) / (2 * _srh)
    _md = (_at_lin(_adr + _srh) - _at_lin(_adr - _srh)) / (2 * _srh)
    R['rough_slope_ratio'] = _md / dmax(_m0, const(0.02, nx))
    off = p['alpha_falloff_offset']
    R['falloff_c_l'] = (at('c', 'lod', a_des - off, pkc) - R['lod_clean']) / R['lod_clean']
    R['falloff_c_r'] = (at('c', 'lod', a_des + off, pkc) - R['lod_clean']) / R['lod_clean']
    R['falloff_r_l'] = (at('r', 'lod', a_des - off, pkr) - R['lod_rough']) / R['lod_rough']
    R['falloff_r_r'] = (at('r', 'lod', a_des + off, pkr) - R['lod_rough']) / R['lod_rough']
    # SECOND L/D-falloff at a wider offset (default 2 deg), imposed as an ADDITIONAL
    # constraint. Kept separate from alpha_falloff_offset because the family yaml hard-sets
    # that to 1.0 (clobbering any default); alpha_falloff_offset_2 is not in the yaml, so it
    # survives. Purpose: hold L/D flat further below design (out to CL(a_des-2deg)) so the
    # laminar-bucket lower knee can't perch right under the operating point.
    off2 = p.get('alpha_falloff_offset_2', None)
    if off2 is not None:
        R['falloff2_c_l'] = (at('c', 'lod', a_des - off2, pkc) - R['lod_clean']) / R['lod_clean']
        R['falloff2_c_r'] = (at('c', 'lod', a_des + off2, pkc) - R['lod_clean']) / R['lod_clean']
        R['falloff2_r_l'] = (at('r', 'lod', a_des - off2, pkr) - R['lod_rough']) / R['lod_rough']
        R['falloff2_r_r'] = (at('r', 'lod', a_des + off2, pkr) - R['lod_rough']) / R['lod_rough']
    # transition-slope: forward march of the clean upper transition point over the
    # +toff-deg window above design. Large => design parked on a laminar cliff.
    if 'xtr_top' in S['c']:
        toff = p.get('xtr_slope_offset', 1.0)
        R['xtr_slope_c'] = at('c', 'xtr_top', a_des, pkc) - at('c', 'xtr_top', a_des + toff, pkc)
    # rough transition location: the rough case FORCES transition at xtp=0.05, so a reported
    # Top/Bot_Xtr > 0.05 means the solver numerically re-laminarized past the trip (an
    # xfoil-family artifact) -- fake laminar drag in a 'rough' polar. Sampled at design AND at
    # every L/D-falloff sample alpha (+-off, +-off2), so the exploit can't be shifted to a
    # nearby alpha that feeds the falloff L/D. Capped in constraint_list.
    if 'xtr_top' in S['r']:
        _off = p['alpha_falloff_offset']; _off2 = p.get('alpha_falloff_offset_2')
        _xoffs = [0.0, -_off, _off] + ([-_off2, _off2] if _off2 is not None else [])
        R['xtr_top_r_caps'] = [at('r', 'xtr_top', a_des + _da, pkr) for _da in _xoffs]
        R['xtr_top_r'] = R['xtr_top_r_caps'][0]
        if 'xtr_bot' in S['r']:
            R['xtr_bot_r_caps'] = [at('r', 'xtr_bot', a_des + _da, pkr) for _da in _xoffs]
            R['xtr_bot_r'] = R['xtr_bot_r_caps'][0]
    # moment constraint: clean CM must stay >= -|CMc_min| across a +-cm_alpha_band window
    # around design (matches the GA's 'min CM over the band' via per-alpha point constraints).
    if p.get('CMc_min') is not None:
        _cmb = float(p.get('cm_alpha_band', 5.0))
        _cmoffs = list(np.arange(-_cmb, _cmb + 1e-9, 1.0))
        R['cm_clean_band'] = [at('c', 'cm', a_des + _da, pkc) for _da in _cmoffs]
        if p.get('CMr_min') is not None:
            R['cm_rough_band'] = [at('r', 'cm', a_des + _da, pkr) for _da in _cmoffs]
    # lift-curve linearity: slope[a_des-lo2 -> a_des-1] vs slope[a_des-1 -> a_des];
    # equal (within tol) => straight through design (clean AND rough).
    lo2 = p.get('cl_linearity_offset', 2.0)
    def _lin_disc(tag, pk):
        c0 = at(tag, 'cl', a_des, pk); c1 = at(tag, 'cl', a_des - 1.0, pk)
        c2 = at(tag, 'cl', a_des - lo2, pk)
        s_lo = c1 - c2; s_hi = c0 - c1              # lift slopes over the two 1-deg intervals
        # (s_lo - s_hi)/s_lo = fractional slope mismatch, but with s_lo FLOORED. This is both
        # crash-safe (no ZeroDivisionError when a probed airfoil has a flat segment, s_lo->0)
        # AND properly scaled: the earlier division-free form s_hi-(1+-tol)*s_lo is ~0.002 at
        # the boundary (slopes ~0.1/deg) -- below Ipopt's constr_viol_tol=1e-3, so it had NO
        # TEETH (5-6% violations read as feasible). Normalizing makes the constraint value the
        # true fractional mismatch (~tol at the boundary). floor 0.02 << normal s_lo (~0.1),
        # so the dmax kink is never hit in practice.
        return (s_lo - s_hi) / dmax(s_lo, const(0.02, nx))
    R['cl_lin_disc_c'] = _lin_disc('c', pkc)
    R['cl_lin_disc_r'] = _lin_disc('r', pkr)
    R['cpmin_c_des'] = at('c', 'cpmin', a_des, pkc)
    R['cpmin_r_des'] = at('r', 'cpmin', a_des, pkr)
    if p.get('target_alpha') is not None:
        R['cl_target_rough'] = at('r', 'cl', float(p['target_alpha']), pkr)
    # pre-stall cpmin minimum
    cpm_c = S['c']['cpmin'][0]
    for i in range(1, pkc + 1): cpm_c = dmin(cpm_c, S['c']['cpmin'][i])
    cpm_r = S['r']['cpmin'][0]
    for i in range(1, pkr + 1): cpm_r = dmin(cpm_r, S['r']['cpmin'][i])
    R['cpmin_c_prestall'] = cpm_c; R['cpmin_r_prestall'] = cpm_r

    # ---- geometry constraint quantities ----------------------------------
    for name in ('tau', 'area', 'Ixx', 'Iyy', 'Izz', 'ler_u', 'ler_l',
                 'taumax_psi', 'taumax_psi_upper', 'taumax_psi_lower',
                 'te_cone', 'concave_sum', 'aftcurve_sum',
                 'concave_terms', 'aft_terms', 'te_cone_terms',
                 'lower_signflip_terms',
                 'minrad_loc_u', 'minrad_loc_l',
                 'hmin', 'toothpick_h'):
        R[name] = G[name]

    # curvature-acceleration envelope terms (only when enabled; gated on ctx to avoid
    # the extra Dual work on every solve). Two-sided per-station: E-d2k>=0 AND E+d2k>=0.
    if ctx.get('curvature_accel', False):
        R['curv_accel_hi'], R['curv_accel_lo'] = curvature_accel_terms(up, lo, p, nx)
    else:
        R['curv_accel_hi'], R['curv_accel_lo'] = [], []

    # lower-surface secondary-bulge guard (no rising interior curvature hump); gated on ctx
    if ctx.get('bulge', False):
        R['bulge_lo'] = bulge_terms(lo, p, nx)
    else:
        R['bulge_lo'] = []

    diag = dict(alpha_design=a_des.v, lod_clean=R['lod_clean'].v, lod_rough=R['lod_rough'].v,
                cl_max_clean=R['cl_max_clean'].v, stall_c=R['stall_margin_clean'].v,
                stall_r=R['stall_margin_rough'].v, tau=R['tau'].v, area=R['area'].v,
                Ixx=R['Ixx'].v, taumax_psi=R['taumax_psi'].v, hmin=R['hmin'].v,
                cpmin_c=R['cpmin_c_des'].v)
    return Result(R, diag)


def objective(R, which='clean'):
    """Single objective (minimize the negative L/D at the design point):
        which='clean'  -> maximize clean L/D
        which='rough'  -> maximize rough  L/D
    R is an evaluate() Result. Returns a Dual. The three optimization modes are
    (1) which='clean', (2) which='rough', (3) which='clean' plus a rough-L/D
    floor constraint (pass rough_lod_min to constraint_list) — the
    epsilon-constraint method, which traces the clean-vs-rough Pareto front by
    sweeping the floor."""
    if which == 'clean':
        return -R.lod_clean
    if which == 'rough':
        return -R.lod_rough
    raise ValueError(f"which must be 'clean' or 'rough', got {which!r}")


# The optimization constraints, grouped under human-readable labels that mirror
# the fitness function's sections. Each label can be switched on/off from the
# caller via the `enabled` dict (see CONSTRAINT_GROUPS for the full list and
# what each does). This is the gradient-NLP analogue of the fitness function's
# per-term weights (a None/0 weight there = "term off"). Constraints are always
# written as  g(z) >= 0  ('ineq')  or  h(z) == 0  ('eq').
CONSTRAINT_GROUPS = {
    'non_intersection':      'upper & lower surfaces do not cross (min thickness > 0)',
    'reach_design_cl':       'clean CL_max >= design CL (the design point exists)',
    'thickness':             'max thickness tau == target (equality)',
    'stall_margin':          'clean & rough stall margin >= target_stall_margin',
    'moments_of_inertia':    'Ixx, Iyy, Izz >= structural targets',
    'area':                  'enclosed area >= A_con',
    'leading_edge_radius':   'upper & lower LE radius >= targets',
    'radii_skew':            'upper/lower LE radii not too dissimilar',
    'max_thickness_location':'max-thickness x/c not too far forward',
    'te_cone':               'trailing edge stays inside the cone wedge',
    'curvature':             'no upper-surface concavity',
    'aft_curvature':         'upper-surface aft (TE) curvature >= curvature_bound',
    'lower_signflip':        'lower-surface curvature changes sign at most once, via a free split location psi* (needs n_aux>=1)',
    'curvature_accel':       '|d2kappa/ds2(x)| <= frozen envelope E(x/c) at each station x>=XSTART, both surfaces (needs ctx["curvature_accel"]=True)',
    'bulge':                 'lower-surface curvature may not rise into a secondary mid-chord hump: |kappa(x_band)| <= |kappa(2%)| + margin (needs ctx["bulge"]=True)',
    'min_radius_location':   'min radius-of-curvature stays near the LE (needs targets)',
    'rough_xtr_cap':         'rough reported Top/Bot_Xtr <= forced trip (no numeric re-laminarization exploit)',
    'moment':                'clean (and optional rough) CM >= -|CM*_min| across +-cm_alpha_band of design',
    'roughness_delta_cl':    '|clean->rough CL drop at design| <= threshold',
    'slope_ratio':           'rough lift-curve slope at design >= rough_slope_ratio_min x zero-lift slope (surrogate-trust: design reached before the curve rounds off)',
    'lod_falloff':           'L/D falloff within +/- offset of design <= threshold (asymmetric down/up)',
    'lod_falloff_2':         'ADDITIONAL L/D falloff at a wider offset (alpha_falloff_offset_2, default 2deg)',
    'transition_slope':      'clean upper transition point does not snap forward with alpha near design',
    'cl_linearity':          'clean & rough lift-curve slope consistent across a_des-2 -> a_des-1 -> a_des',
    'cpmin':                 'min pressure coefficient >= floor (design & pre-stall)',
    'cl_target':             'rough CL at target_alpha >= target_cl',
}


def constraint_list(R, p, rough_lod_min=None, clean_lod_min=None, enabled=None):
    """The fitness function's requirements as hard constraints. `enabled` is a
    {group_label: bool} dict toggling each CONSTRAINT_GROUPS entry (missing
    label => on). Pass rough_lod_min to add the epsilon-constraint floor
    LoD_rough >= rough_lod_min (mode 3: max clean s.t. rough >= floor); pass
    clean_lod_min for the dual epsilon-constraint LoD_clean >= clean_lod_min
    (mode 4: max rough s.t. clean >= floor). The two are used to approach the
    front from opposite ends -- the rough end with an active rough floor, the
    clean end with an active clean floor -- so the binding constraint is always
    the tight one and neither corner is left loosely bounded."""
    en = {} if enabled is None else enabled
    def on(label):
        return en.get(label, True)
    C = []
    def add(name, kind, dual):
        C.append((name, kind, dual))

    # epsilon-constraint floor. Default 'ineq' (LoD_rough >= eps): the max-clean solve may
    # drift to a HIGHER rough than eps, which lets several eps collapse onto the rough corner.
    # With p['eps_floor_equality'] the floor becomes an EQUALITY (LoD_rough == eps), pinning
    # each point to its own eps so the front is evenly spaced along rough (at the cost of
    # making a point INFEASIBLE where no airfoil sits at exactly that rough -- an honest gap
    # instead of a duplicate). Applies to both sweep directions.
    _floor_kind = 'eq' if (p is not None and p.get('eps_floor_equality', False)) else 'ineq'
    if rough_lod_min is not None:
        add('rough_lod_floor', _floor_kind, R.lod_rough - rough_lod_min)
    if clean_lod_min is not None:                                  # dual epsilon-constraint floor
        add('clean_lod_floor', _floor_kind, R.lod_clean - clean_lod_min)

    if on('non_intersection'):
        add('non_intersection', 'ineq', R.hmin - 1e-4)
    if on('reach_design_cl'):
        add('reach_design_CL', 'ineq', R.lift_margin_clean)
    if on('thickness'):
        add('thickness', 'eq', R.tau - p['tau'])

    if on('stall_margin'):
        add('stall_margin_clean', 'ineq', R.stall_margin_clean - p['target_stall_margin'])
        add('stall_margin_rough', 'ineq', R.stall_margin_rough - p['target_stall_margin'])
    if on('moments_of_inertia'):
        add('Ixx', 'ineq', R.Ixx - p['Ixx_con'])
        add('Iyy', 'ineq', R.Iyy - p['Iyy_con'])
        # Izz == Ixx + Iyy exactly (perpendicular-axis theorem for a planar section),
        # so grad(Izz) == grad(Ixx) + grad(Iyy) identically. Imposing all three makes
        # the active-set Jacobian rank-deficient whenever they bind -> non-unique KKT
        # multipliers -> Ipopt cannot drive dual infeasibility to zero (and the bad
        # multipliers corrupt the search direction). Add the polar-moment row ONLY when
        # it is not already implied by the other two; when Izz_con <= Ixx_con + Iyy_con
        # it is redundant (Izz = Ixx+Iyy >= Ixx_con+Iyy_con >= Izz_con) and dropped.
        if p['Izz_con'] > p['Ixx_con'] + p['Iyy_con']:
            add('Izz', 'ineq', R.Izz - p['Izz_con'])
    if on('area'):
        add('area', 'ineq', R.area - p['A_con'])
    if on('leading_edge_radius'):
        add('ler_upper', 'ineq', R.ler_u - p['ler_con_upper'])
        add('ler_lower', 'ineq', R.ler_l - p['ler_con_lower'])
    if on('radii_skew'):
        add('radii_skew', 'ineq',
            p['ler_skew_factor'] * dmin(R.ler_u, R.ler_l) - dmax(R.ler_u, R.ler_l))
    if on('max_thickness_location'):
        add('max_thickness_loc', 'ineq', R.taumax_psi - p['max_thickness_loc'])
        if p.get('max_thickness_loc_upper') is not None:
            add('max_thickness_loc_upper', 'ineq', R.taumax_psi_upper - p['max_thickness_loc_upper'])
        if p.get('max_thickness_loc_lower') is not None:
            add('max_thickness_loc_lower', 'ineq', R.taumax_psi_lower - p['max_thickness_loc_lower'])
    # NOTE: te_cone / curvature / aft_curvature are imposed as one
    # constraint PER STATION (all built from the linear d2zeta Duals or linear cone
    # clearances) rather than a single relu-sum aggregate. An aggregate goes flat
    # (value 0 AND gradient 0) once satisfied, dropping its Jacobian row to zero and
    # breaking LICQ -> singular KKT -> Ipopt divergence. Per-station linear rows keep
    # the active-set Jacobian full-rank. Same feasible set: aggregate == 0  <=>  every
    # station individually satisfied.
    if on('te_cone'):
        for i, d in enumerate(R.te_cone_terms):
            add(f'te_cone_{i}', 'ineq', d)
    if on('curvature'):
        for i, d in enumerate(R.concave_terms):
            add(f'upper_concavity_{i}', 'ineq', d)       # -d2zeta_upper >= 0
    if on('aft_curvature'):
        # upper-surface d2zeta/dpsi2 >= curvature_bound over the aft region (psi >=
        # ec_cutoff), imposed at each aft station.
        for i, d in enumerate(R.aft_terms):
            add(f'aft_curvature_{i}', 'ineq', d)         # d2zeta_upper - curvature_bound >= 0
    if on('lower_signflip'):
        # exact "<=1 curvature sign change" via a FREE split psi* (aux variable):
        # convex before psi*, concave after (per station). Requires ctx['n_aux']>=1
        # so psi* exists; otherwise R.lower_signflip_terms is empty (no-op).
        for i, d in enumerate(R.lower_signflip_terms):
            add(f'lower_signflip_{i}', 'ineq', d)
    if on('curvature_accel'):
        # |d2kappa/ds2(x_i)| <= frozen E(x/c) at each station (both surfaces), imposed
        # two-sided so |.| stays smooth: E - d2k >= 0 AND E + d2k >= 0. Empty (no-op)
        # unless ctx['curvature_accel'] was set so evaluate() built the terms.
        for i, d in enumerate(getattr(R, 'curv_accel_hi', [])):
            add(f'curv_accel_hi_{i}', 'ineq', d)
        for i, d in enumerate(getattr(R, 'curv_accel_lo', [])):
            add(f'curv_accel_lo_{i}', 'ineq', d)
    if on('bulge'):
        for i, d in enumerate(getattr(R, 'bulge_lo', [])):
            add(f'bulge_lo_{i}', 'ineq', d)
    if on('min_radius_location'):
        if p.get('min_radius_location_upper') is not None:
            add('minrad_loc_u', 'ineq', p['min_radius_location_upper'] - R.minrad_loc_u)
        if p.get('min_radius_location_lower') is not None:
            add('minrad_loc_l', 'ineq', p['min_radius_location_lower'] - R.minrad_loc_l)
    if on('slope_ratio'):
        # rough lift-curve slope at design >= rough_slope_ratio_min x its zero-lift slope
        add('rough_slope_ratio', 'ineq', R.rough_slope_ratio - p['rough_slope_ratio_min'])
    if on('rough_xtr_cap') and hasattr(R, 'xtr_top_r_caps'):
        # rough reported transition must not exceed the forced trip at design or at any
        # falloff sample alpha (no numerical re-laminarization -> no fake laminar rough drag).
        xm = p.get('rough_xtr_max', 0.05)
        for i, xt in enumerate(R.xtr_top_r_caps):
            add(f'rough_xtr_top_{i}', 'ineq', xm - xt)
        if hasattr(R, 'xtr_bot_r_caps'):
            for i, xb in enumerate(R.xtr_bot_r_caps):
                add(f'rough_xtr_bot_{i}', 'ineq', xm - xb)
    if on('moment') and hasattr(R, 'cm_clean_band'):
        cmlim = -abs(p['CMc_min'])                       # CM >= -|CMc_min| (not more nose-down)
        for i, cmv in enumerate(R.cm_clean_band):
            add(f'cm_clean_{i}', 'ineq', cmv - cmlim)
        if hasattr(R, 'cm_rough_band'):
            cmlimr = -abs(p['CMr_min'])
            for i, cmv in enumerate(R.cm_rough_band):
                add(f'cm_rough_{i}', 'ineq', cmv - cmlimr)
    if on('roughness_delta_cl'):
        thr = p['percent_delta_cl_from_roughness_threshold']
        add('delta_cl_hi', 'ineq', thr - R.delta_cl_pct)
        add('delta_cl_lo', 'ineq', thr + R.delta_cl_pct)
    if on('lod_falloff'):
        # asymmetric: tight bound on the DOWNSIDE (left, a_des-off) L/D falloff -- a
        # robustness floor for below-design excursions -- and a loose bound on the
        # UPSIDE (right), which legitimately falls as design nears the upper cliff.
        tf_dn = p.get('percent_LoD_falloff_threshold_down', p['percent_LoD_falloff_threshold'])
        tf_up = p.get('percent_LoD_falloff_threshold_up', p['percent_LoD_falloff_threshold'])
        for nm, tf in (('falloff_c_l', tf_dn), ('falloff_r_l', tf_dn),
                       ('falloff_c_r', tf_up), ('falloff_r_r', tf_up)):
            add(nm + '_hi', 'ineq', tf - getattr(R, nm))
            add(nm + '_lo', 'ineq', tf + getattr(R, nm))
    if on('lod_falloff_2') and hasattr(R, 'falloff2_c_l'):
        # additional wider-offset (2 deg) falloff, same asymmetric down/up bounds
        tf_dn2 = p.get('percent_LoD_falloff_threshold_2_down', p.get('percent_LoD_falloff_threshold_down', 0.15))
        tf_up2 = p.get('percent_LoD_falloff_threshold_2_up', p.get('percent_LoD_falloff_threshold_up', 0.30))
        for nm, tf in (('falloff2_c_l', tf_dn2), ('falloff2_r_l', tf_dn2),
                       ('falloff2_c_r', tf_up2), ('falloff2_r_r', tf_up2)):
            add(nm + '_hi', 'ineq', tf - getattr(R, nm))
            add(nm + '_lo', 'ineq', tf + getattr(R, nm))
    if on('transition_slope') and hasattr(R, 'xtr_slope_c'):
        # forbid the clean upper transition point snapping forward with alpha near
        # design (one-sided: only the forward march is a cliff).
        add('xtr_slope_c', 'ineq', p['xtr_slope_threshold'] - R.xtr_slope_c)
    if on('cl_linearity') and hasattr(R, 'cl_lin_disc_c'):
        tl_c = p.get('cl_linearity_tol_clean', p.get('cl_linearity_tol', 0.01))
        tl_r = p.get('cl_linearity_tol_rough', p.get('cl_linearity_tol', 0.01))
        # |disc| <= tol, disc = (s_lo-s_hi)/max(s_lo,floor) = fractional lift-slope mismatch,
        # O(tol) at the boundary so Ipopt can actually enforce it (the raw slope-difference
        # form was ~constr_viol_tol and toothless).
        add('cl_lin_c_hi', 'ineq', tl_c - R.cl_lin_disc_c)
        add('cl_lin_c_lo', 'ineq', tl_c + R.cl_lin_disc_c)
        add('cl_lin_r_hi', 'ineq', tl_r - R.cl_lin_disc_r)
        add('cl_lin_r_lo', 'ineq', tl_r + R.cl_lin_disc_r)
    if on('cpmin') and p.get('cp_min_design') is not None:
        add('cpmin_c_design', 'ineq', R.cpmin_c_des - p['cp_min_design'])
        add('cpmin_r_design', 'ineq', R.cpmin_r_des - p['cp_min_design'])
        if p.get('cp_min_prestall') is not None:
            add('cpmin_c_prestall', 'ineq', R.cpmin_c_prestall - p['cp_min_prestall'])
            add('cpmin_r_prestall', 'ineq', R.cpmin_r_prestall - p['cp_min_prestall'])
    if on('cl_target') and p.get('target_alpha') is not None and p.get('target_cl') is not None:
        add('cl_target_rough', 'ineq', R.cl_target_rough - p['target_cl'])

    return C
