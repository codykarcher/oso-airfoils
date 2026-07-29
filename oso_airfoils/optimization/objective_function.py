import numpy as np
from metafoil.core.kulfan import Kulfan
from pint import UnitRegistry
units = UnitRegistry()
import copy
import math

import pathlib
path_to_here = pathlib.Path(__file__).parent.resolve()

# The aerodynamic solver is injected into core_fitness_function rather than being
# a module global chosen by an if/elif on params['tool'] -- see solvers.make_solver.
# The individual xfoil/qfoil/neuralfoil wrappers are imported lazily there, so a
# neuralfoil run never pays to import the external-solver wrappers, and vice versa.
from oso_airfoils.optimization.solvers import make_solver

from oso_airfoils.optimization.geometry_functions import (
    TE_gap_function,
    cone_angle_function,
    Ixx_function,
    Iyy_function,
    Izz_function,
    area_function,
    ler_function,
    min_radius_location_upper_function,
    min_radius_location_lower_function)


def _drop_unconverged(cl, cd, cm, cpmin, alpha, xtr_top=None, xtr_bot=None):
    """Keep only the converged points of a polar sweep.

    metafoil's in-memory xfoil/qfoil sweep returns NaN cl/cd for the alphas
    that did not converge (the old file-I/O xfoil simply omitted them from the
    polar). A non-converged sweep point is expected and fine, but it must not
    leak into the design-point interpolations downstream (np.interp through a
    NaN yields NaN, which would then poison a final reported value and get the
    whole design rejected). Dropping them here restores the old "converged
    points only" polar. Arrays stay mutually aligned; returns lists.

    ``xtr_top``/``xtr_bot`` (reported transition x/c per alpha) are optional; when
    the solver provides them they are dropped in the same mask and returned too,
    so the rough transition-cap and clean transition-slope constraints can read
    them. Absent (e.g. a solver that reports no transition), they come back None.
    """
    cl = np.atleast_1d(np.asarray(cl, dtype=float))
    cd = np.atleast_1d(np.asarray(cd, dtype=float))
    cm = np.atleast_1d(np.asarray(cm, dtype=float))
    cpmin = np.atleast_1d(np.asarray(cpmin, dtype=float))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    m = np.isfinite(cl) & np.isfinite(cd) & np.isfinite(alpha)
    xt = list(np.atleast_1d(np.asarray(xtr_top, dtype=float))[m]) if xtr_top is not None else None
    xb = list(np.atleast_1d(np.asarray(xtr_bot, dtype=float))[m]) if xtr_bot is not None else None
    return list(cl[m]), list(cd[m]), list(cm[m]), list(cpmin[m]), list(alpha[m]), xt, xb


#: Rejection codes returned in the ``alpha_design`` slot (index 4) when an airfoil is
#: rejected before the design point can be evaluated.
#:
#: A rejected row is ``[pid, inf, inf, False, CODE, 0, 0, ...]`` -- the objectives are
#: inf and ``con_tag`` is False, and this slot carries WHY. The values are deliberately
#: absurd as angles of attack: a design alpha of -80 degrees is self-evidently not a
#: real answer, so a garbage row announces itself instead of blending in with plausible
#: ones.
#:
#: The trap for post-processing: read this column WITHOUT first masking on
#: ``obj1 == inf`` (or ``con_tag``) and you are averaging status codes together with
#: real angles, which yields a meaningless number.
REJECTION_CODES = {
    -10: 'self-intersecting geometry (negative thickness somewhere)',
    -20: 'upper Kulfan coefficient magnitude > 2',
    -30: 'lower Kulfan coefficient magnitude > 2',
    -60: 'rough sweep did not reach target_alpha (high alphas did not converge)',
    -70: 'no stall peak found in the sweep',
    -80: 'design CL is above the airfoil CL_max',
    -85: 'design CL is below the start of the (extended) sweep',
    -90: 'exception during evaluation, or a NaN in the reported values',
}


def airfoil_fitness(x, solver=None, kulfan=None):
    """Evaluate one individual, retrying up to ``N_tries`` times on a solver failure.

    ``solver`` / ``kulfan`` are the injected aerodynamic solver and geometry class
    (see :func:`core_fitness_function`); leaving them ``None`` builds the defaults
    from ``x['params']``, which is what a plain single-airfoil call wants.
    """
    N_tries = x['params']['N_tries']
    if N_tries is None:
        N_rtr = 1
    else:
        N_rtr = N_tries

    for i in range(0,N_rtr):
        res = core_fitness_function(x, solver=solver, kulfan=kulfan)
        if res[3] > -10:
            return res
    return res

def core_fitness_function(x, solver=None, kulfan=None):
    """Objectives + 31 constraints for one design vector.

    Parameters
    ----------
    x : dict
        ``{'pid': int, 'individual': design vector, 'params': case params}``.
    solver : callable, optional
        An AeroSolver (see ``solvers.make_solver``). Defaults to the solver implied
        by ``params['tool']``. Injecting it is what lets the GPU-batched evaluator
        serve every polar in a generation from one forward without this function
        knowing anything about batching.
    kulfan : class, optional
        Geometry class constructed as ``kulfan(TE_gap=...)``. Defaults to metafoil's
        ``Kulfan``; the batched path passes a registry-backed drop-in instead.
    """
    # ----------------------
    # unpack
    # ----------------------
    pid = x['pid']
    K_upper = x['individual'][0:int(len(x['individual'])/2)]
    K_lower = x['individual'][int(len(x['individual'])/2):]
    # te_gap = x['individual'][-1]
    tau         = x['params']['tau']
    CL_in       = x['params']['CL']
    CMc_in      = x['params']['CMc_min']
    CMr_in      = x['params']['CMr_min']
    Re_in       = x['params']['Re']

    N_reported    = 16
    N_constraints = 31

    # for iiiiii in range (0,1):
    try:

        cl_design = CL_in
        Re = Re_in

        if 'TE_gap' not in x['params'] or x['params']['TE_gap'] is None:
            te_gap = TE_gap_function(tau)
        else:
            te_gap = x['params']['TE_gap']
        
        if 'cone_angle' not in x['params'] or x['params']['cone_angle'] is None:
            cone_angle = cone_angle_function(tau)
            te_frac = 0.95
        else:
            cone_angle = x['params']['cone_angle']
            te_frac    = x['params']['te_frac']

        if 'Ixx_con' not in x['params'] or x['params']['Ixx_con'] is None:
            Ixx_con = Ixx_function(tau)
        else:
            Ixx_con = x['params']['Ixx_con']

        if 'Iyy_con' not in x['params'] or x['params']['Iyy_con'] is None:
            Iyy_con = Iyy_function(tau)
        else:
            Iyy_con = x['params']['Iyy_con']

        if 'Izz_con' not in x['params'] or x['params']['Izz_con'] is None:
            Izz_con = Izz_function(tau)
        else:
            Izz_con = x['params']['Izz_con']

        if 'A_con' not in x['params'] or x['params']['A_con'] is None:
            A_con = area_function(tau)
        else:
            A_con = x['params']['A_con']

        if 'ler_con_upper' not in x['params'] or x['params']['ler_con_upper'] is None:
            ler_con_upper = ler_function(tau)
        else:
            ler_con_upper = x['params']['ler_con_upper']

        if 'ler_con_lower' not in x['params'] or x['params']['ler_con_lower'] is None:
            ler_con_lower = ler_function(tau)
        else:
            ler_con_lower = x['params']['ler_con_lower']
        
        if 'min_radius_location_upper' not in x['params'] or x['params']['min_radius_location_upper'] is None:
            min_radius_location_upper = min_radius_location_upper_function(tau)
        else:
            min_radius_location_upper = x['params']['min_radius_location_upper']

        if 'min_radius_location_lower' not in x['params'] or x['params']['min_radius_location_lower'] is None:
            min_radius_location_lower = min_radius_location_lower_function(tau)
        else:
            min_radius_location_lower = x['params']['min_radius_location_lower']

        if 'min_radius_location_cutoff' not in x['params'] or x['params']['min_radius_location_cutoff'] is None:
            min_radius_location_cutoff = 0.08
        else:
            min_radius_location_cutoff = x['params']['min_radius_location_cutoff']

        target_stall_margin                       = x['params']['target_stall_margin']
        percent_delta_cl_from_roughness_threshold = x['params']['percent_delta_cl_from_roughness_threshold']
        alpha_falloff_offset                      = x['params']['alpha_falloff_offset']
        percent_LoD_falloff_threshold             = x['params']['percent_LoD_falloff_threshold']
        max_thickness_loc                         = x['params']['max_thickness_loc']
        max_thickness_loc_upper                   = x['params']['max_thickness_loc_upper']
        max_thickness_loc_lower                   = x['params']['max_thickness_loc_lower']
        ler_skew_factor                           = x['params']['ler_skew_factor']
        cl_max_limit_clean                        = x['params']['cl_max_limit_clean']
        cl_max_limit_rough                        = x['params']['cl_max_limit_rough']
        if 'target_cl' not in x['params'] or x['params']['target_cl'] is None:
            target_cl = None
        else:
            target_cl = x['params']['target_cl']
        if 'target_alpha' not in x['params'] or x['params']['target_alpha'] is None:
            target_alpha = None
        else:
            target_alpha = x['params']['target_alpha']
        curvature_bound                           = x['params']['curvature_bound']
        ec_cutoff                                 = x['params']['ec_cutoff']
        cm_alpha_band                             = x['params']['cm_alpha_band']
        cp_min_design                             = x['params']['cp_min_design']
        cp_min_at_alpha_offset                    = x['params']['cp_min_at_alpha_offset']
        cp_min_alpha_offset                       = x['params']['cp_min_alpha_offset']
        cp_min_prestall                           = x['params']['cp_min_prestall']
        toothpick_height                          = x['params']['toothpick_height']
        toothpick_location                        = x['params']['toothpick_location']

        stall_margin_clean_weighting        = x['params']['stall_margin_clean_weighting']
        stall_margin_rough_weighting        = x['params']['stall_margin_rough_weighting']
        lift_margin_clean_weighting         = x['params']['lift_margin_clean_weighting']
        cl_max_limit_clean_weighting        = x['params']['cl_max_limit_clean_weighting']
        cl_max_limit_rough_weighting        = x['params']['cl_max_limit_rough_weighting']
        delta_cl_from_roughness_weighting   = x['params']['delta_cl_from_roughness_weighting']
        LoD_falloff_weighting               = x['params']['LoD_falloff_weighting']
        ixx_weighting                       = x['params']['ixx_weighting']
        iyy_weighting                       = x['params']['iyy_weighting']
        izz_weighting                       = x['params']['izz_weighting']
        a_weighting                         = x['params']['a_weighting']
        leading_edge_radius_upper_weighting = x['params']['leading_edge_radius_upper_weighting']
        leading_edge_radius_lower_weighting = x['params']['leading_edge_radius_lower_weighting']
        max_thickness_weighting             = x['params']['max_thickness_weighting']
        max_thickness_upper_weighting       = x['params']['max_thickness_upper_weighting']
        max_thickness_lower_weighting       = x['params']['max_thickness_lower_weighting']
        radii_skew_weighting                = x['params']['radii_skew_weighting']
        curvature_weighting                 = x['params']['curvature_weighting']
        lower_surface_curvature_weighting   = x['params']['lower_surface_curvature_weighting']
        te_cone_violation_weighting         = x['params']['te_cone_violation_weighting']
        CL_target_weighting                 = x['params']['CL_target_weighting']
        clean_moment_weighting              = x['params']['clean_moment_weighting']
        rough_moment_weighting              = x['params']['rough_moment_weighting']
        cpmin_design_weighting              = x['params']['cp_min_design_weighting']
        cpmin_at_alpha_offset_weighting     = x['params']['cp_min_at_alpha_offset_weighting']
        cpmin_prestall_weighting            = x['params']['cp_min_prestall_weighting']
        infeasibility_penalty               = x['params']['infeasibility_penalty']
        min_radius_location_upper_weighting = x['params']['min_radius_location_upper_weighting']
        min_radius_location_lower_weighting = x['params']['min_radius_location_lower_weighting']
        toothpick_weighting                 = x['params']['toothpick_weighting']

        N_crit_clean     = x['params']['N_crit_clean']
        N_crit_rough     = x['params']['N_crit_rough']
        alpha_min_clean  = x['params']['alpha_min_clean']
        alpha_max_clean  = x['params']['alpha_max_clean']
        alpha_step_clean = x['params']['alpha_step_clean']
        alpha_min_rough  = x['params']['alpha_min_rough']
        alpha_max_rough  = x['params']['alpha_max_rough']
        alpha_step_rough = x['params']['alpha_step_rough']
        xtp_u_clean      = x['params']['xtp_u_clean']
        xtp_l_clean      = x['params']['xtp_l_clean']
        xtp_u_rough      = x['params']['xtp_u_rough']
        xtp_l_rough      = x['params']['xtp_l_rough']

        # Solver and geometry class are injected (see the docstring). Building them
        # here when absent keeps a bare core_fitness_function(x) call working.
        if solver is None:
            solver = make_solver(x['params'])
        if kulfan is None:
            kulfan = Kulfan

        # ----------------------
        # reject too large coefficients
        # ----------------------
        if max(abs(np.array(K_upper)))>2.0:
            return [pid, np.inf, np.inf, False, -20] + [0]*N_reported + [0]*N_constraints
        if max(abs(np.array(K_lower)))>2.0:
            return [pid, np.inf, np.inf, False, -30] + [0]*N_reported + [0]*N_constraints

        # ----------------------
        # build airfoil
        # ----------------------
        afl_geo = kulfan(TE_gap = te_gap)
        afl_geo.upperCoefficients = K_upper
        afl_geo.lowerCoefficients = K_lower
        afl_geo.chord = 1.0*units.m
        
        # ----------------------
        # reject self intersecting
        # ----------------------
        hts = afl_geo.getNormalizedHeight()
        if any(hts<0):
            return [pid, np.inf, np.inf, False, -10] + [0]*N_reported + [0]*N_constraints

        # ----------------------
        # Sweep + unconverged-drop, factored so the clean/rough sweeps AND the
        # on-demand downward polar extension below can all reuse it. Returns
        # (cl, cd, cm, cpmin, alpha) lists of the converged points only.
        # ----------------------
        def _sweep(kind, amin, amax, astep):
            nc, xu, xl = ((N_crit_clean, xtp_u_clean, xtp_l_clean) if kind == 'clean'
                          else (N_crit_rough, xtp_u_rough, xtp_l_rough))
            r = solver(K_upper, K_lower, [amin, amax, astep], N_crit=nc, xtp_u=xu, xtp_l=xl)
            if r is None:
                raise RuntimeError("solver failed")
            # Drop non-converged sweep points (NaN cl/cd) so they can't poison the
            # design-point interpolations below — see _drop_unconverged. xtr_top/xtr_bot
            # (reported transition x/c) ride along when the solver provides them.
            cl, cd, cm, cp, a, xt, xb = _drop_unconverged(
                r['cl'], r['cd'], r['cm'], r['cpmin'], r['alpha'],
                r.get('xtr_top'), r.get('xtr_bot'))
            return list(cl), list(cd), list(cm), list(cp), list(a), xt, xb

        # first index (scanning forward from alpha~0) where CL stops rising = stall
        # peak (the intentional first-roll-over definition). If CL never declines in
        # range, the airfoil simply hasn't stalled within the sweep -> use the last
        # index (the sweep edge = a large stall margin), matching the gradient model's
        # _positive_peak_index. Previously this raised -> reject (-70), which rejected
        # high-camber airfoils whose rough CL rises past the sweep top; the gradient
        # (now-working reference) keeps them, so this stays consistent with it.
        def _peak(cl, alpha):
            mid = int(np.argmin(abs(np.array(alpha))))
            for i in range(mid + 1, len(cl)):
                if cl[i] <= cl[i - 1]:
                    return i - 1
            return len(cl) - 1

        # ----------------------
        # Run Clean Data
        # ----------------------
        cl_clean, cd_clean, cm_clean, cpmin_clean, alpha_clean, xtr_top_clean, xtr_bot_clean = _sweep('clean', alpha_min_clean, alpha_max_clean, alpha_step_clean)
        LoD_clean   = [cl_clean[i]/cd_clean[i] for i in range(0,len(cl_clean))]

        # ----------------------
        # Run Rough Data
        # ----------------------
        cl_rough, cd_rough, cm_rough, cpmin_rough, alpha_rough, xtr_top_rough, xtr_bot_rough = _sweep('rough', alpha_min_rough, alpha_max_rough, alpha_step_rough)
        LoD_rough   = [cl_rough[i]/cd_rough[i] for i in range(0,len(cl_rough))]
        
        # ----------------------
        # Find the stall locations
        # ----------------------
        try:
            positive_peak_index_clean = _peak(cl_clean, alpha_clean)
            positive_peak_index_rough = _peak(cl_rough, alpha_rough)
        except:
            # no peak present in the sweep
            return [pid, np.inf, np.inf, False, -70] + [0]*N_reported + [0]*N_constraints

        # ----------------------
        # On-demand DOWNWARD polar extension (mirrors metafoil oso_gradient.evaluate).
        # If the design CL is below the clean sweep's starting CL, the design point
        # sits at NEGATIVE alpha, off the low end of the sweep. Sweep back from the
        # current alpha_min down to alpha_min_extend (using the clean/rough step) and
        # PREPEND to BOTH the clean and rough polars, then redo stall detection, so
        # every downstream design-point interpolation uses the extended data. Only
        # runs when needed; in-range designs skip it entirely.
        # ----------------------
        alpha_min_extend = x['params'].get('alpha_min_extend', -10.0)
        if cl_design < cl_clean[0] and alpha_clean[0] > alpha_min_extend + 1e-9:
            ec = _sweep('clean', alpha_min_extend, alpha_clean[0] - alpha_step_clean, alpha_step_clean)
            er = _sweep('rough', alpha_min_extend, alpha_rough[0] - alpha_step_rough, alpha_step_rough)
            cl_clean, cd_clean = ec[0] + cl_clean, ec[1] + cd_clean
            cm_clean, cpmin_clean, alpha_clean = ec[2] + cm_clean, ec[3] + cpmin_clean, ec[4] + alpha_clean
            cl_rough, cd_rough = er[0] + cl_rough, er[1] + cd_rough
            cm_rough, cpmin_rough, alpha_rough = er[2] + cm_rough, er[3] + cpmin_rough, er[4] + alpha_rough
            # keep the transition arrays aligned with the prepended polars (when present)
            if xtr_top_clean is not None and ec[5] is not None: xtr_top_clean = ec[5] + xtr_top_clean
            if xtr_bot_clean is not None and ec[6] is not None: xtr_bot_clean = ec[6] + xtr_bot_clean
            if xtr_top_rough is not None and er[5] is not None: xtr_top_rough = er[5] + xtr_top_rough
            if xtr_bot_rough is not None and er[6] is not None: xtr_bot_rough = er[6] + xtr_bot_rough
            LoD_clean = [cl_clean[i]/cd_clean[i] for i in range(0, len(cl_clean))]
            LoD_rough = [cl_rough[i]/cd_rough[i] for i in range(0, len(cl_rough))]
            try:
                positive_peak_index_clean = _peak(cl_clean, alpha_clean)
                positive_peak_index_rough = _peak(cl_rough, alpha_rough)
            except:
                return [pid, np.inf, np.inf, False, -70] + [0]*N_reported + [0]*N_constraints

        # ----------------------
        # Find alpha_design. The (possibly extended) clean pre-stall sweep must
        # BRACKET the design CL. Above CL_max -> -80; still below the sweep start
        # (even after extension) -> -85. Do not silently clamp.
        # ----------------------
        if cl_clean[positive_peak_index_clean] <= cl_design:
            # design CL above CL_max -- airfoil cannot reach target CL
            return [pid, np.inf, np.inf, False, -80] + [0]*N_reported + [0]*N_constraints
        if cl_design < cl_clean[0]:
            # design CL below the (extended) sweep start -- design point off the low end
            return [pid, np.inf, np.inf, False, -85] + [0]*N_reported + [0]*N_constraints
        alpha_design = np.interp(cl_design,
                                 np.array(cl_clean)[0:positive_peak_index_clean],
                                 np.array(alpha_clean)[0:positive_peak_index_clean] )
        
        # ----------------------
        # Begin objective function and constraints
        # ----------------------
        conpen = 0.0
        cons = []

        # ----------------------
        # Pure L/D performance
        # ----------------------
        LoD_clean_at_design_alpha = np.interp(alpha_design, 
                                     np.array(alpha_clean)[0:positive_peak_index_clean], 
                                     np.array(LoD_clean)[0:positive_peak_index_clean] )

        LoD_rough_at_design_alpha = np.interp(alpha_design, 
                                     np.array(alpha_rough)[0:positive_peak_index_rough], 
                                     np.array(LoD_rough)[0:positive_peak_index_rough] )
        
        obj1 = -1*LoD_clean_at_design_alpha

        obj2 = -1*LoD_rough_at_design_alpha

        # ----------------------
        # Stall margin
        # ----------------------
        stall_margin_clean = alpha_clean[positive_peak_index_clean] - alpha_design
        stall_margin_rough = alpha_rough[positive_peak_index_rough] - alpha_design
        
        # target_stall_margin = 4.0
        # 0 if valid, otherwise 4.0-target_stall_margin
        if stall_margin_clean_weighting is not None and stall_margin_clean_weighting != 0.0:
            cons.append(target_stall_margin-(min([target_stall_margin,stall_margin_clean])))
            conpen += stall_margin_clean_weighting*(target_stall_margin-min([target_stall_margin, stall_margin_clean]))
        else:
            cons.append(0.0)
        if stall_margin_rough_weighting is not None and stall_margin_rough_weighting != 0.0:
            cons.append(target_stall_margin-(min([target_stall_margin,stall_margin_rough])))
            conpen += stall_margin_rough_weighting*(target_stall_margin-min([target_stall_margin, stall_margin_rough]))
        else:
            cons.append(0.0)
        
        # ----------------------
        # lift margin
        # ----------------------
        lift_margin_clean = cl_clean[positive_peak_index_clean] - cl_design
        
        if lift_margin_clean_weighting is not None and lift_margin_clean_weighting != 0.0:
            if lift_margin_clean >=0:
                # expected behavior
                conpen += lift_margin_clean_weighting*lift_margin_clean
            else:
                # in some weird near-stall location, penalize extra
                conpen += 100*lift_margin_clean_weighting*abs(lift_margin_clean)

        
        # ----------------------
        # limit CL max
        # ----------------------
        if cl_max_limit_clean is not None and cl_max_limit_clean_weighting is not None and cl_max_limit_clean_weighting != 0.0:
            cl_max_violation_clean = cl_clean[positive_peak_index_clean] - cl_max_limit_clean
            if cl_max_violation_clean > 0:
                conpen += cl_max_limit_clean_weighting * cl_max_violation_clean
                cons.append(cl_max_violation_clean)
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)


        if cl_max_limit_rough is not None and cl_max_limit_rough_weighting is not None and cl_max_limit_rough_weighting != 0.0:
            cl_max_violation_rough = cl_rough[positive_peak_index_rough] - cl_max_limit_rough
            if cl_max_violation_rough > 0:
                conpen += cl_max_limit_rough_weighting * cl_max_violation_rough
                cons.append(cl_max_violation_rough)
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)

        # ----------------------
        # change in cl at alpha design due to roughness
        # ----------------------
        delta_cl_clean_to_rough_at_alpha_design = cl_design - np.interp(alpha_design, 
                                                                        np.array(alpha_rough)[0:positive_peak_index_rough], 
                                                                        np.array(cl_rough)[0:positive_peak_index_rough] )

        percent_delta_cl_clean_to_rough_at_alpha_design = delta_cl_clean_to_rough_at_alpha_design/cl_design
        
        # penalize if greater than 10%
        # assert(percent_delta_cl_clean_to_rough_at_alpha_design >=0)
        if delta_cl_from_roughness_weighting is not None and delta_cl_from_roughness_weighting != 0.0:
            conpen += delta_cl_from_roughness_weighting * (max([percent_delta_cl_from_roughness_threshold, abs(percent_delta_cl_clean_to_rough_at_alpha_design)])-percent_delta_cl_from_roughness_threshold)
        
        # ----------------------
        # clean L/D curve fall off
        # ----------------------
        LoD_clean_1degree_left = np.interp(alpha_design - alpha_falloff_offset, 
                                             np.array(alpha_clean)[0:positive_peak_index_clean], 
                                             np.array(LoD_clean)[0:positive_peak_index_clean] )

        LoD_clean_1degree_right = np.interp(alpha_design + alpha_falloff_offset, 
                                             np.array(alpha_clean)[0:positive_peak_index_clean], 
                                             np.array(LoD_clean)[0:positive_peak_index_clean] )

        percent_change_LoD_clean_left  = (LoD_clean_1degree_left-LoD_clean_at_design_alpha)/LoD_clean_at_design_alpha
        percent_change_LoD_clean_right = (LoD_clean_1degree_right-LoD_clean_at_design_alpha)/LoD_clean_at_design_alpha
     
        # penalize if greater than 15%
        # assert(percent_delta_LoD_clean_to_rough_at_alpha_design >=0)
        if LoD_falloff_weighting is not None and LoD_falloff_weighting != 0.0:
            conpen += LoD_falloff_weighting * (max([percent_LoD_falloff_threshold, abs(percent_change_LoD_clean_left )])-percent_LoD_falloff_threshold)
            conpen += LoD_falloff_weighting * (max([percent_LoD_falloff_threshold, abs(percent_change_LoD_clean_right)])-percent_LoD_falloff_threshold)

        # ----------------------
        # rough L/D curve fall off
        # ----------------------
        LoD_rough_1degree_left = np.interp(alpha_design - alpha_falloff_offset, 
                                             np.array(alpha_rough)[0:positive_peak_index_rough], 
                                             np.array(LoD_rough)[0:positive_peak_index_rough] )

        LoD_rough_1degree_right = np.interp(alpha_design + alpha_falloff_offset, 
                                             np.array(alpha_rough)[0:positive_peak_index_rough], 
                                             np.array(LoD_rough)[0:positive_peak_index_rough] )

        percent_change_LoD_rough_left  = (LoD_rough_1degree_left-LoD_rough_at_design_alpha)/LoD_rough_at_design_alpha
        percent_change_LoD_rough_right = (LoD_rough_1degree_right-LoD_rough_at_design_alpha)/LoD_rough_at_design_alpha

        # penalize if greater than 15%
        # assert(percent_delta_LoD_clean_to_rough_at_alpha_design >=0)
        if LoD_falloff_weighting is not None and LoD_falloff_weighting != 0.0:
            conpen += LoD_falloff_weighting * (max([percent_LoD_falloff_threshold, abs(percent_change_LoD_rough_left )])-percent_LoD_falloff_threshold)
            conpen += LoD_falloff_weighting * (max([percent_LoD_falloff_threshold, abs(percent_change_LoD_rough_right)])-percent_LoD_falloff_threshold)

        # ======================================================================
        # Constraints ported from the gradient optimizer (gradient_objective.py) to
        # keep the GA fitness consistent with it. All are AERO-performance guards, so
        # they follow the same SOFT-PENALTY convention as delta_cl / lod_falloff above
        # (conpen only, not in cons[]), and each is individually TOGGLABLE via its own
        # *_weighting param (None/0 -> no effect). All new params are read with .get()
        # so runfiles that predate them are unaffected.
        # ======================================================================
        # Inclusive pre-stall interpolation (through the peak point), matching the
        # gradient optimizer's `at(...)` [:pk+1] convention. Guarded below on a valid
        # peak index (>= 2): a degenerate polar (no proper peak -> _peak returns a
        # non-positive index) simply skips these constraints rather than reading a
        # backwards/empty slice.
        _cpk = positive_peak_index_clean
        _rpk = positive_peak_index_rough
        def _pre_interp(a, ap, vals, pk):
            return np.interp(a, np.array(ap)[0:pk + 1], np.array(vals)[0:pk + 1])

        # --- (1) SECOND L/D falloff at a wider offset (default 2 deg): holds L/D flat
        # further below design so the laminar-bucket lower knee can't perch under the
        # operating point. Mirrors gradient_objective's lod_falloff_2. ---------------
        LoD_falloff_2_weighting = x['params'].get('LoD_falloff_2_weighting', None)
        alpha_falloff_offset_2  = x['params'].get('alpha_falloff_offset_2', 2.0)
        percent_LoD_falloff_threshold_2 = x['params'].get('percent_LoD_falloff_threshold_2', percent_LoD_falloff_threshold)
        if LoD_falloff_2_weighting is not None and LoD_falloff_2_weighting != 0.0:
            for ap, lod, lod_des, pk in ((alpha_clean, LoD_clean, LoD_clean_at_design_alpha, _cpk),
                                         (alpha_rough, LoD_rough, LoD_rough_at_design_alpha, _rpk)):
                if pk < 2:
                    continue
                for da in (-alpha_falloff_offset_2, alpha_falloff_offset_2):
                    pc = (_pre_interp(alpha_design + da, ap, lod, pk) - lod_des) / lod_des
                    conpen += LoD_falloff_2_weighting * (max([percent_LoD_falloff_threshold_2, abs(pc)]) - percent_LoD_falloff_threshold_2)

        # --- (2) ROUGH LIFT-SLOPE RATIO: slope(dCL/dalpha) @ design / @ zero-lift on the
        # rough polar. A rough curve that has rounded off (lost attached slope) before
        # design => low ratio => the tool is extrapolating through incipient separation.
        # Reference-free (anchors CL=0 and CL=CL_design; central FD, half-step
        # slope_ratio_h). Needs the rough polar to bracket zero-lift AND reach
        # slope_ratio_alpha_min so the +/-h differences aren't clamped at the sweep edge
        # (the grid bug that doubles the ratio). For the batched path set the runfile's
        # alpha_min_extend <= slope_ratio_alpha_min so this extension stays cache-served.
        # Mirrors gradient_objective's rough_slope_ratio. ---------------------------
        slope_ratio_weighting = x['params'].get('slope_ratio_weighting', None)
        rough_slope_ratio_min = x['params'].get('rough_slope_ratio_min', None)
        if (slope_ratio_weighting is not None and slope_ratio_weighting != 0.0
                and rough_slope_ratio_min is not None and _rpk >= 2):
            srh = x['params'].get('slope_ratio_h', 0.5)
            sr_amin = x['params'].get('slope_ratio_alpha_min', -14.0)
            clr = list(cl_rough[0:_rpk + 1])
            alr = list(alpha_rough[0:_rpk + 1])
            if alr and alr[0] > sr_amin + 1e-9:            # reach low enough to bracket zero-lift +/- h
                ext = _sweep('rough', sr_amin, alr[0] - alpha_step_rough, alpha_step_rough)
                clr = ext[0] + clr
                alr = ext[4] + alr
            clr = np.array(clr); alr = np.array(alr)
            if clr.size and clr.min() <= 0.0 <= clr.max():             # zero-lift bracketed
                a0 = np.interp(0.0, clr, alr)              # rough zero-lift alpha
                ad = np.interp(cl_design, clr, alr)        # rough design alpha
                m0 = (np.interp(a0 + srh, alr, clr) - np.interp(a0 - srh, alr, clr)) / (2 * srh)
                md = (np.interp(ad + srh, alr, clr) - np.interp(ad - srh, alr, clr)) / (2 * srh)
                rough_slope_ratio = md / max(m0, 0.02)
                conpen += slope_ratio_weighting * (rough_slope_ratio_min - min([rough_slope_ratio_min, rough_slope_ratio]))

        # --- (3) ROUGH TRANSITION-LOCATION CAP: the rough case forces transition at
        # xtp~0.05, so a reported Top/Bot_Xtr above that trip is numerical
        # re-laminarization (fake laminar rough drag). Sampled at design and the falloff
        # offsets. Mirrors gradient_objective's rough_xtr_cap. Needs solver xtr. -------
        rough_xtr_cap_weighting = x['params'].get('rough_xtr_cap_weighting', None)
        if (rough_xtr_cap_weighting is not None and rough_xtr_cap_weighting != 0.0
                and xtr_top_rough is not None and _rpk >= 2):
            xm = x['params'].get('rough_xtr_max', 0.05)
            _o2 = x['params'].get('alpha_falloff_offset_2', None)
            xoffs = [0.0, -alpha_falloff_offset, alpha_falloff_offset] + ([-_o2, _o2] if _o2 is not None else [])
            for xarr in (xtr_top_rough, xtr_bot_rough):
                if xarr is None:
                    continue
                for da in xoffs:
                    xt = _pre_interp(alpha_design + da, alpha_rough, xarr, _rpk)
                    conpen += rough_xtr_cap_weighting * (max([xm, xt]) - xm)

        # --- (4) CLEAN UPPER TRANSITION-SLOPE: forward march of the clean upper
        # transition point over the +xtr_slope_offset window above design. Large =>
        # design parked on a laminar cliff. Mirrors gradient_objective's
        # transition_slope. One-sided (only the forward march is a cliff). Needs xtr. --
        transition_slope_weighting = x['params'].get('transition_slope_weighting', None)
        xtr_slope_threshold        = x['params'].get('xtr_slope_threshold', None)
        if (transition_slope_weighting is not None and transition_slope_weighting != 0.0
                and xtr_slope_threshold is not None and xtr_top_clean is not None and _cpk >= 2):
            toff = x['params'].get('xtr_slope_offset', 1.0)
            xtr_slope_c = (_pre_interp(alpha_design, alpha_clean, xtr_top_clean, _cpk)
                           - _pre_interp(alpha_design + toff, alpha_clean, xtr_top_clean, _cpk))
            conpen += transition_slope_weighting * (max([xtr_slope_threshold, xtr_slope_c]) - xtr_slope_threshold)

        # ----------------------
        # structure surrogates
        # ----------------------
        # metafoil's Kulfan returns plain (chord-normalized) floats, not pint
        # quantities, so these are already magnitudes.
        Ixx = afl_geo.Ixx
        Iyy = afl_geo.Iyy
        Izz = afl_geo.Izz
        A   = afl_geo.area
        
        Ixx_target = Ixx_con 
        Iyy_target = Iyy_con 
        Izz_target = Izz_con 
        A_target   = A_con

        # 0 if valid, otherwise target-val
        if ixx_weighting is not None and ixx_weighting != 0.0:
            cons.append(Ixx_target-(min([Ixx_target,Ixx])))
            conpen += ixx_weighting * (Ixx_target-min([Ixx_target, Ixx]))
        else:
            cons.append(0.0)
        
        if iyy_weighting is not None and iyy_weighting != 0.0:
            cons.append(Iyy_target-(min([Iyy_target,Iyy])))
            conpen += iyy_weighting * (Iyy_target-min([Iyy_target, Iyy]))
        else:
            cons.append(0.0)
        
        if izz_weighting is not None and izz_weighting != 0.0:
            cons.append(Izz_target-(min([Izz_target,Izz])))
            conpen += izz_weighting * (Izz_target-min([Izz_target, Izz]))
        else:
            cons.append(0.0)
        
        if a_weighting is not None and a_weighting != 0.0:
            cons.append(A_target-(min([A_target,A])))
            conpen += a_weighting * (A_target-min([A_target, A]))
        else:
            cons.append(0.0)
        
        # ----------------------
        # reject leading edges with too tight radii
        # ----------------------
        leading_edge_radius_upper, leading_edge_radius_lower = afl_geo.leadingEdgeRadius()
        
        leruViolation = -1*min([0,leading_edge_radius_upper-ler_con_upper])
        lerlViolation = -1*min([0,leading_edge_radius_lower-ler_con_lower])
        
        if leading_edge_radius_upper_weighting is not None and leading_edge_radius_upper_weighting != 0.0:
            cons.append(leruViolation)
            conpen += leading_edge_radius_upper_weighting * leruViolation
        else:
            cons.append(0.0)
        if leading_edge_radius_lower_weighting is not None and leading_edge_radius_lower_weighting != 0.0:
            cons.append(lerlViolation)
            conpen += leading_edge_radius_lower_weighting * lerlViolation
        else:
            cons.append(0.0)

        # ----------------------
        # reject violations of TE_cone
        # ----------------------
        
        height_upper_at_98percent = np.interp(te_frac,afl_geo.psi,afl_geo.zetaUpper)
        height_lower_at_98percent = np.interp(te_frac,afl_geo.psi,afl_geo.zetaLower)

        midpoint_at_98_percent = ( height_upper_at_98percent + height_lower_at_98percent )/2

        upper_10deg_cone = midpoint_at_98_percent + np.tan(cone_angle/2/180*np.pi)*(1-te_frac)
        lower_10deg_cone = midpoint_at_98_percent - np.tan(cone_angle/2/180*np.pi)*(1-te_frac)

        teViolation = 0
        for i,psi_val in enumerate(afl_geo.psi):
            if psi_val >= te_frac:
                zeta_upper_val = afl_geo.zetaUpper[i]
                zeta_lower_val = afl_geo.zetaLower[i]

                h_upper = upper_10deg_cone - (psi_val-te_frac)*(upper_10deg_cone/(1-te_frac)) + afl_geo.zetaUpper[-1]
                h_lower = lower_10deg_cone - (psi_val-te_frac)*(lower_10deg_cone/(1-te_frac)) + afl_geo.zetaLower[-1]

                if zeta_upper_val < h_upper:
                    teViolation += (h_upper-zeta_upper_val)                
                if zeta_lower_val > h_lower:
                    teViolation += (zeta_lower_val-h_lower)
        
        # Account for a small numerical issues
        if teViolation < 1e-8:
            teViolation = 0

        if te_cone_violation_weighting is not None and te_cone_violation_weighting != 0.0:
            cons.append(teViolation)
            conpen += te_cone_violation_weighting*teViolation
        else:
            cons.append(0.0)
       
        # ----------------------
        # reject if the max thickness is further forward than 25%
        # ----------------------
        tau_loc = afl_geo.taumax_psi
        if max_thickness_weighting is not None and max_thickness_weighting != 0.0 and tau_loc < max_thickness_loc:
            cons.append(max_thickness_loc-tau_loc)
            conpen += max_thickness_weighting * (max_thickness_loc-tau_loc) 
        else:
            cons.append(0)

        # ----------------------
        # reject if the max thickness on upper surface is further forward than 25%
        # ----------------------
        tau_loc_u = afl_geo.taumax_psi_upper
        if max_thickness_upper_weighting is not None and max_thickness_upper_weighting != 0.0 and tau_loc_u < max_thickness_loc_upper:
            cons.append(max_thickness_loc_upper-tau_loc_u)
            conpen += max_thickness_upper_weighting * (max_thickness_loc_upper-tau_loc_u) 
        else:
            cons.append(0)

        # ----------------------
        # reject if the max thickness on lower surface is further forward than 25%
        # ----------------------
        tau_loc_l = afl_geo.taumax_psi_lower
        if max_thickness_lower_weighting is not None and max_thickness_lower_weighting != 0.0 and tau_loc_l < max_thickness_loc_lower:
            cons.append(max_thickness_loc_lower-tau_loc_l)
            conpen += max_thickness_lower_weighting * (max_thickness_loc_lower-tau_loc_l) 
        else:
            cons.append(0)

        # ----------------------
        # attempt to stop significantly asymmetric leading edges, should not be true as this is very conservative
        # ----------------------
        radii = afl_geo.leadingEdgeRadius()
        if radii_skew_weighting is not None and radii_skew_weighting != 0.0 and max(radii) > ler_skew_factor*min(radii):
            conpen += radii_skew_weighting * (max(radii)-ler_skew_factor*min(radii))
            cons.append(max(radii)-ler_skew_factor*min(radii))
        else:
            cons.append(0)

        # ----------------------
        # Verify thickness, though this should be true by construction (except when nans were introduced)
        # ----------------------
        # negate because 0 is an unviolated constraint
        cons.append(int(not abs(afl_geo.tau-tau)<1e-4))

        # ----------------------
        # Penalize upper surface concavity (positive curvature)
        # ----------------------
        # Upper-surface curvature = the exact analytic d2(zeta)/dpsi2 (metafoil
        # Kulfan.d2zeta_dpsi2), evaluated on the interior psi grid. This replaces
        # the old grid finite difference  Delta(dzeta/dpsi) / Delta(Delta psi),
        # which divided by the *change* in grid spacing and so blew up ~1e3-1e4x
        # near the trailing edge (a pure grid artifact, not a real curvature).
        # The closed form is grid-independent, so curvature_bound now means the
        # same thing on any grid.
        second_derivative_approx = np.asarray(afl_geo.d2zeta_dpsi2(afl_geo.psi[1:-1], 'upper'))
        positive_curvature = []
        for i in range(0, len(second_derivative_approx)):
            if second_derivative_approx[i] >0:
                positive_curvature.append(second_derivative_approx[i])
        if curvature_weighting is not None and curvature_weighting != 0.0:
            conpen += curvature_weighting * sum(positive_curvature)
            cons.append(sum(positive_curvature))
        else:
            cons.append(0.0)
        # ----------------------
        # Penalize the aft curvature
        excess_curvature = second_derivative_approx - curvature_bound
        ec_psi = afl_geo.psi[1:-1] # approximate, but close
        ec_sum = 0
        if curvature_weighting is not None and curvature_weighting != 0.0:
            for tst_idx, ecpsi in enumerate(ec_psi):
                if ecpsi >= ec_cutoff:
                    if excess_curvature[tst_idx] < 0:
                        ec_sum += curvature_weighting * abs(excess_curvature[tst_idx])
            conpen += ec_sum
            cons.append(ec_sum/curvature_weighting)
        else:
            cons.append(0.0)

        # ----------------------
        # Penalize lower surface curvature changes if it happens more than once
        # ----------------------
        # Lower-surface curvature = analytic d2(zeta)/dpsi2 (see upper-surface note).
        second_derivative_approx_l = np.asarray(afl_geo.d2zeta_dpsi2(afl_geo.psi[1:-1], 'lower'))
        sflips = 0
        sgn = second_derivative_approx_l[0]/abs(second_derivative_approx_l[0])
        for i in range(0, len(second_derivative_approx_l)):
            if second_derivative_approx_l[i]/abs(second_derivative_approx_l[i]) != sgn:
                sgn = second_derivative_approx_l[i]/abs(second_derivative_approx_l[i])
                sflips += 1
        if lower_surface_curvature_weighting is not None and lower_surface_curvature_weighting != 0.0 and sflips > 1:
            conpen += lower_surface_curvature_weighting * (sflips-1)
            cons.append(sflips-1)
        else:
            cons.append(0)

        # ----------------------
        # Ensure 36% airfoil has sufficient CL for rough case
        # ----------------------
        if target_alpha is not None and CL_target_weighting is not None and CL_target_weighting != 0.0:
            if max(alpha_rough)<target_alpha:
                #  Higher alphas did not converge
                return [pid, np.inf, np.inf, False, -60] + [0]*N_reported + [0]*N_constraints

            cl_at_target_degrees_rough = np.interp(target_alpha, 
                                               np.array(alpha_rough)[0:positive_peak_index_rough], 
                                               np.array(cl_rough)[0:positive_peak_index_rough] )

            if cl_at_target_degrees_rough <= target_cl:
                conpen += CL_target_weighting * abs(target_cl-cl_at_target_degrees_rough)
                cn = target_cl-cl_at_target_degrees_rough
            else:
                cn = 0.0

            cons.append(cn)
        else:
            cons.append(0.0)

        # ----------------------
        # Add a moment constraint (clean)
        # ----------------------
        if CMc_in is not None and clean_moment_weighting is not None and clean_moment_weighting != 0.0:
            try:
                cm_design_clean = -1*abs(CMc_in)
                alpha_design_range_indicies = [i for i,alpha_val in enumerate(alpha_clean) if alpha_val >= alpha_design-cm_alpha_band and alpha_val <= alpha_design+cm_alpha_band]
                min_cm_clean = min([cm_clean[i] for i in alpha_design_range_indicies])
                if min_cm_clean < cm_design_clean:
                    conpen += clean_moment_weighting * abs(cm_design_clean - min_cm_clean)
                    cons.append(abs(cm_design_clean - min_cm_clean))
                else:
                    cons.append(0.0)
            except:
                # something strange happened
                conpen += clean_moment_weighting * 100
                cons.append(clean_moment_weighting)
        else:
            cons.append(0.0)
            
        # ----------------------
        # Add a moment constraint (rough)
        # ----------------------
        if CMr_in is not None and rough_moment_weighting is not None and rough_moment_weighting != 0.0:
            try:
                cm_design_rough = -1*abs(CMr_in)
                alpha_design_range_indicies = [i for i,alpha_val in enumerate(alpha_rough) if alpha_val >= alpha_design-cm_alpha_band and alpha_val <= alpha_design+cm_alpha_band]
                min_cm_rough = min([cm_rough[i] for i in alpha_design_range_indicies])
                if min_cm_rough < cm_design_rough:
                    conpen += rough_moment_weighting * abs(cm_design_rough - min_cm_rough)
                    cons.append(abs(cm_design_rough - min_cm_rough))
                else:
                    cons.append(0.0)
            except:
                # something strange happened
                conpen += rough_moment_weighting * 100
                cons.append(rough_moment_weighting)
        else:
            cons.append(0.0)

        # ----------------------
        # Add a minimum pressure coefficient constraint at design condition
        # ----------------------
        cpmin_swept_clean_design = np.interp(alpha_design, alpha_clean[0:positive_peak_index_clean], cpmin_clean[0:positive_peak_index_clean])
        cpmin_swept_rough_design = np.interp(alpha_design, alpha_rough[0:positive_peak_index_rough], cpmin_rough[0:positive_peak_index_rough])

        if cp_min_design is not None and cpmin_design_weighting is not None and cpmin_design_weighting != 0.0:
            if cpmin_swept_clean_design < cp_min_design:
                conpen += cpmin_design_weighting * abs(cp_min_design - cpmin_swept_clean_design)
                cons.append(abs(cp_min_design - cpmin_swept_clean_design))
            else:
                cons.append(0.0)

            if cpmin_swept_rough_design < cp_min_design:
                conpen += cpmin_design_weighting * abs(cp_min_design - cpmin_swept_rough_design)
                cons.append(abs(cp_min_design - cpmin_swept_rough_design))
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)
            cons.append(0.0)

        # ----------------------
        # Add a minimum pressure coefficient constraint at an offset from design condition
        # ----------------------
        
        if cp_min_at_alpha_offset is not None and cpmin_at_alpha_offset_weighting is not None and cpmin_at_alpha_offset_weighting != 0.0:
            cpmin_swept_clean_design = np.interp(alpha_design+cp_min_alpha_offset, alpha_clean[0:positive_peak_index_clean], cpmin_clean[0:positive_peak_index_clean])
            cpmin_swept_rough_design = np.interp(alpha_design+cp_min_alpha_offset, alpha_rough[0:positive_peak_index_rough], cpmin_rough[0:positive_peak_index_rough])

            if cpmin_swept_clean_design < cp_min_at_alpha_offset:
                conpen += cpmin_at_alpha_offset_weighting * abs(cp_min_at_alpha_offset - cpmin_swept_clean_design)
                cons.append(abs(cp_min_at_alpha_offset - cpmin_swept_clean_design))
            else:
                cons.append(0.0)

            if cpmin_swept_rough_design < cp_min_at_alpha_offset:
                conpen += cpmin_at_alpha_offset_weighting * abs(cp_min_at_alpha_offset - cpmin_swept_rough_design)
                cons.append(abs(cp_min_at_alpha_offset - cpmin_swept_rough_design))
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)
            cons.append(0.0)

        # ----------------------
        # Add a minimum pressure coefficient constraint pre-stall
        # ----------------------
        cpmin_swept_clean_prestall = min( np.array(cpmin_clean)[0:positive_peak_index_clean] )
        cpmin_swept_rough_prestall = min( np.array(cpmin_rough)[0:positive_peak_index_rough] )

        if cp_min_prestall is not None and cpmin_prestall_weighting is not None and cpmin_prestall_weighting != 0.0:
            if cpmin_swept_clean_prestall < cp_min_prestall:
                conpen += cpmin_prestall_weighting * abs(cp_min_prestall - cpmin_swept_clean_prestall)
                cons.append(abs(cp_min_prestall - cpmin_swept_clean_prestall))
            else:
                cons.append(0.0)

            if cpmin_swept_rough_prestall < cp_min_prestall:
                conpen += cpmin_prestall_weighting * abs(cp_min_prestall - cpmin_swept_rough_prestall)
                cons.append(abs(cp_min_prestall - cpmin_swept_rough_prestall))
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)
            cons.append(0.0)

        # ----------------------
        # Constrain the locations of the min radius of curvature
        # ----------------------
        delta_zeta_upper = afl_geo.zetaUpper[1:] - afl_geo.zetaUpper[0:-1]
        delta_psi = afl_geo.psi[1:] - afl_geo.psi[0:-1]
        first_derivative_approx = (delta_zeta_upper / delta_psi)
        first_derivative_average = (first_derivative_approx[1:] + first_derivative_approx[0:-1]) / 2.0
        # analytic second derivative (grid-independent; see upper-surface note above)
        second_derivative_approx = np.asarray(afl_geo.d2zeta_dpsi2(afl_geo.psi[1:-1], 'upper'))
        radius_of_curvature_approx =  (1+first_derivative_average**2)**1.5 / abs(second_derivative_approx)
        chopped_roc = radius_of_curvature_approx[afl_geo.psi[1:-1] <= min_radius_location_cutoff]
        computed_min_radius_loc_upper = afl_geo.psi[1:-1][afl_geo.psi[1:-1] <= min_radius_location_cutoff][np.argmin(chopped_roc)]
        if min_radius_location_upper_weighting is not None and min_radius_location_upper_weighting != 0.0 and computed_min_radius_loc_upper > min_radius_location_upper:
            conpen += min_radius_location_upper_weighting * (computed_min_radius_loc_upper - min_radius_location_upper)
            cons.append(computed_min_radius_loc_upper - min_radius_location_upper)
        else:
            cons.append(0.0)

        delta_zeta_lower = afl_geo.zetaLower[1:] - afl_geo.zetaLower[0:-1]
        delta_psi = afl_geo.psi[1:] - afl_geo.psi[0:-1]
        first_derivative_approx = (delta_zeta_lower / delta_psi)
        first_derivative_average = (first_derivative_approx[1:] + first_derivative_approx[0:-1]) / 2.0
        # analytic second derivative (grid-independent; see upper-surface note above)
        second_derivative_approx = np.asarray(afl_geo.d2zeta_dpsi2(afl_geo.psi[1:-1], 'lower'))
        radius_of_curvature_approx =  (1+first_derivative_average**2)**1.5 / abs(second_derivative_approx)
        chopped_roc = radius_of_curvature_approx[afl_geo.psi[1:-1] <= min_radius_location_cutoff]
        computed_min_radius_loc_lower = afl_geo.psi[1:-1][afl_geo.psi[1:-1] <= min_radius_location_cutoff][np.argmin(chopped_roc)]
        if min_radius_location_lower_weighting is not None and min_radius_location_lower_weighting != 0.0 and computed_min_radius_loc_lower > min_radius_location_lower:
            conpen += min_radius_location_lower_weighting * (computed_min_radius_loc_lower - min_radius_location_lower)
            cons.append(computed_min_radius_loc_lower - min_radius_location_lower)
        else:
            cons.append(0.0)


        # ----------------------
        # Toothpick constraint
        # ----------------------
        if toothpick_weighting is not None and toothpick_weighting != 0.0:
            toothpick_height_at_location = afl_geo.getNormalizedHeight(toothpick_location)
            if toothpick_height_at_location < toothpick_height:
                conpen += toothpick_weighting * (toothpick_height - toothpick_height_at_location)
                cons.append(toothpick_height - toothpick_height_at_location)
            else:
                cons.append(0.0)
        else:
            cons.append(0.0)

        # ----------------------
        # return
        # ----------------------
        con_tag = all([c==0 for c in cons])

        # Add penalty for violating the con tag
        # Helps at the end of the run
        if infeasibility_penalty is not None and infeasibility_penalty != 0.0:
            conpen += (1-con_tag) * infeasibility_penalty
        
        r_list = [
                pid,
                obj1 + conpen,
                obj2 + conpen,
                con_tag,
                alpha_design,
                LoD_clean_at_design_alpha,
                LoD_rough_at_design_alpha,
                stall_margin_clean,
                stall_margin_rough,
                lift_margin_clean,
                delta_cl_clean_to_rough_at_alpha_design,
                LoD_clean_1degree_left,
                LoD_clean_1degree_right,
                afl_geo.tau,
                leading_edge_radius_upper,
                leading_edge_radius_lower,
                Ixx,
                Iyy,
                Izz,
                A,
                min([cpmin_swept_clean_design, cpmin_swept_rough_design]),
            ] + cons 

        for rix, rtv in enumerate(r_list):
            if isinstance(rtv, units.Quantity):
                assert(rtv.units == units.dimensionless)
                r_list[rix] = rtv.to('dimensionless').magnitude

        # NSGA_sort decides dominance with `>` and `<`, and both are False
        # against NaN, so a NaN individual is never dominated and lands in
        # front 1. Reject the design outright instead.
        # Reject the design only if one of the FINAL reported/constraint values
        # (the ones written to the JSON) is NaN — a non-converged sweep point on
        # its own is fine and was already filtered out above.
        if any(np.isnan(rtv) for rtv in r_list if isinstance(rtv, (float, np.floating))):
            return [pid, np.inf, np.inf, False, -90] + [0]*N_reported + [0]*N_constraints

        return r_list
    except:
        return [pid, np.inf, np.inf, False, -90] + [0]*N_reported + [0]*N_constraints



if __name__ == '__main__':
    import time
    t0 = time.time()
    import matplotlib.pyplot as plt
    import yaml
    airfoil_seeds = [
        # [3.040459184728062092e-01, 3.865158898232732843e-01, 1.316538699269504120e-01 ,3.139820595993685348e-01 ,-2.593332834032758827e-01 ,-1.465580816167938449e-01 ,-4.868709152015524566e-01, 3.030523043601500155e-01],
        # [2.436455855485705202e-01, 5.627052809097938812e-01, 1.841844605039789085e-01 ,3.441977319654409007e-01 ,-2.291428721230603649e-01 ,-1.068366555679247792e-01 ,-2.774852502080942251e-01, 3.132586099910192323e-01],
        [.1,.1,.1,-.1,-.1,-.1,-.1,0.2],
        # [0.2895529384519609, 0.5707988120123219, 0.34602093353667224, 0.7675267435685507, 0.47442973419723866, 0.8705928599649991, 0.4336960256541951, 0.4663320196324578, -0.15104922613749502, -0.05073712004691763, -0.25913307521796636, 0.2605533024015594, -0.5160313871452309, 0.44225025302983123, 0.14956381466541888, 0.3174515752976585],
        # [2.418060954461536127e-01, 3.178580125622825769e-01, 4.933624366764263192e-01, 2.544814149655084679e-01, 6.142654843995031255e-01, 6.123154352426792846e-01, 6.045152386153840318e-01, 5.694143537925553389e-01, -1.974544732709675832e-01, -1.457661744725805009e-01, -2.960661180175324050e-06, -4.416861340205950892e-01, 3.961037349798383206e-02, 2.442436568921833862e-01, 3.393085532873446053e-01, 3.997600771488830085e-01],
        # [0.2759924824591029, 0.36505371575699025, 0.3807132511161153, 0.5050076534019967, 0.4032220194306991, 0.2769723823719758, 0.3983287124814639, 0.35820334426455686, -0.22510024324231004, -0.07854031784219763, -0.36994764134972546, -0.26914104884527795, -0.07071138196387987, 0.11279515592192352, 0.25641832721484725, 0.25250387348994047],
        # [0.27200518835149395, 0.5349785727756038, 0.08526832357342562, 0.8798051137895084, -0.013288258418964789, 0.6398114717677857, 0.19013282950501767, 0.40686735458401174, -0.19584565656757194, -0.249756231310527, -0.19629487910041066, -0.2311328308373759, -0.17125182017854618, -0.06893729120760746, 0.17150653015921413, 0.20454160490141127],
        # 
        # 
        [0.27200518835149395, 0.5349785727756038, 0.08526832357342562, 0.8798051137895084, -0.013288258418964789, 0.6398114717677857, 0.19013282950501767, 0.40686735458401174, -0.25, -0.249756231310527, -0.19629487910041066, -0.2311328308373759, -0.17125182017854618, -0.06893729120760746, 0.17150653015921413, 0.20454160490141127]
    ]

    labels = [
        'pid',
        'obj1',
        'obj2',
        'con_tag',
        'alpha_design',
        'LoD_clean_at_design',
        'LoD_rough_at_design',
        'stall_margin_clean',
        'stall_margin_rough',
        'lift_margin_clean',
        'delta_cl_from_roughness',
        'LoD_c_1d_left',
        'LoD_c_1d_right',
        'tau',
        'ler_upper' ,
        'ler_lower',
        'Ixx',
        'Iyy',
        'Izz',
        'A',
        'cpmin',
        'con_sm_clean',
        'con_sm_rough',
        'con_clmax_clean',
        'con_clmax_rough',
        'con_ixx',
        'con_iyy',
        'con_izz',
        'con_a',
        'con_leru',
        'con_lerl',
        'con_te_cone',
        'con_max_tau',
        'con_max_tau_u',
        'con_max_tau_l',
        'con_ler_skew',
        'con_tau',
        'con_concave',
        'con_aftcurve',
        'con_lower_flips',
        'con_10deg',
        'con_mom_c',
        'con_mom_r',
        'con_cpmin_design_clean',
        'con_cpmin_design_rough',
        'con_cpmin_offset_clean',
        'con_cpmin_offset_rough',
        'con_cpmin_prestall_clean',
        'con_cpmin_prestall_rough',
        'con_min_rad_loc_upper',
        'con_min_rad_loc_lower',
        'con_toothpick',
    ]

    # params = yaml.safe_load(open("example_hydro.yaml"))
    params = yaml.safe_load(open("run_hydro_003.json"))

    for i,ind in enumerate(airfoil_seeds):
        K_upper = ind[0:int(len(ind)/2)]
        K_lower = ind[int(len(ind)/2):]

        afl_geo = Kulfan(TE_gap = params['TE_gap'])
        afl_geo.upperCoefficients = K_upper
        afl_geo.lowerCoefficients = K_lower
        afl_geo.scaleThickness(params['tau'])
        ind = afl_geo.upperCoefficients.tolist() + afl_geo.lowerCoefficients.tolist()

        x = {}
        x['pid'] = i
        x['individual'] = ind
        x['params'] = params

        res = airfoil_fitness(x)
        # print(res)
        print(len(res))
        print(len(labels))
        assert(len(res)==len(labels))
        # print(time.time()-t0)
        for ii,r in enumerate(res):
            print(labels[ii].ljust(30),': ',r)
        print()


