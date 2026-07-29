"""
live_gradient.py — the live dashboard, driven by the GRADIENT Pareto solver.

The GA counterpart is live_ga.py; this is the same dashboard, the same
`state.json` contract, the same `server.py` and `index.html`, and the same
renderer. Only the optimizer underneath differs, so it REUSES live_ga.Plotter and
live_ga._render_worker rather than duplicating them -- the figures, the reference
family overlay and the snapshot browser all behave identically.

    python -m oso_airfoils.live.server 8777
    python -m oso_airfoils.live.live_gradient --thickness 21

WHAT AN "ITERATION" MEANS HERE

The GA has generations: a natural, evenly-spaced clock. The gradient solver does
not -- it runs M x K independent Ipopt sub-solves, each taking ~67 s and finishing
whenever it finishes. So:

  * updates are pushed on a TIMER (--push-every, default 10 s), not per event;
  * the "iteration" counter is the number of COMPLETED sub-solves, which is the
    honest measure of progress through the run;
  * results stream back with imap_unordered, so the front fills in as points land
    rather than appearing all at once at the end;
  * where a point has several multi-start seeds, the displayed front takes the
    BEST FEASIBLE one so far -- exactly the selection rule pareto_gold applies at
    the end, just applied continuously.

Phases appear in the status line in the order the algorithm runs them: the
endpoint locators first (they fix the achievable rough range and everything else
depends on them), then the cross-seed pair, then the interior sweep.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import sys
import threading
import time

import numpy as np

from oso_airfoils.optimization import pareto_gold as pg
from oso_airfoils.live import live_ga

HERE = pathlib.Path(__file__).resolve().parent
FEAS = 1e-3


class FrontState:
    """Best-so-far front, updated as sub-solves land. Thread-safe.

    Holds TWO populations, because the run has two phases with different shapes:

      * phase 1 produces free-standing locator results -- there is no epsilon grid
        yet (its spacing depends on the rough range phase 1 is computing), so they
        accumulate in a plain list;
      * phase 2 produces one result per epsilon level, indexed, with multi-start
        seeds competing for each slot.

    Both are shown. The locators are genuine front points -- max-clean and
    max-rough are its extremes -- so leaving them off until phase 2 started meant
    the dashboard sat empty through the first ten sub-solves and several minutes
    of real work.
    """

    def __init__(self, params):
        self.lock = threading.Lock()
        self.te_gap = params["TE_gap"]
        self.CL = params["CL"]; self.Re = params["Re"]; self.tau = params["tau"]
        self.loc = []                      # phase-1: [(clean, rough, viol, z)]
        self.eps = np.zeros(0)
        self.M = 0
        self.clean = np.zeros(0); self.rough = np.zeros(0)
        self.viol = np.zeros(0); self.z = []
        self.done = 0
        self.total = 0
        self.phase = "locators"
        self.span = None            # (r_lo, r_hi): fixes the colour mapping
        self._r_lo = self._r_hi = None
        self.t0 = time.time()
        self.times = []             # wall time per completed sub-solve, for ETA

    def begin_sweep(self, eps_levels):
        """Switch to the indexed phase-2 structure once the epsilon grid exists."""
        with self.lock:
            self.eps = np.asarray(eps_levels, float)
            # Authoritative colour anchor, and it OVERWRITES whatever phase 1
            # guessed. The epsilon grid spans exactly the rough range phase 1
            # settled on (r_lo from the best max-clean solve, r_hi from the best
            # max-rough solve), which is only known here -- after multi-start and
            # cross-seeding. The provisional span set while locators were still
            # landing came from whichever two sub-solves finished first, which is
            # a worker race, not a measurement.
            self.span = (float(self.eps[0]), float(self.eps[-1]))
            self.M = len(self.eps)
            self.clean = np.full(self.M, np.nan)
            self.rough = np.full(self.M, np.nan)
            self.viol = np.full(self.M, np.inf)
            self.z = [None] * self.M
            self.phase = "sweep"

    def record_locator(self, z, d, which=None):
        """A phase-1 result: no epsilon slot to compete for, so just keep it.

        `which` ('clean'/'rough') feeds a PROVISIONAL colour anchor, used only
        while phase 1 is still running. It cannot be final: multi-start means the
        first sub-solve of a corner to return is rarely the best one, so a span
        frozen on the first clean/rough pair is set by worker scheduling. Every
        point drawn during phase 1 is muted scaffolding anyway, so a rough anchor
        costs nothing; begin_sweep replaces it with the real range.
        """
        with self.lock:
            self.done += 1
            self.times.append(time.time() - self.t0)
            self.loc.append((d["lod_clean"], d["lod_rough"], d["violation"],
                             np.asarray(z, float)))
            if which == 'clean':
                self._r_lo = min(self._r_lo, d["lod_rough"]) \
                    if self._r_lo is not None else d["lod_rough"]
            elif which == 'rough':
                self._r_hi = max(self._r_hi, d["lod_rough"]) \
                    if self._r_hi is not None else d["lod_rough"]
            if self._r_lo is not None and self._r_hi is not None \
                    and self._r_hi > self._r_lo:
                self.span = (self._r_lo, self._r_hi)

    def record(self, idx, z, d):
        """Keep this seed only if it beats the incumbent for its point.

        Selection mirrors pareto_gold's phase-2 rule: prefer FEASIBLE, and among
        feasible prefer the highest clean L/D. An infeasible result replaces the
        incumbent only while nothing feasible has been seen for that point.
        """
        with self.lock:
            self.done += 1
            self.times.append(time.time() - self.t0)
            cl, rg, v = d["lod_clean"], d["lod_rough"], d["violation"]
            cur_ok = self.viol[idx] <= FEAS
            new_ok = v <= FEAS
            better = (new_ok and (not cur_ok or cl > self.clean[idx])) or \
                     (not cur_ok and not new_ok and v < self.viol[idx])
            if better:
                self.clean[idx], self.rough[idx], self.viol[idx] = cl, rg, v
                self.z[idx] = np.asarray(z, float)

    def snapshot(self):
        """A live_ga-shaped snapshot of the points solved so far."""
        with self.lock:
            # phase-2 slots that have a result, plus every phase-1 locator
            pts = [(float(self.clean[i]), float(self.rough[i]),
                    float(self.viol[i]), self.z[i], False)
                   for i in range(self.M) if self.z[i] is not None]
            # Phase-1 locators are SCAFFOLDING, not results: pareto_gold discards
            # the max-rough airfoil outright and re-solves both corners inside the
            # uniform epsilon sweep, so none of these points appear in the final
            # front. They stay visible (they show where the corners are) but mute
            # to grey once the sweep begins, so the coloured points are exactly
            # the ones that will be reported.
            muted = (self.phase == "sweep")
            pts += [(c, r, v, zz, muted) for (c, r, v, zz) in self.loc]
            pts.sort(key=lambda t: t[1])                     # by rough L/D
            cl = [t[0] for t in pts]
            rg = [t[1] for t in pts]
            zs = [t[3] for t in pts]
            vs = [t[2] for t in pts]
            mu = [bool(t[4]) for t in pts]
            # The renderer reads exactly these keys (grepped from dash_figure.py
            # and live_ga.py, rather than guessed -- two earlier attempts each
            # supplied a subset and died on the first one missing, KeyError 'N_k'
            # then KeyError 'pop_rough', with the optimizer running fine
            # throughout because the two halves fail independently):
            #   ip, front_upper, front_lower, front_clean, front_rough,
            #   pop_clean, pop_rough, pop_front, population, input_parameters
            #
            # The pop_* series are the GA's cloud of non-front individuals behind
            # the front. A gradient run has no population, so they mirror the
            # front itself: the scatter then coincides with the front line, which
            # is the honest picture -- every design this solver produces IS a
            # front candidate. pop_front is the GA's front-membership rank; 1
            # marks all of ours as on-front.
            return dict(
                ip=dict(TE_gap=self.te_gap, N_k=16, CL=self.CL, Re=self.Re,
                        tau=self.tau),
                input_parameters=dict(TE_gap=self.te_gap, N_k=16, CL=self.CL,
                                      Re=self.Re, tau=self.tau),
                front_upper=[zz[:8].tolist() for zz in zs],
                front_lower=[zz[8:16].tolist() for zz in zs],
                front_clean=cl, front_rough=rg, front_muted=mu,
                pop_clean=cl, pop_rough=rg, pop_front=[1] * len(zs),
                population=[zz[:16].tolist() for zz in zs],
                n_feasible=int(sum(v <= FEAS for v in vs)),
                done=self.done, total=self.total, phase=self.phase,
                # Colour anchor. During phase 1 the endpoints ARE the extremes, so
                # the span is whatever has been seen; once the epsilon grid exists
                # it is fixed, and colours stop shifting between frames.
                # Fixed once known, NEVER recomputed from the visible points: an
                # anchor derived from min/max of whatever had landed so far moved
                # on every new result, so previously-plotted points changed colour
                # between frames.
                front_span=list(self.span) if self.span else None,
                # Extra signal for the human between sub-solves (~30 s apart):
                # coverage of the epsilon grid, throughput, and a remaining-time
                # estimate from the observed per-solve rate.
                solved=len([1 for i in range(self.M) if self.z[i] is not None]),
                pending=max(self.total - self.done, 0),
                rate_per_min=(60.0 * self.done / max(time.time() - self.t0, 1e-9)),
                eta_s=((time.time() - self.t0) / max(self.done, 1)
                       * max(self.total - self.done, 0)) if self.done else None,
                grid=[('feasible' if (self.M and self.z[i] is not None
                                      and self.viol[i] <= FEAS)
                       else 'solved' if (self.M and self.z[i] is not None)
                       else 'pending') for i in range(self.M)])


def _stream_solve(tasks, wcfg, workers, on_result):
    """Run sub-solves and hand each back AS IT LANDS.

    pareto_gold._parallel_solve uses a blocking pool.map, which is right for the
    batch driver but means phase 1's ten sub-solves -- several minutes of real
    work, including both front endpoints -- would finish before the dashboard saw
    anything. imap_unordered streams instead.
    """
    ctxmp = mp.get_context("spawn")
    out = []
    with ctxmp.Pool(workers, initializer=pg._init_worker, initargs=(wcfg,)) as pool:
        for tag, z, d in pool.imap_unordered(pg._run_task, tasks, chunksize=1):
            out.append((tag, z, d))
            on_result(tag, z, d)
    return out


def _pusher(state, plotter, every, stop):
    """Push a dashboard frame on a fixed cadence, not per completed sub-solve.

    Sub-solves land irregularly (~67 s each, several in flight), so an
    event-driven push would either flood the renderer at the start of a wave or
    go silent for a minute inside one. A timer decouples the two.
    """
    it = 0
    while not stop.is_set():
        stop.wait(every)
        snap = state.snapshot()
        if not snap["front_upper"]:
            continue
        it += 1
        plotter.submit(dict(
            gen=snap["done"], snapshot=snap, best=None,
            lod_clean=(max(snap["front_clean"]) if snap["front_clean"] else 0.0),
            lod_rough=(max(snap["front_rough"]) if snap["front_rough"] else 0.0),
            n_feasible=snap["n_feasible"],
            status=f"{snap['phase']}  {snap['done']}/{snap['total']} sub-solves",
            s_per_gen=None,
            eta_s=snap.get('eta_s'), rate_per_min=snap.get('rate_per_min'),
            grid=snap.get('grid'), solved=snap.get('solved'),
            pending=snap.get('pending')))


def _park(port, serving):
    """Keep the dashboard reachable after the run finishes.

    serve() runs on a DAEMON thread, so returning from main() tears the HTTP
    server down with the process -- the run would complete and the page would
    immediately start failing to fetch state.json, which reads as "the dashboard
    broke" rather than "the run finished". Park here instead so the final front
    stays inspectable until the user is done with it.
    """
    if not serving:
        return
    import time as _t
    print(f"\nrun complete — dashboard still served at http://localhost:{port}"
          f"\n(Ctrl-C to stop serving)", flush=True)
    try:
        while True:
            _t.sleep(3600)
    except KeyboardInterrupt:
        print("stopped.", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thickness", type=int, default=21)
    ap.add_argument("--n", type=int, default=18,
                    help="interior points; +2 endpoints (default 18 -> 20-point front)")
    ap.add_argument("--tool", default="nqfoil",
                    choices=["nqfoil", "neuralfoil", "cxfoil", "cqfoil"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--sweep-starts", type=int, default=4,
                    help="multi-start seeds per front point. Matches pareto_gold's\n"
                         "default: the rough tail is multi-modal, so different seeds\n"
                         "reach different feasible clean L/D at the same floor and the\n"
                         "best is kept. Dropping to 2 halves the runtime and visibly\n"
                         "degrades the rough end -- do not.")
    ap.add_argument("--locator-starts", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--proj-max-iter", type=int, default=60)
    ap.add_argument("--push-every", type=float, default=10.0,
                    help="seconds between dashboard updates (default 10)")
    ap.add_argument("--n-polar", type=int, default=9)
    ap.add_argument("--oso-root", default=pg.DEFAULT_OSO_ROOT)
    ap.add_argument("--port", type=int, default=8777,
                    help="serve this run's dashboard here. Each concurrent run "
                         "needs its own port AND its own --run-dir: state.json "
                         "and frames/ are per-run, so two runs sharing a "
                         "directory overwrite each other's output.")
    ap.add_argument("--run-dir", default=None,
                    help="where this run's state.json and frames/ go "
                         "(default: live/runs/<tag>, derived from the port and "
                         "thickness so concurrent runs never collide)")
    ap.add_argument("--no-serve", action="store_true",
                    help="do not host a server; just write into --run-dir")
    ap.add_argument("--out", default=None,
                    help="explicit state.json path (overrides --run-dir)")
    args = ap.parse_args(argv)
    if args.model is None:
        args.model = "xxlarge" if args.tool == "nqfoil" else "xxxlarge"

    params, cfg = pg.load_family_params(args.thickness, args.oso_root)
    seed_path = pg.family_dat(args.thickness, args.oso_root)
    from oso_airfoils.optimization.airfoil_io import load_airfoil_dat
    from metafoil.core.kulfan_geometry import fit_kulfan_to_coordinates
    from metafoil.core.kulfan import Kulfan
    fit = fit_kulfan_to_coordinates(*load_airfoil_dat(seed_path).T[:2],
                                    fit_order=8, n_pts=160)
    seed = Kulfan(upper_coefficients=np.asarray(fit["upper_coefficients"], float),
                  lower_coefficients=np.asarray(fit["lower_coefficients"], float),
                  te_gap=params["TE_gap"])
    z0 = np.concatenate([seed.upper_coefficients, seed.lower_coefficients, [0.35]])

    wcfg = dict(tool=args.tool, model=args.model, params=params,
                max_iter=args.max_iter, proj_max_iter=args.proj_max_iter)
    workers = args.workers or (os.cpu_count() or 1)
    meta = dict(tau=params["tau"], CL=params["CL"], Re=params["Re"],
                n_polar=args.n_polar, backend=args.tool, model=args.model,
                case=os.path.basename(cfg or ""), solver="gradient (Ipopt+ma27)",
                # index.html is shared with the GA; these drive its wording. A
                # gradient run has no generations -- the counter is completed
                # sub-solves, which are neither evenly spaced nor a population
                # step, so calling them generations would misdescribe the run.
                # Reference-airfoil buttons come from meta['families'] / ['library'],
                # filled in asynchronously below -- the scan is ~95 s and must not
                # sit between process start and the server coming up.
                families=None, library=None,
                started=time.time(),
                solver_label="OSO gradient", iter_word="sub-solve",
                iter_unit="solve",
                # whole run: locator multi-starts + the cross-seed pair +
                # the sweep. Counting only the sweep made the progress
                # denominator disagree with the status line (40 vs 50).
                target_gens=(2 * max(2, args.locator_starts) + 2
                             + (args.n + 2) * args.sweep_starts))

    # Per-run directory, named with oso's own <filecode>__<timestamp> convention
    # (run_dirs.for_gradient). The timestamp is the point: keying off port or
    # thickness alone would make a rerun reuse the directory and clobber the
    # previous run's frames while a dashboard may still be showing them.
    from oso_airfoils.live import run_dirs
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else \
        run_dirs.for_gradient(params, args.thickness, args.n + 2,
                              args.sweep_starts, args.tool)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)
    state_path = args.out or str(run_dir / "state.json")
    if not args.no_serve:
        from oso_airfoils.live import server as _srv
        _srv.serve(args.port, run_dir, background=True)
    # live_ga.Plotter writes frames to its module-level FRAMES; point it at this
    # run's directory so two runs do not scribble over each other's SVGs.
    live_ga.FRAMES = run_dir / "frames"
    from oso_airfoils.live import ref_library
    ref_library.populate_async(
        meta, params["tau"],
        on_done=lambda nf, nl: print(f"  [ref] reference library ready: "
                                     f"{nf} families, {nl} airfoils", flush=True))
    plotter = live_ga.Plotter(state_path, meta)
    plotter.start()
    # Seed state.json before any solving: the page then shows the header, phase
    # banner and runtime clock straight away instead of a blank "waiting" screen
    # for the first minute.
    plotter.phase_text = "starting — loading the model and seeding the corner solves"
    plotter.write_state("starting", 0, None)
    print(f"dashboard: http://localhost:{args.port}   (state -> {state_path})",
          flush=True)

    # ---- phase 1: endpoint locators, all seeds in parallel -----------------
    rng = np.random.default_rng(0)
    lo_b = np.full(17, -2.0); hi_b = np.full(17, 2.0)
    lo_b[16], hi_b[16] = 0.10, 0.98

    def perturb(z, s=0.05):
        zc = np.array(z, float); zc[:16] += rng.normal(0, s, 16)
        return np.clip(zc, lo_b, hi_b)

    seeds = [z0] + [perturb(z0) for _ in range(max(2, args.locator_starts) - 1)]
    tasks = [((i, w), w, s, None)
             for i, (w, s) in enumerate([(w, s) for s in seeds
                                         for w in ("clean", "rough")])]
    # State and pusher come up FIRST, so the dashboard shows phase 1 rather than
    # sitting empty through ten sub-solves. The locators are front endpoints.
    state = FrontState(params)
    state.total = len(tasks) + 2 + (args.n + 2) * args.sweep_starts
    stop = threading.Event()
    th = threading.Thread(target=_pusher, args=(state, plotter, args.push_every, stop),
                          daemon=True)
    th.start()
    plotter.phase_text = ("finding the endpoints — multi-start solves for max clean L/D "
                          "and max rough L/D, which fix the achievable rough range")
    print(f"phase 1: {len(tasks)} locator sub-solves on {workers} workers", flush=True)
    res = _stream_solve(tasks, wcfg, workers,
                        lambda tag, z, d: state.record_locator(z, d, which=tag[1]))
    best = {}
    for tag, z, d in res:
        w = tag[1]
        if w not in best or d[f"lod_{w}"] > best[w][1][f"lod_{w}"]:
            best[w] = (z, d)
    zc1, dc1 = best["clean"]; zr1, dr1 = best["rough"]

    state.phase = "cross-seed"
    plotter.phase_text = ("cross-seeding the endpoints — re-solving each corner from the "
                          "other's optimum, which is what escapes the local max-rough optimum")
    xres = _stream_solve([("xc", "clean", zr1, None), ("xr", "rough", zc1, None)],
                         wcfg, workers,
                         lambda tag, z, d: state.record_locator(
                             z, d, which=('clean' if tag == 'xc' else 'rough')))
    xd = {t: (z, d) for t, z, d in xres}
    if xd["xc"][1]["lod_clean"] > dc1["lod_clean"]:
        zc1, dc1 = xd["xc"]
    if xd["xr"][1]["lod_rough"] > dr1["lod_rough"]:
        zr1, dr1 = xd["xr"]
    r_lo, r_hi = dc1["lod_rough"], dr1["lod_rough"]
    print(f"phase 1 done: rough range [{r_lo:.1f}, {r_hi:.1f}]", flush=True)

    # ---- phase 2: interior sweep, streaming --------------------------------
    M, K = args.n + 2, args.sweep_starts
    eps_levels = np.linspace(r_lo, r_hi, M)
    state.begin_sweep(eps_levels)

    ptasks = []
    for j in range(M):
        w = j / max(M - 1, 1)
        base = (1 - w) * np.asarray(zc1, float) + w * np.asarray(zr1, float)
        for k in range(K):
            zz = base if k == 0 else perturb(base, 0.04)
            ptasks.append(((j, k), "clean", zz, float(eps_levels[j])))
    plotter.phase_text = (f"searching the Pareto front — maximising clean L/D at each of "
                          f"{M} rough-L/D floors, {K} multi-start seed(s) per point")
    print(f"phase 2: {len(ptasks)} sub-solves (M={M} x K={K})", flush=True)
    t0 = time.time()
    ctxmp = mp.get_context("spawn")
    with ctxmp.Pool(workers, initializer=pg._init_worker, initargs=(wcfg,)) as pool:
        for tag, z, d in pool.imap_unordered(pg._run_task, ptasks, chunksize=1):
            state.record(tag[0], z, d)
            print(f"  [{state.done:3d}/{state.total}] point {tag[0]:2d} seed {tag[1]} "
                  f"L/D=({d['lod_clean']:.1f},{d['lod_rough']:.1f}) "
                  f"viol={d['violation']:.1e}  [{time.time()-t0:.0f}s]", flush=True)
    stop.set(); th.join(timeout=2)

    state.phase = "done"
    plotter.phase_text = "complete — front assembled from the best feasible seed at each floor"
    snap = state.snapshot()
    plotter.submit(dict(gen=state.done, snapshot=snap, best=None,
                        lod_clean=max(snap["front_clean"]),
                        lod_rough=max(snap["front_rough"]),
                        n_feasible=snap["n_feasible"],
                        status=f"done  {snap['n_feasible']}/{M} feasible",
                        s_per_gen=None))
    time.sleep(1.0)
    plotter.stop = True

    # Persist the front. state.json carries only rendered SVG filenames, so
    # without this the geometry dies with the process and the run cannot be
    # re-checked against the constraint stack afterwards. Same schema as
    # pareto_gold's own output, so downstream tooling reads either one.
    with state.lock:
        keep = [i for i in range(state.M)
                if state.z[i] is not None and state.viol[i] <= FEAS]
        front_json = str(run_dir / "front.json")
        n = pg.write_front_json(
            front_json,
            meta=dict(tool=args.tool, model=args.model, thickness=args.thickness,
                      n_points=args.n, sweep_starts=args.sweep_starts,
                      solver="gradient/ipopt", constraints=dict(pg.CONSTRAINTS),
                      runtime_s=round(time.time() - t0, 1)),
            labels=[f"eps={state.eps[i]:.1f}" for i in keep],
            coeffs=[state.z[i] for i in keep],
            clean_lod=[state.clean[i] for i in keep],
            rough_lod=[state.rough[i] for i in keep],
            viol=[state.viol[i] for i in keep],
            eps_list=[state.eps[i] for i in keep],
            te_gap=params["TE_gap"])
    print(f"front -> {front_json}  ({n} feasible airfoils)", flush=True)

    print(f"\nDONE {time.time()-t0:.0f}s   feasible {snap['n_feasible']}/{M}", flush=True)
    _park(args.port, not args.no_serve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
