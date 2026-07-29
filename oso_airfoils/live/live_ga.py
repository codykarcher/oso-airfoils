"""
live_ga.py -- run an OSO GA case and stream the project's OWN plots per generation
to a local dashboard.

Each generation produces the two figures the project already makes:

  * the Pareto-evolution frame  (postprocessing.pareto_frame -- the frame that
    generate_gif.py strings into pareto_shapes_evolution.gif)
  * the polar rainbow           (postprocessing.runners.run_and_plot_polars_rainbow
    -- the same call that produces polar_plot.png)

The GA itself runs through the optimization package's phase API
(produce_children / evaluate / finish_generation), so this is a real run, not a mock.

Plotting happens on a background thread and renders EVERY generation in order. The
GA never waits on matplotlib; rendering is slower than a generation, so the figures
lag the run and keep arriving after it finishes.

    python live_ga.py <case.yaml> [--gens 150] [--pop 200]
                      [--backend nqfoil] [--model xxlarge]
"""

import argparse
import concurrent.futures as _cf
import json
import os
import pathlib
import queue
import random
import shutil
import sys
import threading
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

from oso_airfoils.optimization.case import Case
from oso_airfoils.optimization.config import read_input_file
from oso_airfoils.optimization.evaluators import GPUBatchEvaluator
from oso_airfoils.optimization.generation import (
    finish_generation, initialize_sort, produce_children,
)
from oso_airfoils.postprocessing.pareto_frame import compute_limits, render_pareto_frame
from oso_airfoils.postprocessing.runners import run_and_plot_polars_rainbow

from oso_airfoils.live import families

HERE = pathlib.Path(__file__).parent.resolve()
FRAMES = HERE / 'frames'
REPO = HERE.parent.parent            # <repo>/oso_airfoils/live -> <repo>
DATA = REPO / 'oso_airfoils' / 'data'
_GEN_RE = __import__('re').compile(r'_g(\d+)\.json$')

TURB_CASES = [[9.0, 1.0, 1.0], [3.0, 0.05, 0.05]]   # clean, rough (as oso_polar uses)
SWEEP_RANGE = (-5, 25, 0.5)
VECTOR_EXT = 'svg'     # vector output -- crisp at any zoom, no resampling


def _worker_init():
    """Pin the render process to a single compute thread.

    Rendering is figure construction, not linear algebra -- but importing torch and
    numpy in the child spins up their default thread pools (4 each here), which then
    oversubscribe the same 8 cores the GA is using. Left alone this made renders ~9x
    slower than they are standalone.
    """
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    import matplotlib
    matplotlib.use('Agg')


def _render_worker(job, meta, limits, frames_dir, selection, sel_kind):
    """Render ONE dashboard figure. Runs in a separate process.

    The figure object is kept alive between calls (the pool has a single, persistent
    worker), so axes are created once and only the data is redrawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import numpy as _np
    from metafoil.core.kulfan import Kulfan
    from oso_airfoils.live.dash_figure import (
        DashFigure, batched_polars, sample_front,
    )
    from oso_airfoils.live import families as _fam

    global _DASH
    _t = {}
    _t0 = time.time()
    gen, snap = job['gen'], job['snapshot']
    ip = snap['ip']
    frames_dir = pathlib.Path(frames_dir)

    idx = sample_front(snap, meta['n_polar'])
    kulfans = []
    for i in idx:
        a = Kulfan(TE_gap=ip['TE_gap'])
        a.upperCoefficients = snap['front_upper'][i]
        a.lowerCoefficients = snap['front_lower'][i]
        kulfans.append(a)

    ref_afl = ref_label = None
    if selection:
        hit = (_fam.kulfan_for_stem(selection) if sel_kind == 'airfoil'
               else _fam.kulfan_for(selection, meta['tau']))
        if hit is not None:
            stem, t, ref_afl = hit
            ref_label = f'{stem} (t/c {t:.3f})'

    _t['setup'] = time.time() - _t0; _t0 = time.time()
    alphas = _np.arange(SWEEP_RANGE[0], SWEEP_RANGE[1] + 1e-9, SWEEP_RANGE[2])
    todo = kulfans + ([ref_afl] if ref_afl is not None else [])
    pol = batched_polars(todo, meta['Re'], TURB_CASES, alphas,
                         meta['backend'], meta['model']) if todo else []

    _t['polars'] = time.time() - _t0; _t0 = time.time()
    if _DASH is None:
        _DASH = DashFigure(ip, n_front=meta['n_polar'])
    _t['figinit'] = time.time() - _t0; _t0 = time.time()
    reference = None
    if ref_afl is not None:
        reference = (ref_label, pol[-1], (ref_afl.xcoordinates, ref_afl.ycoordinates))
        pol = pol[:-1]
    _DASH.draw(snap, idx, {i: p for i, p in enumerate(pol)}, reference=reference)

    _t['draw'] = time.time() - _t0; _t0 = time.time()
    sel = ''.join(c if c.isalnum() else '-' for c in (selection or 'none'))
    out = frames_dir / f'dash_{gen:04d}_{sel}.{VECTOR_EXT}'
    _DASH.save(str(out))
    _t['save'] = time.time() - _t0
    print(f'    [worker pid={os.getpid()}] ' +
          '  '.join(f'{k}={v:.2f}s' for k, v in _t.items()), flush=True)
    return gen, out.name, out.name


_DASH = None


class Plotter(threading.Thread):
    """Renders the two figures off the GA's critical path.

    EVERY generation is rendered -- jobs queue up in order rather than being
    coalesced. Rendering (~2-4 s) is slower than a generation (~1.5 s), so the
    figures lag the GA and keep arriving after it finishes; that is the cost of not
    skipping any. The GA itself never blocks on matplotlib either way.
    """

    def __init__(self, state_path, meta):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.state_path = state_path
        self.meta = meta
        self.limits = None
        self.done_gens = []
        self.stop = False
        self.pool = _cf.ProcessPoolExecutor(max_workers=1, initializer=_worker_init)
        self.save_path = None         # where snapshots are being written
        self.save_status = None       # result of the last relocation request
        self.gen_times = []           # every generation's wall time, for the mean
        self.last_job = None          # replayed when the reference family changes
        self.selection = None
        self.sel_kind = None
        self._sel_mtime = None
        self.sel_path = pathlib.Path(state_path).parent / 'selection.json'

    def submit(self, job):
        self.q.put(job)

    def run(self):
        while not self.stop:
            try:
                job = self.q.get(timeout=0.4)
            except queue.Empty:
                # idle: pick up a reference-family change and redraw the current
                # generation, so the button responds without waiting for the next one
                if self._selection_changed() and self.last_job is not None:
                    try:
                        self._render(self.last_job, force=True)
                    except Exception as e:
                        print(f"  [plot] re-render failed: {type(e).__name__}: {e}", flush=True)
                continue
            if job is None:
                continue
            try:
                self._render(job)
            except Exception as e:
                print(f"  [plot] generation {job['gen']} failed: "
                      f"{type(e).__name__}: {e}", flush=True)

    def _expand_limits(self, snap):
        """Axis limits that only ever GROW as the run proceeds.

        generate_gif.py fixes its axes off the LAST generation so frames don't
        jitter -- which a live run can't do, since the final front doesn't exist yet.
        Seeding from generation 0 instead is worse than useless: the front is a
        single point there, so every later generation falls outside the axes and the
        Pareto panel renders empty. Taking the running union keeps frames stable
        while guaranteeing the data is always in view.
        """
        cur = compute_limits(snap)
        if cur is None:
            return self.limits
        if self.limits is None:
            self.limits = cur
            return self.limits
        out = {}
        for k, v in cur.items():
            prev = self.limits.get(k)
            if v is None:
                out[k] = prev
            elif prev is None:
                out[k] = v
            else:
                out[k] = (min(prev[0], v[0]), max(prev[1], v[1]))
        self.limits = out
        return self.limits

    def _selection_changed(self):
        try:
            m = self.sel_path.stat().st_mtime
        except OSError:
            return False
        if m == self._sel_mtime:
            return False
        self._sel_mtime = m
        try:
            d = json.loads(self.sel_path.read_text())
            self.selection = d.get('airfoil') or d.get('family') or None
            self.sel_kind = 'airfoil' if d.get('airfoil') else (
                'family' if d.get('family') else None)
        except Exception:
            self.selection, self.sel_kind = None, None
        return True

    def _reference(self):
        """Reference entry for the selected family: geometry from the store, polar
        computed fresh by the same surrogate (see families.py)."""
        if not self.selection:
            return None
        hit = (families.kulfan_for_stem(self.selection) if self.sel_kind == 'airfoil'
               else families.kulfan_for(self.selection, self.meta['tau']))
        if hit is None:
            return None
        stem, t, afl = hit
        return [[f'{stem} (t/c {t:.3f})', afl, 'k']]

    def _render(self, job, force=False):
        gen, snap = job['gen'], job['snapshot']
        t0 = time.time()
        self.last_job = job
        # Refresh the reference selection HERE, not only on the idle tick: while the
        # GA is feeding jobs the queue is rarely empty, so an idle-only check would
        # leave a newly-clicked family unapplied until the run slowed down.
        self._selection_changed()

        _, pareto_name, polar_name = self.pool.submit(
            _render_worker, job, self.meta, self.limits, str(FRAMES),
            self.selection, self.sel_kind).result()
        pareto = FRAMES / pareto_name
        polar = FRAMES / polar_name

        entry = dict(gen=gen, lod_clean=job['lod_clean'], lod_rough=job['lod_rough'],
                     n_feasible=job['n_feasible'], pareto=pareto.name,
                     polar=polar.name, reference=self.selection,
                     # optional per-frame extras (gradient run supplies these)
                     **{k: job[k] for k in ('eta_s', 'rate_per_min', 'grid',
                                            'solved', 'pending') if k in job})
        for i, e in enumerate(self.done_gens):
            if e['gen'] == gen:
                self.done_gens[i] = entry
                break
        else:
            self.done_gens.append(entry)
        print(f"  [plot] generation {gen} rendered in {time.time()-t0:.2f}s", flush=True)
        self.write_state(job['status'], job['gen'], job['s_per_gen'])

    def write_state(self, status, gen, spg):
        # Report a ROLLING MEAN, not the last generation's time. Generation cost
        # alternates sharply (fast when the plotter is idle, ~4x slower while a
        # render is competing for CPU), so the instantaneous value swings and reads
        # as far faster than the run actually is.
        payload = dict(meta=self.meta, status=status, generation=gen,
                       phase_text=getattr(self, 'phase_text', None),
                       updated=time.time(), s_per_gen=spg,
                       s_per_gen_mean=(sum(self.gen_times) / len(self.gen_times)
                                       if self.gen_times else None),
                       frames=self.done_gens, selection=self.selection)
        tmp = str(self.state_path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, separators=(',', ':'))
        os.replace(tmp, self.state_path)


def front_airfoils(snap, n=9):
    """The Pareto front as Kulfan objects, sampled evenly along the front.

    Two things matter here:

    * The rainbow is meant to show the whole FRONT, not one airfoil -- the trade
      between clean and rough L/D is the point of the plot, so it needs several
      members spread across it. Sampled by arc length (as the GIF frame does) so
      the picks are spread over the trade rather than bunched where the front is
      dense.
    * They are passed as Kulfan OBJECTS, not (upper, lower) tuples. The tuple form
      in ``runners._resolve_spec`` builds ``Kulfan()`` with the DEFAULT TE gap of
      zero, which silently sharpens the trailing edge -- this case runs a 0.262%
      chord gap, so the tuple form would plot (and analyse) the wrong geometry.
    """
    from metafoil.core.kulfan import Kulfan
    te_gap = snap['input_parameters']['TE_gap']
    pts = sorted([p for p in snap['population'] if p['pareto_index'] == 1],
                 key=lambda p: p['LoD_rough_at_design'])
    if not pts:
        return []
    rough = np.array([p['LoD_rough_at_design'] for p in pts], float)
    clean = np.array([p['LoD_clean_at_design'] for p in pts], float)
    xs = float(np.ptp(rough)) or 1.0
    ys = float(np.ptp(clean)) or 1.0
    arc = np.concatenate(([0.0], np.cumsum(np.sqrt(
        np.diff(rough / xs) ** 2 + np.diff(clean / ys) ** 2))))
    if arc[-1] <= 0 or len(pts) < 2:
        idx = np.unique(np.linspace(0, len(pts) - 1, n).astype(int))
    else:
        idx = np.unique([int(np.argmin(np.abs(arc - t)))
                         for t in np.linspace(0, arc[-1], n)])
    out = []
    for i in idx:
        afl = Kulfan(TE_gap=te_gap)
        afl.upperCoefficients = pts[i]['K_upper']
        afl.lowerCoefficients = pts[i]['K_lower']
        out.append([f"rough L/D {rough[i]:.0f}", afl])
    return out


def run_output_dir(case):
    """Where this run's snapshots go, following the data tree's own convention:
    ``data/cases_<lo>_to_<hi>/case_<N>/<filecode>__<timestamp>/``.

    An existing bucket that already holds this case number wins, because the
    buckets are not perfectly regular (there is a ``cases_91_to_99`` and a bare
    ``cases_100``), so computing the name blindly would scatter runs into new
    directories beside the ones they belong with.
    """
    n = int(case.case_number)
    for bucket in sorted(DATA.glob('cases_*')):
        if (bucket / f'case_{n}').is_dir():
            return bucket / f'case_{n}' / case.folderstr
    lo = ((n - 1) // 10) * 10 + 1
    return DATA / f'cases_{lo}_to_{lo + 9}' / f'case_{n}' / case.folderstr


def prune_snapshots(dirpath, keep):
    """Keep only the newest ``keep`` snapshots, ordered by GENERATION number.

    Sorted numerically, not lexically: with the runner's zero-padding, g1000
    sorts before g999 as a string, so a lexical prune would delete the newest
    file once a run passes the padding width.
    """
    snaps = [f for f in pathlib.Path(dirpath).glob('population_*_g*.json')
             if _GEN_RE.search(f.name)]
    snaps.sort(key=lambda f: int(_GEN_RE.search(f.name).group(1)))
    for f in snaps[:-keep] if keep > 0 else []:
        try:
            f.unlink()
        except OSError:
            pass


def relocate_run(case, want, keep):
    """Move this run's saved snapshots to ``want`` and write there from now on.

    Returns ``(ok, message)``. Refuses rather than risks data:

    * a target already holding OTHER ``population_*.json`` files is rejected --
      merging two runs' snapshots into one directory would produce a folder that
      looks like a single run and isn't;
    * on any failure the run keeps writing where it was, so a bad path costs a
      message rather than the record.
    """
    old = pathlib.Path(case.outdir)
    new = pathlib.Path(os.path.expanduser(str(want))).resolve()
    if not new.is_absolute():
        new = (REPO / new).resolve()
    if new == old.resolve():
        return True, f'already writing to {new}'
    try:
        if new.exists() and not new.is_dir():
            return False, f'not a directory: {new}'
        mine = {f.name for f in old.glob('population_*_g*.json')} if old.is_dir() else set()
        if new.is_dir():
            theirs = [f for f in new.glob('population_*_g*.json') if f.name not in mine]
            if theirs:
                return False, (f'{new.name}/ already holds {len(theirs)} snapshot(s) '
                               'from another run — choose an empty directory')
        new.mkdir(parents=True, exist_ok=True)
        moved = 0
        if old.is_dir():
            for f in sorted(old.iterdir()):
                if f.is_file():
                    shutil.move(str(f), str(new / f.name))
                    moved += 1
            try:
                old.rmdir()                      # only succeeds if now empty
            except OSError:
                pass
        case.outdir = str(new)
        prune_snapshots(case.outdir, keep)
        return True, f'moved {moved} file(s) to {new}'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def best_member(pop, N_k, labels):
    """Best member: highest clean L/D among feasible, else best on front 1."""
    arr = np.asarray(pop, float)
    rest = labels[N_k:]
    ci, li, ri, pi = (N_k + rest.index(k) for k in
                      ('con_tag', 'LoD_clean_at_design', 'LoD_rough_at_design', 'pareto_index'))
    feas = arr[arr[:, ci] >= 1.0]
    src = feas if len(feas) else arr[arr[:, pi] == 1]
    if not len(src):
        src = arr
    row = src[np.argmax(src[:, li])]
    half = N_k // 2
    return (dict(K_upper=row[:half].tolist(), K_lower=row[half:N_k].tolist()),
            float(row[li]), float(row[ri]), int(len(feas)))


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
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--gens', type=int, default=150)
    ap.add_argument('--pop', type=int, default=200)
    ap.add_argument('--backend', default='nqfoil')
    ap.add_argument('--model', default='xxlarge')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n-polar', type=int, default=9,
                    help='airfoils sampled across the Pareto front for the rainbow')
    ap.add_argument('--every', type=int, default=5,
                    help='render every Nth generation (1 = all). A frame costs ~6 s '
                         'while the GA runs, against ~1 s per generation, so N=5 puts '
                         'the two at rough parity and keeps the lag bounded. Lower it '
                         'for a denser record (frames then arrive after the run ends).')
    ap.add_argument('--port', type=int, default=8777,
                    help="serve this run's dashboard here. Each concurrent run "
                         "needs its own port AND its own --run-dir: state.json "
                         "and frames/ are per-run, so two runs sharing a "
                         "directory overwrite each other's output.")
    ap.add_argument('--run-dir', default=None,
                    help="where this run's state.json and frames/ go "
                         "(default: live/runs/ga_p<port>, so concurrent GA and "
                         "gradient runs never collide)")
    ap.add_argument('--no-serve', action='store_true',
                    help='do not host a server; just write into --run-dir')
    ap.add_argument('--out', default=None,
                    help='explicit state.json path (overrides --run-dir)')
    ap.add_argument('--save-every', type=int, default=10,
                    help='write a population snapshot every Nth generation '
                         '(0 disables saving)')
    ap.add_argument('--keep', type=int, default=5,
                    help='snapshots to retain; older ones are pruned as the run goes')
    args = ap.parse_args(argv)

    params = read_input_file(args.case)
    params.update(N_pop=args.pop, N_generations=args.gens,
                  surrogate_backend=args.backend, neuralfoil_model=args.model)
    params.pop('continuation_file', None)
    import yaml
    tmp_case = HERE / '_live_case.yaml'
    yaml.safe_dump(params, open(tmp_case, 'w'), sort_keys=False)

    case = Case(str(tmp_case), create_output=False)
    if args.save_every:
        case.outdir = str(run_output_dir(case))
        os.makedirs(case.outdir, exist_ok=True)
        shutil.copy(args.case, os.path.join(case.outdir, os.path.basename(args.case)))
        print(f'[live] snapshots -> {case.outdir} '
              f'(every {args.save_every} generations, keeping {args.keep})', flush=True)
    # Per-run directory + server, so a GA run and a gradient run (or two GA runs)
    # can be live at once. state.json and frames/ were previously fixed at the
    # package dir, so concurrent runs silently overwrote each other's output --
    # the port alone was never enough to separate them.
    #
    # Derived HERE rather than at argument-parse time because the name comes from
    # `case.folderstr` (<filecode>__<timestamp>, the data tree's own convention),
    # which does not exist until the Case is built.
    global FRAMES
    from oso_airfoils.live import run_dirs
    run_dir = pathlib.Path(args.run_dir) if args.run_dir else run_dirs.for_case(case)
    run_dir.mkdir(parents=True, exist_ok=True)
    FRAMES = run_dir / 'frames'
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True, exist_ok=True)
    state_path = args.out or str(run_dir / 'state.json')
    if not args.no_serve:
        from oso_airfoils.live import server as _srv
        _srv.serve(args.port, run_dir, background=True)
    print(f'dashboard: http://localhost:{args.port}   (state -> {state_path})',
          flush=True)

    ev = GPUBatchEvaluator(backend=args.backend, model_size=args.model, device=args.device)

    meta = dict(case=os.path.basename(args.case), tau=case.tau, CL=case.CL, Re=case.Re,
                N_pop=case.N_pop, N_k=case.N_k, backend=args.backend, model=args.model,
                device=ev.device, target_gens=args.gens, n_polar=args.n_polar,
                every=args.every, families=None, library=None,
                solver_label='OSO GA', iter_word='generation', iter_unit='gen',
                started=time.time())
    from oso_airfoils.live import ref_library
    ref_library.populate_async(
        meta, case.tau,
        on_done=lambda nf, nl: print(f'  [ref] reference library ready: '
                                     f'{nf} families, {nl} airfoils', flush=True))
    plotter = Plotter(state_path, meta)
    plotter.phase_text = 'evolving the population — NSGA sort, crossover and mutation each generation'
    plotter.save_path = case.outdir if args.save_every else None
    plotter.start()
    plotter.write_state('starting', 0, 0.0)

    nv, et, lb, ub = [1] * case.N_k, [float] * case.N_k, [-2.0] * case.N_k, [2.0] * case.N_k

    save_req = {'mtime': None}

    def check_savepath():
        f = HERE / 'savepath.json'
        try:
            m = f.stat().st_mtime
        except OSError:
            return
        if m == save_req['mtime']:
            return
        save_req['mtime'] = m
        try:
            want = json.loads(f.read_text()).get('path')
        except Exception:
            return
        if not want:
            return
        ok, msg = relocate_run(case, want, args.keep)
        plotter.save_path = case.outdir
        plotter.save_status = ('ok' if ok else 'error') + ': ' + msg
        print(f'[live] save path {"->" if ok else "REJECTED"} {msg}', flush=True)

    def snapshot(gen, spg, status='running'):
        plotter.gen_times.append(spg)
        if args.save_every:
            check_savepath()
        render_this = (gen % args.every == 0) or gen == args.gens or gen == 0
        # COMPACT payload. The figure needs the front's coefficients, the front's
        # L/D pairs and the population's L/D scatter -- a few hundred floats. Shipping
        # the full population snapshot (200 members x 68 fields, nested dicts) cost
        # 6-7 s per frame just to pickle across the process boundary, which was the
        # entire render cost.
        arr = np.asarray(case.pop, float)
        rest = case.labels[case.N_k:]
        ci, li, ri, pi = (case.N_k + rest.index(k) for k in
                          ('con_tag', 'LoD_clean_at_design', 'LoD_rough_at_design',
                           'pareto_index'))
        half = case.N_k // 2
        f1 = arr[arr[:, pi] == 1]
        order = np.argsort(f1[:, ri]) if len(f1) else []
        f1 = f1[order] if len(f1) else f1
        snap = dict(
            ip=dict(TE_gap=case.te_gap, N_k=case.N_k, CL=case.CL, Re=case.Re,
                    tau=case.tau),
            front_upper=f1[:, :half].tolist(),
            front_lower=f1[:, half:case.N_k].tolist(),
            front_rough=f1[:, ri].tolist(),
            front_clean=f1[:, li].tolist(),
            pop_rough=arr[:, ri].tolist(),
            pop_clean=arr[:, li].tolist(),
            pop_front=arr[:, pi].tolist(),
        )
        # Persist the real population snapshot on its own cadence, independent of
        # the display cadence -- the figure is disposable, this is the run record.
        if args.save_every and (gen % args.save_every == 0 or gen == args.gens):
            case.save(gen)
            prune_snapshots(case.outdir, args.keep)

        best, lc, lr, nf = best_member(case.pop, case.N_k, case.labels)
        # The stat boxes report the FRONT's extremes -- best clean L/D anywhere on
        # it and best rough L/D anywhere on it -- which are the two corners the
        # front spans. best_member returns ONE individual's (clean, rough) pair,
        # which is a different quantity: no single member holds both maxima
        # (that is what makes it a trade-off), so showing its pair understated
        # whichever objective that member was not best at.
        fc_l, fr_l = snap.get('front_clean') or [], snap.get('front_rough') or []
        if fc_l:
            lc, lr = max(fc_l), max(fr_l)
        if render_this:
            plotter.submit(dict(gen=gen, snapshot=snap, best=best, lod_clean=lc,
                                lod_rough=lr, n_feasible=nf, status=status, s_per_gen=spg))
        print(f"gen {gen:4d}  cleanL/D={lc:7.2f}  roughL/D={lr:7.2f}  "
              f"feasible={nf:4d}/{case.N_pop}  {spg:.2f}s", flush=True)

    random.seed(0); np.random.seed(0)
    case.init_population()
    t0 = time.time()
    res = ev.evaluate([(case, case.pop)])
    case.pop = initialize_sort(case.pop, res[case.uid], case.N_k, case.params)
    snapshot(0, time.time() - t0)

    for g in range(1, args.gens + 1):
        t0 = time.time()
        sd, ch = produce_children(case.pop, nv, et, lb, ub, case.params)
        case.surrogate_cache = None
        res = ev.evaluate([(case, ch)])
        case.pop = finish_generation(case.pop, sd, ch, res[case.uid], nv, case.params)
        snapshot(g, time.time() - t0, 'running' if g < args.gens else 'finishing')

    # let the plotter drain the last submission before declaring done
    while not plotter.q.empty():
        print(f'  [plot] draining {plotter.q.qsize()} queued generation(s)…', flush=True)
        time.sleep(2.0)
    time.sleep(1.0)
    plotter.write_state('done', args.gens, 0.0)
    print('done', flush=True)
    _park(args.port, not args.no_serve)


if __name__ == '__main__':
    sys.exit(main())
