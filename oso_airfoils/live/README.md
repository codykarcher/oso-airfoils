Live GA Dashboard
=================

Watch an optimization run in a browser while it happens: the Pareto front, the
airfoils on it, their Kulfan parameters and their polars, all refreshed every
generation. Intended for watching a run develop and for spotting a bad one early
— not as a replacement for the publication figures in `postprocessing/`.

Everything is local. Nothing is uploaded or published.


Quick start
-----------

Two terminals, both from anywhere in the repo.

**1 — serve the page:**

```bash
python -m oso_airfoils.live.server 8777
```

**2 — start a run:**

```bash
python -m oso_airfoils.live.live_ga oso_airfoils/runfiles/t21_neuralfoil.yaml
```

Then open **http://localhost:8777**.

The run writes `state.json` and `frames/*.svg` next to these sources; the page
polls `state.json` and swaps in each new frame. Both are gitignored.


Options
-------

```bash
python -m oso_airfoils.live.live_ga <case.yaml> [options]
```

| Flag | Default | Meaning |
| :--- | :------ | :------ |
| `--gens` | 150 | generations to run |
| `--pop` | 200 | population size (overrides the case file) |
| `--backend` | `nqfoil` | surrogate: `nqfoil` or `nxfoil` |
| `--model` | `xxlarge` | model size; nqfoil tops out at `xxlarge`, nxfoil at `xxxlarge` |
| `--device` | `cpu` | `cpu`, `cuda`, or `mps` |
| `--n-polar` | 9 | airfoils sampled across the front for the polars |
| `--every` | 1 | render every Nth generation |
| `--save-every` | 10 | write a population snapshot every Nth generation (`0` disables) |
| `--keep` | 5 | snapshots retained; older ones pruned as the run proceeds |
| `--out` | `state.json` | dashboard state file |

A production-shaped run:

```bash
python -m oso_airfoils.live.live_ga oso_airfoils/runfiles/t21_neuralfoil.yaml \
    --gens 1000 --pop 752 --backend nqfoil --model xxlarge
```

At 752 members a generation costs ~5 s and a frame ~1 s, so rendering keeps up
with no lag. Raise `--every` if you put the dashboard on a machine where the
frame is slower than a generation — the log prints both, so the ratio is visible.


Saved output
------------

The dashboard's frames are disposable, but the run itself is recorded. Every
`--save-every` generations a full population snapshot is written into the data
tree, in exactly the layout and schema a normal run produces:

```
oso_airfoils/data/cases_<lo>_to_<hi>/case_<N>/<filecode>__<timestamp>/
    <case>.yaml
    population_<filecode>_g<NNN>.json
```

Only the newest `--keep` snapshots are retained — a long live run would otherwise
fill the data tree with hundreds of megabytes of intermediate populations. The
retained files are ordinary snapshots, so `oso-gif`, `oso-polar` and the rest of
`postprocessing/` read them directly, and any of them can seed a `continuation_file`.

The bucket is chosen by looking for an existing `case_<N>` directory first, since
the buckets aren't perfectly regular (there is a `cases_91_to_99` and a bare
`cases_100`); only if none exists is a new `cases_<lo>_to_<hi>` name computed.

Set `--save-every 0` for a purely throwaway run.

### Changing where it saves, mid-run

The **Save to** box at the top of the page is prefilled with the directory above.
Type a new path and press Enter (or click away): everything already written is
moved there, and every later snapshot goes to the new location. Relative paths
resolve against the repo root and `~` expands. Missing directories are created.

Two things it refuses, reporting the reason next to the box instead:

- a path that is not a directory;
- a directory already holding **another run's** `population_*.json`. Merging two
  runs' snapshots produces a folder that looks like one run and isn't, and the
  files interleave by generation number, so this is unrecoverable by inspection.

On a refusal — or any failure mid-move — the run keeps writing where it was, so a
typo costs a message rather than the record. Press Escape to abandon an edit.

The move is performed by the GA process, not the web server: the GA owns those
files and is the only thing that can retarget its own writer without racing it.
The server only records the request; the GA picks it up at its next save tick, so
the box acts on a `--save-every` boundary rather than instantly.


What you see
------------

Top: **the airfoils across the front**, coloured along the turbo ramp from the
rough-biased end to the clean-biased end. Below, left to right:

- **Pareto front** over **Kulfan shape parameters**, sharing an x-axis (rough L/D).
  The big dots are the sampled airfoils drawn above, in matching colours.
- **C_L and C_M vs alpha** — clean solid, rough dashed. C_M is confined to the
  lower half of the pane so it can't tangle with C_L.
- **L/D vs C_L**.
- **C_p,min vs C_L** — appears only when a cp_min constraint is active in the case.

Controls: **Follow live** tracks the newest generation, **Replay** loops through
every rendered frame (picking up new ones as it goes), and the slider scrubs.

### Reference airfoils

The bar above the figure compares the front against a published airfoil. Family
buttons pick the member closest in thickness to the case; a family with nothing
within tolerance is greyed out and says why. The search box on the right reaches
any airfoil in the store regardless of thickness. Buttons are radio-style —
clicking the active one clears it.

**Only geometry comes from the store.** The reference's polar is computed fresh
through the same surrogate at the same Reynolds and transition conditions as the
optimized airfoils, so the curves are directly comparable. Using the stored polar
data instead would mix solvers and operating points and make the comparison
meaningless.


How it works
------------

`live_ga.py` drives the GA through the optimization package's own phase API
(`produce_children` → `evaluate` → `finish_generation`), so this is a real run,
not a simulation — the same selection, constraints and batched surrogate a normal
run uses. It differs from `python -m oso_airfoils.optimization` only in that it
emits a frame between phases.

Rendering happens in a **separate process**. Matplotlib's figure construction is
mostly pure Python and holds the GIL, so as a thread it slowed the GA by ~2.5x.

The payload handed to that process is deliberately small — the front's
coefficients and the population's L/D scatter, a few hundred floats — rather than
the full population snapshot.

| file | role |
| :--- | :--- |
| `live_ga.py` | GA driver, render process, state file |
| `dash_figure.py` | the figure; persistent, redrawn per generation |
| `families.py` | reference-airfoil geometry lookup (cached) |
| `server.py` | static server plus a `/select` endpoint for the reference bar |
| `index.html` | the page |


Notes
-----

- The polars are **surrogate predictions**. nqfoil runs ~8% below nxfoil on the
  same designs and neither has been validated against QFOIL for optimized shapes.
  Read the numbers as relative, not absolute.
- Frames accumulate at roughly 300 KB each, so a 1000-generation run at
  `--every 1` leaves ~300 MB in `frames/`. Delete it between runs.
- `families.py` caches its geometry lookups. The performance JSONs carry every
  recorded polar run and are multi-megabyte, so re-parsing a family per frame
  cost ~13 s — by far the largest cost this dashboard ever had.
