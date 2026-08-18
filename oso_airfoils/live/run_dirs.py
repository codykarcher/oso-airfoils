"""
run_dirs.py — per-run dashboard directories, using oso's own naming convention.

Both live drivers write `state.json`, `frames/*.svg` and `selection.json` into a
directory of their own, so several dashboards can be live at once without
overwriting each other. The names follow the data tree's convention rather than a
new one:

    <filecode>__<timestamp>

with `filecode` in `case.py`'s shape (`c<case>_t<tau*100>_k<N_k>_n<N_pop>_l<CL*10>_e<Re/1e5>`)
and `timestamp` from `case._timestamp()` — imported, not reimplemented, so the
centisecond suffix and the `%Y_%m_%d_%H-%M-%S` layout stay in one place.

The timestamp is what makes repeat runs safe: keying a directory off the port or
the thickness alone means relaunching the same case reuses the directory, and the
GA path wipes `frames/` on start — so a rerun would delete the previous run's
figures out from under a dashboard still displaying them.
"""
from __future__ import annotations

import pathlib

from oso_airfoils.optimization.case import _timestamp

HERE = pathlib.Path(__file__).parent.resolve()
RUNS = HERE / "runs"


def for_case(case):
    """GA run directory. `case.folderstr` is already `<filecode>__<timestamp>`."""
    return RUNS / case.folderstr


def for_gradient(params, thickness, n_points, n_seeds, tool):
    """Gradient run directory, in the same shape as a GA one.

    `g` prefix rather than `c` so the two solvers are distinguishable at a glance
    in `runs/`; the rest mirrors case.py's filecode, with the multi-start seed
    count in place of a population size.
    """
    tau = params.get("tau")
    code = f"g{tool}_t{int(round(tau * 100))}_k16_n{n_points}x{n_seeds}"
    if params.get("CL") is not None:
        code += f"_l{int(params['CL'] * 10)}"
    if params.get("Re") is not None:
        code += f"_e{int(params['Re'] / 1e5)}"
    return RUNS / f"{code}__{_timestamp()}"
