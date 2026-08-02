Core
----

Solver wrappers and data utilities.

The aerodynamic solvers themselves live in the companion **metafoil** package. The
wrappers here are thin adapters that preserve this project's call signature and return
contract:

| Module | Role |
| :----- | :--- |
| `xfoil_wrapper.py` | Dispatches to `metafoil.xfoil` (in-memory, alpha mode) or `metafoil.xfoil.wrappers.xfoil_fileio` (CL mode, flaps, explicit binary path) |
| `qfoil_wrapper.py` | Same split over `metafoil.qfoil` |
| `neuralfoil_wrapper.py` | NeuralFoil surrogate |
| `sweep.py`, `ingest_data.py`, `data_utils.py`, `airfoil_family.py` | Sweep driving, performance-data ingestion/merging, and lookup |

Removed in favour of metafoil: the `*_inmem_wrapper.py` family (`xfoil`, `qfoil`, `cfoil`,
`cxfoil`, `cqfoil`) and `_xfoil_inmem_base.py`, which loaded `.so` files through a local
CMPLXFOIL checkout, plus `cfoil_wrapper.py` (superseded by `metafoil.cxfoil`, which also
provides complex-step gradients). Use metafoil directly for those.


Image Processing
----------------

This folder shows the process of fitting airfoils from png files.  

- The original airfoil files are located in the `png_files` folder
- The most up to date script is the `process_airfoils.py` script
    - Images should not have a vertical line connecting upper and lower trailing edges
    - This was used for the MHKF1 hydrofoils
- The `process_airfoils_with_green_dot.py` script was previously used to process the wind turbine airfoils, where a green dot was manually placed at the desired (0,0)
    - This is now a legacy script
- The `verification_plots` folder stores some helpful debugging plots
- The `fitted_datfiles` folder contains the raw fits to the data
- The `corrected_datfiles` folder contains the final dat files, which have been thickness corrected