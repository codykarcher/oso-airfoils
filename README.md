<img src="logo/oso_logo.svg" alt="OSO Logo" width="500"/>

Wind Turbine Airfoil Design Tools for Open-Source Offshore (OSO) Airfoils
=========================================================================

This repository contains the design tools used to produce the Open-Source Offshore (OSO) Airfoils.  This project was led by Sandia National Laboratories, in collaboration with California State University, Long Beach (CSULB) and the National Renewable Energy Laboratory (NREL).


Installation
------------

From within the cloned repository:
```
pip install -e .
```

We strongly recommend a sparse checkout to minimize the required hard drive space. The bulk of the repository size comes from the optimization data stored in `oso_airfoils/data/cases_*/`. To clone without these large directories:
```
git clone --filter=blob:none --no-checkout git@github.com:sandialabs/oso-airfoils.git
cd oso-airfoils
git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/oso_airfoils/data/cases_*/'
git checkout main
```

To add data from a specific set of cases after cloning:
```
git sparse-checkout add 'oso_airfoils/data/cases_101_to_110/'
```

Note that the SSH link above may need to be replaced with the HTTPS link depending on your setup:
```
https://github.com/sandialabs/oso-airfoils.git
```

A normal clone is still possible:
```
git clone <html_or_ssh_link>
```
but will take up significant hard drive space.

At present, this code has been tested on MacOS and on a Windows machine running WSL (eg, Linux).  Native Windows support is not currently expected nor guaranteed.  Windows users are advised to set up WSL, at which point the code should run with no issues.

After installing dependencies, the airfoil optimization may be run simply by following the directions in the `runfiles` directory.

The following is a quickstart:
```
cd oso_airfoils/runfiles
mpirun -n 8 python -m mpi4py -m oso_airfoils.optimization quickstart.json
```

Note that this will copy the `quickstart.json` file into the run directory.

On a 2022 M1 MacBook Air, this runs roughly 500 generations in roughly 4 hours.

There is a single entry point, `python -m oso_airfoils.optimization`, which accepts one
or more case files. It picks how to run them automatically: MPI when launched under
`mpirun`, a batched GPU path when the case uses a neural-network surrogate and a CUDA
device is present, and a plain serial loop otherwise. To pin the choice, pass
`--execution serial|mpi|gpu-batched`. Passing several case files runs them in lockstep
so that a GPU evaluates the whole fleet in one batch per generation:

```
python -m oso_airfoils.optimization t21.yaml t24.yaml t27.yaml
```

See `oso_airfoils/optimization/README.md` for the execution modes, the CLI options, and
the package architecture.

Run outputs and airfoil performance data live under `oso_airfoils/data/`: completed runs
in `data/cases_<lo>_to_<hi>/case_<N>/`, and per-family aerodynamic data in
`data/<family>/performance_data/` with plots in `data/<family>/polar_plots/`.
`oso_airfoils/runfiles/` holds only run configurations and the scripts that generate them.


Relationship to metafoil
------------------------

The aerodynamic solvers live in the companion **metafoil** package, which this project
depends on.  `oso_airfoils.core`'s `xfoil_wrapper` and `qfoil_wrapper` are thin adapters:
they dispatch alpha sweeps to metafoil's in-memory solvers and CL mode, flap deflections
and explicitly-pointed binaries to its file-I/O solvers, keeping only the argument
reconciliation this project's call sites depend on.  Geometry comes from
`metafoil.core.kulfan` (per-airfoil) and `metafoil.core.kulfan_torch` (batched,
differentiable).  Use metafoil directly for complex-step gradients (`metafoil.cxfoil`,
`metafoil.cqfoil`) and for the surrogate nets (`metafoil.nxfoil`, `metafoil.nqfoil`).


Dependencies
------------

The use of these tools assumes the following dependencies, all of which should be installable with `pip install <package>` or `conda install <package>` as appropriate.

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `mpi4py`
- `pint`
- `torch`
- `pyyaml`
- `natsort`
- `Pillow`
- `neuralfoil`

We also assume that there is an `xfoil` executable located somewhere in your path.  EG: if you type `which xfoil` in a terminal, a pathname should be printed.  

You may choose to compile XFOIL on your own from source (https://web.mit.edu/drela/Public/web/xfoil/), however, we recommend simply obtaining XFOIL through a distribution of Engineering Sketch Pad (https://acdl.mit.edu/ESP/) (readme is here: https://acdl.mit.edu/ESP/ESPreadme.txt)

NeuralFoil (https://github.com/peterdsharpe/NeuralFoil) is included as a required dependency and will be installed automatically. Though NeuralFoil is valuable for quick passes and produces reasonable results, we caution that optimized shapes obtained using NeuralFoil are notably and meaningfully different from those obtained using XFOIL.  NeuralFoil should not be trusted for final results in our experience.

Citations
---------

If referencing this work, please cite the following paper:

```
@inbook{karcher2025design, 
    author={Karcher, Cody J. and Maniaci, David C. and Kelley, Chris and Hsieh, Alan and deVelder, Nathaniel and Gupta, Anurag}, 
    title={Design of a Preliminary Family of Airfoils for High Reynolds Number Wind Turbine Applications}, 
    booktitle={AIAA SCITECH 2025 Forum}, 
    publisher={American Institute of Aeronautics and Astronautics}, 
    place={Orlando, Florida}, 
    month={Jan},
    year={2025},
    DOI={10.2514/6.2025-0840},
    ISBN={978-1-62410-723-8}
    }
```

A pdf of this paper is included in the `publications` folder.


License
-------

Use and distribution of this work is subject to the included MIT License.


Copyright
---------

Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.


Funding
-------

We are grateful for the funding that made this work possible provided by the U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Wind Energy Technologies Office.
