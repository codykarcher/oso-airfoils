#!/usr/bin/env bash
# Run all generated YAML cases through the OSO airfoil optimizer.
#   - neuralfoil cases : 96  MPI processes
#   - xfoil cases      : 188 MPI processes

cd "$(dirname "${BASH_SOURCE[0]}")"

# --- neuralfoil ---
# mpirun -n 96  python -m oso_airfoils.optimization.runner t27_neuralfoil.yaml
# mpirun -n 96  python -m oso_airfoils.optimization.runner t18_neuralfoil.yaml
# mpirun -n 96  python -m oso_airfoils.optimization.runner t21_neuralfoil.yaml
mpirun -n 96  python -m oso_airfoils.optimization.runner t24_neuralfoil.yaml
# mpirun -n 96  python -m oso_airfoils.optimization.runner t27_neuralfoil.yaml
# mpirun -n 96  python -m oso_airfoils.optimization.runner t30_neuralfoil.yaml
mpirun -n 96  python -m oso_airfoils.optimization.runner t33_neuralfoil.yaml
mpirun -n 96  python -m oso_airfoils.optimization.runner t36_neuralfoil.yaml

# # --- xfoil ---
# mpirun -n 188 python -m oso_airfoils.optimization.runner t18_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t21_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t24_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t27_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t30_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t33_xfoil.yaml
# mpirun -n 188 python -m oso_airfoils.optimization.runner t36_xfoil.yaml
