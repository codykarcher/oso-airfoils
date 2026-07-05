from oso_airfoils.postprocessing.polars import polars_compare, polars_rainbow
from oso_airfoils.postprocessing.boundary_layer import (
    boundary_layer_compare,
    boundary_layer_rainbow,
)
from oso_airfoils.postprocessing.runners import (
    run_and_plot_polars_compare,
    run_and_plot_polars_rainbow,
    run_and_plot_boundary_layer_compare,
    run_and_plot_boundary_layer_rainbow,
)

__all__ = [
    'polars_compare',
    'polars_rainbow',
    'boundary_layer_compare',
    'boundary_layer_rainbow',
    'run_and_plot_polars_compare',
    'run_and_plot_polars_rainbow',
    'run_and_plot_boundary_layer_compare',
    'run_and_plot_boundary_layer_rainbow',
]
