"""
airfoil_io.py — re-exports metafoil.core.airfoil_io's load_airfoil_dat for
these notebooks' `from airfoil_io import load_airfoil_dat` imports.

The actual loader (and the raw_airfoil_files name-lookup it shares code
with) now lives in core/airfoil_io.py, since Kulfan(airfoil=...) needs it
too, not just these examples.
"""
from metafoil.core.airfoil_io import load_airfoil_dat  # noqa: F401
