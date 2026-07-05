FFA Airfoils
------------

This repository contains two folders:

- `original` -- Contains the original FFA airfoils as they appear in the source publication (the `.txt` files), along with files that have been reordered into standard coordinate form (the `.dat`) files
- `fitted` -- Contains fitted FFA airfoils that offer a more standardized family.  Kulfan CST shape function with 10 parameters was fit to both the top and bottom surfaces of each airfoil and a cosine spacing applied to increase the desitiy of points at the leading edge.



Source
------

1. Björck, A., “Coordinates and Calculations for the FFA-W1-xxx, FFA-W2-xxx, and FFA-W3-xxx Series of Airfoils for Horizontal Axis Wind Turbines,” Technical Report FFA TN 1990-15, The Aeronautical Research Institute of Sweden, Stockholm, 1990.

2. Kulfan, Brenda M. "Universal parametric geometry representation method." Journal of Aircraft 45.1 (2008): 142-158.