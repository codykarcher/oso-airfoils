# IEA 22MW Blade Re-Twist with OSO Airfoils

This repo contains in-progress files using WEIS to re-twist the IEA 22MW with updated airfoils

## Case 1: Baseline2

Run WEIS optimization with original windio: IEA-22-280-RWT.yaml

## Case 2: FFA_Opt_1 

Run WEIS optimization with rfoil generated FFA windio: IEA-22-280-RWT-FFA.yaml

## Case 2: OSO_Opt_3 

Run WEIS optimization with OSO windio: IEA-22-280-RWT-GA.yaml


# Known Issues

- OSO case fails after 12 minor iterations with openfast startup error 
- airfoilprep.py writes occasional duplicate values and should be replaced (when regenerating windio)
- Cases with more optimizer iterations return unlikely twist

