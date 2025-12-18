Sweep through Reynolds numbers of the OSO-2025-WT2 airfoil family
Airfoil_Name = {'OSO_2025_WT2_T21';'OSO_2025_WT2_T24';'OSO_2025_WT2_T27';'OSO_2025_WT2_T30';'OSO_2025_WT2_T33';'OSO_2025_WT2_T36'};
Analysis performed using RFoil v. 3.0.  
Clean Case uses N_factor of 9.0 clean with free transition.
Rough case uses N_factor of 3.0 with transition fixed at x/c 0.05 on the suction and pressure surfaces.
Re = 3, 6, 10, 15, 20 million chordwise Reynolds number
delta AoA = 0.1 deg., AoA sweeps run form -24 to 24 deg., but many cases do not reach the extremes
Mach = 0.0
cr for the rotational augmentation model was set to 0.0
Naming convention:
OSO_2025_WT2_T21_r3_cleanPol_RF6f_DragOff.dat

'OSO_2025_WT2' is the airfoil family 
T21 is thickness of 21%
r3 is 3.0e6 chordwise Reynolds number
cleanPol is the clean polar, as defined above
RF6f is RFoil, run with version 6f of the Sandia scripting code (RFOILnew_Robot6f_scripter_MultiAF_OSOWT2_20251211_ReSweep_2a.m)
DragOff is the drag correction model turned off in RFoil.  All cases were run with DragOn and DragOff.  DragOn will typically increase the drag, based on empirical tuning to a range of airfoil wind tunnel test data performed by TNO.

