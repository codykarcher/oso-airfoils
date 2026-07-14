import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from oso_airfoils.core.colors import default_color_cycle as dcc
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=dcc)
from metafoil.core.kulfan import Kulfan
import pathlib
path_to_here = pathlib.Path(__file__).parent.resolve()
path_to_oso = path_to_here.parent.parent.parent

import copy

fls = os.listdir(path_to_here)
fls = sorted([f for f in fls if f.endswith('.txt')])

for fl in fls:
    print(fl)
    fl_leader = fl.split('.')[0]

    plt.figure(figsize=(22,4))
    data = np.genfromtxt(path_to_here / fl)
    
    raw_psi = []
    raw_zeta_upper = []
    raw_zeta_lower = []

    if data[0][0] != 0.0 and data[0][1]!=0.0 and data[0][2]!=0.0:
        raw_psi.append(0)
        raw_zeta_upper.append(0)
        raw_zeta_lower.append(0)

    elif data[0][0] == 0.0 and data[0][1] == 0.0 and data[0][2] == 0.0:
        # do nothing, this will load correctly
        pass

    elif data[0][0] != 0.0 and data[0][1]==data[0][2]:
        data[:,0] = data[:,0] - data[0][0]
        data[:,1] = data[:,1] - data[0][1]
        data[:,2] = data[:,2] - data[0][2]
        data *= 1.0/max(data[:,0])
    
    else:
        raise ValueError('Check the dat file, data may be corrupted')
    
    for dr in data:
        raw_psi.append(dr[0])
        raw_zeta_upper.append(dr[1])
        raw_zeta_lower.append(dr[2])

    raw_psi = np.array(list(reversed(raw_psi)) + raw_psi[1:])
    raw_zeta = np.array(list(reversed(raw_zeta_upper)) + raw_zeta_lower[1:])
    
    afl = Kulfan()
    afl.fit2coordinates(copy.deepcopy(raw_psi), copy.deepcopy(raw_zeta), fit_order=10)
    # Adjusts the TE to be exactly on centerline, which helps analysis
    # This is a very minor change for most airfoils, but does deviate from the 'true' FFAs
    afl.constants.TE_shift = 0.0
    
    plt.figure(figsize=(22,4))
    plt.plot(raw_psi, raw_zeta, 'x', label='Raw Data')
    plt.plot(afl.xcoordinates, afl.ycoordinates, '.-', color= dcc.green, label='Fitted CST10')
    plt.grid(1)
    plt.legend()
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(path_to_here / (fl_leader+'_CST10fit.png'), dpi=300)
    plt.close()

    afl.write2file(str(path_to_here.parent / (fl_leader+'.dat')))
