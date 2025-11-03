import numpy as np
import re
import pandas as pd
import warnings
import scipy.stats
from scipy import interpolate
import matplotlib.pyplot as plt
from subprocess import check_output
from scipy.interpolate import interp1d
import matplotlib.patches as patches
import glob
from datetime import datetime
import traceback
import sys
import shutil
import ruamel.yaml as yaml


def remove_last_line(path):

    # Remove last line
    with open(path, "r") as file:
        lines = file.readlines()

    # Remove the last line
    if lines:  # Ensure the file is not empty
        lines = lines[:-1]

    # Write the updated content back to the file
    with open(path, "w") as file:
        file.writelines(lines) 

def closest_index(lst, target):
    return min(range(len(lst)), key=lambda i: abs(lst[i] - target))


def replace_regex(filepath,variable,newval):
    #Regex to match any integer, float, or sci notation (unused because allreg works fine)
    numregfinal = "[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?"

    # Regex to match any continuous non-whitespace
    allreg = "\\S+"

    # Regex to match any continuous whitespace
    spreg = "\\s+"

    search_text = allreg + spreg + str(variable)
    replace_text = str(newval) + "          " + str(variable)

    with open(filepath,'r+') as f:
        file = f.read()
        file = re.sub(search_text,replace_text, file)
        f.seek(0)
        f.write(file)
        f.truncate()
    
    return str(variable) + " Replaced"

def load_aerodyn_v13(filepath):

    n=14

    a13data = pd.read_csv(filepath, sep='\\s+', skiprows=(range(n-1)), names=['Alpha','Cl','Cd','Cm'])

    with open(filepath) as afile:
        a13list=[re.split(r'[\t ]+',s.strip(), maxsplit=1) for s in afile.readlines()[4:n-1]] #put here the interval you want

    a13header = pd.DataFrame(a13list,columns=['value','description'])

    return a13header,a13data

def writeAerodynLatest(header,data,outpath):

    # Write new header file
    shutil.copy('/tscratch/ndeveld/airfoil/Airfoil_Process/aerodyn_header.dat',outpath)

    # Make replacements
    replace_regex(outpath,'alpha0',float(header.iloc[3].value))
    replace_regex(outpath,'alpha1',float(header.iloc[5].value))
    replace_regex(outpath,'alpha2',float(header.iloc[6].value))
    replace_regex(outpath,'C_nalpha',float(header.iloc[4].value))
    replace_regex(outpath,'NumAlf',int(len(data.Alpha)))
    replace_regex(outpath,'NumCoords','0')
    replace_regex(outpath,'BL_file','0')
    replace_regex(outpath,'Re ',float(header.iloc[0].value))

    # Append data
    data.to_csv(outpath,mode='a',sep=' ',index=False,header=False)

    # Remove "EOT" from end of file
    remove_last_line(outpath)

def get_zl_aoa(polar_object):

    a = 50

    cl = polar_object.cl[a:-a]
    aoa = polar_object.alpha[a:-a]

    f_lin = interp1d(cl,aoa)
    
    clvals = np.unique(np.round(cl,2))

    if len(clvals) > 1:
        return f_lin(0.0)
    else:
        return 0.0

def flist(x):
    retval = yaml.comments.CommentedSeq(x)
    retval.fa.set_flow_style()  # fa -> format attribute
    return retval

def modify_windio(airfoils,polars,turbine,windioyaml):
    
    for r,rep in enumerate(windioyaml['replacements']):

        templist = turbine['components']['blade']['outer_shape_bem']['airfoil_position']['labels']
        newlist = flist([rep['new'] if item == rep['old'] else item for item in templist])
        turbine['components']['blade']['outer_shape_bem']['airfoil_position']['labels'] = newlist

    unique_rep = list(set([r['new'] for r in windioyaml['replacements']]))
    for r,rep in enumerate(unique_rep):

        this_airfoil = airfoils[rep]
        this_polar = polars[rep]['airfoil_extrap'].polars[0]

        #print(vars(this_polar))

        cl = flist([float(v) for v in this_polar.cl])
        cd = flist([float(v) for v in this_polar.cd])
        cm = flist([float(v) for v in this_polar.cm])
        alpha = flist([float(v)*np.pi/180.0 for v in this_polar.alpha])

        xlist = flist(this_airfoil.coord_data.x.values.tolist())
        ylist = flist(this_airfoil.coord_data.y.values.tolist())
        coord_dict = {'x':xlist,'y':ylist}

        polars_dict = {'configuration':'Xfoil Clean',
                       're':float(this_airfoil.Re),
                       'c_l':{'grid':alpha,'values':cl},
                       'c_d':{'grid':alpha,'values':cd},
                       'c_m':{'grid':alpha,'values':cm}}
                       

        this_airfoil_dict = {"name":this_airfoil.name,
                             'coordinates':coord_dict,
                             'relative_thickness':float(this_airfoil.thick)/100.0,
                             'aerodynamic_center':0.25,
                             'polars':[polars_dict]}
                             

        turbine['airfoils'].append(this_airfoil_dict)

    return turbine

def read_of_bladefile(bladefile):
    headskip = [0,1,2,3,5]
    print("Reading blade file",bladefile)
    blade_data = pd.read_csv(bladefile,sep='\\s+',skiprows=(headskip), header=(0),skipinitialspace=True)
    return blade_data

def setup_summary_plot(r,c,w,h,inp):

    plt.rcParams.update({'font.size': 18})
    #xmin = inp['common']['plot_x_bounds'][0]
    #xmax = inp['common']['plot_x_bounds'][1]

    fig, ax = plt.subplots(r,c,figsize=(w,h))


    ax[0].set_title('Twist (deg)')
    ax[1].set_title('Chord (m)')


    for i in range(r):
            ax[i].minorticks_on()
            ax[i].grid(visible=True, which='minor', axis='both',linestyle='-',color="#eee")
            ax[i].grid(visible=True, which='major', axis='both',linestyle='-',color="#888")
            ax[i].set_axisbelow(True)
            #ax[i,j].set_xlim([xmin,xmax])

    return fig,ax

def setup_log_plot(r,c,w,h,title):

    plt.rcParams.update({'font.size': 18})

    fig, ax = plt.subplots(1,1,figsize=(w,h))
    ax.set_title(title)

    for i in range(r):
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', axis='both',linestyle='-',color="#eee")
            ax.grid(visible=True, which='major', axis='both',linestyle='-',color="#888")
            ax.set_axisbelow(True)

    return fig,ax
