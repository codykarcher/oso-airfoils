import os
import re
import numpy as np
from wisdem.ccblade.Polar import Polar


run_dir = os.path.dirname(os.path.abspath(__file__))


coords_folder = os.path.join(run_dir, 'datfiles')
polar_folder = os.path.join(run_dir, 'rfoil_data')

coords =   ['OSO_2025_WT2_T21.dat', 
            'OSO_2025_WT2_T24.dat',
            'OSO_2025_WT2_T27.dat',
            'OSO_2025_WT2_T30.dat',
            'OSO_2025_WT2_T33.dat',
            'OSO_2025_WT2_T36.dat']

polar_files = [f for f in os.listdir(polar_folder) if f.endswith('.dat')]

airfoils = [{} for _ in range(len(coords))]

for i in range(len(coords)):
    print('Processing airfoil %s' % (coords[i]))
    coord = coords[i]
    airfoil_name = coord.split('.')[0]
    coordinates = np.loadtxt(os.path.join(coords_folder, coord))
    airfoils[i] = {}
    airfoils[i]['name'] = airfoil_name
    airfoils[i]['coordinates'] = {}
    airfoils[i]['coordinates']['x'] = coordinates[:,0].tolist()
    airfoils[i]['coordinates']['y'] = coordinates[:,1].tolist()
    airfoils[i]['aerodynamic_center'] = 0.25
    thick_label = coords[i].split('_')[-1].split('.')[0]
    airfoils[i]['description'] = '%d%% thick airfoil developed in 2025 by Cody Karcher (CSULB) and David Maniaci (SNL) as part of the Holistic Systems Engineering project' % int(thick_label[1:])

    airfoils[i]['polars'] = [{} for _ in range(4)]
    matching_files = [f for f in polar_files if thick_label in f]
    for j in range(4):
        polar_path = os.path.join(polar_folder, matching_files[j])
        # Extract Re number from the line using regex
        with open(polar_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if lines.index(line) == 9:  # line 10 (0-based index)
                    # Example line: ' Mach =      0.00000000     Re =     12.00000000 e 6     Ncrit =      3.00000000\n'
                    match = re.search(r"Re\s*=\s*([-\d\.]+)\s*e\s*(\d+)", line)
                    if match:
                        base = float(match.group(1))
                        exponent = int(match.group(2))
                        reynolds = base * 10**exponent
                    else:
                        raise ValueError(f"Could not extract Reynolds number from line: {line}")

        airfoils[i]['polars'][j]['re_sets'] = [{}]
        airfoils[i]['polars'][j]['re_sets'][0]['re'] = reynolds
        polars = np.loadtxt(os.path.join(polar_folder, matching_files[j]), skiprows=13)
        airfoils[i]['polars'][j]['re_sets'][0]['cl'] = {}
        airfoils[i]['polars'][j]['re_sets'][0]['cd'] = {}
        airfoils[i]['polars'][j]['re_sets'][0]['cm'] = {}

        aoa = polars[:,0]
        cl = polars[:,1]
        cd = polars[:,2]
        cm = polars[:,4]

        mypolar = Polar(Re=reynolds, alpha=aoa, cl=cl, cd=cd, cm=cm, compute_params=True, radians=False)
        mypolar_extended = mypolar.extrapolate(1.5, cdmin=0.001)
        aoa_extended = mypolar_extended.alpha
        cl_extended = mypolar_extended.cl
        cd_extended = mypolar_extended.cd
        cm_extended = mypolar_extended.cm

        # Enforce minimum drag at extreme AOA
        cd_extended[aoa_extended < -60] = np.maximum(cd_extended[aoa_extended < -60], 0.05)
        cd_extended[aoa_extended > +60] = np.maximum(cd_extended[aoa_extended > +60], 0.05)


        airfoils[i]['polars'][j]['re_sets'][0]['cl']['grid'] = aoa_extended.tolist()
        airfoils[i]['polars'][j]['re_sets'][0]['cd']['grid'] = aoa_extended.tolist()
        airfoils[i]['polars'][j]['re_sets'][0]['cm']['grid'] = aoa_extended.tolist()
        airfoils[i]['polars'][j]['re_sets'][0]['cl']['values'] = cl_extended.tolist()
        airfoils[i]['polars'][j]['re_sets'][0]['cd']['values'] = cd_extended.tolist()
        airfoils[i]['polars'][j]['re_sets'][0]['cm']['values'] = cm_extended.tolist()
        if 'clean'in matching_files[j] and 'DragOff' in matching_files[j]:
            airfoils[i]['polars'][j]['configuration'] = 'Clean and drag off'
        elif 'clean' in matching_files[j] and 'DragOn' in matching_files[j]:
            airfoils[i]['polars'][j]['configuration'] = 'Clean and drag on'
        elif 'rough' in matching_files[j] and 'DragOff' in matching_files[j]:
            airfoils[i]['polars'][j]['configuration'] = 'Rough and drag off'
        elif 'rough' in matching_files[j] and 'DragOn' in matching_files[j]:
            airfoils[i]['polars'][j]['configuration'] = 'Rough and drag on'
        else:
            raise ValueError('Could not determine configuration from file name: %s' % matching_files[j])

data = {}
data['airfoils'] = airfoils


import matplotlib.pyplot as plt

for i in range(len(airfoils)):
    fig, axs = plt.subplots(2, 2, figsize=(7, 5), sharex=True)

    for j in range(4):
        airfoil_name = airfoils[i]['name']
        config_label = airfoils[i]['polars'][j]['configuration']
        aoa_extended = airfoils[i]['polars'][j]['re_sets'][0]['cl']['grid']
        cl_extended = airfoils[i]['polars'][j]['re_sets'][0]['cl']['values']
        cd_extended = airfoils[i]['polars'][j]['re_sets'][0]['cd']['values']
        cm_extended = airfoils[i]['polars'][j]['re_sets'][0]['cm']['values']
        axs[0,0].plot(aoa_extended, cl_extended, label=config_label)
        axs[0,1].plot(aoa_extended, cd_extended, label=config_label)
        axs[1,0].plot(aoa_extended, cm_extended, label=config_label)
        efficiency = np.array(cl_extended) / np.array(cd_extended)
        axs[1,1].plot(aoa_extended, efficiency, label=config_label)

    axs[1,0].set_xlabel('Angle of Attack (deg)')
    axs[1,1].set_xlabel('Angle of Attack (deg)')
    axs[0,0].set_ylabel('Cl')
    axs[0,1].set_ylabel('Cd')
    axs[1,0].set_ylabel('Cm')
    axs[1,1].set_ylabel('Cl/Cd')
    axs[1,1].set_ylim([-100, 250])

    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True)
    plt.subplots_adjust(right=0.75)
    axs[1,0].legend()
    plt.suptitle(f"Airfoil: {airfoil_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.savefig(os.path.join(run_dir, "Figures", f"{airfoil_name}_polars_full_aoa.png"))
    
    ax.set_xlim([-10, 20])
    axs[0,0].set_ylim([-1, 2.5])
    axs[0,1].set_ylim([0, 0.05])
    axs[1,0].legend()
    plt.savefig(os.path.join(run_dir, "Figures", f"{airfoil_name}_polars_zoomed_aoa.png"))
    # plt.show()
    plt.close()


for j in range(4):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7 ), sharex=True)

    config_label = airfoils[0]['polars'][j]['configuration']
    for i in range(len(airfoils)):
        airfoil_name = airfoils[i]['name']
        for k in range(4):
            if airfoils[i]['polars'][k]['configuration'] == config_label:
                break
        
        aoa_extended = airfoils[i]['polars'][k]['re_sets'][0]['cl']['grid']
        cl_extended = airfoils[i]['polars'][k]['re_sets'][0]['cl']['values']
        cd_extended = airfoils[i]['polars'][k]['re_sets'][0]['cd']['values']
        cm_extended = airfoils[i]['polars'][k]['re_sets'][0]['cm']['values']
        axs[0,0].plot(aoa_extended, cl_extended, label=airfoil_name)
        axs[0,1].plot(aoa_extended, cd_extended, label=airfoil_name)
        axs[1,0].plot(aoa_extended, cm_extended, label=airfoil_name)
        efficiency = np.array(cl_extended) / np.array(cd_extended)
        axs[1,1].plot(aoa_extended, efficiency, label=airfoil_name)

    axs[1,0].set_xlabel('Angle of Attack (deg)')
    axs[1,1].set_xlabel('Angle of Attack (deg)')
    axs[0,0].set_ylabel('Cl')
    axs[0,1].set_ylabel('Cd')
    axs[1,0].set_ylabel('Cm')
    axs[1,1].set_ylabel('Cl/Cd')
    axs[1,1].set_ylim([-100, 250])

    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True)
    # plt.subplots_adjust(bottom=0.75)
    
    # Move legend outside and below the figure
    axs[1,0].legend()
    plt.suptitle(f"Configuration: {config_label}")
    plt.tight_layout()

    plt.savefig(os.path.join(run_dir, "Figures", f"{config_label}_polars_full_aoa.png"))
    
    ax.set_xlim([-10, 20])
    axs[0,0].set_ylim([-1, 2.5])
    axs[0,1].set_ylim([0, 0.05])
    plt.savefig(os.path.join(run_dir, "Figures", f"{config_label}_polars_zoomed_aoa.png"))
    plt.show()


import windIO
windIO.yaml.write_yaml(data, os.path.join(run_dir, 'OSO_2025_WT2_airfoils_windIO_2p0p1.yaml'))