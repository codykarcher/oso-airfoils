from airfoilprep import Polar, Airfoil
import numpy as np
import os,sys,importlib
import warnings
import shutil

import ruamel.yaml as yaml
yml = yaml.YAML(typ='rt', pure=True)
yml2 = yaml.YAML(typ='rt', pure=True)

import xfoil as x
importlib.reload(sys.modules['xfoil'])

import blade as b
importlib.reload(sys.modules['blade'])

import helpers as util
importlib.reload(sys.modules['helpers'])

def main():

    airfoils = {}
    polars = {}
    aerodyn_airfoils = {}

    with open('input.yaml', 'r') as stream:
        loadyaml = yml.load(stream)
    
    airfoil_files = [af['polar'] for af in loadyaml['new']['airfoils']]
    airfoil_types = [af['type'] for af in loadyaml['new']['airfoils']]
    airfoil_thick = [af['thickness'] for af in loadyaml['new']['airfoils']]
    anames = [af['name'] for af in loadyaml['new']['airfoils']]

    print(airfoil_thick)

    # Model directory setup
    ##########################################   
    input_of_model = loadyaml['old']['directory']
    blade_file = os.path.join(input_of_model,loadyaml['old']['bladefile'])
    aerodyn_file = os.path.join(input_of_model,loadyaml['old']['aerodynfile'])
    of_airfoil_dir = input_of_model+'/IEA-22-280-RWT/Airfoils'

    print("blade_file",input_of_model)

    new_of_model = loadyaml['new']['directory']
    new_blade_file = new_of_model+'/IEA-22-280-RWT/IEA-22-280-RWT_AeroDyn15_blade.dat'
    new_aerodyn_file = new_of_model+'/IEA-22-280-RWT-Monopile/IEA-22-280-RWT_AeroDyn15.dat'
    new_of_airfoil_dir = new_of_model+'/IEA-22-280-RWT/Airfoils'  

    shutil.copytree(input_of_model, new_of_model,dirs_exist_ok=True)

    # Read Blade Props and create output blade
    ##########################################
    iea22blade = b.blade('iea22mw_orig',aerodyn_file,blade_file,of_airfoil_dir, intent="input")
    new22blade = b.blade('iea22mw_new',aerodyn_file,new_blade_file,new_of_airfoil_dir, intent="output")
    
    # Read Airfoil Data
    ###################
    for i,af in enumerate(airfoil_files):
        airfoils[anames[i]] = {}
        polars[anames[i]] = {}
        if(airfoil_types[i] == 'xfoil'):
            airfoils[anames[i]] = x.xfoil(anames[i],af,loadyaml['new']['airfoils'][i]['coords'],airfoil_thick[i])
            polars[anames[i]]['polar_orig'] = Polar(airfoils[anames[i]].Re, airfoils[anames[i]].alpha, airfoils[anames[i]].cl, airfoils[anames[i]].cd, airfoils[anames[i]].cm)
            polars[anames[i]]['airfoil_obj'] = Airfoil([polars[anames[i]]['polar_orig']])
            if(airfoils[anames[i]].alpha[-1] < 180.0):
                polars[anames[i]]['airfoil_extrap'] = polars[anames[i]]['airfoil_obj'].extrapolate(airfoils[anames[i]].cdmax)
            else:
                polars[anames[i]]['airfoil_extrap'] = polars[anames[i]]['airfoil_obj']

    # Create OpenFAST Blade With Polars and Write Files
    ###################################################
    mapi = []
    mapr = []
    nroot = int(loadyaml['new']['numroot'])
    maxai = int(len(airfoil_files)-1)

    # Create fixed points for ga airfoils
    for j,t in enumerate(airfoil_thick):
        if j == 0:
            mapi.append(nroot-1)
            mapr.append(iea22blade.span[nroot-1])
        else:
            mapi.append(util.closest_index(iea22blade.thickness*100,t))
            mapr.append(iea22blade.span[mapi[j]])           

    # Create polar and airfoil objects
    for bs,f in enumerate(iea22blade.blade_airfoilpaths):
        for i,ga in enumerate(mapi):

            if bs <= mapi[0]:
                new22blade.add_airfoilprep_object_single(airfoils[anames[0]],bs)
                new22blade.add_coordinates_single(airfoils[anames[0]].coord_data,airfoils[anames[0]].x_over_c)
                #print(bs,airfoils[anames[0]].x_over_c,airfoils[anames[0]].name)
                break

            if bs == ga:
                new22blade.add_airfoilprep_object_single(airfoils[anames[i]],bs)
                new22blade.add_coordinates_single(airfoils[anames[i]].coord_data,airfoils[anames[i]].x_over_c)
                #print(bs,airfoils[anames[i]].x_over_c,airfoils[anames[i]].name)
                break

            if bs >= mapi[-1]:
                new22blade.add_airfoilprep_object_single(airfoils[anames[-1]],bs)
                new22blade.add_coordinates_single(airfoils[anames[-1]].coord_data,airfoils[anames[-1]].x_over_c)
                #print(bs,airfoils[anames[-1]].x_over_c,airfoils[anames[-1]].name)
                break

            if bs > mapi[0] and bs < mapi[-1] and bs > mapi[i-1] and bs < mapi[i]:
                r = iea22blade.span[bs]
                arl = mapr[i-1]
                aru = mapr[i]
                frac = (r-arl)/(aru-arl)
                new22blade.add_airfoilprep_object_blended(airfoils[anames[i-1]],airfoils[anames[i]],bs,frac)
                new22blade.add_coordinates_blended(airfoils[anames[i-1]].coord_data,airfoils[anames[i]].coord_data,frac,airfoils[anames[i]].x_over_c)
                #print(bs,airfoils[anames[i]].x_over_c,airfoils[anames[i]].name)
                break
    
    for bs,airfoil in enumerate(new22blade.ap_airfoils_extrap):

        # Write to Aerodyn 13 format
        a13_out_path = os.path.join(new_of_airfoil_dir,'v13_airfoil'+str(bs)+'.dat')
        airfoil.writeToAerodynFile(a13_out_path)

        # Read Aerodyn 13 for conversion
        a13header,a13data = util.load_aerodyn_v13(a13_out_path)
        alatest_out_path = os.path.join(new_of_airfoil_dir,'IEA-22-280-RWT_AeroDyn15_Polar_'+str(bs).zfill(2)+'.dat')
        util.writeAerodynLatest(a13header,a13data,alatest_out_path)

        # Replace Aerodyn15 template values with actual values
        util.replace_regex(alatest_out_path,'Re ',np.round(new22blade.ap_airfoils_extrap[bs].polars[0].Re/1000000.0,2))
        util.replace_regex(alatest_out_path,'alpha0',np.round(util.get_zl_aoa(new22blade.ap_airfoils_extrap[bs].polars[0]),2))

        os.remove(a13_out_path)

        # Write coordinate files
        #print("Write Coordinate Files")
        coord_out_path=os.path.join(new_of_airfoil_dir,"IEA-22-280-RWT_AF"+str(bs).zfill(2)+"_Coords.txt")

        with open(coord_out_path, "w") as file:
            nc = len(new22blade.coord_data[bs])
            xc = new22blade.x_over_c[bs]
            file.write(new22blade.coord_file_header(nc,xc))

        new22blade.coord_data[bs].to_csv(coord_out_path, mode='a', sep='\t', index=None, header=False)

    # Work with windio file labels
    with open(loadyaml['windio']['turbine_input'], 'r') as stream:
        turbine = yml.load(stream)

    turbine = util.modify_windio(airfoils,polars,turbine,loadyaml['windio'])

    yml.width = 4096
    yml.default_flow_style = False

    with open(loadyaml['windio']['turbine_output'], 'w') as file:
        yml.dump(turbine, file)

if __name__ == "__main__":
    main()



