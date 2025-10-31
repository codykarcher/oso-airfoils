import pandas as pd
import numpy as np
import os,sys,importlib
import warnings
import re
from airfoilprep import Polar, Airfoil
from scipy.interpolate import interp1d

class blade:

    def __init__(self,name,afilepath="",bfilepath="",of_airfoil_dir="",intent="input"):

        self.name = name
        self.afilepath = afilepath
        self.bfilepath = bfilepath
        self.data = None
        self.span = None
        self.curve_ac = None
        self.sweep = None
        self.curve_ang = None
        self.twist = None
        self.chord = None
        self.afid = None
        self.blade_airfoilpaths = None
        self.of_airfoil_dir = of_airfoil_dir
        self.n = None
        self.coord_files = None
        self.thickness = None

        self.ap_polars = []
        self.ap_airfoils = []
        self.ap_airfoils_extrap = []
        self.ap_idx = []

        self.coord_data = []
        self.x_over_c = []

        if intent == "input":
            self.parse_blade(bfilepath)
            self.parse_aerodyn(afilepath)
            self.get_coord_files(self.blade_airfoilpaths)
            self.calc_thickness(self.coord_files,self.chord,self.of_airfoil_dir)


    def parse_blade(self,bfilepath):

        headskip = [0,1,2,3,5]
        self.data = pd.read_csv(bfilepath,sep='\\s+',skiprows=(headskip), header=(0),skipinitialspace=True)
        print('Reading',bfilepath)

        self.span = np.array(self.data.BlSpn)
        self.curve_ac = np.array(self.data.BlCrvAC)
        self.sweep = np.array(self.data.BlSwpAC)
        self.curve_ang = np.array(self.data.BlCrvAng)
        self.twist = np.array(self.data.BlTwist)
        self.chord = np.array(self.data.BlChord)
        self.afid = np.array(self.data.BlAFID)

    def parse_aerodyn(self,afilepath):

        airfoils = []
        ai = 0
        nline = 0
        nairfoils = 0

        with open(afilepath) as f:
            for num, line in enumerate(f, 1):
                if " NumAFfiles" in line:
                    nline = num
                    nairfoils = int(line.split()[0])
                if (num > nline) & (num <= (nline + nairfoils)) & (nairfoils > 0) & (nline > 0):
                    airfoils.append(line.split()[0].replace('"','').split('/')[-1])

        self.blade_airfoilpaths = airfoils
        self.n = len(airfoils)

    def get_coord_files(self,afdatfiles):

        airfoil_files = []

        for i,a in enumerate(afdatfiles):
            af_genfile = os.path.join(self.of_airfoil_dir,a)
            with open(af_genfile) as f:
                for num, line in enumerate(f, 1):
                    if "NumCoords" in line:
                        airfoil_files.append(line.split()[0].replace("@","").replace('"',''))

        self.coord_files = airfoil_files

    def calc_thickness(self,coordfiles,chord,apath):

        thick = []
        headskip = [0,1,2,3,4,5,6,7]

        for i,cf in enumerate(coordfiles):
            thispath = os.path.join(apath,cf)
            rawcoord = pd.read_csv(thispath,skiprows=(headskip),sep='\\s+',header=None)
            rawcoord.columns = ['x_over_c','y_over_c']
            yvals = rawcoord.y_over_c
            rawthick = np.max(yvals)-np.min(yvals)
            thick.append(rawthick)

        self.thickness = np.array(np.round(thick,4))

    def add_airfoilprep_object_single(self,polarinput,span_idx):
        self.ap_idx.append(span_idx)

        # Create polar
        temp_polar_object = Polar(polarinput.Re, polarinput.alpha, polarinput.cl, polarinput.cd, polarinput.cm)
        self.ap_polars.append(temp_polar_object)

        # Temp airfoil obj
        temp_airfoil_object = Airfoil([temp_polar_object])
        self.ap_airfoils.append(temp_airfoil_object)

        # Extrapolate polars if necessary
        if(polarinput.alpha[-1] < 180.0):
            self.ap_airfoils_extrap.append(temp_airfoil_object.extrapolate(polarinput.cdmax))
        else:
            self.ap_airfoils_extrap.append(temp_airfoil_object)


    def add_airfoilprep_object_blended(self,polarinputlow,polarinputhigh,span_idx,blend_frac):
        self.ap_idx.append(span_idx)

        # Create polar
        temp_polar_object_low = Polar(polarinputlow.Re, polarinputlow.alpha, polarinputlow.cl, polarinputlow.cd, polarinputlow.cm)
        temp_polar_object_high = Polar(polarinputhigh.Re, polarinputhigh.alpha, polarinputhigh.cl, polarinputhigh.cd, polarinputhigh.cm)
        temp_polar_blended = temp_polar_object_low.blend(temp_polar_object_high,blend_frac)
        self.ap_polars.append(temp_polar_blended)

        blended_cdmax = np.max(temp_polar_blended.cd)

        # Temp airfoil obj
        temp_airfoil_object = Airfoil([temp_polar_blended])
        self.ap_airfoils.append(temp_airfoil_object)

        # Extrapolate polars if necessary
        if(temp_polar_blended.alpha[-1] < 180.0):
            self.ap_airfoils_extrap.append(temp_airfoil_object.extrapolate(blended_cdmax))
        else:
            self.ap_airfoils_extrap.append(temp_airfoil_object)

    def add_coordinates_single(self,coord_data,xc):
        self.coord_data.append(coord_data)
        self.x_over_c.append(xc)

    def add_coordinates_blended(self,coord_data_low,coord_data_high,blend_frac,xc):
        self.x_over_c.append(xc)

        if coord_data_low.iloc[0:100].x.equals(coord_data_high.iloc[0:100].x):
            temp_coord_data = blend_frac*coord_data_high + (1.0-blend_frac)*coord_data_low
        else:
            x = coord_data_low.x
            y = coord_data_low.y

            f = interp1d(x, y, kind='linear')

            x_new = coord_data_high.x 
            y_new = f(x_new)  # Interpolated y values

            new_coord_data_low = pd.DataFrame({'x':x_new,'y':y_new})
            temp_coord_data = blend_frac*coord_data_high + (1.0-blend_frac)*new_coord_data_low 

        self.coord_data.append(temp_coord_data)

    

    def coord_file_header(self,numcoords,xoverc):

        header_text = """{num}   NumCoords   ! The number of coordinates in the airfoil shape file (including an extra coordinate for airfoil reference).  Set to zero if coordinates not included.
! ......... x-y coordinates are next if NumCoords > 0 .............
! x-y coordinate of airfoil reference
!  x/c        y/c
{x_over_c}       0
! coordinates of airfoil shape
! interpolation to 200 points
!  x/c        y/c
"""
        
        updated_string = header_text.format(num=numcoords,x_over_c=xoverc)

        return updated_string