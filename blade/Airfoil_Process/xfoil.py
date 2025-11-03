import pandas as pd
import numpy as np
import os,sys,importlib
import warnings
import re
from scipy.interpolate import interp1d

class xfoil:

    def __init__(self,name,filepath,coordfilepath,thick):

        self.name = name
        self.filepath = filepath
        self.coordfilepath = coordfilepath
        self.data = None
        self.Re = None
        self.alpha = None
        self.cl = None
        self.cd = None
        self.Re = None
        self.cm = None
        self.Sxtr = None
        self.Pxtr = None
        self.CDp = None
        self.thick = thick
        self.cdmax = None

        self.zerolift_aoa = None
        
        self.coord_data = None 
        self.x_over_c = None 

        self.parse_xfoil(filepath)
        self.read_coords(coordfilepath)
        self.get_x_over_c(thick)
        self.set_zl_aoa()

    def parse_xfoil(self,filepath):

        headskip = [0,1,2,3,4,5,6,7,8,9,10,12]
        self.data = pd.read_csv(filepath,sep='\\s+',skiprows=(headskip), header=(0),skipinitialspace=True)
        print('Reading',filepath)

        self.alpha = np.array([np.round(a,2) for a in self.data.alpha])
        self.cl = np.array(self.data.CL)
        self.cd = np.array(self.data.CD)
        self.Re = np.mean(np.array(self.data['Re(CL)']))*1e6
        self.cm = np.array(self.data.CM)
        self.Sxtr = np.array(self.data.S_xtr)
        self.Pxtr = np.array(self.data.P_xtr)
        self.CDp = np.array(self.data.CDp)

        #self.thick = float(re.search(r'_ga\d\d_', filepath).group(0).replace('_','').replace('ga',''))

        self.cdmax = np.max(self.cd)

    def read_coords(self,filepath):
        temp_coords = pd.read_csv(filepath,sep='\\s+',header=None,skipinitialspace=True)
        temp_coords.columns = ['x','y']

        if temp_coords['y'].iloc[0:40].mean() > 0:
            self.coord_data = temp_coords.iloc[::-1].reset_index(drop=True)
        else:
            self.coord_data = temp_coords

    def get_x_over_c(self,thickness):
        if thickness > 90.0:
            self.x_over_c = 0.5
        else:
            self.x_over_c = 0.25

    def set_zl_aoa(self):

        f_lin = interp1d(self.cl,self.alpha)
        
        clvals = np.unique(np.round(self.cl,2))

        if len(clvals) > 1:
            self.zerolift_aoa = f_lin([0.0])
        else:
            self.zerolift_aoa = 0.0