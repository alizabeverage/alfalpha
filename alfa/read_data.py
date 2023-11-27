import numpy as np
import pandas as pd
import os
ALFA_HOME = os.environ['ALFA_HOME']

class Data():
    def __init__(self,filename=None):
        self.filename = filename
        if self.filename is None: 
            return
            
        tmp = np.array(pd.read_csv(f"{ALFA_HOME}indata/{self.filename}.dat",
                                       delim_whitespace=True, header=None, comment='#'))
        self.wave = tmp[:,0]
        self.flux = tmp[:,1]
        self.err = tmp[:,2]
        self.mask = tmp[:,3]
        self.ires = tmp[:,4]

        # this is how I'm dealing with masked values. Make nan
        # then it's not included in the log prob
        self.err[self.mask == 0] = np.nan

        self.fitting_regions = []
        self.get_fitting_regions()


    def get_fitting_regions(self):
        f = open(f"{ALFA_HOME}indata/{self.filename}.dat",'r')
        lines = f.readlines()
        f.close()
        for line in lines[:50]:
            if '#' in line: 
                x1 = float(line.split()[1])
                x2 = float(line.split()[2])
                self.fitting_regions.append([x1,x2])
        self.fitting_regions = 1e4*np.array(self.fitting_regions)
