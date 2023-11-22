from alfa.grids import Grids
from alfa.utils import correct_abundance
import numpy as np

inst_res = 0
outwave = None

G = Grids(inst_res=inst_res)
params = {'velz':0, 'sigma':200, 'logage':0.4, 'zH':0,
             'feh':0, 'ah':0, 'ch':0, 'nh':0, 'nah':0, 'mgh':0, 'sih':0,
                  'kh':0, 'cah':0, 'tih':0, 'vh':0, 'crh':0, 'mnh':0, 'coh':0,
                  'nih':0, 'cuh':0, 'srh':0, 'bah':0, 'euh':0}


logage = [0.1,0.2,0.3,0.4]
zh = [-0.1,0.1]
feh = []
mgh = []
model = G.get_model(params,outwave=outwave)

