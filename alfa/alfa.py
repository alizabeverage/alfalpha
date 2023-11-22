from grids import Grids
from read_data import Data
from polynorm import polynorm
import numpy as np
import pandas as pd
import csv
import emcee
# import corner
from multiprocessing import Pool
import matplotlib.pyplot as plt
from schwimmbad import MPIPool
from setup_params import setup_params, get_properties
import os, sys
from utils import correct_abundance

# must have alfa_home defined in bash profile
ALFA_HOME = os.environ['ALFA_HOME']
ALFA_OUT = os.environ['ALFA_OUT']

# define which parameters to fit
# you must have logage and zH
parameters_to_fit = ['velz', 'sigma', 'logage', 'zH', 'feh', 'ah', 'ch', 
                'nh', 'nah', 'mgh', 'sih', 'kh', 'cah', 'tih', 'vh', 
                'crh']

# get the positions and priors of parameters_to_fit
default_pos, priors = setup_params(parameters_to_fit)

# ~~~~~~~~~~~~~~~~~~~~~~~ probability stuff ~~~~~~~~~~~~~~~~~~~~~~~ #

def lnlike(theta): #, data, grids):
    # generate model according to theta
    params = get_properties(theta,parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)

    #poly norm
    poly, mfluxnorm = polynorm(data, mflux)

    
    return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2))
    
def lnprior(theta):
    check = np.array([prio[0] < p < prio[1] for (p,prio) in zip(theta,priors.values())])
    if False not in check:
        return 1.0
    return -np.inf

def lnprob(theta): #, data, grids):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp*lnlike(theta) #, data, grids)




# ~~~~~~~~~~~~~~~~~~~~~~~ Run fitting tool ~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":    
    nwalkers = 256
    nsteps = 10000
    nsteps_save = 1000

    # use command arguments to get filename
    if len(sys.argv)>1:
        filename = sys.argv[1] # the script name is 0  
    
    # manually set filename (overwrites above!)
    else:
        filename = 'test'

    # set up data object
    print(f"Loading {filename}...")
    data = Data(filename)
    
    # you still need to add wavelength dependent ires
    # for now, just take the average of the data
    inst_res = np.average(data.ires) 

    # set up grid object
    print(f"Loading grids...")
    grids = Grids(inst_res=inst_res)
    

    #~~~~~~~~~~~~~~~~~~~~~ emcee ~~~~~~~~~~~~~~~~~~~~~~~ #
    print("fitting with emcee...")
    
    # initialize walkers
    pos = np.array(default_pos) + \
                    1e-4 * np.random.randn(nwalkers, len(parameters_to_fit))
    nwalkers, ndim = pos.shape

    # open file for saving steps
    backend = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
    backend.reset(nwalkers, ndim)

    #sample
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, backend=backend #, args=(data,grids)
    )
    sampler.run_mcmc(pos, nsteps, progress=True);

    
    #~~~~~~~~~~~~~~~~~~~~~ post-process ~~~~~~~~~~~~~~~~~~~~~~~ #
    
    samples = sampler.get_chain(flat=False, thin=15,discard=nsteps-nsteps_save)
    np.save(f"{ALFA_OUT}{filename}_mcmc.npy",samples)

    flat_samples = sampler.get_chain(discard=nsteps-nsteps_save,flat=True, thin=15)
    # make summary file with abundances corrected
    dict_results = {}
    for i,param in enumerate(parameters_to_fit):
        dict_results[param+'16'] = [np.percentile(flat_samples[:,i],16)]
        dict_results[param+'50'] = [np.median(flat_samples[:,i])]
        dict_results[param+'84'] = [np.percentile(flat_samples[:,i],64)]
    
        if param in ['feh','ah','ch','nh','nah','mgh','sih',
                              'kh','cah','tih','vh','crh','mnh','coh',
                              'nih','cuh','srh','bah','euh']:
            dist = correct_abundance(flat_samples[:,parameters_to_fit=='zH'].ravel(),
                                     flat_samples[:,i],param)
            param_st = '['+param[:-1].capitalize()+'/H]'
            dict_results[param_st+'16'] = [np.percentile(dist,16)]
            dict_results[param_st+'50'] = [np.median(dist)]
            dict_results[param_st+'64'] = [np.percentile(dist,64)]
    
    
    df = pd.DataFrame.from_dict(dict_results)
    df.to_csv(f"{ALFA_OUT}{filename}.sum", 
              float_format='%10.3f', sep=" ", 
              quoting=csv.QUOTE_NONE, escapechar=" ")
    




