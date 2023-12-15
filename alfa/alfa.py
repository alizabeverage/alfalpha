from grids import *
from read_data import Data
from polynorm import polynorm
import numpy as np
import pandas as pd
import csv
import emcee
import corner
from multiprocessing import Pool
import matplotlib.pyplot as plt
#from schwimmbad import MPIPool
from setup_params import setup_params, get_properties
import os, sys
from utils import correct_abundance
from plot_outputs import plot_outputs

# must have alfa_home defined in bash profile
ALFA_HOME = os.environ['ALFA_HOME']
ALFA_OUT = os.environ['ALFA_OUT']

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
    nsteps = 1000
    nsteps_save = 300
    thin = 1
    post_process = True

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
    grids = Grids(inst_res=inst_res,kroupa_shortcut=False)

    # define which parameters to fit
    # you must have logage and zH
    # parameters_to_fit = ['velz', 'sigma', 'logage', 'zH', 'feh', 'ah', 'ch', 
    #                 'nh', 'nah', 'mgh', 'sih', 'kh', 'cah', 'tih', 'vh', 
    #                 'crh']

    parameters_to_fit = ['velz', 'sigma', 'logage', 'zH', 'feh', 'mgh', 'cah']
    
    # fit emission lines if HM 56163 or 23351
    # if '56163' in filename or '23351' in filename:
    #     parameters_to_fit+=list(np.unique(emline_strs[(wave_emlines<5600)&(wave_emlines>3800)]))
    #     parameters_to_fit+=['velz2','sigma2']

    # get the positions and priors of parameters_to_fit
    default_pos, priors = setup_params(parameters_to_fit)
    

    #~~~~~~~~~~~~~~~~~~~~~ emcee ~~~~~~~~~~~~~~~~~~~~~~~ #
    print("fitting with emcee...")

    # with Pool() as pool:    
    # initialize walkers
    pos = np.array(default_pos) + \
            1e-2 * np.random.randn(nwalkers, len(parameters_to_fit))
    nwalkers, ndim = pos.shape

    # open file for saving steps
    backend = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
    backend.reset(nwalkers, ndim)

    #sample
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, backend=backend#, pool=pool #, args=(data,grids)
          )
    sampler.run_mcmc(pos, nsteps, progress=True);

    # #~~~~~~~~~~~~~~~~~~~~~ post-process ~~~~~~~~~~~~~~~~~~~~~~~ #

    if post_process:
        reader = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
        flat_samples = reader.get_chain(discard=nsteps-nsteps_save, thin=thin, flat=True)


        # corner plot
        sel = np.ones(len(parameters_to_fit)).astype(bool)
        fig = corner.corner(
            flat_samples[:,sel], labels=np.array(parameters_to_fit)[sel],
            show_titles=True
        )
        plt.savefig(f"{ALFA_OUT}{filename}_corner.jpeg")
    
        
        # best-fit spectra
        plt.figure(figsize=(10,3))
        inds = np.random.randint(len(flat_samples), size=10)
        for ind in inds:
            sample = flat_samples[ind]
            
            params = get_properties(sample,parameters_to_fit)
            mflux = g.get_model(params,outwave=data.wave)
        
            #poly norm
            poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
            plt.plot(data.wave,data_r, 'C0', lw=1)
            plt.plot(data.wave,mfluxnorm, 'C1')
            
        
        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
    
        plt.savefig(f"{ALFA_OUT}{filename}_bestspec.jpeg")
    
    
        # save outputs in summary file
        parameters_to_fit = np.array(parameters_to_fit)
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
    
    


