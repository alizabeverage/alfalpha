from alfa.grids import *
from alfa.read_data import Data
from alfa.polynorm import polynorm
import numpy as np
import pandas as pd
import csv
import emcee
import corner
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
#from schwimmbad import MPIPool
from alfa.setup_params import setup_params,get_properties, setup_initial_position, setup_initial_position_diff_ev,get_init_pos_bounds
import os, sys
from alfa.utils import correct_abundance
from alfa.plot_outputs import plot_outputs
from scipy.optimize import differential_evolution

multip = False

# must have alfa_home defined in bash profile
ALFA_HOME = os.environ['ALFA_HOME']
ALFA_OUT = os.environ['ALFA_OUT']

diff_ev_parameters = ['velz','sigma','logage','zH']

parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh',
                    'ch', 'nh', 'mgh', 'nah', 'ah', 'sih', 'cah',
                    'tih', 'crh', 'teff','jitter','logemline_h', 
                    'velz2', 'sigma2'])


default_pos, priors = setup_params(parameters_to_fit)
priors['jitter'] = [1.6,1.66]
bounds = get_init_pos_bounds()
bounds['jitter'] = [1.6,1.66]
ncpu = cpu_count()
# ~~~~~~~~~~~~~~~~~~~~~~~ probability stuff ~~~~~~~~~~~~~~~~~~~~~~~ #


def lnlike(theta): # multiprocessing
    # generate model according to theta
    params = get_properties(theta,parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)
    
    #poly norm
    poly, mfluxnorm = polynorm(data, mflux, deg=None)
    
    if 'jitter' in parameters_to_fit:
        # copied from alf
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2*params['jitter']**2) \
                            + np.log(2*np.pi*data.err**2*params['jitter']**2))
    else:
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2))
        
def lnprior(theta):
    check = np.array([prio[0] < p < prio[1] for (p,prio) in zip(theta,priors.values())])
    if False not in check:
        return 1.0
    return -np.inf
    
def lnprob(theta): # multiprocessing
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp*lnlike(theta) # multiprocessing


def diff_ev_objective_function(theta):
    # Generate model according to theta
    params = get_properties(theta,diff_ev_parameters)
    mflux = grids.get_model(params,outwave=data.wave)

    # Perform polynomial normalization
    poly, mfluxnorm = polynorm(data, mflux)

    return np.nansum((data.flux - mfluxnorm)**2/(data.err**2))



if __name__ == "__main__":  
    filename = sys.argv[1] # the script name is 0  
    
    # set up data object
    print(f"Loading {filename}...")
    data = Data(filename, filename_exact=True)

    # set up grid object
    print(f"Loading grids...")
    grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)

    #~~~~~~~~~~~~~~~~~~~~~ run differential evolution ~~~~~~~~~~~~~~~~~~ #
    print(f"Running differential evolution... diff_ev_parameters: {diff_ev_parameters}")
    _,prior = setup_params(diff_ev_parameters)
    bounds = list(prior.values())  

    # Run differential evolution optimization
    result = differential_evolution(diff_ev_objective_function, bounds,disp=False,updating='deferred')
    
    diff_ev_result = get_properties(result.x,diff_ev_parameters)
    print(f"Differential evolution result: {diff_ev_result}")
    print(f"Differential evolution success: {result.success}")

    #~~~~~~~~~~~~~~~~~~~~~ emcee ~~~~~~~~~~~~~~~~~~~~~~~ #
    # Now, run emcee, fix the starting position to the result of the differential evolution
    nwalkers = 256
    nsteps = 8000
    nsteps_save = 100
    thin = 1
    post_process = True
    print("fitting with emcee...")

    # initialize walkers
    if result.success:
        pos = setup_initial_position_diff_ev(nwalkers,parameters_to_fit,diff_ev_result=diff_ev_result,
                                             init_pos=bounds)

    else:
        pos = setup_initial_position(nwalkers,parameters_to_fit,init_pos=bounds)
    
    
    nwalkers, ndim = pos.shape

    # open file for saving steps
    filename = filename.split('/')[-1]
    backend = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
    backend.reset(nwalkers, ndim)
    with Pool() as pool: 
        #sample
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, lnprob, backend=backend 
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
            mflux = grids.get_model(params,outwave=data.wave)
        
            #poly norm
            poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
            plt.plot(data.wave,data_r, 'C0', lw=1)
            plt.plot(data.wave,mfluxnorm, 'C1')
            
        
        plt.xlabel('Wavelength ($\mathring{\mathrm{A}}$)')
        plt.ylabel('F$_\lambda$')
        plt.tight_layout()
    
        plt.savefig(f"{ALFA_OUT}{filename}_bestspec.jpeg",dpi=200)
        
    
        # save outputs in summary file
        parameters_to_fit = np.array(parameters_to_fit)
        dict_results = {}
        # define Fe for retrieving [X/Fe]
        Fe = correct_abundance(flat_samples[:,parameters_to_fit=='zH'].ravel(),
                                         flat_samples[:,parameters_to_fit=='feh'].ravel(),'feh')
        for i,param in enumerate(parameters_to_fit):
            dict_results[param+'16'] = [np.percentile(flat_samples[:,i],16)]
            dict_results[param+'50'] = [np.median(flat_samples[:,i])]
            dict_results[param+'84'] = [np.percentile(flat_samples[:,i],84)]
        
            if param in ['feh','ah','ch','nh','nah','mgh','sih',
                                  'kh','cah','tih','vh','crh','mnh','coh',
                                  'nih','cuh','srh','bah','euh']:
                dist = correct_abundance(flat_samples[:,parameters_to_fit=='zH'].ravel(),
                                         flat_samples[:,i].ravel(),param)
                param_st = '['+param[:-1].capitalize()+'/H]'
                dict_results[param_st+'16'] = [np.percentile(dist,16)]
                dict_results[param_st+'50'] = [np.median(dist)]
                dict_results[param_st+'84'] = [np.percentile(dist,84)]

                param_st = '['+param[:-1].capitalize()+'/Fe]'
                dict_results[param_st+'16'] = [np.percentile(dist-Fe,16)]
                dict_results[param_st+'50'] = [np.median(dist-Fe)]
                dict_results[param_st+'84'] = [np.percentile(dist-Fe,84)]
        
        
        df = pd.DataFrame.from_dict(dict_results)
        np.savetxt(
            f"{ALFA_OUT}{filename}.sum",
            df.values,
            fmt='%10.3f',
            header=''.join([f'{col:20}' for col in df.columns]),
            comments=''
        )
    
    


