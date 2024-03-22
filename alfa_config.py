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
from alfa.setup_params import setup_params,get_properties, setup_initial_position, setup_initial_position_diff_ev
import os, sys
from alfa.utils import correct_abundance
from alfa.plot_outputs import plot_outputs
from scipy.optimize import differential_evolution
from alfa.fitting_info import Info
from alfa.post_process import post_process


'''
~~~~~~~~~~~~~~~~~~~~~ Parameters to define ~~~~~~~~~~~~~~~~~~~~~~ 

        Define the parameters to fit and which sampler

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# must have alfa_home defined in bash profile
ALFA_HOME = os.environ['ALFA_HOME']
ALFA_OUT = os.environ['ALFA_OUT']


# instantiate the Info class
fitting_info = Info()

# which sampler do you want to use?
fitting_info.sampler = 'emcee' # 'dynesty' or 'emcee'
if fitting_info.sampler == 'emcee':
    # emcee parameters
    fitting_info.nwalkers = 256
    fitting_info.nsteps = 8000
    fitting_info.nsteps_save = 100


# which parameters (if any) do you want to "pre-fit"?
# if diff_ev_parameters is empty, then the code will skip this step
fitting_info.diff_ev_parameters = ['velz','sigma','logage','zH']

# which parameters do you want to fit?
# you are required to have at least 'velz', 'sigma', 'zH', and 'feh' in the list
# if you want to fit emission lines, you include which line (e.g., 'logemline_h')
# *and* 'velz2' and 'sigma2'
fitting_info.parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh',
                    'ch', 'nh', 'mgh', 'nah', 'ah', 'sih', 'cah',
                    'tih', 'crh', 'teff','jitter','logemline_h', 
                    'velz2', 'sigma2'])

# Grab the default positions and the priors of the parameters (set in setiup_params.py)
_, fitting_info.priors = setup_params(fitting_info.parameters_to_fit)

# you can alter the prior ranges here
fitting_info.priors['jitter'] = [1.6,1.66]

# set the polynomial degree for normalization
fitting_info.poly_degree = 'default' # 'default' or int


'''
~~~~~~~~~~~~~~~~~~~~~~~ probability stuff ~~~~~~~~~~~~~~~~~~~~~~~ 

       Define the likelihood, prior, and posterior functions

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def lnlike(theta): # multiprocessing
    # generate model according to theta
    params = get_properties(theta,fitting_info.parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)
    
    # perform polynomial normalization
    if fitting_info.poly_degree == 'default': deg = None
    else: deg = fitting_info.poly_degree
    poly, mfluxnorm = polynorm(data, mflux, deg=deg)
    
    if 'jitter' in fitting_info.parameters_to_fit:
        # copied from alf
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2*params['jitter']**2) \
                            + np.log(2*np.pi*data.err**2*params['jitter']**2))
    else:
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2))
        
def lnprior(theta):
    check = np.array([prio[0] < p < prio[1] for (p,prio) in zip(theta,fitting_info.priors.values())])
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
    params = get_properties(theta,fitting_info.diff_ev_parameters)
    mflux = grids.get_model(params,outwave=data.wave)

    # Perform polynomial normalization
    poly, mfluxnorm = polynorm(data, mflux)

    return np.nansum((data.flux - mfluxnorm)**2/(data.err**2))



if __name__ == "__main__":  
    filename = sys.argv[1] # the script name is 0  
    fitting_info.filename = filename
    
    # set up data object
    print(f"Loading {filename}...")
    data = Data(filename)

    # add the data info to fitting_info
    fitting_info.data_wave = data.wave
    fitting_info.data_flux = data.flux
    fitting_info.data_err = data.err
    fitting_info.data_mask = data.mask
    fitting_info.data_ires = data.ires
    fitting_info.data_fitting_regions = data.fitting_regions

    # save fitting_info to a file
    fitting_info.save_settings()

    # set up grid object
    print(f"Loading grids...")
    grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)

    #~~~~~~~~~~~~~~~~~~~~~ run differential evolution and setup initial positions ~~~~~~~~~~~~~~~~~~ #
    
    if fitting_info.sampler == 'emcee':
        if (len(fitting_info.diff_ev_parameters)>0):
            print(f"Running differential evolution... diff_ev_parameters: {fitting_info.diff_ev_parameters}")
            _,prior = setup_params(fitting_info.diff_ev_parameters)
            bounds = list(prior.values())  

            # Run differential evolution optimization
            result = differential_evolution(diff_ev_objective_function, bounds,disp=False,updating='deferred')
            
            diff_ev_result = get_properties(result.x,fitting_info.diff_ev_parameters)
            print(f"Differential evolution result: {diff_ev_result}")
            print(f"Differential evolution success: {result.success}")

            fitting_info.diff_ev_results = diff_ev_result
            fitting_info.diff_ev_success = result.success

            if result.success:
                fitting_info.pos = setup_initial_position_diff_ev(fitting_info.nwalkers,fitting_info.parameters_to_fit,diff_ev_result=diff_ev_result,
                                                priors=fitting_info.priors)   
        
        else:
                fitting_info.pos = setup_initial_position(fitting_info.nwalkers,fitting_info.parameters_to_fit,priors=fitting_info.priors)

        nwalkers, ndim = fitting_info.pos.shape

        filename = filename.split('/')[-1]
        backend = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
        backend.reset(nwalkers, ndim)
        with Pool() as pool: 
            #sample
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, backend=backend 
                )
            sampler.run_mcmc(fitting_info.pos, fitting_info.nsteps, progress=True);
    

    elif fitting_info.sampler == 'dynesty':
        raise NotImplementedError("Dynesty not implemented yet")
    

    #~~~~~~~~~~~~~~~~~~~~~ post processing ~~~~~~~~~~~~~~~~~~~~~~~ #
    
    post_process(fitting_info)




