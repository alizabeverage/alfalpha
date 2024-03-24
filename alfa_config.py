from alfa.grids import *
from alfa.read_data import Data
from alfa.polynorm import polynorm
import numpy as np
import emcee
from multiprocessing import Pool, cpu_count
#from schwimmbad import MPIPool
from alfa.setup_params import setup_params,get_properties, setup_initial_position, setup_initial_position_diff_ev
import os, sys
from scipy.optimize import differential_evolution
from alfa.fitting_info import Info
from alfa.post_process import post_process
from dynesty import DynamicNestedSampler
import pickle
import time


'''
~~~~~~~~~~~~~~~~~~~~~ Parameters to define ~~~~~~~~~~~~~~~~~~~~~~ 

        Define the parameters to fit and which sampler

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# must have alfa_out defined in bash profile
ALFA_OUT = os.environ['ALFA_OUT']


# instantiate the Info class
fitting_info = Info()

# which sampler do you want to use?
fitting_info.sampler = 'dynesty' # 'dynesty' or 'emcee'

if fitting_info.sampler == 'emcee':
    # emcee parameters
    fitting_info.nwalkers = 40 #256
    fitting_info.nsteps = 50 #8000
    fitting_info.nsteps_save = 10#100


# which parameters (if any) do you want to "pre-fit"?
# if diff_ev_parameters is empty, then the code will skip this step
fitting_info.diff_ev_parameters = ['velz','sigma','logage','zH']

# which parameters do you want to fit?
# you are required to have at least 'velz', 'sigma', 'zH', and 'feh' in the list
# if you want to fit emission lines, you include which line (e.g., 'logemline_h')
# *and* 'velz2' and 'sigma2'
fitting_info.parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh',
                                        'ch', 'nh', 'mgh', 'nah', 'ah', 'sih', 'cah',
                                        'tih', 'crh', 'teff','jitter',
                                        'logemline_h', 'logemline_ni', 
                                        'logemline_nii', 'logemline_oii',
                                    'logemline_oiii', 'logemline_sii',
                                        'velz2', 'sigma2'])


# you can alter the prior ranges here, but they're already set in Info() by "setup_params"
# fitting_info.priors['jitter'] = [1.6,1.66]

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

# dyensty functions
def prior_transform(theta):
    """
    A function defining the tranform between the parameterisation in the unit hypercube
    to the true parameters.

    Args:
        theta (tuple): a tuple containing the parameters.
        
    Returns:
        tuple: a new tuple or array with the transformed parameters.

    mmu = 0.     # mean of Gaussian prior on m
    msigma = 10. # standard deviation of Gaussian prior on m
    m = mmu + msigma*ndtri(mprime) # convert back to m
    """

    return np.array([p*(prio[1]-prio[0]) + prio[0] for (p,prio) in zip(theta,fitting_info.priors.values())])

def loglikelihood_dynesty(theta): # multiprocessing
    # generate model according to theta
    params = get_properties(theta,fitting_info.parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)

    #poly norm
    poly, mfluxnorm = polynorm(data, mflux,deg=None)

    if 'jitter' in fitting_info.parameters_to_fit:
        # copied from alf
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2*params['jitter']**2) \
                        + np.log(2*np.pi*data.err**2*params['jitter']**2))
    else:
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2))



if __name__ == "__main__":  
    filename = sys.argv[1] # the script name is 0  
    
    # set up data object
    print(f"Loading {filename}...")
    data = Data(filename)

    filename = filename.split('/')[-1]
    fitting_info.filename = filename

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

            # print(fitting_info.nwalkers,fitting_info.parameters_to_fit,diff_ev_result)
            if result.success:
                fitting_info.pos = setup_initial_position_diff_ev(fitting_info.nwalkers,fitting_info.parameters_to_fit,diff_ev_result=diff_ev_result,
                                                priors=fitting_info.priors)   
        
        else:
                fitting_info.pos = setup_initial_position(fitting_info.nwalkers,fitting_info.parameters_to_fit,priors=fitting_info.priors)

        nwalkers, ndim = fitting_info.pos.shape

        backend = emcee.backends.HDFBackend(f"{fitting_info.ALFA_OUT}{fitting_info.filename}.h5")
        backend.reset(nwalkers, ndim)
        with Pool() as pool: 
            #sample
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, backend=backend 
                )
            sampler.run_mcmc(fitting_info.pos, fitting_info.nsteps, progress=True);
    

    elif fitting_info.sampler == 'dynesty':
        dsampler = DynamicNestedSampler(loglikelihood_dynesty, prior_transform, len(fitting_info.parameters_to_fit))

        t0 = time.time()
        dsampler.run_nested()
        t1 = time.time()
        
        timedynesty = (t1-t0)
        
        print("Time taken to run 'dynesty' (in static mode) is {} seconds".format(timedynesty))

        res = dsampler.results # get results dictionary from sampler

        with open(f'{fitting_info.ALFA_OUT}{fitting_info.filename}.pkl', 'wb') as f:
            pickle.dump(res, f)
    

    #~~~~~~~~~~~~~~~~~~~~~ post processing ~~~~~~~~~~~~~~~~~~~~~~~ #
    
    post_process(fitting_info, plot_corner=True, plot_bestspec=True)




