from grids import *
from read_data import Data
from polynorm import polynorm
import numpy as np
import pandas as pd
import csv
import emcee
import corner
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
#from schwimmbad import MPIPool
from setup_params import setup_params, get_properties, setup_initial_position
import os, sys
from utils import correct_abundance
from plot_outputs import plot_outputs
import dynesty
from time import time
from dynesty import DynamicNestedSampler
import pickle
from scipy.stats import gaussian_kde


# must have alfa_home defined in bash profile
ALFA_HOME = os.environ['ALFA_HOME']
ALFA_OUT = os.environ['ALFA_OUT']
#ALFA_OUT = '/Users/alizabeverage/Research/chem_ev/mock_spectra/smooth_burst/results/'

#parameters_to_fit = ['velz', 'sigma', 'logage', 'zH', 'feh',
#                     'ah', 'ch', 'nh', 'mgh', 'sih', 'kh', 'cah',
#                     'tih', 'vh', 'crh', 'mnh', 'coh', 'nih',
#                     'cuh', 'srh', 'bah', 'euh', 'teff', 'jitter']



# parameters_to_fit = ['velz', 'sigma', 'logage', 'zH', 'feh',
#                      'mgh','jitter']
#
parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh',
                     'ch', 'nh', 'mgh', 'sih', 'kh', 'cah',
                     'tih', 'vh', 'crh','teff','jitter','logemline_h', 
                     'logemline_oiii', 'logemline_ni','velz2', 'sigma2'])

default_pos, priors = setup_params(parameters_to_fit)

ncpu = cpu_count()

# ~~~~~~~~~~~~~~~~~~~~~~~ probability stuff ~~~~~~~~~~~~~~~~~~~~~~~ #

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

    return np.array([p*(prio[1]-prio[0]) + prio[0] for (p,prio) in zip(theta,priors.values())])

def loglikelihood_dynesty(theta): # multiprocessing
    # generate model according to theta
    params = get_properties(theta,parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)

    #poly norm
    poly, mfluxnorm = polynorm(data, mflux)

    if 'jitter' in parameters_to_fit:
        # copied from alf
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2*params['jitter']**2) \
                        + np.log(2*np.pi*data.err**2*params['jitter']**2))
    else:
        return -0.5*np.nansum((data.flux - mfluxnorm)**2/(data.err**2))


# ~~~~~~~~~~~~~~~~~~~~~~~ Run fitting tool ~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":  
    # use command arguments to get filename
    if len(sys.argv)>1:
        filename = sys.argv[1] # the script name is 0  
    
    # manually set filename (overwrites above!)
    else:
        filename = 'test'

    # set up data object
    print(f"Loading {filename}...")
    data = Data(filename, filename_exact=True)

    # set up grid object
    print(f"Loading grids...")
    grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)

    
    dsampler = DynamicNestedSampler(loglikelihood_dynesty, prior_transform, len(parameters_to_fit))

    t0 = time()
    dsampler.run_nested(checkpoint_file=f'{ALFA_OUT}{filename}.save')
    t1 = time()
    
    timedynesty = (t1-t0)
    
    print("Time taken to run 'dynesty' (in static mode) is {} seconds".format(timedynesty))

    res = dsampler.results # get results dictionary from sampler
    
    with open(f'{ALFA_OUT}{filename}.pkl', 'wb') as f:
        pickle.dump(res, f)
        
    # #~~~~~~~~~~~~~~~~~~~~~ post-process ~~~~~~~~~~~~~~~~~~~~~~~ #

    def plotposts(samples, **kwargs):
        """
        Function to plot posteriors using corner.py and scipy's gaussian KDE function.
        """
        fig = corner.corner(samples, labels=parameters_to_fit, hist_kwargs={'density': True}, **kwargs)
    
        # plot KDE smoothed version of distributions
        for i,samps in enumerate(samples.T):
            axidx = i*(samples.shape[1]+1)
            kde = gaussian_kde(samps)
            xvals = fig.axes[axidx].get_xlim()
            xvals = np.linspace(xvals[0], xvals[1], 100)
            fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')

        return fig

    
    if post_process:
        # get chi2
        # chi2 = np.zeros(samples_dynesty.shape[0])
        # for i in range(samples_dynesty.shape[0]):
        #     chi2[i] = loglikelihood_dynesty(samples_dynesty[i])

    
        with open(f'{ALFA_OUT}{filename}.pkl', 'rb') as f:
            res = pickle.load(f)
            
        # draw posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        samples_dynesty = resample_equal(res.samples, weights)


        # corner plot
        fig = plotposts(samples_dynesty)
        plt.savefig(f"{ALFA_OUT}{filename}_dyncorner.jpeg")
    
        
        # best-fit spectra
        plt.figure(figsize=(10,3))
        inds = np.random.randint(len(samples_dynesty), size=10)
        for ind in inds:
            sample = samples_dynesty[ind]
            
            params = get_properties(sample,parameters_to_fit)
            mflux = grids.get_model(params,outwave=data.wave)
        
            #poly norm
            poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
            plt.plot(data.wave,data_r, 'C0', lw=1)
            plt.plot(data.wave,mfluxnorm, 'C1')
            
        
        plt.xlabel('Wavelength ($\mathring{\mathrm{A}}$)')
        plt.ylabel('F$_\lambda$')
        plt.tight_layout()
    
        plt.savefig(f"{ALFA_OUT}{filename}_dynbestspec.jpeg",dpi=200)
        
    
        # save outputs in summary file
        parameters_to_fit = np.array(parameters_to_fit)
        dict_results = {}
        # define Fe for retrieving [X/Fe]
        Fe = correct_abundance(samples_dynesty[:,parameters_to_fit=='zH'].ravel(),
                                         samples_dynesty[:,parameters_to_fit=='feh'].ravel(),'feh')
        for i,param in enumerate(parameters_to_fit):
            dict_results[param+'16'] = [np.percentile(samples_dynesty[:,i],16)]
            dict_results[param+'50'] = [np.median(samples_dynesty[:,i])]
            dict_results[param+'84'] = [np.percentile(samples_dynesty[:,i],84)]
        
            if param in ['feh','ah','ch','nh','nah','mgh','sih',
                                  'kh','cah','tih','vh','crh','mnh','coh',
                                  'nih','cuh','srh','bah','euh']:
                dist = correct_abundance(samples_dynesty[:,parameters_to_fit=='zH'].ravel(),
                                         samples_dynesty[:,i].ravel(),param)
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
    
    


