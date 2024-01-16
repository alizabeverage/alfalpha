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

multip = False

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

if multip:
    def lnlike(theta, data, grids): # multiprocessing
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
        
    def lnprior(theta):
        check = np.array([prio[0] < p < prio[1] for (p,prio) in zip(theta,priors.values())])
        if False not in check:
            return 1.0
        return -np.inf
    
    def lnprob(theta, data, grids): # multiprocessing
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp*lnlike(theta, data, grids) # multiprocessing

else:
    def lnlike(theta): # multiprocessing
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




# ~~~~~~~~~~~~~~~~~~~~~~~ Run fitting tool ~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":  
    nwalkers = 256
    nsteps = 8000
    nsteps_save = 500
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
    data = Data(filename, filename_exact=True)

    # set up grid object
    print(f"Loading grids...")
    grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)


    # fit emission lines if HM 56163 or 23351
    # if '56163' in filename or '23351' in filename:
    #     parameters_to_fit+=list(np.unique(emline_strs[(wave_emlines<5600)&(wave_emlines>3800)]))
    #     parameters_to_fit+=['velz2','sigma2']

    # get the positions and priors of parameters_to_fit
    # default_pos, priors = setup_params(parameters_to_fit)
    

    #~~~~~~~~~~~~~~~~~~~~~ emcee ~~~~~~~~~~~~~~~~~~~~~~~ #
    print("fitting with emcee...")

    # initialize walkers
    pos = setup_initial_position(nwalkers,parameters_to_fit)
    nwalkers, ndim = pos.shape

    # open file for saving steps
    filename = filename.split('/')[-1]
    backend = emcee.backends.HDFBackend(f"{ALFA_OUT}{filename}.h5")
    backend.reset(nwalkers, ndim)

    if multip:
        with Pool() as pool:
            #sample
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, backend=backend, pool=pool, args=(data,grids) 
                  )
            sampler.run_mcmc(pos, nsteps, progress=True);

    else:
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
    
    


