import emcee
import matplotlib.pyplot as plt
import corner
import numpy as np
from alfa.setup_params import get_properties
from alfa.fitting_info import Info
from alfa.grids import Grids
from alfa.read_data import Data
from alfa.polynorm import polynorm
from alfa.utils import correct_abundance
import pandas as pd
import pickle
from dynesty.utils import resample_equal
from scipy.stats import gaussian_kde


def post_process(fitting_info = None, fname = None, plot_corner=True, plot_bestspec=True):
    if (fitting_info is None) and (fname is None):
        raise ValueError("You must provide either a fitting_info object or a filename")
    elif (fitting_info is not None) and (fname is not None):
        raise ValueError("You must provide either a fitting_info object or a filename, not both")

    if fname is not None:
        # instantiate the Info class
        fitting_info = Info(filename = fname)

    # open walker files
    if fitting_info.sampler == 'emcee':
        reader = emcee.backends.HDFBackend(f"{fitting_info.ALFA_OUT}{fitting_info.filename}.h5")
        flat_samples = reader.get_chain(discard=fitting_info.nsteps-fitting_info.nsteps_save, thin=1, flat=True)

    elif fitting_info.sampler == 'dynesty':
        with open(f'{fitting_info.ALFA_OUT}{fitting_info.filename}.pkl', 'rb') as f:
            res = pickle.load(f)
        
        # draw posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        flat_samples = resample_equal(res.samples, weights)

    # corner plot
    if plot_corner:
        # plot posteriors using corner.py and scipy's gaussian KDE function.
        fig = corner.corner(flat_samples, labels=fitting_info.parameters_to_fit, 
                            show_titles=True, hist_kwargs={'density': True})
    
        # plot KDE smoothed version of distributions
        for i,samps in enumerate(flat_samples.T):
            axidx = i*(flat_samples.shape[1]+1)
            kde = gaussian_kde(samps)
            xvals = fig.axes[axidx].get_xlim()
            xvals = np.linspace(xvals[0], xvals[1], 100)
            fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')
        
        plt.savefig(f"{fitting_info.ALFA_OUT}{fitting_info.filename}_corner.jpeg")
        # sel = np.ones(len(fitting_info.parameters_to_fit)).astype(bool)
        # fig = corner.corner(
        #     flat_samples[:,sel], labels=np.array(fitting_info.parameters_to_fit)[sel],
        #     show_titles=True
        # )
        
    
    # best-fit spectra
    data = Data()
    data.wave = np.array(fitting_info.data_wave)
    data.flux = np.array(fitting_info.data_flux)
    data.err = np.array(fitting_info.data_err)
    data.mask = np.array(fitting_info.data_mask)
    data.ires = np.array(fitting_info.data_ires)
    data.fitting_regions = fitting_info.data_fitting_regions

    
    grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)

    if plot_bestspec:
        plt.figure(figsize=(10,3))
        inds = np.random.randint(len(flat_samples), size=10)
        for ind in inds:
            sample = flat_samples[ind]
            
            params = get_properties(sample,fitting_info.parameters_to_fit)
            mflux = grids.get_model(params,outwave=data.wave)
        
            #poly norm
            poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
            plt.plot(data.wave,data_r, 'C0', lw=1)
            plt.plot(data.wave,mfluxnorm, 'C1')
            
        
        plt.xlabel('Wavelength ($\mathring{\mathrm{A}}$)')
        plt.ylabel('F$_\lambda$')
        plt.tight_layout()

        plt.savefig(f"{fitting_info.ALFA_OUT}{fitting_info.filename}_bestspec.jpeg",dpi=200)
    

    # save best spectrum (mean of each parameter) to file
    params = get_properties(np.mean(flat_samples,axis=0),fitting_info.parameters_to_fit)
    mflux = grids.get_model(params,outwave=data.wave)

    #poly norm
    poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
        
    # write the bestspectrum to file
    if 'jitter' not in fitting_info.parameters_to_fit:
        params['jitter'] = 1.0

    bestspec = {}
    bestspec['wave'] = data.wave
    bestspec['m_flux'] = mfluxnorm # Model spectrum, normalization applied
    bestspec['d_flux'] = data_r # Data spectrum with mask & fitting regions applied
    bestspec['unc'] =data.err*params['jitter'] # data err with jitter applied
    bestspec['poly'] = poly # Polynomial used to create m_flux

    df = pd.DataFrame(bestspec)
    np.savetxt(f"{fitting_info.ALFA_OUT}{fitting_info.filename}.bestspec",
               df.values,
               fmt='%11.3f %11.3e %11.3e %11.3e %11.3e',
               header=''.join([f'{col:20}' for col in df.columns]),
               comments='')


    # save outputs in summary file
    fitting_info.parameters_to_fit = np.array(fitting_info.parameters_to_fit)
    dict_results = {}
    # define Fe for retrieving [X/Fe]
    Fe = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                                        flat_samples[:,fitting_info.parameters_to_fit=='feh'].ravel(),'feh')
    for i,param in enumerate(fitting_info.parameters_to_fit):
        dict_results[param+'16'] = [np.percentile(flat_samples[:,i],16)]
        dict_results[param+'50'] = [np.median(flat_samples[:,i])]
        dict_results[param+'84'] = [np.percentile(flat_samples[:,i],84)]
    
        if param in ['feh','ah','ch','nh','nah','mgh','sih',
                                'kh','cah','tih','vh','crh','mnh','coh',
                                'nih','cuh','srh','bah','euh']:
            
            # here we correct the abundances for the elemental enhancement of the stellar library
            dist = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                                        flat_samples[:,i].ravel(),param)
            param_st = '['+param[:-1].capitalize()+'/H]'
            dict_results[param_st+'16'] = [np.percentile(dist,16)]
            dict_results[param_st+'50'] = [np.median(dist)]
            dict_results[param_st+'84'] = [np.percentile(dist,84)]

            param_st = '['+param[:-1].capitalize()+'/Fe]'
            dict_results[param_st+'16'] = [np.percentile(dist-Fe,16)]
            dict_results[param_st+'50'] = [np.median(dist-Fe)]
            dict_results[param_st+'84'] = [np.percentile(dist-Fe,84)]
    
    # add the "metallicity" [Z/H] as defined in Thomas et al. 2003
    mgh = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                            flat_samples[:,fitting_info.parameters_to_fit=='mgh'].ravel(),'mgh')
    feh = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                            flat_samples[:,fitting_info.parameters_to_fit=='feh'].ravel(),'feh')
    mgfe = mgh-feh

    zh = np.array(feh + 0.94*mgfe)
    dict_results['[Z/H]_thomas16'] = [np.percentile(zh,16)]
    dict_results['[Z/H]_thomas50'] = [np.median(zh)]
    dict_results['[Z/H]_thomas84'] = [np.percentile(zh,84)]
    
    df = pd.DataFrame.from_dict(dict_results)
    np.savetxt(
        f"{fitting_info.ALFA_OUT}{fitting_info.filename}.sum",
        df.values,
        fmt='%10.3f',
        header=''.join([f'{col:20}' for col in df.columns]),
        comments=''
    )
