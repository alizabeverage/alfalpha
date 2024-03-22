import emcee
import matplotlib.pyplot as plt
import corner
import numpy as np
from alfa.setup_params import get_properties
from alfa.fitting_info import Info
from alfa.grids import Grids
from alfa.polynorm import polynorm
from alfa.utils import correct_abundance
import pandas as pd


def post_process(fitting_info = None, fname = None):
    if (fitting_info is None) and (fname is None):
        raise ValueError("You must provide either a fitting_info object or a filename")
    elif (fitting_info is not None) and (fname is not None):
        raise ValueError("You must provide either a fitting_info object or a filename, not both")

    if fname is not None:
        # instantiate the Info class
        fitting_info = Info()

        # load the settings
        fitting_info.load_settings(fname = fname)

    elif fitting_info is not None:
        pass

    # open walker files
    if fitting_info.sampler == 'emcee':
        reader = emcee.backends.HDFBackend(f"{fitting_info.ALFA_OUT}{fitting_info.filename}.h5")
        flat_samples = reader.get_chain(discard=fitting_info.nsteps-fitting_info.nsteps_save, thin=1, flat=True)

    elif fitting_info.sampler == 'dynesty':
        raise ValueError("Dynesty not yet implemented in post_process")

    # corner plot
    sel = np.ones(len(fitting_info.parameters_to_fit)).astype(bool)
    fig = corner.corner(
        flat_samples[:,sel], labels=np.array(fitting_info.parameters_to_fit)[sel],
        show_titles=True
    )
    plt.savefig(f"{fitting_info.ALFA_OUT}{fitting_info.filename}_corner.jpeg")

    
    # best-fit spectra
    grids = Grids(inst_res=fitting_info.data_ires,inst_res_wave=fitting_info.data_wave,kroupa_shortcut=False)

    plt.figure(figsize=(10,3))
    inds = np.random.randint(len(flat_samples), size=10)
    for ind in inds:
        sample = flat_samples[ind]
        
        params = get_properties(sample,fitting_info.parameters_to_fit)
        mflux = grids.get_model(params,outwave=fitting_info.data_wave)
    
        #poly norm
        poly, mfluxnorm, data_r = polynorm(fitting_info.data, mflux,return_data=True)
        plt.plot(fitting_info.data.wave,data_r, 'C0', lw=1)
        plt.plot(fitting_info.data.wave,mfluxnorm, 'C1')
        
    
    plt.xlabel('Wavelength ($\mathring{\mathrm{A}}$)')
    plt.ylabel('F$_\lambda$')
    plt.tight_layout()

    plt.savefig(f"{fitting_info.ALFA_OUT}{fitting_info.filename}_bestspec.jpeg",dpi=200)
    
    # write the bestspectrum to file
    if 'jitter' not in fitting_info.parameters_to_fit:
        params['jitter'] = 1.0

    data = {}
    data['wave'] = fitting_info.data_wave
    data['m_flux'] = fitting_info.mfluxnorm # Model spectrum, normalization applied
    data['d_flux'] = fitting_info.data_r # Data spectrum with mask & fitting regions applied
    data['unc'] =fitting_info.data_err*params['jitter'] # data err with jitter applied
    data['poly'] = poly # Polynomial used to create m_flux

    pd = pd.DataFrame(data)
    np.savetxt(f"{fitting_info.ALFA_OUT}{fitting_info.filename}.bestspec",
               pd.values,
               fmt='%11.3f %11.3e %11.3e %11.3f %11.3f',
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
    
    
    df = pd.DataFrame.from_dict(dict_results)
    np.savetxt(
        f"{fitting_info.ALFA_OUT}{fitting_info.filename}.sum",
        df.values,
        fmt='%10.3f',
        header=''.join([f'{col:20}' for col in df.columns]),
        comments=''
    )
