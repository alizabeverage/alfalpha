import emcee
import matplotlib.pyplot as plt
import corner
import numpy as np
from alfa.setup_params import get_properties
from alfa.fitting_info import Info
from alfa.read_data import Data
from alfa.polynorm import polynorm
from alfa.utils import correct_abundance
import pandas as pd
import pickle
from dynesty.utils import resample_equal
from scipy.stats import gaussian_kde
from alfa.utils import get_grids_class


def _two_process_summary(fitting_info, flat_samples, dict_results):
    """Two-process-mode reporting (replaces the post-fit correct_abundance path).

    Reports the derived truths (feh_true, mgfe_true) and the two-process
    predicted [X/H]/[X/Fe] for every element the model constrains -- these are
    exactly the slaved predictions the real-data crosscheck compares against a
    full-element fit. It also asserts the standard library correction did NOT
    run (the correction now lives inside the forward model).
    """
    from alfa.two_process_mode import TwoProcessConfig
    from alfa.two_process import two_process_abundances

    p = np.array(fitting_info.parameters_to_fit)
    zH = flat_samples[:, p == 'zH'].ravel()
    feh = flat_samples[:, p == 'feh'].ravel()
    mgh = flat_samples[:, p == 'mgh'].ravel()
    cfg = TwoProcessConfig.from_dict(getattr(fitting_info, 'two_process_config', None))

    feh_true = zH + feh
    mgh_true = zH + mgh
    mgfe_true = mgh - feh

    def _pct(name, arr):
        dict_results[name + '16'] = [np.nanpercentile(arr, 16)]
        dict_results[name + '50'] = [np.nanmedian(arr)]
        dict_results[name + '84'] = [np.nanpercentile(arr, 84)]

    _pct('feh_true', feh_true)
    _pct('mgh_true', mgh_true)
    _pct('mgfe_true', mgfe_true)

    # two-process predicted [X/H] and [X/Fe] for every constrained element.
    pred = two_process_abundances(feh_true, mgfe_true, ref='H', mgfe_pl=cfg.mgfe_pl)
    fe_h = np.asarray(pred['Fe'], dtype=float)
    for elem, xh in pred.items():
        xh = np.asarray(xh, dtype=float)
        _pct('[%s/H]_tp' % elem, xh)
        _pct('[%s/Fe]_tp' % elem, xh - fe_h)

    # policy-flag fractions over the posterior (logged per fit).
    n_lo, n_hi = cfg.n_valid_mgh
    dict_results['frac_N_clamped'] = [float(np.mean((mgh_true < n_lo) | (mgh_true > n_hi)))]
    c_raw = np.asarray(two_process_abundances(feh_true, mgfe_true, ref='H',
                                              mgfe_pl=cfg.mgfe_pl)['C'], dtype=float)
    c_base = np.asarray(two_process_abundances(zH, cfg.lib_mgfe(zH), ref='H',
                                               mgfe_pl=cfg.mgfe_pl)['C'], dtype=float)
    with np.errstate(invalid='ignore'):
        c_exp = np.abs(c_raw - c_base) / 0.15
    dict_results['frac_C_capped'] = [float(np.nanmean(c_exp > cfg.c_max_exponent))]

    # ASSERT the old post-fit library correction was bypassed: no standard
    # (non-_tp) [X/...] keys may have been written in this mode.
    std_keys = [k for k in dict_results
                if k.startswith('[') and ('_tp' not in k) and k[-2:] in ('16', '50', '84')]
    assert not std_keys, (
        "two-process mode: post-fit correct_abundance path was NOT bypassed; "
        "stray standard keys: %s" % std_keys)
    print("[two-process mode] post-fit correct_abundance bypassed; "
          "N-clamp frac=%.3f, C-cap frac=%.3f"
          % (dict_results['frac_N_clamped'][0], dict_results['frac_C_capped'][0]))


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

    elif (fitting_info.sampler == 'dynesty')|(fitting_info.sampler == 'dynesty_test'):
        with open(f'{fitting_info.ALFA_OUT}{fitting_info.filename}.pkl', 'rb') as f:
            res = pickle.load(f)
        
        # draw posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        flat_samples = resample_equal(res.samples, weights)

    elif fitting_info.sampler == 'nuts':
        from alfa.nuts_sampler import load_nuts_samples
        flat_samples = load_nuts_samples(fitting_info)

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
    
    # import correct model grid
    two_process_mode = bool(getattr(fitting_info, 'two_process_mode', False))
    Grids = get_grids_class(fitting_info.model)
    if two_process_mode:
        from alfa.two_process_mode import TwoProcessConfig
        tp_config = TwoProcessConfig.from_dict(getattr(fitting_info, 'two_process_config', None))
        grids = Grids(inst_res=data.ires, inst_res_wave=data.wave, kroupa_shortcut=False,
                      two_process_mode=True, two_process_config=tp_config)
    else:
        grids = Grids(inst_res=data.ires,inst_res_wave=data.wave,kroupa_shortcut=False)

    if plot_bestspec:
        plt.figure(figsize=(10,3))
        inds = np.random.randint(len(flat_samples), size=10)
        for ind in inds:
            sample = flat_samples[ind]
            
            params = get_properties(sample,fitting_info.parameters_to_fit)
            mflux = grids.get_model(params,outwave=data.wave)
        
            #poly norm
            if fitting_info.poly_degree == 'default': deg = None
            else: deg = fitting_info.poly_degree
            poly, mfluxnorm, data_r = polynorm(data, mflux, deg=deg, return_data=True)
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
    if fitting_info.poly_degree == 'default': deg = None
    else: deg = fitting_info.poly_degree
    poly, mfluxnorm, data_r = polynorm(data, mflux, deg=deg,return_data=True)
        
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

    if (fitting_info.model == 'Conroy18') and (not two_process_mode):
        # define Fe for retrieving [X/Fe]
        Fe = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                                            flat_samples[:,fitting_info.parameters_to_fit=='feh'].ravel(),'feh')
        Mg = correct_abundance(flat_samples[:,fitting_info.parameters_to_fit=='zH'].ravel(),
                                        flat_samples[:,fitting_info.parameters_to_fit=='mgh'].ravel(),'mgh')

    for i,param in enumerate(fitting_info.parameters_to_fit):
        dict_results[param+'16'] = [np.percentile(flat_samples[:,i],16)]
        dict_results[param+'50'] = [np.median(flat_samples[:,i])]
        dict_results[param+'84'] = [np.percentile(flat_samples[:,i],84)]

        # In two-process mode the per-element library correction is applied
        # INSIDE the forward model, so the post-fit correct_abundance path must
        # NOT run here (see the TP reporting block below).
        if (not two_process_mode) and (param in ['feh','ah','ch','nh','nah','mgh','sih',
                                'kh','cah','tih','vh','crh','mnh','coh',
                                'nih','cuh','srh','bah','euh']):

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

            param_st = '['+param[:-1].capitalize()+'/Mg]'
            dict_results[param_st+'16'] = [np.percentile(dist-Mg,16)]
            dict_results[param_st+'50'] = [np.median(dist-Mg)]
            dict_results[param_st+'84'] = [np.percentile(dist-Mg,84)]

    if two_process_mode:
        _two_process_summary(fitting_info, flat_samples, dict_results)

    if (fitting_info.model == 'Conroy18') and (not two_process_mode):
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
