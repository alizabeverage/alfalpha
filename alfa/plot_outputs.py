import emcee 
import matplotlib.pyplot as plt
import corner
import numpy as np
from alfa.read_data import Data
from alfa.grids import Grids
from alfa.polynorm import polynorm
from alfa.setup_params import get_properties
import os
from scipy.stats import gaussian_kde

ALFA_OUT = os.environ['ALFA_OUT']

# filename = 'HeavyMetal3.JHK.104779.1d'
# data = Data(filename)
# inst_res = np.average(data.ires) 

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams["xtick.major.size"] = 4.5
plt.rcParams["ytick.major.size"] = 4.5
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.width"] = 0.8
plt.rcParams["ytick.minor.width"] = 0.8
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.top"] = True
plt.rc('font', size=13)

def plot_outputs(data, grids, parameters_to_fit, filename, reader=None, inst_res=0, thin=1, discard=0):

    if reader is None:
        reader = emcee.backends.HDFBackend(f'{ALFA_OUT}/{filename}.h5')
    
    
    # ~~~~~~~~~~~~~~~~~~~~~~~ process outputs and plot! ~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # burn in steps
    samples = reader.get_chain(flat=False, thin=thin,discard=discard)
    ndim = samples.shape[2]
    fig, axes = plt.subplots(len(parameters_to_fit), figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(parameters_to_fit[i])
        # ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(f'{ALFA_OUT}/{filename}_burnin.png',dpi=100)
    
    flat_samples = reader.get_chain(discard=discard,flat=True, thin=thin)
    # corner plot
    fig = corner.corner(
        flat_samples, labels=parameters_to_fit
    )
    plt.savefig(f'{ALFA_OUT}/{filename}_corner.png',dpi=100)
    
    
    plt.figure()
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        
        params = get_properties(sample,parameters_to_fit)
        mflux = grids.get_model(params,outwave=data.wave)
    
        #poly norm
        poly, mfluxnorm, data_r = polynorm(data, mflux,return_data=True)
        plt.plot(data.wave,data_r, 'C0', lw=1)
        plt.plot(data.wave,mfluxnorm, 'C1')
        
    
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    
    plt.savefig(f'{ALFA_OUT}/{filename}_spectrum.png',dpi=200)


def plotposts(samples, parameters_to_fit, **kwargs):
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