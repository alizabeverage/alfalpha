import emcee 
import matplotlib.pyplot as plt
import corner
import numpy as np
from alfa.read_data import Data
from alfa.grids import Grids
from alfa.polynorm import polynorm
from setup_params import get_properties
import os

ALFA_OUT = os.environ['ALFA_OUT']

filename = 'HeavyMetal3.JHK.104779.1d'
data = Data(filename)
inst_res = np.average(data.ires) 

# set up grid object
print(f"Loading grids...")
grids = Grids(inst_res=inst_res)

reader = emcee.backends.HDFBackend(f'{ALFA_OUT}/{filename}.h5')

parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh', 'ah', 'ch', 'nh', 'nah',
       'mgh', 'sih', 'kh', 'cah', 'tih', 'vh', 'crh'])
# ~~~~~~~~~~~~~~~~~~~~~~~ process outputs and plot! ~~~~~~~~~~~~~~~~~~~~~~~ #

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

# burn in steps
samples = reader.get_chain(flat=False, thin=15)
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

flat_samples = reader.get_chain(discard=8000,flat=True, thin=15)
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