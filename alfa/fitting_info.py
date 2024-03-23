import os
import numpy as np
from alfa.setup_params import setup_params
import json


class Info:
    def __init__(self):
        self.ALFA_OUT = os.environ['ALFA_OUT']

        # which sampler do you want to use?
        self.sampler = 'emcee' # 'dynesty' or 'emcee'
        if self.sampler == 'emcee':
            # emcee parameters
            self.nwalkers = 256
            self.nsteps = 8000
            self.nsteps_save = 100

        # which parameters (if any) do you want to "pre-fit"?
        # if diff_ev_parameters is empty, then the code will skip this step
        self.diff_ev_parameters = ['velz','sigma','logage','zH']

        # which parameters do you want to fit?
        # you are required to have at least 'velz', 'sigma', 'zH', and 'feh' in the list
        # if you want to fit emission lines, you include which line (e.g., 'logemline_h')
        # *and* 'velz2' and 'sigma2'
        self.parameters_to_fit = np.array(['velz', 'sigma', 'logage', 'zH', 'feh',
                            'ch', 'nh', 'mgh', 'nah', 'ah', 'sih', 'cah',
                            'tih', 'crh', 'teff','jitter','logemline_h', 
                            'velz2', 'sigma2'])

        # Grab the default positions and the priors of the parameters (set in setiup_params.py)
        _, self.priors = setup_params(self.parameters_to_fit)

        # set the polynomial degree for normalization
        self.poly_degree = 'default' # 'default' or int

        # you'll end up putting the data here
        self.filename = None
        self.data_wave = None
        self.data_flux = None
        self.data_err = None
        self.data_mask = None
        self.data_ires = None
        self.data_fitting_regions = None

        # you'll put results of diff_ev here
        self.diff_ev_results = {}
        self.diff_ev_success = True

    def save_settings(self, fname = None):
        # json doesn't like numpy arrays
        for key in self.__dict__:
            if isinstance(self.__dict__[key], np.ndarray):
                self.__dict__[key] = self.__dict__[key].tolist()

        if fname is None:
            fname = f"{self.ALFA_OUT}{self.filename}"

        with open(f"{fname}.json", "w") as f:
            json.dump(self.__dict__, f, indent=4)
    
    def load_settings(self, fname = None):
        if fname is None:
            fname = f"{self.ALFA_OUT}{self.filename}"
        with open(f"{fname}.json", "r") as f:
            settings = json.load(f)
        self.__dict__.update(settings)

        # convert lists back to numpy arrays
        for key in self.__dict__:
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = np.array(self.__dict__[key])