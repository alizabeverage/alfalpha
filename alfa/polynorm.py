from utils import remap_array, fit_poly
import numpy as np
import math

def polynorm(data_obj, model, return_data=False):
    # normalize model to data
    poly = np.nan*np.zeros(len(model))
    model_norm = np.nan*np.zeros(len(model))
    data_r = np.nan*np.zeros(len(model))
    
    # fit a polynomial to each wavelength regime
    for region in data_obj.fitting_regions:
        s = (data_obj.wave >= region[0])&(data_obj.wave <= region[1])
        
        # one degree for every 100 AA
        polydegree = math.ceil(np.diff(region)/100)
        
        if polydegree>9: polydegree = 9
            
        poly[s] = fit_poly(data_obj.wave[s],data_obj.flux[s]/model[s],deg = polydegree)
        model_norm[s] = model[s]*poly[s]
        data_r[s] = data_obj.flux[s]

    if return_data:
        return poly, model_norm, data_r
    else:
        return poly, model_norm