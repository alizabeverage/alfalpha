from .utils import wave_to_x
import numpy as np
import math
from numpy.polynomial.chebyshev import chebval, chebvander

def fit_poly(wave, flux, err, mask, model, order=None):
    # This function is adapted from prospector. The 
    # coefficients of the maximum likelihood Cheb polynomial
    # is found using least-squares method
    
    # map unmasked wavelengths to the interval -1, 1
    # masked wavelengths may have x>1, x<-1
    mask = mask.astype(bool)
    x = wave_to_x(wave, mask)
    y = (flux / model)[mask] - 1.0
    yerr = (err / model)[mask]
    yvar = yerr**2
    
    A = chebvander(x[mask], order)
    ATA = np.dot(A.T, A / yvar[:, None])
    ATAinv = np.linalg.inv(ATA)
    c = np.dot(ATAinv, np.dot(A.T, y / yvar))
    Afull = chebvander(x, order)
    
    poly = np.dot(Afull, c)
    poly_coeffs = c

    return (1.0 + poly)

def polynorm(data_obj, model, return_data=False):
    # normalize model to data
    poly = np.nan*np.zeros(len(model))
    model_norm = np.nan*np.zeros(len(model))
    data_r = np.nan*np.zeros(len(model))
    
    # fit a polynomial to each wavelength regime using least-squares method
    for region in data_obj.fitting_regions:
        s = (data_obj.wave >= region[0])&(data_obj.wave <= region[1])
        
        # one degree for every 100 AA
        polydegree = math.ceil(np.diff(region)/100)
        if polydegree>=14: polydegree = 14
        polydegree = 8
            
        poly[s] = fit_poly(data_obj.wave[s], data_obj.flux[s], data_obj.err[s], 
                            data_obj.mask[s], model[s], order=polydegree)
        model_norm[s] = model[s]*poly[s]
        data_r[s] = data_obj.flux[s]

    if return_data:
        return poly, model_norm, data_r
    else:
        return poly, model_norm
















