import numpy as np
from numpy.polynomial.chebyshev import Chebyshev as C
from scipy.interpolate import interp1d

def get_grids_class(model_name):
    if model_name == 'sMILES':
        from alfa.grids_sMILES import Grids
    elif model_name == 'Conroy18':
        from alfa.grids import Grids
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Grids

def remap_array(data, mapping=[0,1], vmax=None, vmin=None):
    data = np.array(data)
    if vmax is None: vmax = max(data)
    if vmin is None: vmin = min(data)
    m = np.diff(mapping)/(vmax-vmin)
    b = np.max(mapping) - vmax*m
    
    return m*data+b


# def fit_poly(wave, flux, deg=7):
#     wave_remap = remap_array(wave,mapping=[-1,1])
#     p, coeff = C.fit(wave_remap,flux,deg,full=True)
#     poly = p(wave_remap)
#     return poly

def wave_to_x(wavelength=None, mask=slice(None), **extras):
        """Map unmasked wavelengths to the interval (-1, 1). Masked wavelengths may have x>1, x<-1

        :param wavelength:
            The input wavelengths.  ndarray of shape ``(nwave,)``

        :param mask: optional
            The mask.  slice or boolean array with ``True`` for unmasked elements.
            The interval (-1, 1) will be defined only by unmasked wavelength points

        :returns x:
            The wavelength vector, remapped to the interval (-1, 1).
            ndarray of same shape as  ``wavelength``
        """
        x = wavelength - (wavelength[mask]).min()
        x = 2.0 * (x / (x[mask]).max()) - 1.0
        return x


def correct_abundance(zh,xh,element):
    '''
    zh: value(s) of zH
    xh: value(s) of raw elemental abundnace
    element: string indicating what element xh is 
            (lowercase with h after, e.g. Mg should be "mgh")
    
    '''
    elements = np.array(['feh','ah', 'ch', 'nh', 'nah', 'mgh','sih',
                    'kh', 'cah', 'tih','vh', 'crh','mnh', 
                    'coh', 'nih', 'cuh', 'srh','bah','euh'])
    if element not in elements:
        raise "check your element"
    
    if element not in ['cah', 'tih', 'sih','mgh','ah']:
        return xh+zh
    
    lib_feh = [-1.6, -1.4, -1.2, -1.0, -0.8,
                   -0.6, -0.4, -0.2, 0.0, 0.2]
    lib_ofe = [0.6, 0.5, 0.5, 0.4, 0.3, 0.2,
                   0.2, 0.1, 0.0, 0.0]
    lib_mgfe = [0.4, 0.4, 0.4, 0.4, 0.34, 0.22,
                   0.14, 0.11, 0.05, 0.04]
    lib_cafe = [0.32, 0.3, 0.28, 0.26, 0.26,
                   0.17, 0.12, 0.06, 0.0, 0.0]
    
    del_alfe = interp1d(lib_feh, lib_ofe,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value='extrapolate')
    del_mgfe = interp1d(lib_feh, lib_mgfe,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value='extrapolate')
    del_cafe = interp1d(lib_feh, lib_cafe,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value='extrapolate')
    
    
    al_corr = del_alfe(zh)
    mg_corr = del_mgfe(zh)
    ca_corr = del_cafe(zh)
    
    if element=='mgh':
        return xh + zh + mg_corr

    elif element=='ah':
        return xh + zh + al_corr
    
    elif element in ['cah', 'tih', 'sih']:
        return xh + zh + ca_corr
    
    print("something's not right")


