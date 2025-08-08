import numpy as np
import pandas as pd
import os
from .smoothing import smoothspec
from scipy.interpolate import RegularGridInterpolator
import os
from astropy.io import fits

ckms = 2.998e5
ALFA_INFILES = os.environ['ALFA_INFILES'] # location of model files

# define alpha elements used if fitting in lockstep
# currently defined by vazdekis 2015 models
alpha_elements = ['ah','mgh','sih','cah','tih']

wave_emlines = np.array([4102.89, 4341.69, 4862.71, 4960.30, 5008.24, 
                         5203.05, 6549.86, 6564.61, 6585.27, 6718.29, 
                         6732.67, 3727.10, 3729.86, 3751.22, 3771.70, 
                         3798.99, 3836.49, 3890.17, 3971.20])
emline_strs = np.array(['logemline_h','logemline_h','logemline_h', 
                          'logemline_oiii','logemline_oiii','logemline_ni',
                              'logemline_nii', 'logemline_h', 'logemline_nii', 
                              'logemline_sii', 'logemline_sii', 'logemline_oii', 'logemline_oii', 
                              'logemline_h', 'logemline_h', 'logemline_h', 
                              'logemline_h', 'logemline_h', 'logemline_h'])
emnormall = np.array([1./11.21, 1./6.16, 1./2.87, 1./3., 
                           1., 1., 1./2.95, 
                           1., 1., 1., 0.77, 
                           1., 1.35, 1./65., 1./55., 1./45., 1./35., 
                           1./25., 1./18.])

class Grids():
    '''
    Initialize the grids object
    
    Ssp: read in the SSPs and (as of now) select only the Kroupa IMF
        --> "kroupa_shortcut": read in .npy file with grid already sorted
        --> ssp.ssp_grid: the grid [spectrum, age, metal]
        --> ssp.agegrid: the age array 
        --> ssp.logzgrid: the metal array 
        --> ssp.wave: wavelengths corresponding to spectra
        --> ssp.ssp_interp: interpolant function for ssp_grid
                e.g., ssp.ssp_interp(age,Z)

    Rfn: read in the response functions for a Kroupa IMF
        --> rfn.[element][m or p]: a grid [spectrum, age, metal] of 
                response spectra for given element
        --> rfn.agegrid: the age array 
        --> rfn.logzgrid: the metal array 
        --> rfn.wave: wavelengths corresponding to spectra
        --> rfn.[element][p/m]: the response function for +/-0.3
        --> rfn.[element]_interp: interpolant function
                e.g., rfn.mg_interp(age,Z,abund)
                
    Eventually: add nuisance parameters (skylines, atm transmission, hot stars, M7III stars)
    '''
    
    def __init__(self,kroupa_shortcut=True,inst_res=0,inst_res_wave=None):
        self.ssp = Ssp()

        # smooth all grids to instrumental resolution
        self.model_res = 2.5/inst_res_wave * ckms # km/s
        self.inst_res = inst_res
        self.inst_res_wave = inst_res_wave

        # interpolate instrumental resolution for entire wavelength grid
        # use min/max inst_res around targetted wavelength range
        if np.size(inst_res)>1:
            assert type(inst_res_wave) == np.ndarray
            assert np.size(inst_res) == np.size(inst_res_wave)
            self.inst_res = np.interp(self.ssp.wave,inst_res_wave,self.inst_res,
                             left=np.min(self.inst_res),right=np.max(self.inst_res))

            print(f"Smoothing grids to wave dependent instrumental resolution")
        else:
            print(f"Smoothing grids to instrumental resolution = {self.inst_res:.1f} km/s")
        
        self.smooth_to_inst()

    def smooth_to_inst(self):
        '''
        Smooth the model grids to the instrumental resolution.
        
        If the instrumental resolution is wavelength dependent, 
            you must have a wavelength array defined (self.inst_res_wave)
            corresponding to the inst_res variable

        self.inst_res must be in km/s (this is sigma not FWHM)
        If it is wavelength dependent, I have to convert to \delta\lambda
            for the smoothing function to work. I do this below...
            
        '''

        # don't smooth if instrumental resolution is 0
        if np.size(self.inst_res)==1:
            if self.inst_res==0: 
                return
            else:
                smoothtype = 'vel'
                resolution = self.inst_res
            
        # # don't do wavelength dependent smoothing unless you really need it
        # elif np.max(self.inst_res)-np.min(self.inst_res) < 10:
        #     smoothtype = 'vel'
        #     resolution = np.median(self.inst_res)

        # wavelength dependent smoothing
        # resolution variable converted to \delta \lambda
        else:
            smoothtype = 'lsf'
            resolution = self.inst_res*self.ssp.wave/ckms

        # smooth ssp grid to instrumental resolution
        for j in range(len(self.ssp.agegrid)):
            for k in range(len(self.ssp.logzgrid)):
                for i in range(len(self.ssp.afegrid)):
                    self.ssp.ssp_grid[:,j,k,i] = smoothspec(self.ssp.wave,
                                                            self.ssp.ssp_grid[:,j,k,i],
                                                        resolution=resolution,
                                                        smoothtype=smoothtype)

        # smooth rfn grid to instrumental resolution (loop through elements)
        # for col in self.rfn.rfn_cols_use:
        #     tmp = getattr(self.rfn,col)
        #     for j in range(len(self.rfn.agegrid)):
        #         for k in range(len(self.rfn.logzgrid)):
        #             tmp[:,j,k] = smoothspec(self.rfn.wave, tmp[:,j,k], 
        #                                     resolution=resolution,
        #                                    smoothtype=smoothtype)
        #    setattr(self.rfn,col,tmp)

    
    def get_model(self, params, outwave=None):   

        # get SSP corresopnding to age and Z
        spec = 10**self.ssp.ssp_interp([params['logage'],params['zH'],params['afe']])
        spec = spec[0]

        # add emission lines if logemline is one of the keys
        if np.array(['logemline' in p for p in params.keys()]).sum()>0:
            spec = self.add_emlines(spec,params)

        # smooth to desired sigma
        spec = smoothspec(self.ssp.wave, spec,
                            inres=np.average(self.model_res),resolution=params['sigma'])

        # redshift the model and interpolate to data wavelength
        oneplusz = (1+params['velz']/ckms)

        if outwave is None:
            outwave = self.ssp.wave
        spec = np.interp(outwave, self.ssp.wave*oneplusz, spec)

    
        return spec


    def add_emlines(self,spec,params):
        # this way you only add the emission lines that are included in the 
        # params dict
        # if you enter this you must have 'velz2' and 'sigma2' also defined
        # velz2 is in addition to velz
        for p,val in params.items():
            if 'logemline' not in p:
                continue
            wave_emlines_tmp = wave_emlines[emline_strs == p]
            emnormall_tmp = emnormall[emline_strs == p]

            ve   = wave_emlines_tmp/(1+params['velz2']/ckms)
            lsig = np.max([ve*params['sigma2']/ckms, np.ones(len(ve))],axis=0)  #min dlam=1.0A

            for i in range(len(ve)):
                spec+=10**val*emnormall_tmp[i]*np.exp(-(self.ssp.wave-ve[i])**2/lsig[i]**2/2.0)

        return spec
        

        

    
class Ssp():
    def __init__(self, kroupa_shortcut=True):            
        self.agegrid = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 
                                 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 
                                 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 1.75, 
                                 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 
                                 4.00, 4.50, 5.00, 5.50, 6.00, 6.50, 7.00, 7.50, 
                                 8.00, 8.50, 9.00, 9.50, 10.0, 10.5, 11.0, 11.5, 
                                 12.0, 12.5, 13.0, 13.5, 14.0])
        self.logagegrid = np.log10(self.agegrid)
        self.logzgrid = np.array([-1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25, 0.06, 0.15, 0.26])
        self.afegrid = np.array([-0.2,0.0, 0.2, 0.4, 0.6])
        self.afe_isochrone = np.array([0.0,0.0,0.0,0.4,0.4])
        self.afegrid_str = np.array(['02','00', '02', '04', '06'])
        
        self.populate_grid()
        self.smooth_models(sigma=100)
        self.set_up_interpolator()

    def populate_grid(self):
        
        self.ssp_grid = np.zeros([4300, len(self.agegrid),
                                        len(self.logzgrid),
                                        len(self.afegrid)])
        
        for age_i,age in enumerate(self.agegrid):
            age = f'{age:.4f}'.zfill(7)
            
            for zh_i,zh in enumerate(self.logzgrid):
                if zh>=0: pm_zh = 'p'
                else: pm_zh = 'm'
                zh = f'{np.abs(zh):0.2f}'
                
                for afe_i,afe in enumerate(self.afegrid):
                    afe_iso = self.afe_isochrone[afe_i]
                    if afe>=0: pm_afe = 'p'
                    else: pm_afe = 'm'
                    afe = self.afegrid_str[afe_i]
                    
                    
                    self.ssp_grid[:,age_i,zh_i,afe_i] = fits.open(f'{ALFA_INFILES}/sMILES_SSP/'\
                      f'Universal_Kroupa/aFe{pm_afe}{afe}/'\
                      f'Mku1.30Z{pm_zh}{zh}T{age}_iTp{afe_iso:.2f}_ACFep00_aFe{pm_afe}{afe}.fits')[0].data

        self.wave = np.linspace(3540.5,7409.6,4300)

    
    def smooth_models(self, sigma=100):
        '''
        Smooth the model grids to a constant dispersion
        '''
        # calculate how much you need to smooth the model to get 
        # to the desired resolution of 100 km/s
        smooth = np.sqrt(sigma**2-(3e5*2.54/2.355/self.wave)**2)
        smooth = smooth*self.wave/3e5

        for j in range(len(self.agegrid)):
            for k in range(len(self.logzgrid)):
                for i in range(len(self.afegrid)):
                    self.ssp_grid[:,j,k,i] = smoothspec(self.wave, self.ssp_grid[:,j,k,i],
                                                resolution=smooth, smoothtype='lsf')



    def set_up_interpolator(self):
        # Here you take the log of the grid before interpolating!!
        # Interpolate in log-space!! 
        # This was an annoying bug.
        self.ssp_interp = RegularGridInterpolator((self.logagegrid, self.logzgrid, self.afegrid), 
                        np.transpose(np.log10(self.ssp_grid), (1,2,3,0)), 
                    method='linear', bounds_error=False, fill_value=None)


