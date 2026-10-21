import numpy as np
import pandas as pd
import os
from .smoothing import smoothspec
from scipy.interpolate import RegularGridInterpolator
import os

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
        --> ssp.ssp_grid: the grid [spectrum, age, metal] or [spectrum, imf1, imf2, age, metal]
        --> ssp.agegrid: the age array
        --> ssp.logzgrid: the metal array
        --> ssp.wave: wavelengths corresponding to spectra
        --> ssp.ssp_interp: interpolant function for ssp_grid
                e.g., ssp.ssp_interp(age,Z) or ssp.ssp_interp(imf1,imf2,age,Z)

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

    def __init__(self,kroupa_shortcut=True,inst_res=0,inst_res_wave=None,imf_vary=False,
                 two_process_mode=False,two_process_config=None):
        self.ssp = Ssp(kroupa_shortcut=kroupa_shortcut, imf_vary=imf_vary)
        self.rfn = Rfn()

        # two-process-constrained abundance mode (see alfa/two_process_mode.py).
        # When on, get_model derives every element response from (zH, feh, mgh);
        # all other elements are slaved. Standard mode leaves this False and is
        # completely untouched.
        self.two_process_mode = two_process_mode
        self.two_process_config = two_process_config
        self._last_tp_info = None
        if self.two_process_mode:
            from alfa.two_process_mode import describe_mapping, TwoProcessConfig
            if self.two_process_config is None:
                self.two_process_config = TwoProcessConfig()
            print(describe_mapping(self.two_process_config))

        # smooth all grids to instrumental resolution
        self.model_res = 100 # km/s
        self.inst_res = inst_res
        self.inst_res_wave = inst_res_wave

        # interpolate instrumental resolution for entire wavelength grid
        # use min/max inst_res around targetted wavelength range
        if np.size(inst_res)>1:
            inst_res_wave = np.array(inst_res_wave)
            inst_res = np.array(inst_res)
            
            assert np.size(inst_res) == np.size(inst_res_wave)
            self.inst_res = np.interp(self.ssp.wave,inst_res_wave,self.inst_res,
                             left=np.min(self.inst_res),right=np.max(self.inst_res))

            print(f"Smoothing grids to wave dependent instrumental resolution")
        else:
            print(f"Smoothing grids to instrumental resolution = {self.inst_res:.1f} km/s")
        
        self.smooth_to_inst()
        self.ssp.set_up_interpolator()
        self.rfn.set_up_interpolators()

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
        if self.ssp.imf_vary:
            for i1 in range(len(self.ssp.imfgrid)):
                for i2 in range(len(self.ssp.imfgrid)):
                    for j in range(len(self.ssp.agegrid)):
                        for k in range(len(self.ssp.logzgrid)):
                            self.ssp.ssp_grid[:,i1,i2,j,k] = smoothspec(self.ssp.wave,
                                                                self.ssp.ssp_grid[:,i1,i2,j,k],
                                                                resolution=resolution,
                                                                smoothtype=smoothtype)
        else:
            for j in range(len(self.ssp.agegrid)):
                for k in range(len(self.ssp.logzgrid)):
                    self.ssp.ssp_grid[:,j,k] = smoothspec(self.ssp.wave,
                                                            self.ssp.ssp_grid[:,j,k],
                                                          resolution=resolution,
                                                         smoothtype=smoothtype)

        # smooth rfn grid to instrumental resolution (loop through elements)
        for col in self.rfn.rfn_cols_use:
            tmp = getattr(self.rfn,col)
            for j in range(len(self.rfn.agegrid)):
                for k in range(len(self.rfn.logzgrid)):
                    tmp[:,j,k] = smoothspec(self.rfn.wave, tmp[:,j,k], 
                                            resolution=resolution,
                                           smoothtype=smoothtype)
            setattr(self.rfn,col,tmp)

    
    def get_model(self, params, outwave=None):
        # TWO-PROCESS MODE: derive every element response from (zH, feh, mgh)
        # and inject them so the standard element loop below applies them via
        # the shipped response functions, unchanged. feh/mgh are OFFSETS on the
        # library; delta_Fe == feh, delta_Mg == mgh - lib_mgfe(zH); all other
        # elements are slaved (see alfa/two_process_mode.py).
        if getattr(self, "two_process_mode", False):
            from alfa.two_process_mode import compute_deltas
            deltas, tp_info = compute_deltas(
                params["zH"], params.get("feh", 0.0), params.get("mgh", 0.0),
                self.two_process_config)
            self._last_tp_info = tp_info
            # overwrite/insert the element responses; kinematics, age, emission,
            # jitter, etc. pass through untouched.
            params = {**params, **deltas}

        # get SSP corresponding to age and Z (and optionally IMF slopes)
        if self.ssp.imf_vary:
            spec = 10**self.ssp.ssp_interp([params['imf1'], params['imf2'],
                                            params['logage'], params['zH']])
        else:
            spec = 10**self.ssp.ssp_interp([params['logage'],params['zH']])
        
        # fit elements in lockstep if "alpha" is in the parameter dictionary
        if 'alpha' in params.keys():
            value = params['alpha']
            for key in alpha_elements:
                interp = getattr(self.rfn, key[:-1]+'_interp')
                spec *= interp([params['logage'],params['zH'],value])

        # add rfns corresponding to age and Z
        for key, value in params.items():
            # if fitting in lockstep, skip the alpha elements
            if ('alpha' in params.keys())&(key in alpha_elements):
                continue
            if key in ['feh','ah','ch','nh','nah','mgh','sih',
                          'kh','cah','tih','vh','crh','mnh','coh',
                          'nih','cuh','srh','bah','euh']:
                
                interp = getattr(self.rfn, key[:-1]+'_interp')
                spec *= interp([params['logage'],params['zH'],value])

            # vary Teff (special case - force use of the 13 Gyr model)
            elif key == 'teff':
                interp = getattr(self.rfn, 'teff_interp')
                spec *= interp([np.log10(13),params['zH'],value])
        
        spec = spec[0]

        # add emission lines if logemline is one of the keys
        if np.array(['logemline' in p for p in params.keys()]).sum()>0:
            spec = self.add_emlines(spec,params)

        # smooth to desired sigma
        spec = smoothspec(self.ssp.wave, spec,
                            inres=self.model_res,resolution=params['sigma'])

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
        

        

    
# Wavelength window read from the VCJ / ATLAS model files (row indices).
# alf uses nstart=100, nend=5830 (0.36-1.10 um). The files run to row 10566
# (2.5 um); NEND = 10566 reads everything, which is needed for rest-frame
# 1.1-1.25 um features (Na I 1.14, K I 1.17, Pa gamma/beta) in z ~ 3 G395M
# spectra. Cached grids are tagged with the window so an old 1.1 um cache is
# never loaded by mistake. Changed 2026-09-17.
NSTART = 99      # 0.36 um
# Default: the whole file. Set the environment variable ALFA_NEND (a row index,
# e.g. 6600 = 1.30 um) before importing alfa to use a shorter grid: model
# evaluation time scales with the grid length, not with the data length.
NEND = int(os.environ.get("ALFA_NEND", 10566))     # 2.5 um (was 5830 = 1.10 um)
# Number of low-slope IMF grid points to skip. alf uses 2 (slopes start at 0.9); the VCJ
# files also hold slopes 0.5 and 0.7. Set ALFA_NIMFOFF=0 before importing alfa to load them.
NIMFOFF = int(os.environ.get("ALFA_NIMFOFF", 2))
GRID_TAG = (f"_{NSTART}_{NEND}" if (NSTART, NEND) != (99, 5830) else "") + (f"_imfoff{NIMFOFF}" if NIMFOFF != 2 else "")


class Ssp():
    def __init__(self, kroupa_shortcut=True, imf_vary=False):
        self.imf_vary = imf_vary
        self.agegrid = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.5])
        self.logagegrid = np.log10(self.agegrid)
        self.logzgrid = np.array([-1.5, -1.0, -0.5, 0.0, 0.25])
        self.nstart = NSTART
        self.nend   = NEND

        # IMF grid: 14 usable slopes from 0.9 to 3.5 in steps of 0.2
        # (nimfoff=2 skips the first two entries of the 16-point file grid)
        self.nimfoff = NIMFOFF
        self.imfgrid = np.array([0.5 + (i - 1 + self.nimfoff) / 5 for i in range(1, 17 - self.nimfoff)])

        if imf_vary:
            cache = f"{ALFA_INFILES}VCJ_IMF_grid_alfa{GRID_TAG}.npy"
            if kroupa_shortcut and os.path.isfile(cache):
                self.ssp_grid = np.load(cache)
                tmp = np.array(pd.read_csv(
                    f"{ALFA_INFILES}VCJ_v8_mcut0.08_t07.0_Zm0.5.ssp.imf_varydoublex.s100",
                    sep=r'\s+', header=None, comment='#'))
                self.wave = tmp[self.nstart:self.nend, 0]
            else:
                self.populate_imf_grid()
        else:
            self.get_kroupa_index()
            if kroupa_shortcut and os.path.isfile(f"{ALFA_INFILES}VCJ_Kroupa_alfa{GRID_TAG}.npy"):
                self.ssp_grid = np.load(f"{ALFA_INFILES}VCJ_Kroupa_alfa{GRID_TAG}.npy")
                tmp = np.array(pd.read_csv(
                    f"{ALFA_INFILES}VCJ_v8_mcut0.08_t07.0_Zm0.5.ssp.imf_varydoublex.s100",
                    sep=r'\s+', header=None, comment='#'))
                self.wave = tmp[self.nstart:self.nend, 0]
            else:
                self.populate_grid()


    def get_kroupa_index(self):
        nimfoff = 2
        imfgrid = np.array([0.5 + (i-1+nimfoff)/5 for i in range(1,15)]) #sspgrid%imfx1
        kroupa_imf1 = 1.3
        kroupa_imf2 = 2.3
        # add two here because nimfoff means the first two grid points are skipped in alf
        # i.e. there are 16 imf1 and imf2 values and only 14 usable - len(imfgrid) = 14
        imfr1 = np.where(imfgrid == kroupa_imf1)[0][0]+2
        imfr2 = np.where(imfgrid == kroupa_imf2)[0][0]+2
        
        self.kroupa_index = imfr1*(len(imfgrid)+nimfoff)+imfr2 #should be 73

    
    def populate_imf_grid(self):
        '''Load full IMF grid: shape (nl, nimf1, nimf2, nage, nzmet).
        Column index in the file: (i1+nimfoff)*16 + (i2+nimfoff), 0-indexed after the wavelength column.
        '''
        nimf = len(self.imfgrid)
        nfile = nimf + self.nimfoff  # = 16
        nl = self.nend - self.nstart
        self.ssp_grid = np.zeros([nl, nimf, nimf, len(self.agegrid), len(self.logzgrid)])
        for j, t in enumerate(self.agegrid):
            for k, z in enumerate(self.logzgrid):
                if z < 0: mp = 'm'
                else: mp = 'p'
                filename = (f"{ALFA_INFILES}VCJ_v8_mcut0.08_t{t:04.1f}"
                            f"_Z{mp}{np.abs(z):.1f}.ssp.imf_varydoublex.s100")
                tmp = np.array(pd.read_csv(filename, sep=r'\s+', header=None, comment='#'))
                for i1 in range(nimf):
                    for i2 in range(nimf):
                        col = (i1 + self.nimfoff) * nfile + (i2 + self.nimfoff)
                        self.ssp_grid[:, i1, i2, j, k] = tmp[self.nstart:self.nend, col + 1]
        self.wave = tmp[self.nstart:self.nend, 0]
        np.save(f"{ALFA_INFILES}VCJ_IMF_grid_alfa{GRID_TAG}.npy", self.ssp_grid)

    def populate_grid(self):
        self.ssp_grid = np.zeros([self.nend-self.nstart,
                                           len(self.agegrid),
                                           len(self.logzgrid)])
        for j,t in enumerate(self.agegrid):
            for k,z in enumerate(self.logzgrid):
                if z<0: mp = 'm'
                else: mp = 'p'
                filename = f"{ALFA_INFILES}VCJ_v8_mcut0.08_t{t:04.1f}"\
                                f"_Z{mp}{np.abs(z):.1f}.ssp.imf_varydoublex.s100"
                tmp = np.array(pd.read_csv(filename, sep=r'\s+', header=None, comment='#'))
                self.ssp_grid[:,j,k] = tmp[self.nstart:self.nend, self.kroupa_index+1]
        self.wave = tmp[self.nstart:self.nend,0]
        
        # save to file
        np.save(f"{ALFA_INFILES}VCJ_Kroupa_alfa{GRID_TAG}.npy",self.ssp_grid)


    def set_up_interpolator(self):
        # Interpolate in log-space (important — was an annoying bug)
        if self.imf_vary:
            # grid shape: (nl, nimf1, nimf2, nage, nzmet) → transpose to (nimf1, nimf2, nage, nzmet, nl)
            self.ssp_interp = RegularGridInterpolator(
                (self.imfgrid, self.imfgrid, self.logagegrid, self.logzgrid),
                np.transpose(np.log10(self.ssp_grid), (1, 2, 3, 4, 0)),
                method='linear', bounds_error=False, fill_value=None)
        else:
            self.ssp_interp = RegularGridInterpolator((self.logagegrid, self.logzgrid),
                            np.transpose(np.log10(self.ssp_grid), (1,2,0)),
                        method='linear', bounds_error=False, fill_value=None)



class Rfn():
    def __init__(self):
        self.agegrid = np.array([1,3,5,9,13])
        self.logagegrid = np.log10(self.agegrid)
        self.logzgrid = np.array([-1.5, -1.0, -0.5, 0.0, 0.25])
        self.nstart = NSTART
        self.nend   = NEND
        
        self.rfn_cols = ['lam','solar','nap','nam','cap','cam','fep','fem',
                      'cp','cm','d1','np','nm','ap','tip','tim','mgp','mgm',
                      'sip','sim','teffp','teffm','crp','mnp','bap','bam',
                      'nip','cop','eup','srp','kp','vp','cup','nap6','nap9']

        # remove lam and d1
        self.rfn_cols_use = ['solar','nap','nam','cap','cam','fep','fem',
                      'cp','cm','np','nm','ap','tip','tim','mgp','mgm',
                      'sip','sim','teffp','teffm','crp','mnp','bap','bam',
                      'nip','cop','eup','srp','kp','vp','cup','nap6','nap9']
        grid_arr = np.zeros([self.nend-self.nstart,
                                       len(self.agegrid),
                                       len(self.logzgrid)])
        for col in self.rfn_cols:
            if col=='lam' or col=='d1': continue
            self.__setattr__(col, np.copy(grid_arr))
        
        self.get_response_spectra()
        # self.set_up_interpolators()
            
    def get_response_spectra(self):
        '''
        Read in response spectra
        '''
        for k,z in enumerate(self.logzgrid):
            for j,t in enumerate(self.agegrid):
                if z<0: mp = 'm'
                else: mp = 'p'
                # Replace the [Z/H]=+0.2 models with the +0.0 models
                # -- as the former are broken
                if z==0.25: z=0.0 
                filename = f"{ALFA_INFILES}/atlas_ssp_t{t:02}"\
                                    f"_Z{mp}{np.abs(z):.1f}.abund.krpa.s100"
                tmp = np.array(pd.read_csv(filename, sep=r'\s+', header=None, comment='#'))
                
                
                for i, col in enumerate(self.rfn_cols):
                    if col == 'd1' or col == 'lam':
                        continue
                    else:
                        getattr(self, col)[:,j,k] = tmp[self.nstart:self.nend, i]
                        
        self.wave = tmp[self.nstart:self.nend, 0]




    def set_up_interpolators(self, age_dep_response = True, met_dep_response=True):
        '''
        set up interpolators
        To do: add age and Z independent interpolators
        '''
        elements = ['na','ca','fe','c','n','a','ti','mg',
              'si','teff','cr','mn','ba','ni','co','eu','sr','k','v','cu']

        for e in elements:
            if e == 'na':
                # set up Na interpolants
                p = np.array([self.nam/self.solar,self.solar/self.solar,
                                self.nap/self.solar,self.nap6/self.solar,self.nap9/self.solar])
                range_ = [-0.3,0,0.3,0.6,0.9] 

            elif e+'p' in self.rfn_cols and e+'m' in self.rfn_cols:
                p = np.array([getattr(self, e+'m')/self.solar,
                                self.solar/self.solar,
                                getattr(self, e+'p')/self.solar])
                range_ = [-0.3,0,0.3]
                if e=='c': range_ = [-0.15,0,0.15]
                if e=='teff': range_ = [-50.0,0,50.0]
                
            elif e+'p' in self.rfn_cols and ~(e+'m' in self.rfn_cols):
                p = np.array([self.solar/self.solar,
                                getattr(self, e+'p')/self.solar])
                range_ = [0,0.3]

            else: print('problem reading in setting up interpolants')

        
            interp = RegularGridInterpolator((self.logagegrid, self.logzgrid,range_), 
                        np.transpose(p, (2,3,0,1)), 
                        method='linear', bounds_error=False, fill_value=None)
            
            setattr(self,e+'_interp',interp)


    


