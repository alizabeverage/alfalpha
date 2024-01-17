import numpy as np
#
def setup_params(parameters_to_fit=None):
    
    default_params = {'velz':0, 'sigma':150, 'logage':0.4, 'zH':0,
                     'feh':0, 'ah':0, 'ch':0, 'nh':0, 'nah':0, 'mgh':0, 'sih':0,
                          'kh':0, 'cah':0, 'tih':0, 'vh':0, 'crh':0, 'mnh':0, 'coh':0,
                          'nih':0, 'cuh':0, 'srh':0, 'bah':0, 'euh':0,
                    'logemline_h':-4, 'logemline_ni':-4, 'logemline_nii':-4, 'logemline_oii':-4,
                   'logemline_oiii':-4, 'logemline_sii':-4, 'velz2':0, 'sigma2':200,'teff':0,
                     'jitter':1}
    '''
    # priors_all = {'velz':[-200,200], 'sigma':[100,500], 'logage':[-0.1,1.2], 'zH':[-1.5,0.2],
    #                 'feh':[-0.3,0.3], 'ah':[-0.3,0.3], 'ch':[-0.15,0.15], 
    #                 'nh':[-0.3,0.3], 'nah':[-0.3,0.3], 'mgh':[-0.3,0.3], 'sih':[-0.3,0.3],
    #                 'kh':[-0.3,0.3], 'cah':[-0.3,0.3], 'tih':[-0.3,0.3], 
    #                 'vh':[-0.3,0.3], 'crh':[-0.3,0.3], 'mnh':[-0.3,0.3], 'coh':[-0.3,0.3],
    #                 'nih':[-0.3,0.3], 'cuh':[-0.3,0.3], 'srh':[-0.3,0.3], 'bah':[-0.3,0.3], 
    #                 'euh':[-0.3,0.3], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
    #                 'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
    #                 'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[50,1000],'teff':[-80,80],
    #                 'jitter':[0.1,10]}
    '''    
    # Heavy Metal -- limit age, extend mgh, feh, zh
    priors_all = {'velz':[-500,500], 'sigma':[100,500], 'logage':[-0.1,0.7], 'zH':[-1.5,0.25],
                    'feh':[-0.8,0.8], 'ah':[-0.3,0.3], 'ch':[-0.15,0.15], 
                    'nh':[-0.3,0.3], 'nah':[-0.3,0.3], 'mgh':[-0.8,0.8], 'sih':[-0.3,0.3],
                    'kh':[-0.3,0.3], 'cah':[-0.3,0.3], 'tih':[-0.3,0.3], 
                    'vh':[-0.3,0.3], 'crh':[-0.3,0.3], 'mnh':[-0.3,0.3], 'coh':[-0.3,0.3],
                    'nih':[-0.3,0.3], 'cuh':[-0.3,0.3], 'srh':[-0.3,0.3], 'bah':[-0.3,0.3], 
                    'euh':[-0.3,0.3], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[0,1000],'teff':[-50,50],
                    'jitter':[0.1,10]}
  

    if parameters_to_fit is None:
        return default_params, priors_all
 
    else:
        default_pos = [default_params[key] for key in parameters_to_fit]
        priors = {key: priors_all[key] for key in parameters_to_fit}
    
        return default_pos, priors

def setup_initial_position(nwalkers,parameters_to_fit):
    default_pos, priors = setup_params(parameters_to_fit)
    pos = np.random.uniform(0,1,(nwalkers,len(parameters_to_fit)))

    # starting positions.... 
    init_pos = {'velz':[-5,5], 'sigma':[200,300], 'logage':[0.2,0.6], 'zH':[-0.3,0.1],
                    'feh':[-0.2,0.2], 'ah':[-0.2,0.2], 'ch':[-0.15,0.15], 
                    'nh':[-0.2,0.2], 'nah':[-0.2,0.2], 'mgh':[-0.2,0.2], 'sih':[-0.2,0.2],
                    'kh':[-0.2,0.2], 'cah':[-0.2,0.2], 'tih':[-0.2,0.2], 
                    'vh':[-0.2,0.2], 'crh':[-0.2,0.2], 'mnh':[-0.2,0.2], 'coh':[-0.2,0.2],
                    'nih':[-0.2,0.2], 'cuh':[-0.2,0.2], 'srh':[-0.2,0.2], 'bah':[-0.2,0.2], 
                    'euh':[-0.2,0.2], 'logemline_h':[-2,0.2], 'logemline_ni':[-2,0.2], 
                    'logemline_nii':[-2,0.2], 'logemline_oii':[-2,0.2], 'logemline_oiii':[-2,0.2],
                    'logemline_sii':[-2,0.2], 'velz2':[-5,5], 'sigma2':[200,300],'teff':[-20,20],
                    'jitter':[0.7,1.3]}

    for i,param in enumerate(parameters_to_fit):
        # check the range is in prior range
        range_ = init_pos[param]
        prior = priors[param]
        
        if range_[0]<prior[0]: range_[0] = prior[0]
        if range_[1]>prior[1]: range_[1] = prior[1]

        pos[:,i] = pos[:,i]*(range_[1] - range_[0]) + range_[0]

    return pos

    

def get_properties(theta,parameters_to_fit):
    return dict(zip(parameters_to_fit,theta))
