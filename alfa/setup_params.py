import numpy as np
#
def setup_params(parameters_to_fit=None,default_priors=False,model='conroy18'):
    
    default_params = {'velz':0, 'sigma':150, 'logage':0.4, 'zH':0,
                     'feh':0, 'alpha':0, 'ah':0, 'ch':0, 'nh':0, 'nah':0, 'mgh':0, 'sih':0,
                          'kh':0, 'cah':0, 'tih':0, 'vh':0, 'crh':0, 'mnh':0, 'coh':0,
                          'nih':0, 'cuh':0, 'srh':0, 'bah':0, 'euh':0,
                    'logemline_h':-4, 'logemline_ni':-4, 'logemline_nii':-4, 'logemline_oii':-4,
                   'logemline_oiii':-4, 'logemline_sii':-4, 'velz2':0, 'sigma2':200,'teff':0,
                     'jitter':1,'afe':0}
    if default_priors&(model=='conroy18'):
        priors_all = {'velz':[-200,200], 'sigma':[100,500], 'logage':[-0.1,1.2], 'zH':[-1.5,0.2],
                      'alpha':[-0.3,0.3],
                    'feh':[-0.3,0.3], 'ah':[-0.3,0.3], 'ch':[-0.15,0.15], 
                    'nh':[-0.3,0.3], 'nah':[-0.3,0.3], 'mgh':[-0.3,0.3], 'sih':[-0.3,0.3],
                    'kh':[-0.3,0.3], 'cah':[-0.3,0.3], 'tih':[-0.3,0.3], 
                    'vh':[-0.3,0.3], 'crh':[-0.3,0.3], 'mnh':[-0.3,0.3], 'coh':[-0.3,0.3],
                    'nih':[-0.3,0.3], 'cuh':[-0.3,0.3], 'srh':[-0.3,0.3], 'bah':[-0.3,0.3], 
                    'euh':[-0.3,0.3], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[50,1000],'teff':[-80,80],
                    'jitter':[0.1,10]}
    elif ~default_priors&(model=='conroy18'):
        # Heavy Metal -- limit age, extend mgh, feh, zh 'logage':[-0.1,0.8], 'zH':[-1.5,0.2]
        priors_all = {'velz':[-500,500], 'sigma':[100,500], 'logage':[-0.1,1.13], 'zH':[-1.5,0.3],
                    'alpha':[-0.5,0.5],
                    'feh':[-0.5,0.5], 'ah':[-0.5,0.5], 'ch':[-0.5,0.5], 
                    'nh':[-0.5,0.5], 'nah':[-0.5,1.0], 'mgh':[-0.5,0.5], 'sih':[-0.5,0.5],
                    'kh':[-0.5,0.5], 'cah':[-0.5,0.5], 'tih':[-0.5,0.5], 
                    'vh':[-0.5,0.5], 'crh':[-0.5,0.5], 'mnh':[-0.5,0.5], 'coh':[-0.5,0.5],
                    'nih':[-0.5,0.5], 'cuh':[-0.5,0.5], 'srh':[-0.5,0.5], 'bah':[-0.5,0.5], 
                    'euh':[-0.5,0.5], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[0,1000],'teff':[-50,50],
                    'jitter':[0.1,10]}

        # SUSPENSE    
        priors_all['logage'] = [-0.3,0.8]

    elif model=='sMILES':
        priors_all = {'velz':[-200,200], 'sigma':[100,500], 'logage':[-0.3,1.2], 'zH':[-1.5,0.26],
                      'afe':[-0.2,0.6], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[50,1000],'teff':[-80,80],
                    'jitter':[0.1,10]}

    if parameters_to_fit is None:
        del default_params['alpha']
        del priors_all['alpha']
        return default_params, priors_all
 
    else:
        default_pos = [default_params[key] for key in parameters_to_fit]
        priors = {key: priors_all[key] for key in parameters_to_fit}
    
        return default_pos, priors
    


def setup_initial_position(nwalkers,parameters_to_fit,priors):
    parameters_to_fit = np.array(parameters_to_fit)

    pos = np.random.uniform(0,1,(nwalkers,len(parameters_to_fit)))

    init_pos = {'velz':[-5,5], 'sigma':[200,300],'logage':[0.2,0.6], 'zH':[-0.3,0.1],
                        'alpha':[-0.2,0.2],
                        'feh':[-0.2,0.2], 'ah':[-0.2,0.2], 'ch':[-0.1,0.1], 
                        'nh':[-0.2,0.2], 'nah':[-0.1,0.6], 'mgh':[-0.2,0.2], 'sih':[-0.2,0.2],
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
        
        if (range_[0]>prior[1])|(range_[0]<prior[0]): range_[0] = prior[0]
        if (range_[1]<prior[0])|(range_[1]>prior[1]): range_[1] = prior[1]

        pos[:,i] = pos[:,i]*(range_[1] - range_[0]) + range_[0]

    return pos


def setup_initial_position_diff_ev(nwalkers,parameters_to_fit,diff_ev_result,priors):
    parameters_to_fit = np.array(parameters_to_fit)
    pos = setup_initial_position(nwalkers,parameters_to_fit,priors)

    for key, value in diff_ev_result.items():
        pos[:,parameters_to_fit==key] = value + 1e-4 * np.random.randn(nwalkers,1)
    
    return pos

    

def get_properties(theta,parameters_to_fit):
    return dict(zip(parameters_to_fit,theta))
