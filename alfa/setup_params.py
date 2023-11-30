def setup_params(parameters_to_fit=None):
    
    default_params = {'velz':0, 'sigma':200, 'logage':0.4, 'zH':0,
                     'feh':0, 'ah':0, 'ch':0, 'nh':0, 'nah':0, 'mgh':0, 'sih':0,
                          'kh':0, 'cah':0, 'tih':0, 'vh':0, 'crh':0, 'mnh':0, 'coh':0,
                          'nih':0, 'cuh':0, 'srh':0, 'bah':0, 'euh':0,
                    'logemline_h':-4, 'logemline_ni':-4, 'logemline_nii':-4, 'logemline_oii':-4,
                   'logemline_oiii':-4, 'logemline_sii':-4, 'velz2':0, 'sigma2':200}
    
    priors_all = {'velz':[-200,200], 'sigma':[100,500], 'logage':[-0.1,1.2], 'zH':[-1,0.2],
                    'feh':[-0.3,0.3], 'ah':[-0.3,0.3], 'ch':[-0.15,0.15], 
                    'nh':[-0.3,0.3], 'nah':[-0.3,0.3], 'mgh':[-0.3,0.3], 'sih':[-0.3,0.3],
                    'kh':[-0.3,0.3], 'cah':[-0.3,0.3], 'tih':[-0.3,0.3], 
                    'vh':[-0.3,0.3], 'crh':[-0.3,0.3], 'mnh':[-0.3,0.3], 'coh':[-0.3,0.3],
                    'nih':[-0.3,0.3], 'cuh':[-0.3,0.3], 'srh':[-0.3,0.3], 'bah':[-0.3,0.3], 
                    'euh':[-0.3,0.3], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[50,1000]}
    '''    
    # Heavy Metal -- limit age, extend mgh, feh, zh
    priors_all = {'velz':[-400,400], 'sigma':[100,500], 'logage':[-0.3,0.7], 'zH':[-1,0.5],
                    'feh':[-0.8,0.8], 'ah':[-0.3,0.3], 'ch':[-0.15,0.15], 
                    'nh':[-0.3,0.3], 'nah':[-0.3,0.3], 'mgh':[-0.8,0.8], 'sih':[-0.3,0.3],
                    'kh':[-0.3,0.3], 'cah':[-0.3,0.3], 'tih':[-0.3,0.3], 
                    'vh':[-0.3,0.3], 'crh':[-0.3,0.3], 'mnh':[-0.3,0.3], 'coh':[-0.3,0.3],
                    'nih':[-0.3,0.3], 'cuh':[-0.3,0.3], 'srh':[-0.3,0.3], 'bah':[-0.3,0.3], 
                    'euh':[-0.3,0.3], 'logemline_h':[-6,1], 'logemline_ni':[-6,1], 
                    'logemline_nii':[-6,1], 'logemline_oii':[-6,1], 'logemline_oiii':[-6,1],
                    'logemline_sii':[-6,1], 'velz2':[-200,200], 'sigma2':[50,1000]}
    '''

    if parameters_to_fit is None:
        return default_params, priors_all
 
    else:
        default_pos = [default_params[key] for key in parameters_to_fit]
        priors = {key: priors_all[key] for key in parameters_to_fit}
    
        return default_pos, priors


def get_properties(theta,parameters_to_fit):
    return dict(zip(parameters_to_fit,theta))
