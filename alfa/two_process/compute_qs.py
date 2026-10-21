import os

import numpy as np
from astropy.table import Table

# W22 tables live next to this file, so compute_qs() works from any cwd
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def correct_roberts(C,N,CN):
    '''
    Enforce that C+N/2 (apogee) = C+N (roberts)
    '''
    rawC, rawN, rawCN = 10**C, 10**N, 10**CN
    # equation 27 in W22
    roberts_sum = (10**(C+8.39) + 10**(N+7.78))/(10**8.39+10**7.78)
    difference_factor = rawCN/roberts_sum
    newC, newN = np.log10(rawC*difference_factor), np.log10(rawN*difference_factor)
    return newC, newN


def compute_CN_tracks(mgh):  
    '''In APOGEE, they report C+N because the red giant abundances do not
       reflect the natal C and N abundances (because of dredge up). To
       get the natal C and N abundances, we use the hi and lo alpha 
       tracks of subgiants from Roberts+24. '''  
    
    mgh = np.array(mgh)
    mgh[mgh<-0.5] = np.nan # bad subgiant coverage at low metallicity

    # C, normalizing offset so that hiIa track passes through the point (0,0)
    b2_lo = 0.544; b1_lo=0.243; b0_lo=-0.202
    b2_hi = 0.34; b1_hi=0.119; b0_hi=-0.0923
    hiIa_C = b2_hi*mgh**2 + b1_hi*mgh
    loIa_C = b2_lo*mgh**2 + b1_lo*mgh + b0_lo - b0_hi

    # N
    b2_lo = 0.491; b1_lo=0.638; b0_lo=-0.375
    b2_hi = 0.43; b1_hi=0.475; b0_hi=-0.0831
    hiIa_N = b2_hi*mgh**2 + b1_hi*mgh
    loIa_N = b2_lo*mgh**2 + b1_lo*mgh + b0_lo - b0_hi

    return hiIa_N, loIa_N, hiIa_C, loIa_C

def compute_galah_qs(mgh, compute_from_tracks = False, mgfe_pl = 0.3):
    if compute_from_tracks:
        hiIa_Fe, loIa_Fe, hiIa_Y, loIa_Y, hiIa_Ba, loIa_Ba, hiIa_Ti, loIa_Ti = get_galah_tracks(mgh)
        Acc = 10**mgh
        Aia_hi = Acc*(10**hiIa_Fe - 10**-mgfe_pl)/(1-10**-mgfe_pl)
        Aia_lo = Acc*(10**loIa_Fe - 10**-mgfe_pl)/(1-10**-mgfe_pl)
        R_hi = (Aia_hi/Acc)
        R_lo = (Aia_lo/Acc)
        qia_Y = (10**loIa_Y - 10**hiIa_Y)/(R_lo - R_hi)
        qcc_Y = 10**hiIa_Y - (10**loIa_Y - 10**hiIa_Y)/(R_lo/R_hi - 1)
        qia_Ba = (10**loIa_Ba - 10**hiIa_Ba)/(R_lo - R_hi)
        qcc_Ba = 10**hiIa_Ba - (10**loIa_Ba - 10**hiIa_Ba)/(R_lo/R_hi - 1)
        qia_Ti = (10**loIa_Ti - 10**hiIa_Ti)/(R_lo - R_hi)
        qcc_Ti = 10**hiIa_Ti - (10**loIa_Ti - 10**hiIa_Ti)/(R_lo/R_hi - 1)

    else:
        mgh_grif = np.array([-0.439, -0.343,-0.249,-0.146,-0.050,0.049,0.145,0.248,0.348,0.439])
        qcc_Y_grif = np.array([0.444,0.406,0.343,0.320,0.302,0.281,0.293,0.325,0.365,0.333])
        qia_Y_grif = np.array([0.510,0.606,0.713,0.734,0.733,0.690,0.622,0.561,0.513,0.588])

        qcc_Ba_grif = np.array([0.345,0.324,0.296,0.296,0.301,0.321,0.312,0.303,0.295,0.266])
        qia_Ba_grif = np.array([0.742,0.817,0.872,0.837,0.755,0.633,0.567,0.514,0.492,0.544])
        
        qcc_Ti_grif = np.array([0.991,0.952,0.919,0.872,0.819,0.750,0.661,0.602,0.529,0.487])
        qia_Ti_grif = np.array([0.260,0.237,0.209,0.197,0.204,0.232,0.296,0.329,0.395,0.439])


        qcc_Y, qia_Y = np.interp(mgh, mgh_grif, qcc_Y_grif), np.interp(mgh, mgh_grif, qia_Y_grif)
        qcc_Ba, qia_Ba = np.interp(mgh, mgh_grif, qcc_Ba_grif), np.interp(mgh, mgh_grif, qia_Ba_grif)
        qcc_Ti, qia_Ti = np.interp(mgh, mgh_grif, qcc_Ti_grif), np.interp(mgh, mgh_grif, qia_Ti_grif)

        qcc_Yagb = [0.4851, 0.45688, 0.41811, 0.40380, 0.386, 0.3577, 0.37840, 0.42002, 0.46516, 0.4648]
        qagb_Y = [0.43620, 0.5163, 0.5964, 0.6136, 0.6132, 0.588, 0.5077, 0.4340, 0.374, 0.3879] 
        
    return qcc_Y, qia_Y, qcc_Ba, qia_Ba, qcc_Ti, qia_Ti



def get_galah_tracks(mgh):
    mgh_grif = np.array([-0.439, -0.343,-0.249,-0.146,-0.050,0.049,0.145,0.248,0.348,0.439])
    hiIa_Fe_grif = np.array([-0.016704075145633657, -0.012714558169103718, -0.0032386444724107344, -0.004745254745254668, -0.010372744138977819, -0.017370616396590433, -0.03261511216056667, -0.03962109319252172, -0.04662869597934527, -0.06598434033498968])
    loIa_Fe_grif = np.array([-0.28731657952437173, -0.28057980980058905, -0.26834204756282687, -0.26711600088223453, -0.2548847256639464, -0.22891880846426294, -0.20157926489095312, -0.19484898218664448, -0.18812032123720435, -0.19099244910933233])
    hiIa_Y_grif = np.array([-0.05694944453033546,-0.03192570179119636,-0.0018442034587096146,-0.002330101375974447,-0.018121783687081616,-0.05942310665459272,-0.09562250149082319,-0.1165602844711443,-0.13237405305121797,-0.13796187909976365])  
    loIa_Y_grif = np.array([-0.3528833624135874, -0.38400291538750375, -0.4099763676922057, -0.43601607880380766, -0.42119619232723027, -0.39617244958809095, -0.36606886498663793, -0.3461691366477462, -0.3109857101839786, -0.31655144996355755])
    hiIa_Ba_grif = np.array([0.006993006993007089,0.04195804195804198,0.05944055944055948,0.0472027972027973,0.005244755244755317,-0.05244755244755239,-0.09965034965034958,-0.13811188811188801,-0.16433566433566427,-0.18356643356643343])
    loIa_Ba_grif = np.array([-0.4230769230769229,-0.43006993006993,-0.43006993006993,-0.43006993006993,-0.4020979020979021,-0.35664335664335645,-0.3426573426573425,-0.35664335664335645,-0.3618881118881119,-0.3706293706293704])
    hiIa_Ti_grif = np.array([0.0860058309037901,0.06705539358600587,0.04664723032069973,0.0276967930029155,0.008746355685131268,-0.011661807580174877,-0.029154518950437303,-0.04956268221574345,-0.06413994169096204,-0.08309037900874627])
    loIa_Ti_grif = np.array([0.00291545189504383,-0.017492711370262315,-0.030612244897959107,-0.05102040816326525,-0.07580174927113703,-0.09475218658892126,-0.129737609329446,-0.15451895043731767,-0.18075801749271125,-0.2011661807580174])

    hiIa_Fe, loIa_Fe = np.interp(mgh, mgh_grif, hiIa_Fe_grif), np.interp(mgh, mgh_grif, loIa_Fe_grif)
    hiIa_Y, loIa_Y = np.interp(mgh, mgh_grif, hiIa_Y_grif), np.interp(mgh, mgh_grif, loIa_Y_grif)
    hiIa_Ba, loIa_Ba = np.interp(mgh, mgh_grif, hiIa_Ba_grif), np.interp(mgh, mgh_grif, loIa_Ba_grif)
    hiIa_Ti, loIa_Ti = np.interp(mgh, mgh_grif, hiIa_Ti_grif), np.interp(mgh, mgh_grif, loIa_Ti_grif)
    
    return hiIa_Fe, loIa_Fe, hiIa_Y, loIa_Y, hiIa_Ba, loIa_Ba, hiIa_Ti, loIa_Ti


def compute_qs(mgfe_pl=0.3, correct_CN=True):

    # read tables from W22
    w22_r = Table.read(os.path.join(DATA_DIR, 'W22_R.csv'))
    w22_qia = Table.read(os.path.join(DATA_DIR, 'W22_qIa.csv'))
    w22_qcc = Table.read(os.path.join(DATA_DIR, 'W22_qCC.csv'))

    # compute tracks from W22 tables
    elements = w22_qcc.colnames[1:]
    loIa = Table()
    loIa['mgh'] = w22_r['mgh']
    
    hiIa = Table()
    hiIa['mgh'] = w22_r['mgh']
    
    for e in elements:
        loIa[e] = np.log10(w22_qcc[e] + w22_r['Low-Ia']*w22_qia[e])
        hiIa[e] = np.log10(w22_qcc[e] + w22_r['High-Ia']*w22_qia[e])

    hiIa['N'], loIa['N'], hiIa['C'], loIa['C'] = compute_CN_tracks(w22_r['mgh'])

    if correct_CN: 
        hiIa['C'], hiIa['N'] = correct_roberts(hiIa['C'], hiIa['N'], hiIa['C+N'])
        loIa['C'], loIa['N'] = correct_roberts(loIa['C'], loIa['N'], loIa['C+N'])

    # compute qs
    elements = hiIa.colnames[1:]
    qia = Table()
    qia['mgh'] = w22_r['mgh']
    
    qcc = Table()
    qcc['mgh'] = w22_r['mgh']

    fcc_lo = Table()
    fcc_lo['mgh'] = w22_r['mgh']
    
    fcc_hi = Table()
    fcc_hi['mgh'] = w22_r['mgh']

    R_hi = (10**hiIa['Fe'] - 10**-mgfe_pl) / (1-10**-mgfe_pl)
    R_lo = (10**loIa['Fe'] - 10**-mgfe_pl) / (1-10**-mgfe_pl)
    
    for e in elements:
        qia[e] = (10**hiIa[e] - 10**loIa[e])/(R_hi - R_lo)
        qcc[e] = 10**loIa[e] - (10**hiIa[e] - 10**loIa[e]) / (R_hi/R_lo - 1)

        fcc_lo[e] = (1+ (R_lo)*(qia[e]/qcc[e]))**-1
        fcc_hi[e] = (1+ (R_hi)*(qia[e]/qcc[e]))**-1

    # get Ba and Y qcc and qia from GALAH
    # if mgfe_pl is not 0.3, we compute the qs from the tracks
    # otherwise, just interpolate the q's provided in griffiths onto mgh values
    if mgfe_pl!=0.3:
        qcc['Y'], qia['Y'], qcc['Ba'], qia['Ba'], qcc['Ti'], qia['Ti']  = compute_galah_qs(qcc['mgh'], compute_from_tracks = True, mgfe_pl = mgfe_pl)
    else:
        qcc['Y'], qia['Y'], qcc['Ba'], qia['Ba'], qcc['Ti'], qia['Ti'] = compute_galah_qs(qcc['mgh'], compute_from_tracks = False)

    qcc['Sr'] = qcc['Y']
    qia['Sr'] = qia['Y']
    return qia, qcc, fcc_lo, fcc_hi
