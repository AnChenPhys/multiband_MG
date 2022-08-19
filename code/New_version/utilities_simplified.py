### Common functions used in notebook
import time as tm
import numpy as np

from scipy import integrate
from scipy.misc import derivative
from scipy.optimize import minimize, newton, brentq
from scipy.special import comb

# define constants
c       = 299792458     # m/s
GN      = 6.674e-11     # m^3 /kg /s^2
Msolar  = 1.989e30      # kg
Mpc     = 3.086e22      # m


# Compute PhenomA waveform given in Robson et al.(2018)
param_a = { "merg" : 2.9740e-1, "ring" : 5.9411e-1, "sigma" : 5.0801e-1, 
            "cut" :  8.4845e-1}
param_b = { "merg" : 4.4810e-2, "ring" : 8.9794e-2, "sigma" : 7.7515e-2, 
            "cut" : 1.2848e-1}
param_c = { "merg" : 9.5560e-2, "ring" : 1.9111e-1, "sigma" : 2.2369e-2, 
            "cut" : 2.7299e-1}

cosmo_params    = { 'OmegaM'    : 0.2999,   'OmegaDE'   : 0.7, 
                    'OmegaR'    : 1.e-4,    'h'         : 0.7 }


#-------------------------------------------------------------------------------
#-------------------------- Some utilities -------------------------------------
#-------------------------------------------------------------------------------

def numerical_derivative(func, par, dx, conv=5e-2, factor=.5, verbose=False):
    """
    Computes the numerical derivative of a function
    """

    ratio   = 1e8
    r_best  = 1e8
    dx_best = 0.
    
    while ratio > conv:
        d1    = derivative(func, par, dx=dx/2, order=5)
        d2    = derivative(func, par, dx=dx *2, order=5)
        
        ld1   = len(d1)
        prod  = d1 *d2
        
        d1    = d1[prod != 0]
        d2    = d2[prod != 0]
        
        r_vec = np.abs( (d1 -d2)/np.mean((d1, d2), axis=0) )
        
        if len(r_vec) < int(ld1/3):
            ratio = 1e6
        else:
            ratio = np.mean(r_vec)
        
        if ratio < r_best:
            dx_best = dx
            r_best  = ratio
        
        dx *= factor  
        
        if dx < 1e-15:
            #print(par, dx)
            #raise ValueError('No convergence')
            ratio = 1e-1 *conv
    
    print('Ratio best = %.2e for dx = %.2e' % (r_best, dx_best))
    return derivative(func, par, dx=dx_best, order=5)


def scalar_product(hf, gf, psd, freqs):
    return 0.8 *2. *integrate.simps( np.real((hf *np.conjugate(gf) 
            +np.conjugate(hf) *gf))/psd, x=freqs)


#-------------------------------------------------------------------------------
#-------------------- Functions for cosmology ----------------------------------
#-------------------------------------------------------------------------------


def Hubble(z, cosmo_params=cosmo_params):
    """
    Returns the Hubble parameter in 1/Mpc (H is in km/s/Mcp)
    """

    Om0 = cosmo_params['OmegaM']
    OL0 = cosmo_params['OmegaDE']
    Or0 = cosmo_params['OmegaR']
    Hub = cosmo_params['h']

    # Define this was so can choose to absorb h into units if desired.
    return Hub *100 *1e3 /c *np.sqrt(OL0 +Om0 *(1 +z)**3 +Or0 *(1 +z)**4)


def get_dc_int_Mpc(z, cosmo_params=cosmo_params):
    """
    This returns the integrand for computing the comoving distance in Mpc
    In particular compare with 2.5 of 2203.00566 after replacing dz = da / a^2
    """

    return 1 /Hubble(z, cosmo_params=cosmo_params)


def get_dc_Mpc(z, cosmo_params=cosmo_params):
    """
    This computes the comoving distance in Mpc
    """

    ### Since dcom_int can work with vectors we define z_vec
    z_vec   = np.linspace(0, z, 1000)

    ### here we get the integrand
    to_int  = get_dc_int_Mpc(z_vec, cosmo_params=cosmo_params)
    
    return integrate.simps(to_int, x=z_vec, axis=0)


def get_dL_Mpc(z, cosmo_params=cosmo_params):
    """
    This computes the comoving distance in Mpc
    """
    
    return (1 +z) *get_dc_Mpc(z, cosmo_params=cosmo_params)


def get_ddL_Mpc_dz(z, cosmo_params=cosmo_params):
    """
    This computes the comoving distance in Mpc
    """
    
    return (1 +z) *get_dc_int_Mpc(z, cosmo_params=cosmo_params) +get_dc_Mpc(
            z, cosmo_params=cosmo_params)


#-------------------------------------------------------------------------------
#-------------------- Functions for cT -----------------------------------------
#-------------------------------------------------------------------------------

# ------- Start with all cT functions 
def cT_step(f, f0=1., dc0=0.): 
    return dc0 *np.heaviside(f -f0, 1.) +1 -dc0


def get_cT(f, f0=1, dc0=0, cT_type='GR'):
    if cT_type=='GR':
        cT  = f**0

    elif cT_type=='step':
        cT  = cT_step(f, f0=f0, dc0=dc0)

    return cT


def dcT_step(f, f0=1., dc0=0.): 
    return dc0 *(1 - np.heaviside(f -f0, 1.) )


def get_dcT(f, f0=1, dc0=0., cT_type='GR'):
    if cT_type=='GR':
        cT  = 0 *f

    elif cT_type=='step':
        cT  = dcT_step(f, f0=f0, dc0=dc0)

    return cT


# ------- Then all d cT / dc0 functions 
def dcT_step_ddc0(f, f0=1., dc0=0.):
    """
    Second derivative of cT_step wrt dc0
    """ 

    return np.heaviside(f -f0, 1.) -1


def get_dcT_ddc0(f, f0=1, dc0=0., cT_type='GR'):
    if cT_type=='GR':
        dcT_dc0 = 0 *f

    elif cT_type=='step':
        dcT_dc0 = dcT_step_ddc0(f, f0=f0, dc0=dc0)

    return dcT_dc0


# ------- Then all d dcT / dc0 functions 
def ddcT_step_ddc0(f, f0=1., dc0=0.):
    """
    Second derivative of cT_step wrt dc0
    """ 

    return 1 - np.heaviside(f -f0, 1.) 


def get_ddcT_ddc0(f, f0=1, dc0=0., cT_type='GR'):
    if cT_type=='GR':
        dcT_dc0 = 0 *f

    elif cT_type=='step':
        dcT_dc0 = ddcT_step_ddc0(f, f0=f0, dc0=dc0)

    return dcT_dc0


#-------------------- Functions for wf
def inspiral_fc(Mtot_z):   
    """
    Input redshifted total mass in second
    """      

    r_ISCO = 6 *Mtot_z                              # in sec
    f_ISCO = np.sqrt(Mtot_z /r_ISCO**3) /2 /np.pi   # in 1/sec

    return 2 *f_ISCO


#-------------------------------------------------------------------------------
#-------------------- Functions for the waveform -------------------------------
#-------------------------------------------------------------------------------


def get_Psis(eta):
    """
    These are 3.11 - 3.14 of 2203.00566
    """

    Q_1     = -(743 /336 +11 /4 *eta) *eta**(-2/5)
    Q_1p5   = (4 *np.pi -0) *eta**(-3 /5)
    Q_2     = (34103 /18144 +13661 /2016 *eta +59 /18 *eta**2) *eta**(-4 /5)
    Q_2p5   = -np.pi /eta *(4159 /672 +189 /8 *eta)

    return Q_1, Q_1p5, Q_2, Q_2p5


def Psi_Delta_inspiral(fo, Mo, eta, tc, psic, cosmo_params=cosmo_params, 
        cT_type='GR'):
    """
    Function to compute the phase for the GW inspiral, 3.15 of 2203.00566
    """

    Ps_1, Ps_1p5, Ps_2, Ps_2p5  = get_Psis(eta)

    uo_arr  = np.pi *Mo *fo
    
    ### These are the factors in the r.h.s. of 3.15 of 2203.00566
    ### since we work with Mo, we use (1 -Delta)^2 / Mz^2 = 1/Mo^2 !!!
    first   = 96 /5 /np.pi /Mo**2 *uo_arr**(11 /3)
    sq_br   = (1 +Ps_1 *uo_arr**(2 /3) +Ps_1p5 *uo_arr +Ps_2 *uo_arr**(4 /3) 
                +Ps_2p5 *uo_arr**(5/3) )

    ### This is the r.h.s of 3.19 of 2203.00566
    to_int  = 1 /first /sq_br

    t_o = np.zeros(len(fo))
    Psi = np.zeros(len(fo))

    for i in range(len(fo)):
        t_o[i] = integrate.simps(-to_int[i:], x=fo[i:])

    ### This is the integrand of 3.20 of 2203.00566
    Psi_int = 2 *np.pi *t_o #*(t_o -tc)

    for i in range(len(fo)):
        Psi[i]  = integrate.simps(-Psi_int[i:], x=fo[i:]) 

    return Psi +2 *np.pi *fo *tc -np.pi /4 -psic


def Psi_MG_dist(fo, z, f0=1., dc0=0., cT_type='GR', 
        cosmo_params=cosmo_params):
    """
    Old distance correlation term in the phase
    """

    cT      = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dcT     = get_dcT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
        
    dc_Mpc  = get_dc_Mpc(z, cosmo_params=cosmo_params)

    return -2 *np.pi *fo *Mpc *dc_Mpc /c *dcT /cT


def dPsi_MG_dist_ddc0(fo, z, f0=1., dc0=0., cT_type='GR', 
        cosmo_params=cosmo_params):
    """
    Derivative of the distance correlation term in the phase wrt c0
    """

    cT          = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dcT         = get_dcT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    
    dcT_dc0     = get_dcT_ddc0(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    ddcT_dc0    = get_ddcT_ddc0(fo, f0=f0, dc0=dc0, cT_type=cT_type)


    dc_Mpc      = get_dc_Mpc(z, cosmo_params=cosmo_params)
    
    return -2 *np.pi *fo *Mpc *dc_Mpc /c *(ddcT_dc0 /cT -dcT/cT**2 *dcT_dc0)


def dPsi_MG_dist_dz(fo, z, f0=1., dc0=0., cT_type='GR', 
        cosmo_params=cosmo_params):
    """
    Derivative of the distance correlation term in the phase wrt dc0
    """

    cT      = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dcT     = get_dcT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    ddc_Mpc = get_dc_int_Mpc(z, cosmo_params=cosmo_params)

    return -2 *np.pi *fo *Mpc *ddc_Mpc /c *dcT /cT


def amp_Delta_inspiral(fo, Mo, eta, z, f0=1., dc0=0., cT_type='GR', 
         cosmo_params=cosmo_params):
    """
    Returns the MG amplitude for inspiraling only
    Mo here is the chirp mass in observer frame defined as in 2203.00566, 
    in GR Mo = Mz
    """

    cT      = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dL_Mpc  = get_dL_Mpc(z, cosmo_params=cosmo_params)
    dL_MG   = cT *dL_Mpc *Mpc /c

    us_arr  = np.pi *Mo *fo

    ### computes 3.8 of 2203.00566 in the limit of fz = fo
    return np.sqrt(5 *np.pi /24) *Mo**2 /dL_MG *us_arr**(-7 /6)   # s


def damp_Delta_inspiral_dz(fo, Mo, eta, z, f0=1., dc0=0., cT_type='GR',
         cosmo_params=cosmo_params):

    cT      = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dL_Mpc  = get_dL_Mpc(z, cosmo_params=cosmo_params)    
    dL_MG   = cT *dL_Mpc *Mpc /c

    dL_MGdz = cT *Mpc /c *get_ddL_Mpc_dz(z, cosmo_params=cosmo_params)

    return -amp_Delta_inspiral(fo, Mo, eta, z, f0=f0, dc0=dc0, cT_type=cT_type, 
            cosmo_params=cosmo_params) /dL_MG *dL_MGdz


def damp_Delta_inspiral_ddc0(fo, Mo, eta, z, f0=1., dc0=0., cT_type='GR',
         cosmo_params=cosmo_params):

    cT      = get_cT(fo, f0=f0, dc0=dc0, cT_type=cT_type)
    dL_Mpc  = get_dL_Mpc(z, cosmo_params=cosmo_params)    
    dL_MG   = cT *dL_Mpc *Mpc /c

    dL_MGdc = get_dcT_ddc0(fo, f0=f0, dc0=dc0, cT_type=cT_type) *dL_Mpc *Mpc /c
    
    return -amp_Delta_inspiral(fo, Mo, eta, z, f0=f0, dc0=dc0, cT_type=cT_type, 
            cosmo_params=cosmo_params) /dL_MG *dL_MGdc


def h_Delta_inspiral(fo, pars, cosmo_params=cosmo_params, cT_type='GR', 
        dist_corr=True):

    Mo      = np.exp(pars[0]) *Msolar *GN /c**3
    eta     = np.exp(pars[1])
    z       = np.exp(pars[2])
    tc      = pars[3]
    psic    = pars[4]
    dc0     = pars[5]
    f0      = pars[6]


    amp = amp_Delta_inspiral(fo, Mo, eta, z, f0=f0, dc0=dc0, 
            cosmo_params=cosmo_params, cT_type=cT_type)

    Psi = Psi_Delta_inspiral(fo, Mo, eta, tc, psic, cosmo_params=cosmo_params, 
            cT_type=cT_type) 

    if cT_type != 'GR' and dist_corr:
        Psi += Psi_MG_dist(fo, z, f0=f0, dc0=dc0, cT_type=cT_type, 
                cosmo_params=cosmo_params)

    return amp *np.exp(1.j *Psi)


#-------------------------------------------------------------------------------
#------------------------ Derivatives and Fisher -------------------------------
#-------------------------------------------------------------------------------


def Fisher_der(fo, pars, cosmo_params=cosmo_params, cT_type='GR', 
        dist_corr=True):
    """
    This function computes the derivatives of the wf wrt the parameters
    """

    t0  = tm.perf_counter()
        
    Mo      = np.exp(pars[0]) *Msolar *GN/ c**3
    eta     = np.exp(pars[1])
    z       = np.exp(pars[2])
    tc      = pars[3]
    psic    = pars[4]
    dc0     = pars[5]
    f0      = pars[6]

    my_Amp  = amp_Delta_inspiral(fo, Mo, eta, z, f0=f0, dc0=dc0, 
                cosmo_params=cosmo_params, cT_type=cT_type)

    # Function to compute numerical derivatives w.r.t. lnM
    lnA_lnMo    = lambda lnMo_func : np.log(amp_Delta_inspiral(fo, 
                    np.exp(lnMo_func) *Msolar *GN/ c**3, eta, z, f0=f0, 
                    dc0=dc0, cosmo_params=cosmo_params, cT_type=cT_type))

    Psi_lnMo    = lambda lnMo_func : Psi_Delta_inspiral(fo, np.exp(lnMo_func
                    ) *Msolar *GN/ c**3, eta, tc, psic, 
                    cosmo_params=cosmo_params, cT_type=cT_type)

    # Function to compute numerical derivatives w.r.t. lneta
    lnA_lneta   = 0

    Psi_lneta   = lambda lneta_func : Psi_Delta_inspiral(fo, Mo, 
                    np.exp(lneta_func), tc, psic, cosmo_params=cosmo_params, 
                    cT_type=cT_type)

    dlnAs   = []
    dPsis   = []
    verbose = False
    
    for i in range(0, len(pars)-1):
        print('-Working on index', i)
        dx = np.abs(.1 *pars[i]) if pars[i] != 0 else 1e-1

        if i == 0:
            dlnA    = numerical_derivative(lnA_lnMo, pars[i], dx=dx, 
                        verbose=verbose)     
            dPsi    = numerical_derivative(Psi_lnMo, pars[i], dx=dx, 
                        verbose=verbose)   

        elif i == 1:
            dlnA    = 0 *fo   
            dPsi    = numerical_derivative(Psi_lneta, pars[i], dx=dx, 
                        verbose=verbose)                 

        elif i == 2:

            dlnA    = damp_Delta_inspiral_dz(fo, Mo, eta, z, f0=f0, dc0=dc0, 
                        cosmo_params=cosmo_params, cT_type=cT_type
                        ) / my_Amp *z

            dPsi    = 0 *fo

            if cT_type != 'GR' and dist_corr:
                dPsi    += dPsi_MG_dist_dz(fo, z, f0=f0, dc0=dc0, 
                            cT_type=cT_type, cosmo_params=cosmo_params)

        elif i == 3:
            # Analytical derivetives w.r.t. tc 
            dlnA    = 0. *fo
            dPsi    = 2. *np.pi *fo

        elif i == 4:
            # Analytical derivetives w.r.t. psic
            dlnA    = 0. *fo
            dPsi    = fo**0

        elif i == 5:
            if cT_type != 'GR':
                dlnA    = damp_Delta_inspiral_ddc0(fo, Mo, eta, z, f0=f0, 
                            dc0=dc0, cT_type=cT_type, cosmo_params=cosmo_params
                            ) / my_Amp 

                dPsi    = 0 *fo

                if dist_corr:
                    dPsi    += dPsi_MG_dist_ddc0(fo, z, f0=f0, dc0=dc0, 
                                cT_type=cT_type, cosmo_params=cosmo_params)

            else:
                continue
                
        else:
            raise ValueError('Cannot use that!!!')

        dlnAs.append(dlnA)
        dPsis.append(dPsi)
    
    #derivatives   = derivatives[:2][::-1] +derivatives[2:]

    print('This took %.2f seconds' % (tm.perf_counter() -t0) )
    return np.array(dlnAs), np.array(dPsis)


def Fisher_build(freqs, psd, h, derivatives):
    """
    This function builds the Fisher matrix given fo, the psd of the detector
    and the derivatives of the wf wrt the parameters
    """

    prefactor   = np.abs(h)**2 
    to_int      = derivatives[:,None] *derivatives[None,:]

    return 0.8 *4. *integrate.simps( prefactor *to_int /psd, x=freqs, axis=-1)