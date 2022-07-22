### Common functions used in notebook
import time as tm
import numpy as np

from scipy import integrate
from scipy.misc import derivative
from scipy.special import comb

# define constants
c       = 299792.458    #km/s
GN      = 6.674e-11     #m^3/kg/s^2
Msolar  = 1.989e30      # kg
Mpc     = 3.086e19      # km


# Compute PhenomA waveform given in Robson et al.(2018)
para_a  = { "merg" : 2.9740e-1, "ring" : 5.9411e-1, "sigma" : 5.0801e-1, 
            "cut" :  8.4845e-1}
para_b  = { "merg" : 4.4810e-2, "ring" : 8.9794e-2, "sigma" : 7.7515e-2, 
            "cut" : 1.2848e-1}
para_c  = { "merg" : 9.5560e-2, "ring" : 1.9111e-1, "sigma" : 2.2369e-2, 
            "cut" : 2.7299e-1}

cosmo_params    = { 'OmegaM'    : 0.2999,
                    'OmegaDE'   : 0.7,
                    'OmegaR'    : 1.e-4,
                    'h'         : 0.7 }

#-------------------- Functions for cosmology

# Hubble parameter
def Hubble(z, cosmo_params=cosmo_params):
    Om0 = cosmo_params['OmegaM']
    OL0 = cosmo_params['OmegaDE']
    Or0 = cosmo_params['OmegaR']
    Hub = cosmo_params['h']

    # Define this was so can choose to absorb h into units if desired.
    return Hub*100*np.sqrt(OL0 +Om0*(1+z)**3 + Or0*(1+z)**4)


# Integrand in computing the comoving distance
def dcom_int(z, f, cosmo_params=cosmo_params):
    Hubfunc = Hubble(z, cosmo_params=cosmo_params)
    ceetee  = np.array([1.])

    if np.size(f)>1:
        result = ceetee[None,:]/Hubfunc[:,None]
    else:
        result = ceetee/Hubfunc

    return result


# compute the comoving distance
def dcom(zem, f, cosmo_params=cosmo_params):
    ### Works with vectors
    z_vec       = np.linspace(0, zem, 1000)
    to_int      = dcom_int(z_vec, f, cosmo_params=cosmo_params)
    Dcomresult  = integrate.simps(to_int, x=z_vec, axis=0)
    return c *Dcomresult


# GW luminosity distance
def DGW(zem, f, cosmo_params=cosmo_params):
    rcom = dcom(zem, f, cosmo_params=cosmo_params)
    return (1 + zem) *rcom   #In Mpc


#-------------------- Functions for wf


def Lorentzian(f, f_ring, sigma):
    """ """
    return sigma /(2 *np.pi) /( (f -f_ring)**2 + 0.25 *sigma**2 )


def get_freq(M, eta, name):
    """ """
    result  = para_a[name] *eta**2 +para_b[name] *eta + para_c[name]

    return result /(np.pi *M)


def inspiral_fc(Mtot_z):                # input redshifted total mass in second
    r_ISCO = 6 * (Mtot_z)                           # in sec
    f_ISCO = np.sqrt(Mtot_z/r_ISCO**3) / 2/np.pi    # in 1/sec

    return 2 *f_ISCO


def get_Qs(eta):
    Q_1 = - (743./336 + 11./4*eta)*eta**(-2./5)
    Q_1p5 = (4*np.pi-0) *eta**(-3./5)
    Q_2 = (34103./18144 + 13661./2016*eta + 59./18*eta**2) *eta**(-4./5)
    Q_2p5 = -np.pi/eta*(4159./672+189./8*eta)

    return Q_1, Q_1p5, Q_2, Q_2p5


### EFT IS NOT USED 
def F(x,c0):
    return x**2/(1 + x**2 - (1 + 2 * (1 - c0**2) * x**2)**(1/2))**(1/2)


### EFT IS NOT USED 
def coefficients_fo(x,c0,z):
    a = 1
    b = -2 * (F(x, c0)*(1 + z))**2
    c = (F(x, c0)*(1 + z))**2 * ((F(x, c0)*(1 + z))**2 - 2)
    d = 2 * c0**2 * (F(x, c0)*(1 + z))**4
    return a,b,c,d


### EFT IS NOT USED 
def cT_EFT(fo, fstar, c0):
    f_rat = fo/fstar
    return np.sqrt(1 + 1/f_rat**2 - 1/f_rat**2*np.sqrt(1+2*(1-c0**2)*f_rat**2))


### EFT IS NOT USED 
def delta_EFT(fo, fstar, c0, z):
    delta=[]
    fo_rat = fo/fstar
    for ii in fo_rat:
        dd= 1 - (1+z)*ii/(np.roots(coefficients_fo(ii,c0,z)))**(1./2)
        delta.append(np.nanmin(dd))
    return np.array(delta)


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def cT_step(farr, fstar, c0, width=0):

    logf    = np.log10(farr)

    step    = smoothstep(x=logf, x_min=np.log10(fstar) -width, 
                x_max=np.log10(fstar) +width, N=5)

    cT_val = (1 -c0) *step +c0

    return cT_val


def Delta_step(farr, fstar, c0h, zem, width=0):

    cT_step_fo = cT_step(farr, fstar, c0h, width=width)
    cT_step_fs = cT_step(farr*(1+zem), fstar, c0h, width=width)

    return 1 -cT_step_fo /cT_step_fs


def get_Delta_ct(fo, fstar, c0, zem, width=0, cT_type='GR'):
    if cT_type=='GR':
        Delta = 0
        cT_fo = 1

    elif cT_type=='EFT':
        Delta = delta_EFT(fo, fstar, c0, zem)
        cT_fo = cT_EFT(fo, fstar, c0)

    elif cT_type=='step':
        Delta = Delta_step(fo, fstar, c0, zem, width)
        cT_fo = cT_step(fo, fstar, c0, width)

    return Delta, cT_fo


def get_Delta_dDelta(fo, fstar, c0, zem, width=0, cT_type='GR'):
    if cT_type=='GR':
        Delta       = 0
        d_Delta_dfo = 0

    elif cT_type=='EFT':
        Delta       = delta_EFT(fo, fstar, c0, zem)
        Delta_dfo   = lambda fo_func: delta_EFT(fo_func, fstar, c0, zem)
        d_Delta_dfo = derivative(Delta_dfo, fo, fo*1e-3)

    elif cT_type=='step':
        Delta       = Delta_step(fo, fstar, c0, zem, width=width)
        Delta_dfo   = lambda fo_func: Delta_step(fo_func, fstar, c0, zem, 
                        width=width)
        d_Delta_dfo = derivative(Delta_dfo, fo, fo*1e-3)

    else:
        Delta = 0.
        print('No Delta setting detected in Psi_Delta_inspiral.')

    return Delta, d_Delta_dfo


# phase for exact Delta
def Psi_Delta_inspiral(fo, fstar, c0, Mz, eta, zem, tc, psic, 
        cosmo_params=cosmo_params, cT_type='EFT', width=0):

    Q_1, Q_1p5, Q_2, Q_2p5  = get_Qs(eta)
    Delta, d_Delta_dfo      = get_Delta_dDelta(fo, fstar, c0, zem, 
                                width=width, cT_type=cT_type)

    Mo_arr  = Mz / (1-Delta)
    uo_arr  = np.pi * Mo_arr * fo
    to_int  = (5*np.pi*(Mz)**2/96 / (1-Delta)**2 
                * (1 + fo/(1-Delta)*d_Delta_dfo) * uo_arr**(-11./3) 
                * (1 - Q_1*uo_arr**(2./3) - Q_1p5*uo_arr 
                    + (Q_1**2-Q_2) *uo_arr**(4./3) 
                    + (2*Q_1*Q_1p5-Q_2p5)*uo_arr**(5./3) ))

    t_o = np.zeros(len(fo))
    Psi = np.zeros(len(fo))

    for i in range(len(fo)):
        t_o[i] = integrate.simps(-to_int[i:], x=fo[i:])

    Psi_int = 2*np.pi * t_o

    for i in range(len(fo)):
    
        # dist_val =  dcom(zem, fo[i], cosmo)   # TB Does this need to be in MG?
        Psi[i]  = integrate.simps(-Psi_int[i:], x=fo[i:]
                    ) - np.pi/4 + 2*np.pi*fo[i]*tc - psic
        # Psi[i] = integrate.simps(-Psi_int[i:], x=fo[i:]) - np.pi/4 + 2*np.pi*fo[i]*(tc + Mpc *dist_val/c*(1-c0)*(1-fstar/fo[i])*np.heaviside(fstar-fo[i],1)) - psic

    return Psi


# distance correlation term in the phase
def Psi_dist_corr(fo, fstar, c0, zem, cosmo_params=cosmo_params):

    result = np.zeros(len(fo))
    for i in range(len(fo)):
        dist_val    =  dcom(zem, fo[i], cosmo_params=cosmo_params)
        result[i] = 2 *np.pi *fo[i] *Mpc *dist_val /c *(1 -c0) *(1 -fstar 
                        /fo[i]) *np.heaviside(fstar -fo[i], 1)

    return result


# amplitude for exact Delta
def amp_Delta_inspiral(fo, fstar, c0, Mz, eta, zem, cosmo_params=cosmo_params, 
        width=0, cT_type='EFT'):

    Delta, cT_fo    = get_Delta_ct(fo, fstar, c0, zem, width=0, cT_type=cT_type)
    
    dl_GR   = DGW(zem, fo, cosmo_params=cosmo_params)
    dl_MG   = dl_GR / np.sqrt(1-Delta) * cT_fo
    amp_MG  = Amplitude(fo, Mz, eta, dl_MG*Mpc/c, False) * (1-Delta)**(2./3)

    return amp_MG


def h_Delta_inspiral(fo, pars, cosmo_params=cosmo_params, width=0, 
        cT_type='GR', dist_corr=True):

    Mz      = np.exp(pars[0]) * Msolar * GN/ (1e3 *c)**3
    eta     = np.exp(pars[1])
    zem     = np.exp(pars[2])
    tc      = pars[3]
    psic    = pars[4]
    c0h     = pars[5]
    fstarh  = pars[6]

    amp = amp_Delta_inspiral(fo, fstarh, c0h, Mz, eta, zem, 
            cosmo_params=cosmo_params, width=width, cT_type=cT_type)

    if dist_corr==True:
        Psi = Psi_Delta_inspiral(fo, fstarh, c0h, Mz, eta, zem, tc, psic, 
                cosmo_params=cosmo_params, width=width, cT_type=cT_type
                ) +Psi_dist_corr(fo, fstarh, c0h, zem, 
                    cosmo_params=cosmo_params)
    else:
        Psi = Psi_Delta_inspiral(fo, fstarh, c0h, Mz, eta, zem, tc, psic, 
                cosmo_params=cosmo_params, width=width, cT_type=cT_type)

    return amp * np.exp(1.j*Psi)


def Amplitude(f, M, eta, Dl, PhenomFlag):

    Mtot = M*eta**(-3./5)

    if PhenomFlag==True:
        # generate phenomA frequency parameters
        f_merg  = get_freq(Mtot, eta, "merg")
        f_ring  = get_freq(Mtot, eta, "ring")
        sigma   = get_freq(Mtot, eta, "sigma")
        f_cut   = get_freq(Mtot, eta, "cut")

        # break up frequency array into pieces
        mask1   = (f<=f_merg)
        mask2   = (f>f_merg) & (f<=f_ring)
        mask3   = (f>f_ring) & (f<f_cut)

        C   = M**(5./6)/Dl/np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5./24)
        w   = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)
        Amp = np.zeros(len(f))

        Amp[mask1]  = C[mask1]*(f[mask1]/f_merg)**(-7./6)
        Amp[mask2]  = C[mask2]*(f[mask2]/f_merg)**(-2./3)
        Amp[mask3]  = C[mask3]*w*Lorentzian(f[mask3], f_ring, sigma)

    elif PhenomFlag==False:
        us_arr  = np.pi * M * f
        Amp     = np.sqrt(5.*np.pi/24) * M**2 / Dl * us_arr**(-7./6)   # s

    return Amp


#numerically compute GW phase
def Psi_GR(fo, Mz, eta, zem, tc, psic):

    Q_1, Q_1p5, Q_2, Q_2p5  = get_Qs(eta)

    uo_arr  = np.pi * Mz * fo

    to_int  = (5*np.pi*(Mz)**2/96 * uo_arr**(-11./3) * (1 - Q_1*uo_arr**(2./3) 
                - Q_1p5*uo_arr + (Q_1**2-Q_2)*uo_arr**(4./3) 
                + (2*Q_1*Q_1p5-Q_2p5)*uo_arr**(5./3)))

    t_o = np.zeros(len(fo))
    Psi = np.zeros(len(fo))

    for i in range(len(fo)):
        t_o[i] = integrate.simps(-to_int[i:], x=fo[i:])

    Psi_int = 2*np.pi * t_o

    for i in range(len(fo)):
        Psi[i]  = integrate.simps(-Psi_int[i:], x=fo[i:]
                    ) - np.pi/4 + 2*np.pi*fo[i]*tc - psic

    return Psi


# Now the input mass is redshifted
def waveform(freq, pars, cosmo_params=cosmo_params):

    Mc      = np.exp(pars[0]) * Msolar * GN/ (1e3 *c)**3
    eta     = np.exp(pars[1])
    z       = np.exp(pars[2])
    t_c     = pars[3]
    Psi_c   = pars[4]

     # Standard GR PN terms
    Q_1, Q_1p5, Q_2, Q_2p5  = get_Qs(eta)

    dL      = DGW(z, freq, cosmo_params=cosmo_params)
    uo_arr  = np.pi * Mc * freq
    Mt      = Mc / eta**(3./5)     # in sec

    # generate phenomA frequency parameters
    f_merg = get_freq(Mt, eta, "merg")
    f_ring = get_freq(Mt, eta, "ring")
    sigma  = get_freq(Mt, eta, "sigma")
    f_cut  = get_freq(Mt, eta, "cut")

    # break up frequency array into pieces
    mask1 = (freq<=f_merg)
    mask2 = (freq>f_merg) & (freq<=f_ring)
    mask3 = (freq>f_ring) & (freq<=f_cut)

    C   = (Mc)**(5./6)/(dL *Mpc/c) /np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5./24)
    w   = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)
    Amp = np.zeros(len(freq))
    psi = np.zeros(len(freq))

    Amp[mask1] = C *(freq[mask1] /f_merg)**(-7./6)
    Amp[mask2] = C *(freq[mask2] /f_merg)**(-2./3)
    Amp[mask3] = C *w *Lorentzian(freq[mask3], f_ring, sigma)

    # the phase could use input cT to avoid derivation of cT in the phase
    psi[mask1]  = Psi_GR(freq[mask1], Mc, eta, z, t_c, Psi_c)

    dpsi_dlnu   = d_psi_dlnu_func(uo_arr[mask1][-1], Q_1, Q_1p5, Q_2, Q_2p5, 
                    t_c, freq[mask1][-1])

    dlnu_df     = 1./freq[mask1][-1]

    slope_new   = (dpsi_dlnu - 2*np.pi*freq[mask1][-1]*t_c)*dlnu_df +2*np.pi*t_c

    # the starting value of merger phase should not depend on pars
    psi[mask2] = psi[mask1][-1] + slope_new*(freq[mask2]-freq[mask1][-1])

    return Amp * np.exp(1.j*psi)


def d_psi_dlnu_func(uo_arr, Q_1, Q_1p5, Q_2, Q_2p5, t_c, farr):

    line1 = -3./8 + 1./2.*Q_1*uo_arr**(2./3) + 3./5*Q_1p5*uo_arr
    line2 = -3./4*(Q_1**2 - Q_2)*uo_arr**(4./3)
    line3 = ( -(2 *Q_1*Q_1p5 - Q_2p5) )*uo_arr**(5./3)

    return 5./48.*uo_arr**(-5./3)*(line1+line2+line3) + 2*np.pi*farr*t_c


### Computes the numerical derivative of a function
def numerical_derivative(func, par, dx, conv=5e-2, factor=.5, verbose=False):
    
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
    return 0.8* 2. *integrate.simps( np.real(
            (hf*np.conjugate(gf)+np.conjugate(hf)*gf))/psd, x=freqs)



def Fisher_der(farr, pars, cosmo_params=cosmo_params, which_wf='GR', 
        MG_flag=False, width=0, dist_corr=True):

    t0 = tm.perf_counter()
    
    Mz      = np.exp(pars[0]) * Msolar * GN/ (1e3 *c)**3
    eta     = np.exp(pars[1])
    zem     = np.exp(pars[2])
    tc      = pars[3]
    psic    = pars[4]
    c0      = pars[5]
    fstar   = pars[6]

    wf      =  h_Delta_inspiral(farr, pars, cosmo_params=cosmo_params, 
                width=width, cT_type=which_wf, dist_corr=dist_corr)

    # Function to compute numerical derivatives w.r.t. lnM
    lnA_lnMz    = lambda lnMz_func: np.log(amp_Delta_inspiral(farr, fstar, c0, 
                    np.exp(lnMz_func) * Msolar * GN/ (1e3 *c)**3, eta, zem, 
                    cosmo_params=cosmo_params, cT_type=which_wf, width=width))

    Psi_lnMz    = lambda lnMz_func: Psi_Delta_inspiral(farr, fstar, c0, 
                    np.exp(lnMz_func) * Msolar * GN/ (1e3 *c)**3, eta, zem, tc, 
                    psic, cosmo_params=cosmo_params, cT_type=which_wf, 
                    width=width)

    # Function to compute numerical derivatives w.r.t. lneta
    lnA_lneta   = 0

    Psi_lneta   = lambda lneta_func: Psi_Delta_inspiral(farr, fstar, c0, Mz, 
                    np.exp(lneta_func), zem, tc, psic, 
                    cosmo_params=cosmo_params, cT_type=which_wf, width=width)

    # Function to compute numerical derivatives w.r.t. c0
    lnA_c0  = lambda c0_func: np.log(amp_Delta_inspiral(farr, fstar, c0_func, 
                Mz, eta, zem, cosmo_params=cosmo_params, cT_type=which_wf, 
                width=width))

    Psi_c0  = lambda c0_func: Psi_Delta_inspiral(farr, fstar, c0_func, Mz, eta, 
                zem, tc, psic, cosmo_params=cosmo_params, cT_type=which_wf, 
                width=width)

    #Distance correlation term in the phase
    if dist_corr == True and max(farr)<fstar:
        Psi_dist_c0 = lambda c0_func: Psi_dist_corr(farr, fstar, c0_func, zem, 
                        cosmo_params=cosmo_params)

    # Analytical derivetives w.r.t. tc and psic
    dh_dtc      = 2.j *np.pi *farr *wf
    dh_dpsic    = -1.j *wf

    # Analytical derivetives w.r.t. ln DL
    dh_dlnDL    = -1. *wf
    
    lnA = [lnA_lnMz, lnA_lneta, lnA_c0]
    Psi = [Psi_lnMz, Psi_lneta, Psi_c0]

    derivatives = []
    verbose     = False
    
    for i in range(0, len(pars)-1):
        print('- Working on index', i)
        dx = np.abs(.1 *pars[i]) if pars[i] != 0 else 1e-1

        if i == 3:
            derivatives.append(dh_dtc)
        elif i == 4:
            derivatives.append(dh_dpsic)
        elif i == 5:
            if MG_flag:
                if min(farr)> fstar:
                    derivatives.append(np.zeros(len(farr)))
                else:
                    dlnA    = numerical_derivative(lnA[-1], pars[i], dx=dx, 
                                verbose=verbose)
                    if dist_corr == True:
                        dPsi_dist_dc0   = numerical_derivative(Psi_dist_c0, 
                                            pars[i], dx=dx, verbose=verbose)

                        derivatives.append((dlnA +1.j *dPsi_dist_dc0) *wf)  
                    else:
                        derivatives.append((dlnA) *wf)
        elif i == 2:
            if dist_corr == True and max(farr)<fstar:
            
                dist_val    =  dcom(zem, farr, cosmo_params=cosmo_params)
                dist_c_term = (2 *np.pi *farr *Mpc *dist_val /(1+zem) /c *(
                                1 -c0) *(1 -fstar /farr) *1.j *wf)

                derivatives.append(dh_dlnDL +dist_c_term)
            else:
                derivatives.append(dh_dlnDL)
                
        else:
            if i == 1:
                dlnA = 0
            else:
                dlnA = numerical_derivative(lnA[i], pars[i], dx=dx, 
                        verbose=verbose)

            dPsi    = numerical_derivative(Psi[i], pars[i], dx=dx, 
                        verbose=verbose)

            derivatives.append((dlnA + 1.j*dPsi) *wf)
    
    derivatives   = derivatives[:2][::-1] + derivatives[2:]
  
    print('This took %.2f seconds' % (tm.perf_counter() -t0) )

    return np.array(derivatives)


def Fisher_build(farr, psd, derivatives):

    Fisher_matrix = np.zeros((len(derivatives), len(derivatives)))

    for i in range(0, len(derivatives)):
        for j in range(0, i +1):
            Fisher_matrix[i,j] = scalar_product(derivatives[i], 
                                    derivatives[j], psd, farr)
            Fisher_matrix[j,i] = Fisher_matrix[i,j]
            
    return Fisher_matrix