### Common functions used in notebook

import numpy as np
import scipy
from scipy import integrate
from scipy.misc import derivative
from scipy.special import comb

# define constants
c = 299792.458           #km/s
GN = 6.674e-11     #m^3/kg/s^2
Msolar = 1.989e30  # kg
Mpc = 3.086e19     # km

# Hubble parameter
def Hubble(z, cosmo_params):
    Om0 = cosmo_params['OmegaM']
    OL0 = cosmo_params['OmegaDE']
    Or0 = cosmo_params['OmegaR']
    Hub = cosmo_params['h']

    # Define this was so can choose to absorb h into units if desired.
    return Hub*100*np.sqrt(Om0*(1+z)**3 + OL0 + Or0*(1+z)**4)

# dark energy
def OmegaL(z,cosmo_params, Hub):
    OL0 = cosmo_params['OmegaDE']
    E2 = (Hubble(z, cosmo_params, Hub)/100./Hub)**2
    return OL0/E2

# Integrand in computing the comoving distance
def dcom_int(z, f, cosmo_params):
    Hubfunc = Hubble(z, cosmo_params)

    ceetee = np.array([1.])

    if np.size(f)>1:
        result = ceetee[None,:]/Hubfunc[:,None]
    else:
        result = ceetee/Hubfunc

    return result

# compute the comoving distance
def dcom(zem, f, cosmo_params):
    #dcomresult = scipy.integrate.quad(dcom_int,0,zem,args=(f, fstar, Bcoeffs_vec, nvals, cosmo_params, Bfunc, GRflag))[0]
    #return c*dcomresult                #In Mpc since H is in km/s/Mpc

    ### Works with vectors
    z_vec       = np.linspace(0, zem, 1000)
    to_int      = dcom_int(z_vec, f, cosmo_params)
    Dcomresult  = integrate.simps(to_int, x=z_vec, axis=0)
    return c *Dcomresult

# GW luminosity distance
def DGW(zem, f, cosmo_params):

    zfac = (1 + zem)

    rcom = dcom(zem, f, cosmo_params)

    return (zfac*rcom)               #In Mpc

# scalar product in log frequency space
def scalar_product_log(hf, gf, psd, freqs):
    logf = np.log10(freqs)
    dlogf = logf[1]-logf[0]
    summ = 0.8*2.*np.log(10)* integrate.simps(np.real((hf*np.conjugate(gf)+np.conjugate(hf)*gf)/psd*freqs))*dlogf
    return summ

def scalar_product(hf, gf, psd, freqs):
    return 0.8* 2. *integrate.simps( np.real((hf*np.conjugate(gf)+np.conjugate(hf)*gf))/psd, x=freqs)

# Compute PhenomA waveform given in Robson et al.(2018)
para_a = np.array([2.9740e-1, 5.9411e-1, 5.0801e-1, 8.4845e-1])
para_b = np.array([4.4810e-2, 8.9794e-2, 7.7515e-2, 1.2848e-1])
para_c = np.array([9.5560e-2, 1.9111e-1, 2.2369e-2, 2.7299e-1])

# PhenomA waveform parameters given in LALSimIMRPhenom.c
# para_a = np.array([6.6389e-01, 1.3278e+00, 1.1383e+00, 1.7086e+00])
# para_b = np.array([-1.0321e-01, -2.0642e-01, -1.7700e-01, -2.6592e-01])
# para_c = np.array([1.0979e-01, 2.1957e-01, 4.6834e-02, 2.8236e-01])

def Lorentzian(f, f_ring, sigma):
    """ """
    return sigma/(2*np.pi)/( (f-f_ring)**2 + 0.25*sigma**2 )

def get_freq(M, eta, name):
    """ """
    if (name == "merg"):
        idx = 0
    elif (name == "ring"):
        idx = 1
    elif (name == "sigma"):
        idx = 2
    elif (name == "cut"):
        idx = 3

    result = para_a[idx]*eta**2 + para_b[idx]*eta + para_c[idx]

    return result/(np.pi*M)

def inspiral_fc(Mtot_z):     # input redshifted total mass in second
    r_ISCO = 6 * (Mtot_z)    # in sec
    f_ISCO = np.sqrt(Mtot_z/r_ISCO**3) / 2/np.pi      # in 1/sec

    fc = 2*f_ISCO
    return fc

def Amplitude(f, M, eta, Dl, PhenomFlag):

    Mtot = M*eta**(-3./5)

    if PhenomFlag==True:
        # generate phenomA frequency parameters
        f_merg = get_freq(Mtot, eta, "merg")
        f_ring = get_freq(Mtot, eta, "ring")
        sigma  = get_freq(Mtot, eta, "sigma")
        f_cut  = get_freq(Mtot, eta, "cut")

        # break up frequency array into pieces
        mask1 = (f<=f_merg)
        mask2 = (f>f_merg) & (f<=f_ring)
        mask3 = (f>f_ring) & (f<f_cut)

        C = M**(5./6)/Dl/np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5./24)
        # C = M**(5./6)/Dl/np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(1./30)
        w = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)

        Amp = np.zeros(len(f))

        Amp[mask1] = C[mask1]*(f[mask1]/f_merg)**(-7./6)
        Amp[mask2] = C[mask2]*(f[mask2]/f_merg)**(-2./3)
        Amp[mask3] = C[mask3]*w*Lorentzian(f[mask3], f_ring, sigma)

    elif PhenomFlag==False:
        us_arr = np.pi * M * f
        Amp = np.sqrt(5.*np.pi/24) * M**2 / Dl * us_arr**(-7./6)           # s

    return Amp

# analytical phase up to 2.5PN
# TB: Currently spin parameters beta_spin and sigma set to zero.
# Now the input mass is redshifted
def phase_PN(farr, Mz, eta, zem, t_c=0, Psi_c=0, free_Psictc=True):

    # Standard GR PN terms
    Q_1 = - (743./336 + 11./4*eta)*eta**(-2./5)
    Q_1p5 = (4*np.pi-0) *eta**(-3./5)
    Q_2 = (34103./18144 + 13661./2016*eta + 59./18*eta**2) *eta**(-4./5)
    Q_2p5 = -np.pi/eta*(4159./672+189./8*eta)

    uarr = np.pi * Mz * farr

    fc = farr[-1]
    uc = uarr[-1]

    if free_Psictc == False:
        t_c = - 5*Mz/96 * uc**(-8./3) * ( -3./8 + 1./2*Q_1*uc**(2./3) + 3./5*Q_1p5*uc - 3./4*(Q_1**2 - Q_2)*uc**(4./3) - (2*Q_1*Q_1p5 - Q_2p5)*uc**(5./3) )

        Psi_c = 5./48 *uc**(-5./3) *( 9./40 - 1./2*Q_1*uc**(2./3) - 9./10*Q_1p5*uc + 9./4*(Q_1**2 - Q_2) *uc**(4./3) - (2*Q_1*Q_1p5 - Q_2p5)*uc**(5./3)*np.log(uc) ) + 2*np.pi*fc*t_c

    Psi = 5./48 *uarr**(-5./3) *( 9./40 - 1./2*Q_1*uarr**(2./3) - 9./10*Q_1p5*uarr + 9./4*(Q_1**2 - Q_2) *uarr**(4./3) - (2*Q_1*Q_1p5 - Q_2p5)*uarr**(5./3)*np.log(uarr) ) + 2*np.pi*farr*t_c - Psi_c - np.pi/4

    return Psi

#numerically compute GW phase
def Psi_GR(fo, Mz, eta, zem, tc, psic):

    Q_1 = - (743./336 + 11./4*eta)*eta**(-2./5)
    Q_1p5 = (4*np.pi-0) *eta**(-3./5)
    Q_2 = (34103./18144 + 13661./2016*eta + 59./18*eta**2) *eta**(-4./5)
    Q_2p5 = -np.pi/eta*(4159./672+189./8*eta)

    uo_arr = np.pi * Mz * fo

    to_int = 5*np.pi*(Mz)**2/96 * uo_arr**(-11./3) * (1 - Q_1*uo_arr**(2./3) - Q_1p5*uo_arr + (Q_1**2-Q_2)*uo_arr**(4./3) + (2*Q_1*Q_1p5-Q_2p5)*uo_arr**(5./3))

    t_o = np.zeros(len(fo))
    Psi = np.zeros(len(fo))

    for i in range(len(fo)):
        t_o[i] = integrate.simps(-to_int[i:], x=fo[i:])

    Psi_int = 2*np.pi * t_o

    for i in range(len(fo)):
        Psi[i] = integrate.simps(-Psi_int[i:], x=fo[i:]) - np.pi/4 + 2*np.pi*fo[i]*tc - psic

    return Psi

# Now the input mass is redshifted
def waveform(freq, pars, cosmo_params):

    Mc          = np.exp(pars[0]) * Msolar * GN/ (1e3 *c)**3
    eta         = np.exp(pars[1])
    z           = np.exp(pars[2])
    t_c          = pars[3]
    Psi_c       = pars[4]

     # Standard GR PN terms
    Q_1 = - (743./336 + 11./4*eta)*eta**(-2./5)
    Q_1p5 = (4*np.pi-0) *eta**(-3./5)
    Q_2 = (34103./18144 + 13661./2016*eta + 59./18*eta**2) *eta**(-4./5)
    Q_2p5 = -np.pi/eta*(4159./672+189./8*eta)

    dL          = DGW(z, freq, cosmo_params)
    uo_arr       = np.pi * Mc * freq

    Mt          = Mc / eta**(3./5)     # in sec

    # generate phenomA frequency parameters
    f_merg = get_freq(Mt, eta, "merg")
    f_ring = get_freq(Mt, eta, "ring")
    sigma  = get_freq(Mt, eta, "sigma")
    f_cut  = get_freq(Mt, eta, "cut")

    # break up frequency array into pieces
    mask1 = (freq<=f_merg)
    mask2 = (freq>f_merg) & (freq<=f_ring)
    mask3 = (freq>f_ring) & (freq<=f_cut)

    C = (Mc)**(5./6)/(dL *Mpc/c) /np.pi**(2./3.)/f_merg**(7./6)*np.sqrt(5./24)

    w = 0.5*np.pi*sigma*(f_ring/f_merg)**(-2./3)

    Amp = np.zeros(len(freq))
    psi = np.zeros(len(freq))

    Amp[mask1] = C*(freq[mask1]/f_merg)**(-7./6)
    Amp[mask2] = C*(freq[mask2]/f_merg)**(-2./3)
    Amp[mask3] = C*w*Lorentzian(freq[mask3], f_ring, sigma)

    # the phase could use input cT to avoid derivation of cT in the phase
    psi[mask1] = Psi_GR(freq[mask1], Mc, eta, z, t_c, Psi_c)

    dpsi_dlnu = d_psi_dlnu_func(uo_arr[mask1][-1], Q_1, Q_1p5, Q_2, Q_2p5, t_c, freq[mask1][-1])

    dlnu_df = 1./freq[mask1][-1]

    slope_new = (dpsi_dlnu - 2*np.pi*freq[mask1][-1]*t_c)*dlnu_df + 2*np.pi*t_c

    # the starting value of merger phase should not depend on pars
    psi[mask2] = psi[mask1][-1] + slope_new*(freq[mask2]-freq[mask1][-1])

    return Amp * np.exp(1.j*psi)

def d_psi_dlnu_func(uo_arr, Q_1, Q_1p5, Q_2, Q_2p5, t_c, farr):

    line1 = -3./8 + 1./2.*Q_1*uo_arr**(2./3) + 3./5*Q_1p5*uo_arr
    line2 = -3./4*(Q_1**2 - Q_2)*uo_arr**(4./3)
    line3 = ( -(2 *Q_1*Q_1p5 - Q_2p5) )*uo_arr**(5./3)

    result = 5./48.*uo_arr**(-5./3)*(line1+line2+line3) + 2*np.pi*farr*t_c

    return result

# EFT case
def left_d(func, x0, dx, order=5):
    ## This is a derivative computed with x - i dx ()
    coeffs  = [-25./12, 4., -3., +4./3, -1./4]
    val     = 0
    for i in range(0, len(coeffs)):
        val     -= coeffs[i] *func(x0 -i*dx)
    return np.array(val) / dx

def F(x,c0):
    return x**2/(1 + x**2 - (1 + 2 * (1 - c0**2) * x**2)**(1/2))**(1/2)

def coefficients_fo(x,c0,z):
    a = 1
    b = -2 * (F(x, c0)*(1 + z))**2
    c = (F(x, c0)*(1 + z))**2 * ((F(x, c0)*(1 + z))**2 - 2)
    d = 2 * c0**2 * (F(x, c0)*(1 + z))**4
    return a,b,c,d

def cT_EFT(fo, fstar, c0):
    f_rat = fo/fstar
    return np.sqrt(1 + 1/f_rat**2 - 1/f_rat**2*np.sqrt(1+2*(1-c0**2)*f_rat**2))

def delta_EFT(fo,c0,z, fstar):
    delta=[]
    fo_rat = fo/fstar
    for ii in fo_rat:
        dd= 1 - (1+z)*ii/(np.roots(coefficients_fo(ii,c0,z)))**(1./2)
        delta.append(np.nanmin(dd))
    return np.array(delta)

# phase for exact Delta
def Psi_Delta_exact(fo, fstar, c0, Mz, eta, zem, cosmo, tc, psic, cT_type='EFT', width=0):

    Q_1 = - (743./336 + 11./4*eta)*eta**(-2./5)
    Q_1p5 = (4*np.pi-0) *eta**(-3./5)
    Q_2 = (34103./18144 + 13661./2016*eta + 59./18*eta**2) *eta**(-4./5)
    Q_2p5 = -np.pi/eta*(4159./672+189./8*eta)

    if cT_type=='GR':
        Delta = 0
        d_Delta_dfo = 0
    elif cT_type=='EFT':
        Delta = delta_EFT(fo,c0,zem,fstar)
        d_Delta_dfo_func = lambda fo_func: delta_EFT(fo_func, c0, zem, fstar)
        d_Delta_dfo = derivative(d_Delta_dfo_func, fo, fo*1e-3)
    elif cT_type=='step':
        Delta = Delta_step(fo,c0,zem,fstar,width)
        d_Delta_dfo_func = lambda fo_func: Delta_step(fo_func, c0, zem, fstar, width)
        d_Delta_dfo = derivative(d_Delta_dfo_func, fo, fo*1e-3)
    else:
        Delta = 0.
        print('No Delta setting detected in Psi_Delta_exact.')

    Mo_arr = Mz / (1-Delta)

    uo_arr = np.pi * Mo_arr * fo

    to_int = 5*np.pi*(Mz)**2/96 / (1-Delta)**2 * (1 + fo/(1-Delta)*d_Delta_dfo) * uo_arr**(-11./3) * (1 - Q_1*uo_arr**(2./3) - Q_1p5*uo_arr + (Q_1**2-Q_2)*uo_arr**(4./3) + (2*Q_1*Q_1p5-Q_2p5)*uo_arr**(5./3))

    t_o = np.zeros(len(fo))
    Psi = np.zeros(len(fo))
    Psi_old = np.zeros(len(fo))

    for i in range(len(fo)):
        t_o[i] = integrate.simps(-to_int[i:], x=fo[i:])

    Psi_int = 2*np.pi * t_o

    for i in range(len(fo)):
    
        # dist_val =  dcom(zem, fo[i], cosmo)                        # TB Does this need to be in MG?
        Psi[i] = integrate.simps(-Psi_int[i:], x=fo[i:]) - np.pi/4 + 2*np.pi*fo[i]*tc - psic
        # Psi[i] = integrate.simps(-Psi_int[i:], x=fo[i:]) - np.pi/4 + 2*np.pi*fo[i]*(tc + Mpc *dist_val/c*(1-c0)*(1-fstar/fo[i])*np.heaviside(fstar-fo[i],1)) - psic

    return Psi

# distance correlation term in the phase
def Psi_dist_corr(fo, fstar, c0, zem, cosmo_params):

    result = np.zeros(len(fo))
    for i in range(len(fo)):
        dist_val =  dcom(zem, fo[i], cosmo_params)

        result[i] = 2*np.pi*fo[i] * Mpc *dist_val/c*(1-c0)*(1-fstar/fo[i])*np.heaviside(fstar-fo[i],1)

    return result

# amplitude for exact Delta
def amp_Delta_exact(fo, fstar, c0, Mz, eta, zem, cosmo_params, cT_type='EFT', width=0):

    if cT_type=='GR':
        Delta = 0
        cT_fo = 1
    elif cT_type=='EFT':
        Delta = delta_EFT(fo,c0,zem, fstar)
        cT_fo = cT_EFT(fo, fstar, c0)
    elif cT_type=='step':
        Delta = Delta_step(fo,c0,zem, fstar, width)
        cT_fo = cT_step(fo, fstar, c0, width)

    dl_GR = DGW(zem, fo, cosmo_params)
    dl_MG = dl_GR / np.sqrt(1-Delta) * cT_fo
    amp_MG = Amplitude(fo, Mz, eta, dl_MG*Mpc/c, False) * (1-Delta)**(2./3)

    return amp_MG

class waveform_delta(object):

    def __init__(self, cT_type, width=0):

        self.cT_type = cT_type
        self.width = width

    def h_Delta_exact(self, fo, pars, cosmo_params, dist_corr=True):

        Mz          = np.exp(pars[0]) * Msolar * GN/ (1e3 *c)**3
        eta         = np.exp(pars[1])
        zem           = np.exp(pars[2])
        tc          = pars[3]
        psic       = pars[4]
        c0h = pars[5]
        fstarh = pars[6]

        amp = amp_Delta_exact(fo, fstarh, c0h, Mz, eta, zem, cosmo_params, self.cT_type, self.width)

        if dist_corr==True:
            Psi = Psi_Delta_exact(fo, fstarh, c0h, Mz, eta, zem, cosmo_params, tc, psic, self.cT_type, self.width) + Psi_dist_corr(fo, fstarh, c0h, zem, cosmo_params)
        else:
            Psi = Psi_Delta_exact(fo, fstarh, c0h, Mz, eta, zem, cosmo_params, tc, psic, self.cT_type, self.width)
    
        return amp * np.exp(1.j*Psi)


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result


def cT_step(farr, fstar, c0, width):

    logf = np.log10(farr)

    step = smoothstep(x=logf, x_min=np.log10(fstar)-width, x_max=np.log10(fstar)+width, N=5)

    cT_val = (1-c0)*step+c0

    return cT_val

def Delta_step(farr, c0h,zem,fstar, width):

    cT_step_fo = cT_step(farr, fstar, c0h, width)
    cT_step_fs = cT_step(farr*(1+zem), fstar, c0h, width)

    return 1-cT_step_fo/cT_step_fs

def ellipse_para(sigma_x_sq, sigma_y_sq, sigma_xy):

    a_sq = (sigma_x_sq+sigma_y_sq)/2 + np.sqrt((sigma_x_sq-sigma_y_sq)**2/4 + sigma_xy**2)
    b_sq = (sigma_x_sq+sigma_y_sq)/2 - np.sqrt((sigma_x_sq-sigma_y_sq)**2/4 + sigma_xy**2)
    #theta = np.arctan(2*sigma_xy / np.abs(sigma_x_sq-sigma_y_sq)) / 2
    theta = np.arctan(2*sigma_xy / (sigma_x_sq-sigma_y_sq)) / 2

    return np.sqrt(a_sq), np.sqrt(b_sq), theta*180/np.pi
