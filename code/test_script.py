import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d

### Choose here your obs frame parameters
z_val   = 0.09
m1      = 36
m2      = 29
phi0    = 0.
tc      = 0.

f_init  = 2e-2
#### Costants
#### This is c in m/s
c       = 299792458
#### This is GN in m^3 kg / s^2
GN      = 6.674e-11
#### This is Msun in kg
Msun    = 1.989e30
#### This is Mpc in m
Mpc     = 3.086e22
#### This is GN in m/Msun
GMsun   = Msun *GN /c**2
#### This is GN in s/Msun = 1 /Msun/Hz
GMsunHz = Msun *GN /c**3
#### Year in seconds
yr      = 365.25 *24 *3600
#### This is A prefactor for the amplitude
A       = 1 /np.pi**(2/3) *(5/24)**(1/2)

#### Somo cosmo params
h           = 0.7                    ## Dimensionless Hubble parameter
H0          = h *100 *1e3 /Mpc          ## Hubble in 1/s (1e3 converts km in m)
Omegab      = 0.02237                   ## Omega Baryons physical (includes h^2)
Omegac      = 0.1200                    ## Omega DM physical (includes h^2)
OmegaM      = 0.3158                    ## Omega Matter (does not include h^2)
OmegaR      = 5.04e-5                   ## Omega Rad (does not include h^2)
OmegaL      = 1 -OmegaM -OmegaR         ## Omega Lambda (does not include h^2)

#3H^2 = rho
#H = np.sqrt( rho / 3 ) = H0 * np.sqrt(rho/rho0)
Hubble      = lambda z : H0 *(OmegaR *(1+z)**4 +OmegaM *(1+z)**3 +OmegaL)**0.5

#a0 / a = 1+z
#d ln(a/a0) / dz = - d ln(1+z) / dz = - 1/(1+z)
#dt /dz = dt/dln(a) dln(a)/dz = - 1/H *  1/(1+z)
dtdz        = lambda z : - 1 /Hubble(z) /(1 +z)

### Dl definition
def Dc(z):
    z_vec   = np.logspace(-10, np.log10(z), 1000)
    return  simps(-dtdz(z_vec) * (1+z_vec), x=z_vec)

DlMpc   = lambda z : Dc(z) *(1+z) / Mpc *c
dl      = DlMpc(z_val)

#### Here I load the data and create an interpolator for the noise
data    = np.loadtxt('../data/LISA_strain.txt')
noise   = interp1d(data[:,0], data[:,1], bounds_error=False, fill_value=1e50)

#### Some functions
Mtot    = lambda m1, m2 : m1 +m2
mu      = lambda m1, m2 : (m1 *m2) /Mtot(m1, m2)
Mc      = lambda m1, m2 : Mtot(m1, m2)**(2/5) *mu(m1, m2)**(3/5)
MMtot   = lambda Mc, mu : (Mc *mu**(-3/5))**(5/2)
nu      = lambda Mc, mu : mu /MMtot(Mc, mu)
### nu is eta
#eta     = lambda Mc, mu : (MMtot(Mc, mu) / Mc )**(-5/3)

my_Mc   = Mc(m1, m2)
my_mu   = mu(m1, m2)
my_nu   = nu(my_Mc, my_mu)

### I drop all fs since they simplify
tau0    = lambda Mc, mu : 5/(256 *np.pi) *(np.pi *MMtot(Mc, mu) *GMsunHz
            )**(-5/3) / nu(Mc, mu)
Phase   = lambda f, Mc, mu, tc, phi0, Dl : 2 *np.pi *f *tc  +2 *np.pi *3 *tau0(
            Mc, mu) /5 *f**(-5/3) -np.pi /4 -phi0

### I tested that this is the same as phenomA (just one of the 2 polarizations)
Amp     = lambda f, Mc, Dl : A *(c /Dl /Mpc) *(GMsunHz *Mc)**(5/6) *f**(-7/6)

wf      = lambda f, Mc, mu, tc, phi0, Dl : Amp(f, Mc, Dl) *np.exp(1j *
            Phase(f, Mc, mu, tc, phi0, Dl) )

dlnwfMc = lambda f, Mc, mu, tc, phi0, Dl : 5 /6 /Mc -5j *(6 *(GMsunHz *Mc**(5/2)
            )**(1/3) +55 *f**(2/3) *GMsunHz *np.pi**(2/3) *mu**(3/2) )/(768
            *f**(5/3) *GMsunHz**2 *Mc**(7/2) *np.pi**(5/3))

dlnwfmu = lambda f, Mc, mu, tc, phi0, Dl : -5j *(743 -1386 *mu**(5/2) /Mc**(5/2)
            )/(32256 *f *GMsunHz *np.pi *mu**2)

dnudmu  = lambda Mc, mu : 5 *np.sqrt(Mc/mu**(3/5)) *mu**(9/5) /2 /Mc**3
ddldz    = lambda z : (Dc(z) +(1+z) *(-dtdz(z) *(1+z)) )/ Mpc *c

dlnwftc = lambda f, Mc, mu, tc, phi0, Dl : 2j *f *np.pi
dlnwfp  = lambda f, Mc, mu, tc, phi0, Dl : -1j *f**0
dlnwfDl = lambda f, Mc, mu, tc, phi0, Dl : -1/Dl *f**0

#dfuncs  = [dlnwfMc, dlnwfmu, dlnwftc, dlnwfp, dlnwfDl]
dfuncs  = [dlnwfmu, dlnwfMc, dlnwfDl, dlnwftc, dlnwfp]


r_ISCO  = lambda Mc, mu : 6 *MMtot(Mc, mu) *GMsunHz
f_ISCO  = lambda Mc, mu : np.sqrt(MMtot(Mc, mu) *GMsunHz /r_ISCO(Mc, mu)**3
            ) /2 /np.pi *2

myfISCO = f_ISCO(my_Mc, my_mu)
print('My f_ISCO', myfISCO)

#### To compute the time in band use 4.18 of Maggiore
#### \dot{f} = 96/5 * pi^{8/3} (G Mc / c^3)^{5/3} f^{11/3}
dfdt    = lambda f, Mc, mu : 96/5 * np.pi**(8/3) *(GMsunHz *Mc)**(5/3) *f**(
            11/3) *(1 -(743/336 +11/4 *nu(Mc, mu)) *nu(Mc, mu)**(-2/5) *(
                np.pi *f *GMsunHz *Mc)**(2/3) )
to_int  = lambda f, Mc, mu : 1./dfdt(f, Mc, mu)

freq    = np.logspace( np.log10(f_init),  np.log10(myfISCO), 100000)
t_merg0 = simps(to_int(freq, my_Mc, my_mu), x=freq) /yr
print('Time to merger from f = %.2f is = %.2f yrs' % (f_init, t_merg0))

my_h    = wf(freq, my_Mc, my_mu, tc, phi0, dl)

def scalar_product(hf, gf, strain, f):
    return 4. *simps( np.real(hf*np.conjugate(gf))/strain, x=f)

### This factor (which multiplies the wf) is the sqrt of what appears in h^2.
### There is a 4/5 due to the average in iota (which takes also care of
### the sum over polarizations)
#print( np.abs(my_h) )
#print(asdasd)

pref    = np.sqrt(4/5)
SNR     = pref**2 *scalar_product(my_h, my_h, noise(freq), freq)
print('SNR at LISA = ', np.sqrt(SNR), '\n')

params  = [dnudmu(my_Mc, my_mu)**(-1), my_Mc, ddldz(z_val) *z_val, 1, 1]
dvec    = np.array([pref *my_h *dfuncs[i](freq, my_Mc, my_mu, tc, phi0, dl
            ) *params[i] for i in range(0, len(dfuncs))])

elem    = 4. *np.real(dvec[:,None] *np.conjugate(dvec[None,:]))
Fisher  = simps( elem /noise(freq), x=freq, axis=-1)
cov     = np.linalg.inv(Fisher)

#print(Fisher)
print(cov[:2,:2])

dlnnu   = np.sqrt(cov[0,0])
dlnMc   = np.sqrt(cov[1,1])
dlnz    = np.sqrt(cov[2,2])

dtc     = np.sqrt(cov[-2,-2])


print('Mc = %.2f \pm %.5f and relative error = %.7f x 10^(-6)' % (
    my_Mc, dlnMc *my_Mc, 1e6 *dlnMc))
print('eta = %.2f \pm %.12f and relative error = %.7f ' % (my_nu, dlnnu *my_nu,
    dlnnu))
print('z = %.2f \pm %.2f and relative error = %.2f' % (z_val, dlnz *z_val, dlnz))
print('tc = %.2f \pm %.4f ' % (tc, dtc))


print('\nWhat I compute is when it merges after leaving the LISA band!')
t_merg  = simps(to_int(freq, my_Mc, my_mu), x=freq) /yr
#print('Time to merger from f = %.2f is = %.2f yrs' % (
 #           f_init, simps(to_int(freq, my_Mc*(1+dlnMc)), x=freq) /yr))

ddmc    = 1e-2
ddnu    = 1e-3
dM      = dlnMc *my_Mc
dnu     = dlnnu *my_nu

####  tc      = d int_f0^fisc to_int so I should differentiate the integrand
#### and the extrema dtc = (int (d to_int /dm) + to_int(f) df_isc/dm ) dm

### here i diff the integrands
dtoimc  = simps((to_int(freq, my_Mc +ddmc, my_mu) -to_int(
            freq, my_Mc -ddmc, my_mu)) /2 /ddmc, x=freq)
### I use dmu = dnu / dnudmu(my_Mc, my_mu)
dtoinu  = simps((to_int(freq, my_Mc, my_mu +ddnu /dnudmu(my_Mc, my_mu)) -to_int(
            freq, my_Mc, my_mu -ddnu /dnudmu(my_Mc, my_mu))) /2 /ddnu, x=freq)

### here i diff the extrema
dfImc   = (f_ISCO(my_Mc +ddmc, my_mu) -f_ISCO(my_Mc -ddmc, my_mu)
            ) /2 /ddmc *to_int(myfISCO, my_Mc, my_mu)

dfInu   = (f_ISCO(my_Mc, my_mu +ddnu /dnudmu(my_Mc, my_mu)) -f_ISCO(my_Mc,
            my_mu -ddnu /dnudmu(my_Mc, my_mu))) /2 /ddnu *to_int(
            myfISCO, my_Mc, my_mu)

dtdM    = dtoimc +dfImc
dtnu    = dtoinu +dfInu

print(dtdM**2 *dM**2, dtnu**2 *dnu**2, 2 *dtdM *dtnu* cov[0,1] *my_Mc *my_nu)
dt      = np.sqrt( dtdM**2 *dM**2 + dtnu**2 *dnu**2
            +2 *dtdM *dtnu* cov[0,1] *my_Mc *my_nu)

print('Time delta t merger in secs %.2f' % (np.abs( dt )))


MMtot_nu   = lambda Mc, nu : Mc*nu**(-3./5)
r_ISCO_nu  = lambda Mc, nu : 6 *MMtot_nu(Mc, nu) *GMsunHz
f_ISCO_nu  = lambda Mc, nu : np.sqrt(MMtot_nu(Mc, nu) *GMsunHz /r_ISCO_nu(Mc, nu)**3
            ) /2 /np.pi *2

print('\nThis is what Alberto predicts:')
fISCOp  = f_ISCO_nu(my_Mc *(1+1e-6), my_nu*(1+8e-3))
freq    = np.logspace(np.log10(f_init), np.log10(fISCOp), 100000)
print('Time delta t merger in secs %.2f' % (np.abs(t_merg*yr -simps(to_int(
        freq, my_Mc *(1+1e-6), my_mu), x=freq))))

fISCOm  = f_ISCO(my_Mc *(1-1e-6), my_mu)
freq    = np.logspace(np.log10(f_init), np.log10(fISCOm), 100000)
print('Time delta t merger in secs %.2f' % (np.abs(t_merg*yr -simps(to_int(
        freq, my_Mc *(1-1e-6), my_mu), x=freq))))
