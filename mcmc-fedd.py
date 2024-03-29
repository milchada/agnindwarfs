import numpy as np
import matplotlib.pylab as plt

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import default_cosmology

from scipy.stats import norm, bernoulli, cumfreq
from scipy.optimize import curve_fit

import pandas as pd
import corner, glob, multiprocessing, emcee
from multiprocessing import Pool
ncpu = multiprocessing.cpu_count()

lcdm = default_cosmology.get()

allpts = pd.read_csv('allpts.csv', header=0)
# amuse = pd.read_csv('AMUSEGalaxies.csv', header=0)
gama_sdss = pd.read_csv('gama_sdss_sigma.csv', header=0)

def f_occ(Mstar, Mstar0): #tested, works fine!
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def log_Mbh(vstar, alpha=8.33, beta=5.77, eps=0.43): #tested, works fine!
    average = alpha + beta*np.log10(vstar/200)
    return np.array([max(0,norm(loc = av, scale=eps).rvs(size=1)[0]) for av in average])    

def log_Mbh_Mstar(Mstar, alpha = 8.56, beta= 1.34, sigma = 0.17):
    average = alpha + beta*np.log10(Mstar/1e11)
    return average

def f_edd(Lx, vstar,kX = 0.1): #tested, works fine!
    Mbh = 10**log_Mbh(vstar)
    Ledd = 1e38 * Mbh
    return Lx/(kX*Ledd)

def pdf(fedd, beta, fedd_min, fedd_max, cum=False): #tested, works fine!
    if beta == 1: #otherwise /0 below
            beta += 1e-5
    if fedd < fedd_min:
        pdf = fedd_min**(-beta)
        if cum:
            return pdf*fedd 
        else:
            return pdf
    elif (fedd <= fedd_max):
        cdf_min = (fedd_min**(1-beta))/(1-beta)
        cdf_max = (fedd_max**(1-beta))/(1-beta)
        cdf_tot = cdf_max - cdf_min
        if cum:
            cdf = (fedd**(1-beta))/(1-beta) - cdf_min
            return cdf/cdf_tot
        else:
            p = fedd**-beta
            return p/cdf_tot
    else:
        return 0

def N(Lx, vstar, beta, fedd_min, fedd_max, fduty): #tested, works fine!
    #likelihood of Lx given Eddington ratio distribution
    fedd = f_edd(Lx, vstar)
    rand = np.random.randn(len(fedd))
    prob = np.zeros(len(fedd))
    prob[rand < fduty] =  [pdf(f, beta, fedd_min, fedd_max, cum=False) for f in fedd[rand < fduty]]
    return prob
    #this returns the PDF, which doesn't add up to 1. Instead, PDF(x)*dx adds up to 1. 

def log_likelihood(Lx, vstar, Mstar, beta, fedd_min, fedd_max, fduty, Mstar0): #same as in other models
    #works fine.. except there is a lot of -inf, so adding must be done carefully
    return np.log10(f_occ(Mstar, Mstar0)) + np.log10(N(Lx, vstar, beta, fedd_min, fedd_max, fduty))

def p_BH(Lx, vstar, Mstar, beta, fedd_min,fedd_max, fduty, Mstar0):
    #tested, works, but I'm not sure it's correct.
    fedd = f_edd(Lx, vstar)
    p_if_on = np.array([pdf(fi, beta, fedd_min, fedd_max, cum=True) for fi in fedd])
    Phi = p_if_on*fduty + (1 - fduty) #i.e. fduty chance that AGN is on but below detection limit, (1-fduty) chance that it is off; either way, BH exists. 
    focc = f_occ(Mstar, Mstar0)
    pbh = focc*Phi/((1 - focc) + focc*Phi)
    pbh[pbh < 0] = 0
    pbh[pbh > 1] = 1
    pbh[np.isnan(pbh)] = 0
    pbh[focc == 0] = 0
    return pbh #forces 0 < p_bh < 1

def latent(Lx, vstar, Mstar, Llim, beta, fedd_min, fedd_max, fduty, Mstar0):
    #tested, works fine!
    I_n = np.zeros(len(Mstar))
    p_true = p_BH(Lx, vstar, Mstar, beta, fedd_min, fedd_max, fduty, Mstar0)
    for i in range(len(I_n)):
        try:
            I_n[i] = bernoulli.rvs(p_true[i], size=1)
        except ValueError:
            I_n[i] = 0
    I_n[Lx > Llim] = 1
    return I_n

def log_prob_Mstar0(Mstar0, I_n, Mstar): 
#same as in other models, tested.
    focc = f_occ(Mstar, Mstar0)
    pi = np.power(focc, I_n) * np.power(1 - focc, 1-I_n)#np.power(, weights[i])
    return np.log10(p)

def log_prior(theta, priors):
    lnprior = 0
    for i in range(len(priors)):
        if (theta[i] < priors[i][0]) or (theta[i] > priors[i][1]):
            lnprior = -np.inf
    return lnprior

def log_pdf(theta, Lx, Mstar, vstar, Llim, priors, weights):
    beta, log_fedd_min, log_fedd_max, logMstar0 = theta
    Mstar0 = 10**logMstar0
    fedd_min = 10**log_fedd_min
    fedd_max = 10**log_fedd_max
    fduty = 1#0**log_fduty
    detect = (Lx > Llim)

    likely = log_likelihood(Lx, vstar, Mstar, beta, fedd_min, fedd_max, fduty, Mstar0)
    likely *= weights
    likely = np.sum(likely[(likely > -np.inf)*detect]) #sum over detections
    
    I_n = latent(Lx, vstar, Mstar, Llim, beta, fedd_min, fedd_max, fduty, Mstar0)
    prob = log_prob_Mstar0(Mstar0, I_n, Mstar) #this is where the infinities are happening
    prob *= weights
    prob = np.sum(prob[(prob > -np.inf) * ~detect]) #sum over non-detections

    logpdf = (likely + prob)/len(Lx) #sets to order 1
    
    if np.isnan(logpdf):
        print('PDF = 0')
        return -np.inf
    else:
        return logpdf + log_prior(theta, priors)

def seed(priors, nwalkers, ndim, right=True):
    p0 = np.random.uniform(low = priors[:,0], high = priors[:, 1], size=(nwalkers, ndim))
    if not right:
        p0 [:int(nwalkers/2)] /= 2
        p0 [int(nwalkers/2):] *= 2
    return p0

def run_emcee(galcat, filename, nwalkers=100, nsteps=int(1e4), nburn=100, 
                priors=np.array([[.1,4],[-6,-2],[0,1],[7,10]]), right=True):
    ndim = len(priors)
    Lx = galcat['Lx']
    Mstar = galcat['Mstar'] 
    vstar = galcat['vstar'] 
    Llim = galcat['Llim']
    try:
        weights = galcat['w_obs']
    except KeyError:
        weights = np.ones(len(Mstar))
    p0 = seed(priors, nwalkers, ndim, right=right)
    backend = emcee.backends.HDFBackend(filename)
    pool = Pool()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf, args=[Lx, Mstar, vstar, Llim, priors, weights],backend=backend, pool=pool)
    #burn in
    state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    return sampler
    
def check(filename, ax):
    # fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    reader = emcee.backends.HDFBackend(filename)
    samples = reader.get_chain()
    med = np.median(samples, axis=1)
    std = np.std(samples, axis=1)
    for i in range(med.shape[1]):
        ax.flatten()[i].cla()
        ax.flatten()[i].plot(np.arange(len(med)), med[:,i],color='tab:blue')
        ax.flatten()[i].plot(np.arange(len(med)), med[:,i]+std[:,i],color='tab:blue',linestyle='dotted')
        ax.flatten()[i].plot(np.arange(len(med)), med[:,i]-std[:,i],color='tab:blue',linestyle='dotted')

def plot(filename, figname):
	reader = emcee.backends.HDFBackend(filename)
	samples = reader.get_chain(flat=True)
	fig = corner.corner(samples,labels=[r'$\beta$', r'log($\lambda_{min}$)', r'log($\lambda_{max}$)', r'log(f$_{duty}$)', r'log($M_{*,0}$)'], 
        range=[[.1,4],[-10,-5],[0,1],[-1,0],[7,10]])
	fig.savefig(figname)