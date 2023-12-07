import numpy as np
import matplotlib.pylab as plt

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import default_cosmology

from scipy.stats import norm, bernoulli, cumfreq
from scipy.optimize import curve_fit

import pandas as pd
import corner, glob, multiprocessing, emcee, os, pickle
from multiprocessing import Pool

ncpu = multiprocessing.cpu_count()

lcdm = default_cosmology.get()

with open('/n/holylfs05/LABS/bhi/Lab/narayan_lab/angelos_sam_table_for_mila/tableForMila.pkl','rb') as f:
   d = pickle.load(f, encoding='latin1')
models = ['blq_popIII_pmerge0.1_072018', 'powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 
          'blq_dcbh_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
keys = ['logFinalHaloMass', 'm_halo', 'treeIndex', 'redshift', 'L_bol', 'cosmologicalWeight', 'm_bh', 
               'isASatellite', 'sigma', 'm_star']

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:blue', 'tab:green', 'tab:orange']
labels = ['Pop III + BLQ', 'Pop III + PL', 'Pop III + AGN-MS', 
          'DCBH + BLQ', 'DCBH + PL', 'DCBH + AGN-MS']

Lsun = 3.828e33 #erg/s
flim = 6.5e-15 #erg/cm**2/s

amuse = pd.read_csv('AMUSEGalaxies.csv', header=0)

def f_occ(Mstar, Mstar0):
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def N(Lx, Mstar, alpha, beta, sigma):
    #likelihood of Lx given scaling relation of slope beta, intercept alpha and scatter sigma
    mu = alpha + beta*np.log10(Mstar)
    rv = norm(loc=mu, scale=sigma)
    return rv.pdf(np.log10(Lx))
    #this returns the PDF, which doesn't add up to 1. Instead, PDF(x)*dx adds up to 1. 

def log_prob_detect(Lx, Mstar, alpha, beta, sigma, fduty, Mstar0):
    return np.log10(f_occ(Mstar, Mstar0)) + np.log10(N(Lx, Mstar, alpha, beta, sigma)) * np.log10(fduty)

def p_BH(Lxi, Mstari, alpha, beta, sigma, fduty, Mstar0):
    rv = norm(loc=0, scale=1)
    logLxPred = (alpha + beta*np.log10(Mstari))
    #here add duty cycle
    p_if_on = rv.cdf((np.log10(Lxi) - logLxPred)/sigma)
    Phi = p_if_on*fduty + (1 - fduty)
    focc = f_occ(Mstari, Mstar0)
#     print(1- focc, focc*Phi)
    return Phi/((1 - focc) + focc*Phi)

def latent(Lx, Mstar, Llim, alpha, beta, sigma, fduty, Mstar0):
    I_n = np.zeros(len(Mstar))
    for i in range(len(I_n)):
        if Lx[i] > Llim[i]:
            I_n[i] = 1
        else:
            p_true = p_BH(Lx[i], Mstar[i], alpha, beta, sigma, fduty, Mstar0)
            try:
                I_n[i] = bernoulli.rvs(p_true, size=1)
            except ValueError:
                if p_true < 0:
                    I_n[i] = 0
                else:
                    I_n[i] = 1
    return I_n

def log_prob_upperlimit(Mstar0, I_n, Mstar): #I_n is a list where for each galaxy I_n[i] = 1 if there is a BH and 0 otherwise
    focc = f_occ(Mstar, Mstar0)
    p = np.power(focc, I_n) * np.power(1 - focc, 1-I_n)
    return np.log10(p)

def log_prior(theta, priors):
    lnprior = 0
    for i in range(len(priors)):
        if (theta[i] < priors[i,0]) or (theta[i] > priors[i,1]):
            lnprior = -np.inf
    return lnprior

def log_pdf(theta, Lx, Mstar, Llim, fit, err, fduty, priors):
    beta, alpha = fit
    sigma = err
    logMstar0, fduty = theta
    Mstar0 = 10**logMstar0
    detect = (Lx > Llim)
    likely = log_prob_detect(Lx, Mstar, alpha, beta, sigma, fduty, Mstar0)
    likely = np.sum(likely[(likely > -np.inf)*(detect)]) #sum over detections

    I_n = latent(Lx, Mstar, Llim, alpha, beta, sigma, fduty, Mstar0)
    prob = log_prob_upperlimit(Mstar0, I_n, Mstar)
    prob = np.sum(prob[(prob > -np.inf)*(~detect)]) #sum over non-detections
    logpdf = (likely + prob)/len(Lx)
    print(likely, prob, logpdf)
    if np.isnan(logpdf) or logpdf == 0:
        return -np.inf
    else:
        return logpdf + log_prior(theta, priors)

def seed(priors, nwalkers):
    p0 = np.zeros((nwalkers, len(priors)))
    for i in range(len(priors)):
        p0[:, i] = np.random.uniform(low = priors[i, 0], high=priors[i, 1], size=nwalkers)
    return p0


def run_emcee(galcat, filename, fit, err, fduty, nwalkers=100, nsteps=int(1e4), nburn=100, Llimsurvey=1e38):
    Lx = galcat['Lx']
    Mstar = galcat['Mstar'] 
    Llim = galcat['Llim']
    priors = np.array([[7,10], [0.95*fduty, min(1.05*fduty, 1)]])
    p0 = seed(priors, nwalkers)
    backend = emcee.backends.HDFBackend(filename)
    pool = Pool()
    sampler = emcee.EnsembleSampler(p0.shape[0], p0.shape[1], log_pdf, args=[Lx, Mstar, Llim, fit, err, fduty, priors],backend=backend, pool=pool)
    #burn in
    state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    return sampler
    
