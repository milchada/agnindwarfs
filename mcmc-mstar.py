import numpy as np
import matplotlib.pylab as plt

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import default_cosmology

from scipy.stats import norm, bernoulli, cumfreq
from scipy.optimize import curve_fit

import pandas as pd
import glob, multiprocessing, emcee
from multiprocessing import Pool
ncpu = multiprocessing.cpu_count()

lcdm = default_cosmology.get()
#amuse = pd.read_csv('AMUSEGalaxies.csv', header=0)
allpts = pd.read_csv('allpts.csv', header=0)
gama_sdss = pd.read_csv('gama_sdss_all.csv', header=0)

def f_occ(Mstar, Mstar0):
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def N(Lx, Mstar, alpha, beta, sigma):
    #likelihood of Lx given scaling relation of slope beta, intercept alpha and scatter sigma
    mu = alpha + beta*np.log10(Mstar)
    rv = norm(loc=mu, scale=sigma)
    return rv.pdf(np.log10(Lx))
    #this returns the PDF, which doesn't add up to 1. Instead, PDF(x)*dx adds up to 1. 

def log_likelihood(Lx, Mstar, alpha, beta, sigma, Mstar0):
    return np.log10(f_occ(Mstar, Mstar0)) + np.log10(N(Lx, Mstar, alpha, beta, sigma))
#            * (1 - f_occ(Mstar, Mstar0))*np.kron(np.log10(Lx) + 1e4)      when is Lx going to be 1e4?? pretty nonsensical I think. 

def p_BH(Lxi, Mstari, alpha, beta, sigma, Mstar0):
    rv = norm(loc=0, scale=1)
    logLxPred = (alpha + beta*np.log10(Mstari))
    Phi = rv.cdf((np.log10(Lxi) - logLxPred)/sigma)
    focc = f_occ(Mstari, Mstar0)
#     print(1- focc, focc*Phi)
    return Phi/((1 - focc) + focc*Phi)

def latent(Lx, Mstar, Llim, alpha, beta, sigma, Mstar0):
    I_n = np.zeros(len(Mstar))
    for i in range(len(I_n)):
        if Lx[i] > Llim[i]:
            I_n[i] = 1
        else:
            p_true = p_BH(Lx[i], Mstar[i], alpha, beta, sigma, Mstar0)
            if (p_true < 0) or not np.isfinite(p_true):
                p_true = 0
            if p_true > 1:
                p_true = 1
#             print(p_true)
            try:
                I_n[i] = bernoulli.rvs(p_true, size=1)
            except ValueError:
                I_n[i] = 0
    return I_n

def log_prob_Mstar0(Mstar0, I_n, Mstar): #I_n is a list where for each galaxy I_n[i] = 1 if there is a BH and 0 otherwise
    focc = f_occ(Mstar, Mstar0)
    p = np.power(focc, I_n) * np.power(1 - focc, 1-I_n)
    return np.log10(p)

def log_prior(theta, priors):
    lnprior = 0
    for i in range(len(priors)):
        if (theta[i] < priors[i][0]) or (theta[i] > priors[i][1]):
            lnprior = -np.inf
    return lnprior

def log_pdf(theta, Lx, Mstar, Llim, weights, priors):
    alpha, beta, sigma, logMstar0 = theta
    Mstar0 = 10**logMstar0
    detect = (Lx > Llim)
    likely = log_likelihood(Lx, Mstar, alpha, beta, sigma, Mstar0)
    likely = np.sum((likely*weights)[detect])
    print("Log likelihood: ", likely/len(Lx))
    I_n = latent(Lx, Mstar, Llim, alpha, beta, sigma, Mstar0)
    print("f_BH: ", sum(I_n)/len(Lx)) #why is this always 1 for everyone?
    prob = log_prob_Mstar0(Mstar0, I_n, Mstar)
    prob = np.sum((prob*weights)[~detect])
    logpdf = (likely + prob)/len(Lx)
    if np.isnan(logpdf):
        return -np.inf
    else:
        return logpdf + log_prior(theta, priors)

def seed(priors, nwalkers, ndim, right=True):
    p0 = np.random.randn(nwalkers, ndim)
    if right:    
        for i in range(ndim):
            prange = (p0[:,i].max() - p0[:,i].min())
            p0[:,i] *= (priors[i,1] - priors[i,0])/prange
            p0[:,i] += (priors[i,0] - p0[:,i].min())
    else:
        for i in range(ndim):
            p0[:int(nwalkers/2.),i] += priors[i,0]
            p0[int(nwalkers/2.):,i] += priors[i,1]
    return p0

#assume a contamination fraction f
#set f*Ntot detections to upper limits
def create_matched_sample(obs, sam, dm=0.2):
   mobs = obs['Mstar']
   msam = sam['Mstar']
   cat = []
   for m in mobs:
      msub = sam[(msam > (1-dm)*m)*(msam < (1+dm)*m)]
      if len(msub) > 1:
         ind = np.random.randint(0, len(msub)-1)
         cat.append(msub.index[ind])
      else:
         cat.append(msub.index)
   match = sam.iloc[cat]
   match.index = np.arange(len(match))
   return match

def run_emcee(galcat, filename, nwalkers=100, ndim=4, nsteps=int(1e4), nburn=100, 
                priors=np.array([[32,40],[.1,2],[.1,3],[7,10]]), right=True):
    Lx = galcat['Lx']
    Mstar = galcat['Mstar'] 
    Llim = galcat['Llim']
    try:
        weights = galcat['w_obs']
    except KeyError:
        weights = np.ones(len(Lx))
    if max(Lx) < 100:
        Lx = 10**Lx
    if max(Mstar) < 100:
        Mstar = 10**Mstar
    if max(Llim) < 100:
        Llim = 10**Llim
    p0 = seed(priors, nwalkers, ndim, right=right)
    backend = emcee.backends.HDFBackend(filename)
    pool = Pool()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf, args=[Lx, Mstar, Llim, weights, priors],backend=backend, pool=pool)
    #burn in
    state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    return sampler

if __name__ == "__main__":
    files = glob.glob('agnms*csv'); files.sort()
    for file in files:
        g = pd.read_csv(file)
        g = g[g['Mstar'] > 1e8]
        g['Llim'] = 1e34
        g.index = np.arange(len(g))
        name = file.split('_pme')[0]
        run_emcee(g, 'lx-mstar-%s-flim1e34.h5' % name, nsteps=2000, nwalkers=48,nburn=1)  
