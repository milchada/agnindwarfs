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
amuse = pd.read_csv('AMUSEGalaxies.csv', header=0)
gama = pd.read_csv('gama_all.csv', header=0)
gama_sdss = pd.read_csv('gama_sdss_all.csv', header=0)

def f_occ(Mstar, Mstar0):
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def N(Lx, Mstar, alpha, beta, sigma):
    #likelihood of Lx given scaling relation of slope beta, intercept alpha and scatter sigma
    mu = alpha + beta*np.log10(Mstar)
    rv = norm(loc=mu, scale=sigma)
    return rv.pdf(np.log10(Lx))

def log_likelihood(Lx, Mstar, alpha, beta, sigma, Mstar0):
    return np.log10(f_occ(Mstar, Mstar0)) + np.log10(N(Lx, Mstar, alpha, beta, sigma))
#            * (1 - f_occ(Mstar, Mstar0))*np.kron(np.log10(Lx) + 1e4)      when is Lx going to be 1e4?? pretty nonsensical I think. 

def p_BH(Lxi, Mstari, alpha, beta, sigma, Mstar0):
    rv = norm(loc=0, scale=1)
    logLxPred = (alpha + beta*np.log10(Mstari))
    Phi = norm.cdf((np.log10(Lxi) - logLxPred)/sigma)
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
    p = 1
    for i in range(len(I_n)):
        focc = f_occ(Mstar[i], Mstar0)
        p *= (np.power(focc, I_n[i]) * np.power(1 - focc, 1-I_n[i]))
    return np.log10(p)


# def proposal(theta, i):
#     if i ==0:
#         low = 32; high = 40
#     elif i == 1:
#         low = 0.1; high = 2
#     elif i == 2:
#         low = 0.1; high = 3
#     else:
#         low = 7; high = 10
#     rand = norm(loc=theta, scale = (high-low)/5.) #well this doesn't take into account the previous step. that's why.
#     rv = rand.rvs(1)
# #     print(rv)
#     while (rv < low) or (rv > high):
#         rv = rand.rvs(1)
#     return rv

# it's because the prior is gone! it was built into the proposal, which emcee doesn't use. 

def log_prior(theta, priors):
	if (theta[0] < priors[0][0]) or (theta[0] > priors[0][1]) or (theta[1] < priors[1][0]) or (theta[1] > priors[1][1]) or (theta[2] < priors[2][0]) or (theta[2] > priors[2][1]) or (theta[3] < priors[3][0]) or (theta[3] > priors[3][1]):
		return -np.inf
	else:
		return 0

def log_pdf(theta, Lx, Mstar, Llim, priors):
    alpha, beta, sigma, logMstar0 = theta
    Mstar0 = 10**logMstar0
    likely = np.sum(log_likelihood(Lx, Mstar, alpha, beta, sigma, Mstar0))
    print("Log likelihood: ", likely/len(Lx))
    I_n = latent(Lx, Mstar, Llim, alpha, beta, sigma, Mstar0)
    print("f_BH: ", sum(I_n)/len(Lx)) #why is this always 1 for everyone?
    prob = log_prob_Mstar0(Mstar0, I_n, Mstar)
    logpdf = (likely + prob)/len(Lx)
    if np.isnan(logpdf):
        return -np.inf
    else:
        return logpdf + log_prior(theta, priors)

def run_emcee(Lx, Mstar, Llim, filename, nwalkers=100, ndim=4, nsteps=int(1e4), nburn=100, priors=[[32,40],[.1,2],[.1,3],[7,10]]):
    p0 = np.random.randn(nwalkers, ndim)
    low = [32, .1, .1, 7]
    high = [40, 2, 3, 10]
    for i in range(ndim):
        prange = (p0[:,i].max() - p0[:,i].min())
        p0[:,i] *= (high[i] - low[i])/prange
        p0[:,i] += (low[i] - p0[:,i].min())
    backend = emcee.backends.HDFBackend(filename)
    pool = Pool()
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf, args=[Lx, Mstar, Llim, priors],backend=backend, pool=pool)
    #burn in
    state = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    return sampler
    
#so just run run_emcee(Lx, Mstar, Llim)

def plot(filename, figname):
	reader = emcee.backends.HDFBackend(filename)
	samples = reader.get_chain(flat=True)
	fig = corner.corner(samples,labels=[r"$\alpha$", r"$\beta$", r"$\sigma$", r"$M_{*,0}$"], range=[[20,50],[-5,5],[-2,6],[5,12]])
	fig.savefig(figname)