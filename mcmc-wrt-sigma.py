#in terms of velocity dispersion

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
# gama = pd.read_csv('gama_all.csv', header=0)
gama_sdss = pd.read_csv('gama_sdss_sigma.csv', header=0)

def log_prior(theta, priors):
	lnprior = 0
	for i in range(len(priors)):
		if (theta[i] < priors[i][0]) or (theta[i] > priors[i][1]):
			lnprior = -np.inf
	return lnprior

def f_occ(Mstar, Mstar0):
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def N(Lx, vdisp, alpha, beta, sigma):
	mu = alpha + beta*vdisp/200
	rv = norm(loc=mu, scale=sigma)
	return rv.pdf(np.log10(Lx))

def log_likelihood(Lx, Mstar, vdisp, Mstar0, alpha, beta, sigma):
	like = N(Lx, vdisp, alpha, beta, sigma)
	return np.nansum(np.log10(f_occ(Mstar, Mstar0))) + np.sum(np.log10(like))

def p_BH(Lxi, vdispi, Mstari, Mstar0, alpha, beta, sigma):
	rv = norm(loc=0, scale=1)
	logLxPred = (alpha + beta*vdispi/200)
	Phi = rv.cdf((np.log10(Lxi) - logLxPred)/sigma)
	focc = f_occ(Mstari, Mstar0)
	return Phi/((1 - focc) + focc*Phi)

def latent(Lx, vstar, Mstar, Llim, Mstar0, alpha, beta, sigma):
    I_n = np.zeros(len(vstar))
    p_true = np.ones(len(vstar))
    # "TEST THIS BELOW LINE"
    p_true[Lx == Llim] = np.array([p_BH(Lx[i], vstar[i], Mstar[i], Mstar0, alpha, beta, sigma) for i in np.argwhere(Lx == Llim)[:,0]])
    p_true[p_true < 0] = 0
    p_true[p_true > 1] = 1
    p_true[~np.isfinite(p_true)] = 0
    print(p_true)
    for i in range(len(I_n)):
        try:
            I_n[i] = bernoulli.rvs(p_true[i], size=1)
        except ValueError:
            I_n[i] = 0
    return I_n

def log_prob_Mstar0(Mstar0, I_n, Mstar): 
    p = 1
    for i in range(len(I_n)):
        focc = f_occ(Mstar[i], Mstar0)
        p *= (np.power(focc, I_n[i]) * np.power(1 - focc, 1-I_n[i]))
    return np.log10(p)

def log_pdf(theta, Lx, Mstar, vstar, Llim, priors):
    alpha, beta, sigma, logMstar0 = theta
    Mstar0 = 10**logMstar0
    likely = np.sum(log_likelihood(Lx, Mstar, vstar, Mstar0, alpha, beta, sigma))
    print("Log likelihood: ", likely/len(Lx))
    I_n = latent(Lx, vstar, Mstar, Llim, Mstar0, alpha, beta, sigma)
    print("f_BH: ", sum(I_n)/len(Lx)) #why is this always 1 for everyone?
    prob = log_prob_Mstar0(Mstar0, I_n, Mstar)
    logpdf = (likely + prob)/len(Lx) + log_prior(theta, priors)
    if not np.isnan(logpdf):
    	return logpdf 
    else:
    	return -np.inf

def seed(priors, nwalkers, ndim, right=True):
    if right:
        p0 = np.random.randn(nwalkers, ndim)
        for i in range(ndim):
            prange = (p0[:,i].max() - p0[:,i].min())
            p0[:,i] *= (priors[i,1] - priors[i,0])/prange
            p0[:,i] += (priors[i,0] - p0[:,i].min())
    else:
        p0 = np.vstack((np.random.randn(int(nwalkers/2), ndim), 
                        np.random.randn(int(nwalkers/2), ndim))) #bimodel
        for i in range(ndim):
            p0[:int(nwalkers/2.),i] += priors[i,0]
            p0[int(nwalkers/2.):,i] += priors[i,1]
    return p0

#set some fraction of detections to upper limits
def turnoff(galcat, ncontam):
    detections = galcat[galcat['Lx'] > galcat['Llim']]
    inds = np.random.randint(low=0, high=len(detections), size=ncontam)
    inds = detections.index[inds]
    newcat = pd.DataFrame()
    for key in galcat.keys():
        newcat.insert(0, key, galcat[key].values)
    newcat.loc[inds, 'Lx'] = newcat.iloc[inds]['Llim'].values
    return newcat

def run_emcee(galcat, filename, nwalkers=100, ndim=4, nsteps=int(1e4), nburn=100, 
                priors=np.array([[35,45],[0,2],[0,2],[7,10]]), right=True, mean_turnoff=None, std_turnoff=None, fromscratch=True):
    if mean_turnoff:
        ncontam = int(np.random.normal(mean_turnoff, std_turnoff))
        galcat = turnoff(galcat, ncontam)
    vstar = galcat['vstar'].values
    Lx = galcat['Lx'].values[vstar>0]
    Mstar = galcat['Mstar'].values[vstar>0]
    Llim = galcat['Llim'].values[vstar>0]
    vstar = vstar[vstar>0]


    backend = emcee.backends.HDFBackend(filename)
    ncpu = multiprocessing.cpu_count()
    pool = Pool(processes=ncpu)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_pdf, args=[Lx, Mstar, vstar, Llim, priors],backend=backend, pool=pool)
    #burn in
    if fromscratch:
        p0 = seed(priors, nwalkers, ndim, right=right)
        state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(state, nsteps, progress=True)
    else:
        sampler.run_mcmc(None, nsteps, progress=True)
    return sampler

#now how do I run this N times on N different processors
# if __name__ == "__main__":
#     import multiprocessing, os
#     ncpu = os.cpu_count()

#     def main(filename):
#         run_emcee(gama_sdss, filename, nsteps=10000, mean_turnoff=12, std_turnoff=4)

#     filenames = ['lx-sigma-randomoff-%d.h5' % i for i in range(int(ncpu*5))]
#     with Pool(ncpu) as pool:
#         results = pool.map(main, (filenames))

def find_incompletes():
    incomplete = []
    for file in files:
        reader = emcee.backends.HDFBackend(file)
        samples = reader.get_chain()
        if len(samples) < 2000:
            incomplete.append(file)
    return incomplete

def finish(filename):
    reader = emcee.backends.HDFBackend(filename)
    samples = reader.get_chain()
    nsteps = 2000 - len(samples)
    run_emcee(gama_sdss, file, nwalkers=48, ndim=4, nsteps=nsteps, 
            priors=np.array([[35,45],[0,2],[0,2],[7,10]]), fromscratch=False)
    #although.. this will have turned off a different subset of detections, so not ideal. 