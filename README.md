# Measuring the occupation fraction of black holes in dwarf galaxies

[Code used to produce this paper](https://ui.adsabs.harvard.edu/abs/2023ApJ...946...51C/abstract).

I use a variety of optical+X-ray catalogues to find X-ray counterparts to optically selected galaxies. Following Miller et al 2015, I treat non-detections as upper limits. I assume a power-law scaling relation for Lx-Mstar, with a scatter independent of mass. The occupation fraction is modeled as a sigmoid function, going from 0 at stellar masses of 1e7 to 1 at 1e10; a scaling parameter Mstar0 determines the occupation fraction evolves at intermediate masses. Using flat priors, a Gaussian sampler for whether or not each galaxy hosts a black hole, and a likelihood function based on the scaling relation, I run an MCMC sampler. The output of this sampler places joint constraints on the 4 parameters of the model - 3 for the scaling relation, one for the occupation function. 
