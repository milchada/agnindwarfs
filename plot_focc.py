import numpy as np
import matplotlib.pylab as plt
import emcee 

Mstar = 10**np.linspace(7, 10, 100)

def f_occ(Mstar, Mstar0):
    return 0.5 * (1+ np.tanh(pow(2.5, abs(8.9 - np.log10(Mstar0)))*np.log10(Mstar/Mstar0)))

def plot(Mstar, dict, lab, ax, color='tab:blue', title=False, linestyle='solid', alpha=0.2, ret=False):
	reader = emcee.backends.HDFBackend(dict)
	samples = reader.get_chain(flat=True)
	logMstar0 = samples[:,-1] #every row, last column
	med = 10**np.percentile(logMstar0, 50)
	low = 10**np.percentile(logMstar0, 16)
	high = 10**np.percentile(logMstar0, 84)
	focc_low = f_occ(Mstar, high)
	focc_high = f_occ(Mstar, low)
	focc_med = f_occ(Mstar, med)
	if ret:
		return focc_med, focc_low, focc_high
	if title:
		ax.set_title(lab)
		ax.plot(Mstar, focc_med, color=color, linestyle=linestyle)
	else:
		ax.plot(Mstar, focc_med, color=color, label=lab, linestyle=linestyle)
	ax.fill_between(Mstar, focc_low, focc_high, alpha=alpha, color=color)
	plt.xscale('log')

def compare_sams(suffix='amuseflim', obs=True):
	fig, ax = plt.subplots()
	plot(Mstar, 'lx-mstar-agnms_dcbh-%s.h5' % suffix, 'AGN-MS', ax, title=False)
	plot(Mstar, 'lx-mstar-blq_dcbh-%s.h5'% suffix, 'BLQ', ax, title=False, color = 'tab:green')
	plot(Mstar, 'lx-mstar-powerLaw_dcbh-%s.h5'% suffix, 'PowerLaw', ax, title=False, color='tab:orange')
	plot(Mstar, 'lx-mstar-agnms_popIII-%s.h5'% suffix, 'AGN-MS', ax, title=False, linestyle='dotted', alpha=0)
	plot(Mstar, 'lx-mstar-blq_popIII-%s.h5'% suffix, 'BLQ', ax, title=False, linestyle='dotted', alpha=0, color='tab:green')
	plot(Mstar, 'lx-mstar-powerLaw_popIII-%s.h5'% suffix, 'PowerLaw', ax, title=False, linestyle='dotted', alpha=0,color='tab:orange')
	if obs:
		med, low, high = plot(Mstar, 'lx-mstar-amuse.h5', 'AMUSE', ax, ret=True)
		ax.plot(Mstar, low, color='k', linestyle='dashed')
		ax.plot(Mstar, high, color='k', linestyle='dashed')
		ax.plot(Mstar, med, color='k', label='AMUSE')
	plt.legend()
	plt.xlabel(r'log($M_*/M_\odot$)')
	plt.ylabel(r'$f_{\rm occ}(M_*)$')
	plt.tight_layout()
	return fig, ax

def compare_models():
	dicts = ['lx-mstar-gama-sdss.h5',
		 'lx-sigma-gama-sdss.h5',
		 'lx-fedd-fduty1-samples.h5',
		 'lx-fedd-fduty0.1-samples.h5']

	labels = [r'L$_X$-M$_*$',
			  r'L$_X$-$\sigma$',
			  r'$\sigma$-M$_\bullet$ + ERDF, f$_{\rm duty}$ = 1',
			  r'$\sigma$-M$_\bullet$ + ERDF, f$_{\rm duty}$ = 0.1']		 

	i = 0
	fig, ax = plt.subplots(nrows = 2, ncols = 2, sharex=True, sharey=True)
	for (dict, lab) in zip(dicts, labels):
		plot(Mstar, dict, lab, ax.flatten()[i], title=True)
		
		i += 1

	# I should probably overplot Amuse results on ax[0][0]
	plot(Mstar, 'amuse_samples.h5', 'AMUSE', ax[0][0], 'k')

	h, l = ax[0][0].get_legend_handles_labels()
	ax[0][0].legend(h,l)
	plt.xscale('log')
	for a in ax[1]:
		a.set_xlabel(r'M$_*$/M$_\odot$')
	for a in ax[:,0]:
		a.set_ylabel(r'f$_{occ}$')
	plt.tight_layout()
	plt.savefig('/n/home07/uchadaya/focc.png')

def completeness():
	dicts = ['lx-fedd-fduty1-samples.h5',
			 'lx-fedd-samples-gtr1e8Msun.h5',
			 'lx-fedd-samples-gtr1e9Msun.h5',
			 'lx-fedd-samples-gama-complete.h5']
	labels = ['All galaxies',
			r'$M_* > 10^8M_\odot$',
			r'$M_* > 10^9M_\odot$',
			r'$M_* > 10^{10.4}M_\odot$ (GAMA complete)']
	i = 0
	fig, ax = plt.subplots()
	colors = plt.cm.tab10
	for (dict, lab) in zip(dicts, labels):
		plot(Mstar, dict, lab, ax, color=colors(i))
		i += 1
	plt.legend()
	plt.xscale('log')
	plt.xlabel(r'M$_*$/M$_\odot$')
	plt.ylabel(r'f$_{occ}$')
	plt.tight_layout()
	plt.savefig('focc_completeness.png')

def fdetect(tab, zs = zs = np.arange(0, .15, .01)):
	fdetect = np.zeros(len(zs) - 1)
	for i in range(len(zs)-1):
		g = tab[(tab['z'] > zs[i])*(tab['z'] < zs[i+1])]
		det = sum(g['Lx'] > g['Llim'])
		fdetect[i] = det/len(g)
	return zs, fdetect

def m_z(tab, zs = np.arange(0, .15, .01)):
	logm_mean = np.zeros(len(zs) - 1)
	for i in range(len(zs)-1):
		g = tab[(tab['z'] > zs[i])*(tab['z'] < zs[i+1])]
		logm_mean[i] = np.nanmean(np.log10(g['Mstar']))
	return zs, logm_mean

