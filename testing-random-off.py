import numpy as np
import h5py, glob, emcee
import matplotlib.pylab as plt
from matplotlib import colors, cm

files = glob.glob('lx-sigma-random*h5'); files.sort()
mean = np.zeros((len(files),4))
median = np.zeros((len(files),4))
std = np.zeros((len(files),4))
i = 0
for file in files:
	reader = emcee.backends.HDFBackend(file)
	samples = reader.get_chain(flat=True)
	mean[i] = np.mean(samples, axis=0)
	median[i] = np.median(samples, axis=0)
	std[i] = np.std(samples, axis=0)
	i += 1

fig, ax = plt.subplots(nrows = 2, ncols = 4, sharey=True)
for i in range(4):
	# ax[0][i].hist(median[:,i], bins = 100)
	# ax[1][i].hist(std[:,i], bins = 100)
	# ax[0][i].vlines(np.mean(median[:,i]), ymin=0, ymax=180, linewidth=1, color='k', linestyle='dotted')
	# ax[1][i].vlines(np.mean(std[:,i]), ymin=0, ymax=180,linewidth=1, color='k', linestyle='dotted')
	xmin, xmax = ax[0][i].get_xlim()
	ymin, ymax = ax[0][i].get_ylim()
	dx = xmax - xmin
	dy = ymax - ymin
	ax[0][i].text(xmin + int(dx/5), ymax - int(dy/5), '%0.1f' % np.mean(median[:,i]))
	xmin, xmax = ax[1][i].get_xlim()
	ymin, ymax = ax[1][i].get_ylim()
	dx = xmax - xmin
	dy = ymax - ymin
	ax[1][i].text(xmin + int(dx/5), ymax - int(dy/5), '%0.1f' % np.mean(std[:,i]))
plt.ylim(0,180)

ax[0][0].set_ylabel('Median')
ax[1][0].set_ylabel('Std dev')
ax[0][0].set_title(r'$\alpha$')
ax[0][1].set_title(r'$\beta$')
ax[0][2].set_title(r'$\sigma$')
ax[0][3].set_title(r'$M_{*,0}$')
plt.tight_layout()
plt.savefig('test-random-off.png', dpi=152)
