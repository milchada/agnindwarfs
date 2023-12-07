import numpy as np
import pandas as pd
from scipy.stats import powerlaw 

allpts = pd.read_csv('allpts.csv').rename({'RA':'ra', 'DEC':'dec'}, axis=1)
gama_mask = (allpts['ra'] > gama['ra'].min())*(allpts['ra'] < gama['ra'].max())*(allpts['dec'] < gama['dec'].max())*(allpts['dec'] > gama['dec'].min())
pts = allpts[gama_mask]

f = np.log10(pts['flux'])

hist, bins, _ = plt.hist(f, range=(np.nanmin(f[f>-np.inf]), np.nanmax(f)), bins=100, label='eFEDS')
oldxlim = bins[np.argmax(hist)]#np.log10(6.5e-15)
start = np.argmax(hist)   #sample complete above this
hist = hist[start:]
bins = bins[start:]
bins = (bins[1:] + bins[:-1])/2. #midbin

newxlim = np.log10((6.5e-15)/5)

def fake_catalog(newxlim, fakefile=None, allfile=None, xmin=-15):
	#fit the histogram with a power law
	fit = np.polyfit(bins[hist>0],np.log10(hist[hist>0]),1)
	xp = np.linspace(int(newxlim-1),-12,100)
	yp = xp*fit[0] + fit[1]
	plt.plot(xp, 10**yp, label='Fit')
	plt.xlim(xmin, -12)

	plt.vlines(newxlim,0,3000, color='k', linestyles = 'dotted') #eRASS 4 pt source flux limit

	#now count the newly detected sources
	nnew = int((10**yp[(xp < oldxlim)*(xp > newxlim)]).sum()) - len(f[f < oldxlim])
	print(nnew, nnew/len(pts))
	"This is 12 times MORE point sources than all the current ones! Just at the faint ends"

	dx = xp[1]-xp[0]
	xn = xp[(xp < (oldxlim + dx))*(xp > newxlim)]
	yn = yp[(xp < oldxlim)*(xp > newxlim)]
	fn = np.array([])
	for i in range(len(yn)):
		n = int(10**yn[i]) - len(f[(f > xn[i])*(f < xn[i+1])]) #i.e. how many there should be total, minus how many eFEDS has already found
		fi = np.random.uniform(low=xn[i], high=xn[i+1], size=n)
		fn = np.concatenate((fn, fi))

	hfake, bfake, _ = plt.hist(fn, range=(np.nanmin(f[f>-np.inf]), np.nanmax(f)), bins=100, label='Fake')
	scale = hfake[np.argmax(hist)-1] / hist[np.argmax(hist)-1]
	print(scale)
	nfake = int(len(fn)/scale)
	ind = np.random.randint(low=0, high=len(fn), size=nfake)
	fn = fn[ind]
	fake = pd.DataFrame()
	fake.insert(0, 'ra', np.random.uniform(gama['ra'].min(), gama['ra'].max(), size = len(fn)))
	fake.insert(1, 'dec', np.random.uniform(gama['dec'].min(), gama['dec'].max(), size = len(fn)))
	fake.insert(2, 'flux', 10**fn)
	if fakefile:
		fake.to_csv(fakefile, index=False)

	allsrc = pd.DataFrame()
	allsrc.insert(0, 'ra', np.concatenate((fake['ra'], pts['ra'])))
	allsrc.insert(1, 'dec', np.concatenate((fake['dec'], pts['dec'])))
	allsrc.insert(2, 'flux', np.concatenate((fake['flux'], pts['flux'])))
	plt.close()
	if allfile:
		allsrc.to_csv(allfile, index=False)
	else:
		if not fakefile:
			return fake, allsrc
		else:
			return allsrc

def plot(f, fn, filename):
	plt.close()
	plt.hist(np.concatenate((f, fn)), range=(np.nanmin(f[f>-np.inf]), np.nanmax(f)), bins=100, label='eFEDS + Fake')
	plt.hist(f, range=(np.nanmin(f[f>-np.inf]), np.nanmax(f)), bins=100, label='eFEDS')
	plt.legend()
	plt.yscale('log')
	plt.xlabel('log(flux [erg/s/cm**2])')
	plt.ylabel('dN/dlog(f)')
	plt.savefig(filename)

def Lx_Llim(newxlim, gama, allsrc, filename=None):
	ind = crossmatch(allsrc, gama, tab=True)
	d = dist(allsrc.iloc[ind], gama, kpc=False)
	print(sum(d<5))
	flim = 10**newxlim #erg/s/cm**2 
	dL2 = 4*np.pi*lcdm.luminosity_distance(gama['z'].values).to('cm')**2
	Llim = (dL2 * flim).value 
	Lx = (dL2 * flim).value 
	Lx[d < 5] = (allsrc.iloc[ind]['flux']*dL2.value)[d<5].values
	gama['Llim'] = Llim
	gama['Lx'] = Lx
	if filename:
		gama.to_csv(filename, index=False)
	else:
		return gama