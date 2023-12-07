from scipy.stats import norm 
import numpy as np
import pickle, os
import pandas as pd
from astropy.cosmology import default_cosmology
import matplotlib.pylab as plt 

lcdm = default_cosmology.get()

with open('angelo_sam.pkl','rb') as f:
   d = pickle.load(f, encoding='latin1')

def average(dict, key):
   for i in range(1, 21):
      dict[key][dict['treeIndex'] == i]

models = ['blq_popIII_pmerge0.1_072018', 'powerLaw_popIII_pmerge0.1_072018', 'agnms_popIII_pmerge0.1_072018', 
          'blq_dcbh_pmerge0.1_072018', 'powerLaw_dcbh_pmerge0.1_072018', 'agnms_dcbh_pmerge0.1_072018']
keys = ['logFinalHaloMass', 'm_halo', 'treeIndex', 'redshift', 'L_bol', 'cosmologicalWeight', 'm_bh', 
               'isASatellite', 'sigma', 'm_star']

colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:blue', 'tab:green', 'tab:orange']
labels = ['Pop III + BLQ', 'Pop III + PL', 'Pop III + AGN-MS', 
          'DCBH + BLQ', 'DCBH + PL', 'DCBH + AGN-MS']
sigmas ={0:0,1:0.5, 2:1.8, 3:0, 4:.65, 5:1.765}
#note: Angelo made a typo, so isASatellite is actually True if it is a central. 

Lsun = 3.828e33 #erg/s
flim = 6.5e-15 #erg/cm**2/s

def Lx_Lbol(Lbol, band = 'soft'):
   if band == 'soft':
      a1 = .239
      a2 = 0.059
      a3 = -0.009
      b = 1.436 
      sigma = 0.26
   else:
      a1 = 0.288
      a2 = .111
      a3 = -0.007
      b = 1.308 
      sigma = 0.26
   L = np.log10(Lbol/Lsun) - 12 
   mean = b + a1*L + a2*(L**2) + a3*(L**3) #Lusso 2012 mean
   Lbol_Lx = 10**mean#logkXinv
   Lx = Lbol / Lbol_Lx
   Lx_min = 10**(np.log10(Lx) - sigma)
   Lx_max = 10**(np.log10(Lx) + sigma)
   return Lx, Lx_min, Lx_max

def Lx_Lbol_Shen(Lbol, band='soft'):
   from math import erf
   if band == 'soft':
      c1, k1, c2, k2 = (5.71, -0.026, 17.67, 0.278) #0.5-2 keV
      sigma1, sigma2, logL0, sigma3 = (0.080, 0.180, 44.16, 1.496) 
   else:
      c1, k1, c2, k2 = (4.073, -0.026, 12.60, 0.278) #2-10 keV
      sigma1, sigma2, logL0, sigma3 = (0.193, 0.066, 42.99, 1.883)
   L = Lbol/(1e10*Lsun)
   arg = (np.log10(Lbol) - logL0)/(np.sqrt(2)*sigma3)
   term = np.array([0.5 + 0.5*erf(a) for a in arg])
   sigma_corr = sigma2 + sigma1*term
   kInv = c1*(L**k1) + c2*(L**k2)
   kerr = np.array([norm(loc = k, scale = s).rvs() for k, s in zip(kInv, sigma_corr)])
   Lx = Lbol/kerr
   Lx_min = 10**(np.log10(Lx) - sigma_corr)
   Lx_max = 10**(np.log10(Lx) + sigma_corr)
   return Lx, Lx_min, Lx_max

def Lx_Lbol_Duras(Lbol):
   #Duras+ 2020: https://arxiv.org/pdf/2001.09984.pdf
   #Table 1, Eq. 2.
   from scipy.stats import norm 
   logL = np.log10(Lbol/Lsun)
   a = 10.96
   b = 11.93
   c = 17.79
   kX = a*(1+ (logL/b)**c)
   sigma = 0.27
   kXmin = kX*(1-sigma)
   kXmax = kX*(1+sigma)
   return Lbol/kX, Lbol/kXmin, Lbol/kXmax 


def histograms(obs, log=True):
   plt.figure()
   model = d[models[0]]
   #select only centrals
   mstar = np.log10(model['m_star'])[model['isASatellite']]
   if log:
      mobs = obs['Mstar']
   else:
      mobs = np.log10(obs['Mstar'])
   nobs, bins,_ = plt.hist(mobs, range=(mobs.min(),12), bins=25, label='Obs', alpha=0.5, density=True)
   nsam, bins,_ = plt.hist(mstar, range=(mobs.min(),12), bins=25, label='SAM', alpha=0.5, density=True)
   plt.legend(fontsize=14)
   plt.xticks(fontsize=14)
   plt.yticks(fontsize=14)
   plt.savefig('angelo_mhist.png')
   plt.close()
   return bins, nobs/nsam

def mock_catalog():
   obs = pd.read_csv('gama_sdss_all.csv')
   obs = obs[obs['vstar'] > obs['sigma_err']]
   obs.index = np.arange(len(obs))
   for model in models:
      m = d[model]
      mask = m['isASatellite'] 
      dat = pd.DataFrame()
      dat.insert(0, 'Mstar', m['m_star'][mask])
      dat.insert(1, 'vstar', m['sigma'][mask])

      z = np.zeros(len(dat))
      i = 0
      for mass in m['m_star'][mask]:
         similar_mass = (obs['Mstar'] > 0.1*mass)*(obs['Mstar'] < 10*mass)
         zsub = obs['z'].values[similar_mass]
         # print(len(zsub))
         if len(zsub) > 1:
            pick = np.random.randint(low=0, high=len(zsub)-1)
            z[i] = zsub[pick]
         else: 
            z[i] = np.random.uniform(low=0.005, high=0.15)
         i += 1
      
      dL2 = 4*np.pi*lcdm.luminosity_distance(z).to('cm')**2
      Lx, _, _ =  Lx_Lbol_Shen(m['L_bol'][mask]*Lsun) #k_X, unit conv
      Llim = flim*dL2.value
      dat.insert(2, 'Lx', Lx)
      dat.insert(3, 'Llim', Llim)
      bins, weights = histograms(obs)
      w = np.interp(np.log10(m['m_star'][mask]), (bins[1:]+bins[:-1])/2., weights)
      dat.insert(4, 'w_obs', w)
      dat.insert(5, 'redshift', z)
      mmin = 20110.3
      dat = dat[(dat['Mstar'] > mmin)*(~np.isinf(w))]
      dat.to_csv('%s.csv' % model, index=False)

def lx_mstars(sigmas = sigmas):
   fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
   norm = colors.LogNorm(vmin=10, vmax=800)
   cmap = cm.gnuplot2
   m = cm.ScalarMappable(norm=norm, cmap=cmap)
   for i in range(len(models)):
      model = pd.read_csv('%s.csv' % models[i])
      Lx =  model['Lx'].values #k_X, unit conv
      try:
         Lx = add_scatter(Lx, sigma=sigmas[i])
      except TypeError:
         Lx = Lx
      Llim = 1e38*np.ones(len(Lx)) #model['Llim'].values
      # ax.flatten()[i].text(1e6, 1e42, models[i].split('_pmer')[0], fontsize=12)
      mask = (Lx > Llim)
      ax.flatten()[i].scatter(model['Mstar'].values[~mask], Lx[~mask], color= m.to_rgba(model['vstar'].values[~mask]), alpha=0.05, s=10)
      im = ax.flatten()[i].scatter(model['Mstar'].values[mask], Lx[mask], cmap=cmap, norm=norm, c = model['vstar'][mask], s=10)
   plt.yscale('log')
   plt.xscale('log')
   for a in ax[1]: 
      a.set_xlabel(r'M$_*/M_\odot$', fontsize=18)
      a.set_xticks([1e5,1e7,1e9,1e11], [r'10$^{5}$',r'10$^{7}$',r'10$^{9}$',r'10$^{11}$'], fontsize=18)
   for a in ax[:,0]: 
      a.set_ylabel(r'L$_x$ (erg/s)', fontsize=18)
      a.set_yticks([1e34,1e37,1e40,1e43,1e46], [r'10$^{34}$',r'10$^{37}$',r'10$^{40}$',r'10$^{43}$', r'10$^{46}$'], fontsize=18)
   plt.ylim(1e34,1e46)
   plt.xlim(1e7,1e12)
   fig.subplots_adjust(right=0.85)
   cbar_ax = fig.add_axes([0.87, .155, .02, .81])
   fig.colorbar(im, cax=cbar_ax)
   plt.yticks(fontsize=14)
   plt.ylabel(r'$\sigma$ (km/s)', fontsize=14)
   return fig, ax

def ledd():
   fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
   for i in range(len(models)):
      Ledd = 1e38 * d[models[i]]['m_bh'] * 0.1 #k_X               
      dL2 = 4*np.pi*lcdm.luminosity_distance(model['redshift']).to('cm')**2
      Lx =  Lx_Lbol_Shen(model['L_bol']*Lsun) #k_X, unit conv
      Llim = flim*dL2.value
      fedd = Lx/Ledd
      mask = [Lx > Llim]
      ax.flatten()[i].hist(np.log10(fedd), range=(-18,2), bins=100, color='tab:blue')
      ax.flatten()[i].hist(np.log10(fedd[mask]), range=(-18,2), bins=100, color='tab:orange')
   for a in ax[1]: 
      a.set_xlabel(r'log$_{10}(\lambda_{Edd})$', fontsize=18)
      # a.set_xticks([-8, -6,-4,-2,0,2], [-8, -6,-4,-2,0,2], fontsize=18)
   for a in ax[:,0]: 
      a.set_ylabel(r'N', fontsize=18)
      a.set_yticks([1, 10, 100], [r'10$^{0}$',r'10$^{1}$',r'10$^{2}$'], fontsize=18)
   plt.yscale('log')
   ax[0][0].text(-8, 100, 'Pop III', fontsize=12)
   ax[1][0].text(-8, 100, 'DCBH', fontsize=12)
   ax[0][0].text(0, 2, 'BLQ', fontsize=12)
   ax[0][1].text(0, 2, 'PL', fontsize=12)
   ax[0][2].text(-1.5, 2, 'AGN-MS', fontsize=12)
   plt.savefig('angelo_fedd_hists.png')

def fedd_sigma():
   fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
   for i in range(len(models)):
      Ledd = 1e38 * d[models[i]]['m_bh'] * 0.1               
      dL2 = 4*np.pi*lcdm.luminosity_distance(model['redshift']).to('cm')**2
      Lx =  Lx_Lbol_Shen(model['L_bol']*Lsun) #k_X, unit conv
      Llim = flim*dL2.value
      fedd = Lx/Ledd
      mask = [Lx > Llim]
      ax.flatten()[i].scatter(np.log10(fedd), d[models[i]]['sigma'], color='tab:blue',alpha=0.1)
      ax.flatten()[i].scatter(np.log10(fedd[mask]), d[models[i]]['sigma'][mask], color='tab:orange')
   for a in ax[1]: 
      a.set_xlabel(r'log$_{10}(\lambda_{Edd})$', fontsize=18)
      a.set_xticks([-8, -6,-4,-2,0,2], [-8, -6,-4,-2,0,2], fontsize=18)
   for a in ax[:,0]: 
      a.set_ylabel(r'$\sigma$ (km/s)', fontsize=18)
      a.set_yticks([10, 100, 1000], [r'10$^{1}$',r'10$^{2}$',r'10$^{3}$'], fontsize=18)
   ax[0][0].text(-8, 400, 'Pop III', fontsize=12)
   ax[1][0].text(-8, 400, 'DCBH', fontsize=12)
   ax[0][0].text(0, 20, 'BLQ', fontsize=12)
   ax[0][1].text(0, 20, 'PL', fontsize=12)
   ax[0][2].text(-1.5, 20, 'AGN-MS', fontsize=12)
   plt.savefig('angelo_fedd_sigma.png')

def mstar_mbh(nbins):
   mbins = np.linspace(6, 12, nbins)
   # bins, ratios = histograms()
   # weights = np.interp(mbins, (bins[1:]+bins[:-1])/2., nobs/nsam)
   fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
   for i, m in enumerate(models):
      model=d[m]
      mstar = np.log10(model['m_star'])[model['isASatellite']]
      mbh = np.log10(model['m_bh'])[model['isASatellite']]
      H, xe, ye = np.histogram2d(mstar, mbh, range=((7,12),(0,10)), bins=nbins)
      im = ax.flatten()[i].imshow(H/H.sum(axis=1), origin='lower', cmap=cm.magma, vmin=0, vmax=1.0)
   for a in ax[1]: 
      a.set_xlabel(r'log$_{10}(M_*/M_\odot)$', fontsize=18)
      a.set_xticks(np.arange(len(xe))[::5], xe[::5], fontsize=18)
   for a in ax[:,0]: 
      a.set_ylabel(r'log$_{10}(M_\bullet/M_\odot$)', fontsize=18)
      a.set_yticks(np.arange(len(ye))[::5], ye[::5], fontsize=18)
   plt.xlim(2, nbins-2); plt.ylim(0, nbins-1)
   cbar_ax = fig.add_axes([0.85, .145, .025, .83])
   fig.colorbar(im, cax=cbar_ax)
   plt.yticks(fontsize=14)
   plt.ylabel(r'$\phi(M_{\bullet}|M_*)$', fontsize=14)
   ax[0][0].text(5, 15, 'Pop III', fontsize=12, color='w')
   ax[1][0].text(5, 15, 'DCBH', fontsize=12, color='w')
   ax[0][0].text(13, 2.5, 'BLQ', fontsize=12, color='w')
   ax[0][1].text(13, 2.5, 'PL', fontsize=12, color='w')
   ax[0][2].text(11, 2.5, 'AGN-MS', fontsize=12, color='w')
   plt.tight_layout(rect=[0,0,.85,1],w_pad=-1)
   plt.savefig('angelo_mstar_mbh.png')
         
def focc_mcut(ax, nbins=8, mcut = [10, 3e3, 1e6]):
   mbins = np.linspace(8, 10.25, nbins)
   colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:blue', 'tab:green', 'tab:orange']
   labels = ['Pop III + BLQ', 'Pop III + PL', 'Pop III + AGN-MS', 
             'DCBH + BLQ', 'DCBH + PL', 'DCBH + AGN-MS']
   for i, m in enumerate(models):
      model=d[m]
      focc = np.zeros((len(mcut), len(np.unique(model['treeIndex'])), len(mbins) - 1))
      if 'popIII' in m:
         linestyle = 'dotted'
         linewidth = 3
      else:
         linestyle = 'solid'      
         linewidth = 1
      mstar = np.log10(model['m_star'])
      sat = ~model['isASatellite']
      mbh = np.log10(model['m_bh'])
      n = model['cosmologicalWeight']
      for tree in range(focc.shape[1]):
         for j in range(focc.shape[2]):
            select = (mstar > mbins[j])*(mstar < mbins[j+1])*sat*(model['treeIndex'] == tree)
            sbh = mbh[select]
            sn = n[select]
            if len(sbh):
               for k in range(len(focc)):
                  focc[k, tree, j] = sum(sn[sbh > np.log10(mcut[k])])/sum(sn)
      for k in range(focc.shape[0]):
         f = focc[k]
         ax.flatten()[k].plot(10**((mbins[:-1]+mbins[1:])/2.), np.mean(f, axis=0), color=colors[i], label=labels[i], linestyle=linestyle, linewidth=linewidth)
         ax.flatten()[k].fill_between(10**((mbins[:-1]+mbins[1:])/2.), np.min(f, axis=0), np.max(f, axis=0), color=colors[i], alpha=0.1)
   plt.legend()
   plt.xscale('log')
   for a in ax.flatten():
      a.set_xlabel(r'$M_*/M_\odot$', fontsize=18)
   plt.tight_layout()   

def Lx_bh(sbh, active_fraction, eps,loc = -1, scale = 0.6, scaling='Lusso', band='soft'):
   fedd = norm(loc, scale)
   Ledd = 1e38 * (10**sbh) 
   fedds = 10**fedd.rvs(len(Ledd))
   # print(fedds)
   Lbol = fedds * Ledd * eps
   if scaling =='Lusso':
      Lx, _, _ = Lx_Lbol(Lbol)
   else:
      Lx, _, _ = Lx_Lbol_Shen(Lbol, band=band)
   rand = np.random.uniform(size=len(Lbol))
   Lx[rand < (1-active_fraction)] = 0
   return Lx

def focc_total(mmin=8, mmax=10):
   fig, ax = plt.subplots(ncols = 2, sharex=True, sharey=True, figsize=(9,4))
   mcut = np.linspace(0, 6, 12)
   for i, m in enumerate(models):   
      model=d[m]
      if 'popIII' in m:
         linestyle = 'dotted'
         linewidth = 3
      else:
         linestyle = 'solid'      
         linewidth = 2
      mstar = np.log10(model['m_star'])
      mbh = np.log10(model['m_bh'])
      n = model['cosmologicalWeight']
      sat = ~model['isASatellite']
      cut = (mstar>mmin)*(mstar < mmax)
      focc = np.zeros(len(mcut))
      focc_cen = np.zeros(len(mcut))
      for j in range(len(focc)):
         central_bh = mbh[cut*sat]
         sn = n[cut*sat]
         if len(central_bh):
            focc_cen[j] = sum(sn[central_bh > mcut[j]])/sum(sn)
            focc[j] = sum(n[cut][mbh[cut] > mcut[j]])/sum(n[cut])
      ax[0].plot(10**mcut, focc, label=labels[i], color=colors[i], linestyle=linestyle, linewidth=linewidth)
      ax[1].plot(10**mcut, focc_cen, label=labels[i], color=colors[i], linestyle=linestyle, linewidth=linewidth)
   plt.xscale('log')      
   plt.xlim(1, 1e6)
   ax[0].set_ylabel(r'$f_{occ} (10^%d < M_*/M_\odot < 10^{%d}$)' %(mmin, mmax), fontsize=16)
   ax[0].set_yticks([0, .2, .4, .6, .8, 1.0])
   ax[0].set_yticklabels([0.0, .2, .4, .6, .8, 1.0], fontsize=16)
   for a in ax.flatten():
      a.set_xlabel(r'$M_{\bullet, min}/M_\odot$', fontsize=16)
      a.set_xticks([1e3, 1e4, 1e5, 1e6])
      a.set_xticklabels([r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'], fontsize=16)
   plt.legend()
   ax[0].set_title('Centrals only')
   ax[1].set_title('All halos')
   plt.xlim(1e3,1e6)
   plt.tight_layout()
   return fig, ax

def focc_Lx_total(mmin=8, mmax=10, scaling='Lusso', band='soft', sigmas=sigmas):
   fig, ax = plt.subplots()
   Lxcut = 10**np.linspace(34, 43, 9)
   for i, m in enumerate(models):   
      model=d[m]
      if 'popIII' in m:
         linestyle = 'dotted'
         linewidth = 3
      else:
         linestyle = 'solid'      
         linewidth = 2
      mstar = np.log10(model['m_star'])
      n = model['cosmologicalWeight']
      sat = ~model['isASatellite']
      cut = (mstar>mmin)*(mstar < mmax)
      focc = np.zeros(len(Lxcut))
      Lbol = model['L_bol']*Lsun 
      if scaling =='Lusso':
         Lx, _, _= Lx_Lbol(Lbol) 
      else:
         Lx, _, _ = Lx_Lbol_Shen(Lbol, band=band)
      if sigmas[i]:
         Lx = add_scatter(Lx, sigmas[i])
      for j in range(len(focc)):
         slx = Lx[cut*sat]
         sn = n[cut*sat]
         if len(sn):
            focc[j] = sum(sn[(slx > Lxcut[j])])/sum(sn) 
      ax.plot(Lxcut, focc, label=labels[i], color=colors[i], linestyle=linestyle, linewidth=linewidth)
   plt.xscale('log')      
   plt.xlim(1e34, 1e43)
   ax.set_ylabel(r'$f_{occ} (10^%d < M_*/M_\odot < 10^{%d}$)' %(mmin, mmax), fontsize=16)
   ax.set_yticks([0, .2, .4, .6, .8, 1.0])
   ax.set_yticklabels([0.0, .2, .4, .6, .8, 1.0], fontsize=16)
   ax.set_xlabel(r'$L_{X, min}$ (erg/s)', fontsize=16)
   ax.set_xticks([1e34, 1e37, 1e40, 1e43])
   ax.set_xticklabels([r'$10^{34}$',r'$10^{37}$',r'$10^{40}$',r'$10^{43}$'], fontsize=16)
   plt.legend()
   plt.title(r'$f_{\rm active}$ = %0.1f' % active_fraction)
   plt.tight_layout()
   return fig, ax

def focc_mdot(m, color, label, fig, ax, nbins = 10, Lxcut = [1e38], scaling='Lusso', band='soft', sigma = None ):   
   mbins = np.linspace(6, 12, nbins)
   focc = np.ndarray((len(Lxcut), len(mbins)-1))
   model=d[m]
   if 'popIII' in m:
      linestyle = 'dotted'
      linewidth = 3
   else:
      linestyle = 'solid'      
      linewidth = 2
   mstar = np.log10(model['m_star'])
   sat = ~model['isASatellite']
   Lbol = model['L_bol']*Lsun 
   if scaling =='Lusso':
      Lx, _, _= Lx_Lbol(Lbol) 
   else:
      Lx, _, _ = Lx_Lbol_Shen(Lbol, band=band)
   if sigma:
      Lx = add_scatter(Lx, sigma)
   n = model['cosmologicalWeight']
   for j in range(len(mbins) - 1):
      select = (mstar > mbins[j])*(mstar < mbins[j+1])*sat
      slx = Lx[select]
      sn = n[select]
      if len(sn):
         for k in range(len(Lxcut)):
            focc[k, j] = sum(sn[slx > Lxcut[k]])/sum(sn)
   for k in range(len(Lxcut)):
      ax[k].plot(10**((mbins[:-1]+mbins[1:])/2.), focc[k], color=color, label=label, linestyle=linestyle, linewidth=linewidth)
   for a in ax.flatten():
      a.set_xlabel(r'$M_*/M_\odot$', fontsize=16)
      a.set_xticks([1e7, 1e8, 1e9, 1e10])
      a.set_xticklabels([r'$10^{7}$',r'$10^{8}$',r'$10^{9}$',r'$10^{10}$'], fontsize=16)
   ax[0].set_ylabel(r'$f_{occ} (M_*$)' , fontsize=16)
   ax[0].set_yticks([0, .2, .4, .6, .8, 1.0])
   ax[0].set_yticklabels([0.0, .2, .4, .6, .8, 1.0], fontsize=16)
   plt.xscale('log')      
   plt.xlim(1e7, 1e10)
   plt.legend()
   plt.tight_layout()
   return fig, ax

def fduty(obs, model, mmin=8, mmax=10, Llim = 0, eps=0.1, fit=None, err=None, log=False, band='soft', sigma = None):
   sat = ~model['isASatellite']
   Lx,_,_ = Lx_Lbol_Shen(model['L_bol']*Lsun, band=band)
   L = Lx[sat]
   if sigma:
      L = add_scatter(L, sigma)
   print(Lx[Lx > 0].min(), Lx.max(), L[L>0].min(), L.max())
   logMstar = np.log10(model['m_star'][sat])
   bins, weights = histograms(obs, log=log)
   w = np.interp(np.log10(model['m_star'][sat]), (bins[1:]+bins[:-1])/2., weights)
   mcut = (logMstar > mmin) * (logMstar < mmax)
   if err:
      logpred = logMstar*fit[0] + fit[1]
      Llim = 10**(logpred - 2*err)
   on = (L > Llim) * mcut
   valid = (w > -np.inf) * (w < np.inf)
   fduty = np.nansum(w[on*valid])/np.nansum(w[mcut*valid])
   return fduty

def param(model, mmin=8, mmax=10, Llim = 0, eps=0.1, trees=True, scaling='Lusso', band='soft'):
   sat = ~model['isASatellite']
   if scaling == 'Lusso':
      Lx,_,_ = Lx_Lbol(model['L_bol']*Lsun)[sat] * eps/0.1
   else:
      Lx,_,_ = Lx_Lbol_Shen(model['L_bol']*Lsun, band=band)[sat] * eps/0.1
   logMstar = np.log10(model['m_star'][sat])
   weight = model['cosmologicalWeight'][sat]
   mcut = (logMstar > mmin) * (logMstar < mmax)
   on = (Lx > Llim) * mcut
   ntrees = len(np.unique(model['treeIndex']))
   if trees:
      fit = np.zeros((ntrees, 2))
      cov = np.zeros((ntrees, 2,2))
      for i in range(ntrees):
         tree = (model['treeIndex'] == i+1)[sat]
         fit[i] = np.polyfit(logMstar[on*tree], np.log10(Lx[on*tree]), 1, w=weight[on*tree])
   else:
      fit = np.polyfit(logMstar[on], np.log10(Lx[on]), 1, w=weight[on])
   return fit

def err(model, fit, eps=0.1, Llim=1e38, mmin=8, mmax=12, scaling='Lusso', band='soft'):
   sat = ~model['isASatellite']
   logMstar = np.log10(model['m_star'][sat])
   if scaling == 'Lusso':
      Lx,_,_ = Lx_Lbol(model['L_bol']*Lsun)[sat] * eps/0.1
   else:
      Lx,_,_ = Lx_Lbol(model['L_bol']*Lsun, band=band)[sat] * eps/0.1
   mcut = (logMstar > mmin) * (logMstar < mmax)
   on = (Lx > Llim) * mcut
   pred = 10**(logMstar[on]*fit[0] + fit[1])
   err = np.sqrt((np.log10(pred) - np.log10(Lx[on]))**2)
   return err.mean()

def fduty_all(scaling='Lusso', band='soft'):
   obs = pd.read_csv('AMUSEGalaxies.csv', header=0)
   eps = {'blq':0.1, 'powerLaw':0.02,'agnms':0.000035}
   for model in models:
      key = model.split('_')[0]
      e = eps[key]
      print(model.split('_')[:2])
      fit = param(d[model], mmax=12, eps=e, trees=False, scaling=scaling, band=band)
      sigma = err(d[model], fit, e, scaling=scaling, band=band)
      fd_obs = fduty(obs, d[model], mmin=8, mmax=12, eps=e, Llim=1e38, scaling=scaling, band=band)
      fd = fduty(obs, d[model], mmin=8, mmax=12, eps=e, fit=fit, err=sigma, scaling=scaling, band=band)
      print(fit, sigma, fd_obs, fd)

def compare_fits(xmax, nbins):
   fig, ax = plt.subplots()
   colors={'blq':'tab:green', 'agnms':'tab:blue', 'powerLaw':'tab:orange'}
   linestyles = {'popIII':'dotted', 'dcbh':'solid'}
   for model in models:
      _,_,err = param(d[model])
      grow = model.split('_')[0]
      seed = model.split('_')[1]
      if seed == 'dcbh':
         _ = ax.hist(err, range=(0, xmax), bins=nbins, label=grow, histtype='step', color=colors[grow], linestyle=linestyles[seed])
      else:
         _ = ax.hist(err, range=(0, xmax), bins=nbins, color=colors[grow], linestyle=linestyles[seed], histtype='step')
   plt.legend()
   return fig, ax

def add_scatter(arr, sigma=1, rescale=False):
   logArr = np.log10(arr)
   ArrNew = 10**norm(loc=logArr, scale=sigma).rvs()
   if rescale:
      scale = np.nanmean(arr[arr > 0])/np.nanmean(ArrNew[ArrNew > 0])
   else:
      scale = 1.
   return ArrNew*scale

def luminosity_function():
   for model in models:
      j = models.index(model)
      data = d[model]
      sat = ~data['isASatellite']
      z0 = (data['redshift'] == 0 )
      Lbol = data['L_bol'][sat*z0] * Lsun
      Lbols = np.zeros((1000, len(Lbol)))
      for i in range(1000):
         Lbols[i] = add_scatter(Lbol, sigmas[models.index(model)], rescale=True)
      dNn_dVdlogM = data['cosmologicalWeight'][sat*z0]
      dlogM = 0.2
      dN_dV = dNn_dVdlogM * dlogM
      dlogL = 1.
      bins = np.arange(34,50,dlogL)
      hists = np.zeros((1000, len(bins)))
      for k in range(1000):
         Lbol = Lbols[k] 
         for i in range(len(bins)-1):
            cut = (Lbol > 10**bins[i])#*(Lbol < 10**bins[i+1])
            hists[k, i] = sum(dN_dV[cut])
      if 'pop' in model:
         linestyle='dashed'
         alpha = 0.1
      else:
         linestyle='solid'
         alpha = 0.2
      mean = np.mean(hists, axis=0)
      std = np.std(hists, axis=0)
      plt.plot(bins, mean/dlogL, color=colors[j], linestyle=linestyle, label=labels[j]) #/sum(dN_dV)
      plt.fill_between(bins, (mean-std)/dlogL, (mean+std)/dlogL, color=colors[j], alpha=alpha)

def bh_mass_function():
   binsize = 0.3
   bins = np.arange(4,10,.5)
   hists = np.zeros((1000,len(bins)))
   # allhists = {}
   for model in models:
      j = models.index(model)
      data = d[model]
      sat = ~data['isASatellite']
      z0 = (data['redshift'] == 0 )
      mass = data['m_bh'][sat*z0] 
      masses = np.zeros((1000, len(mass)))
      for i in range(1000):
         masses[i] = add_scatter(mass, 0.5, rescale=False)
      dNn_dVdlogM = data['cosmologicalWeight'][sat*z0]
      dlogM = 0.2
      dN_dV = dNn_dVdlogM * dlogM
      hists = np.zeros((1000,len(bins)))
      for k in range(1000):
         mass = masses[k] 
         for i in range(len(bins)-1):
            cut = (mass > 10**bins[i])*(mass < 10**bins[i+1])
            hists[k, i] = sum(dN_dV[cut])/binsize
      # allhists[model] = hists
      if 'pop' in model:
         linestyle='dashed'
         alpha = 0.1
      else:
         linestyle='solid'
         alpha = 0.2
      mean = np.mean(hists, axis=0)
      std = np.std(hists, axis=0)
      plt.plot(bins, mean/dlogM, color=colors[j], linestyle=linestyle, label=labels[j]) #/sum(dN_dV)
      plt.fill_between(bins, (mean-std)/dlogM, (mean+std)/dlogM, color=colors[j], alpha=alpha)
      plt.legend()
      plt.ylabel(r'dN(M$_\bullet$)/dV', fontsize=13)
      plt.xlabel(r'log($M_\bullet/M_\odot$)', fontsize=13)
      plt.xticks(fontsize=13)
      plt.yticks(fontsize=13)
      plt.yscale('log')
      plt.xlim(4,9)

def fundamental_plane(ax, scaling='Lusso', seed='dcbh', ylab=False, sigmas=sigmas):
   if seed == 'dcbh':
      inds = range(3,6)
   else:
      inds = range(3)
   for i in inds:
      m = d[models[i]]
      sat = ~m['isASatellite']
      mstar = m['m_star'][sat]
      mbh = m['m_bh'][sat]/1e8
      Lbol = m['L_bol'][sat] * Lsun
      if scaling == 'Lusso':
         Lx,_,_ = Lx_Lbol(Lbol, band='hard')
      else:
         Lx,_,_ = Lx_Lbol_Shen(Lbol, band='hard')
      Lx = add_scatter(Lx, sigmas[i])
      Lx /= 1e40
      mu0 = norm(loc=0.55, scale=0.22).rvs(size=len(Lx))
      xi_mR = norm(loc=1.09, scale=0.10).rvs(size=len(Lx))
      xi_mX = norm(loc=-0.59, scale=0.15).rvs(size=len(Lx))
      sigma_logM = norm(loc=1, scale = np.exp(-0.04)).rvs(size=len(Lx))
      log_LR = (np.log10(mbh) - mu0 - xi_mX*np.log10(Lx))/xi_mR  #where LR in 1e38 erg/s and LX at 2-10keV in 1e40 erg/s
      ax.scatter(mstar, 10**(log_LR+38), color=colors[i], label=labels[i],alpha=0.2)
   plt.xscale('log')
   plt.yscale('log')
   plt.xlim(1e5, 1e12)
   if ylab:
      ax.set_ylabel(r'$\nu L_\nu$ (5 GHz, erg/s)', fontsize=14)
   ax.set_xlabel(r'$M_*/M_\odot$', fontsize=14)

def Lx_obs_Ueda():
   #https://arxiv.org/pdf/1601.06002.pdf
   #Table 3 + Eq 1, z = 0 and no evolution
   logL0 = scipy.stats.norm(loc=43.03, scale=0.30).rvs(1000)
   gamma1 =.62
   gamma2 = scipy.stats.norm(loc = 1.95, scale=0.15).rvs(1000)
   A = 1e-5*scipy.stats.norm(loc = 8.8, scale=1.5).rvs(1000)
   Lx = 10**np.arange(41.,46.5, .5)
   L0 = 10**logL0
   dPhi_dlogL = np.array([Ai/((Lx/L0i)**gamma1 + (Lx/L0i)**gamma2i) for (Ai, L0i, gamma2i) in zip(A, L0, gamma2)])
   mean = np.mean(dPhi_dlogL, axis=0)
   std = np.std(dPhi_dlogL, axis=0)
   c = 7
   d = 0.3
   kbol_2_10keV = c*(Lx/1e42)**d #Netzer 2019
   Lbol = Lx*kbol_2_10keV
return Lbol, mean, std

def Lbol_Lx_Shen(Lx, band='soft'):
   if band == 'soft':
      c1, k1, c2, k2 = (5.71, -0.026, 17.67, 0.278) #0.5-2 keV
      sigma1, sigma2, logL0, sigma3 = (0.080, 0.180, 44.16, 1.496) 
   else:
      c1, k1, c2, k2 = (4.073, -0.026, 12.60, 0.278) #2-10 keV
      sigma1, sigma2, logL0, sigma3 = (0.193, 0.066, 42.99, 1.883)
   L = Lx/(1e10*Lsun)
   arg = (np.log10(L) - logL0)/(np.sqrt(2)*sigma3)
   term = np.array([0.5 + 0.5*erf(a) for a in arg])
   sigma_corr = sigma2 + sigma1*term
   kInv = c1*(L**k1) + c2*(L**k2)
   kerr = np.array([norm(loc = k, scale = s).rvs() for k, s in zip(kInv, sigma_corr)])
   Lbol = Lx*kerr
   return Lbol

def Lx_obs_Ballantyne():
   L2_10 = np.array([41.30, 41.56, 41.82, 42.08, 42.34, 42.60, 42.86, 43.12, 43.38, 43.64, 43.90, 44.16, 
      44.42, 44.68, 44.94, 45.20, 45.46, 45.72, 45.98, 46.24, 46.50, 46.76, 47.02, 47.28, 47.54, 47.80])
   dPhi_dL = np.array([-3.35, -3.54, -3.73, -3.91, -4.11, -4.30, -4.51, -4.76, -5.07, -5.47, -5.98, -6.54, 
      -7.14, -7.75, -8.36, -8.98, -9.59, -10.21, -10.82, -11.44, -12.06, -12.67, -13.29, -13.90, -14.52, -15.14])