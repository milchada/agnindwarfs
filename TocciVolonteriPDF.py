from scipy.integrate import quad
class p_fedd(stats.rv_continuous):  
        
    def _pdf(self, fedd, vstar):
        fedd0 = 1e-5
        fedd_min = 1e-10
        fedd_max = 0.1
        lMbh = logMbh(vstar,eps=0)
        Mfac = (10**(lMbh- 7.5))**(-3.3) 
            #combine equations 14 and 15 of Pesce 2021, ignore z dependence
        delta = -(0.2 + 0.55*Mfac)/(1+Mfac)
        norm = 1/(np.log10(fedd0/fedd_min) + 
                      (fedd_max**delta - fedd0**delta)/(delta*(fedd0**delta)*np.log(10)))
        
        if (fedd < 1e-10) or (fedd > 1):
            return 0
        else:
            if fedd < 1e-5:
                return norm
            elif fedd < 1e-2:
                return norm*(fedd/1e-5)**delta
            else:
                loglc = np.log(.03)
                sigmac = .88
                denom = 2*np.pi*sigmac*fedd
                exp = (np.log(fedd) - loglc)**2 / (2*sigmac**2)
                return np.exp(-exp)/denom
            
    def _cdf(self, fedd, vstar):
        fedd_min = 1e-10
        fedd_max = 0.1
        intg, _ = quad(self._pdf, fedd_min, fedd, args=(vstar))
        nrm, _ = quad(self._pdf, fedd_min, fedd_max, args=(vstar))
        return intg/nrm

def f_edd(Lx, sigma):
    Mbh = 10**log_Mbh(sigma)#the scatter in the M-sigma relationship goes here
    Ledd = 1e38 * Mbh #erg/s, Mbh in Msun
    #also convert this to X-ray luminosity
    Lsun = 4e33 #erg/s
    a = 10.96
    b = 11.93
    c = 17.79 
    kappa = a + (1 + (np.log10(Ledd/Lsun)/b)**c) #Duras+2020
    # print('kappa: ', kappa)
    fedd = kappa*Lx/Ledd
    return fedd

def log_Mbh(vstar, alpha=8.33, beta=5.77, eps=0.43):
    average = alpha + beta*np.log10(vstar/200)
    return np.array([max(0,norm(loc = av, scale=eps).rvs(size=1)[0]) for av in average])    

def log_Mbh_Mstar(Mstar, alpha=8.56,beta= 1.34,eps= 0.17):
    average = alpha + beta*np.log10(Mstar/1e11)
    return np.array([max(0,norm(loc = av, scale=eps).rvs(size=1)[0]) for av in average])    
