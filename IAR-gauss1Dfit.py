import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl

def main():
    
    MEAN = 3
    SIGMA = 1
    
    NOISE_MEAN = -2
    NOISE_SIGMA = .5
    
    #Fits a Gaussian to the sum of two analytic Gaussians
    analyticSumFit(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA)
    
    #
    binSumFit(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA)
    
def gaussianpdf(x, xo, s):
    return np.exp(-(x-xo)**2/(2*s**2)/np.sqrt(2*np.pi))

def analyticSumFit(mean, sig, nmean, nsigma):
    x = np.linspace(-10,10, 10000)
    y = gaussianpdf(x, mean, sig)
    noise = gaussianpdf(x, nmean, nsigma)
    data = y + .5*noise
            
    mu, error = opt.curve_fit(gaussianpdf, x, data, p0=[3, .5])
    newGauss = gaussianpdf(x, mu[0], mu[1])
    
    params = [mean, sig, nmean, nsigma]
    paramStr = "Params (mean1, sig1, mean2, sig2): " + ', '.join([str(param) for param in params])
    statStr = 'Stats of Sum (mean, sigma): ' + ', '.join([str(round(stat, 2)) for stat in mu])
    titleStr = 'Sum of two 1-D Gaussians\n' + paramStr + '\n' + statStr
    fig = pl.figure()
    fig.suptitle(titleStr)
    pl.plot(x, y, color='b', label = 'Original')
    pl.plot(x, .5*noise, color='g', label = 'Second Gaussian')
    pl.plot(x, data, color='r', label='Sum')
    pl.plot(x, newGauss, color='k', label = 'Sum Fit')
    pl.legend(loc=2)
    pl.show()
    
def binSumFit(u, s, nu, ns):
    bins = 50
    x = np.linspace(-10, 10, 1000, endpoint=True)
    
    np.random.seed(0)
    data = np.random.normal(u, s, 1000)
    noise = np.random.normal(nu, ns, 100)
    binSum = np.concatenate((data, noise))
    
    mu, sigma = st.norm.fit(binSum)
    mu2, sigma2 = st.norm.fit(data)
    gaussFit = st.norm.pdf(x, mu, sigma)
    gaussData = gaussianpdf(x, mu2, sigma2)
    
    params = [u, s, nu, ns]
    stats = [mu, sigma]
    paramStr = "Params (mean1, sig1, mean2, sig2): " + ', '.join([str(param) for param in params])
    statStr = 'Stats of Sum (mean, sigma): ' + ', '.join([str(np.round(stat, 2)) for stat in stats])
    titleStr = 'Sum of two 1-D Gaussian Samples\n' + paramStr + '\n' + statStr
    
    fig = pl.figure()
    fig.suptitle(titleStr)
    
    pl.hist(binSum, bins, normed=True, range=[min(x), max(x)], color='b', label = 'Sum')    
    pl.hist(data, bins, normed=True, range=[min(x), max(x)], color='r', label = 'Data Sample', alpha = .5)
    pl.hist(noise, bins, normed=True, range=[min(x), max(x)], color='g', label = 'Noise Sample', alpha = .5)
    pl.plot(x, gaussFit, color='b', label = 'Sum Fit')
    pl.plot(x, gaussData, color='r', label = 'Data Gauss')
    pl.legend(loc=2)
    pl.show()
    
if __name__ == "__main__":
		main()