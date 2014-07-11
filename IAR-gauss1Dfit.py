import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl

def main():
    #makeOneGaussian(7, 1)
    sim1DData()
    
def gaussianpdf(x, xo, s):
    return np.exp(-(x-xo)**2/(2*s**2)/np.sqrt(2*np.pi))

def sim1DData():
    x = np.linspace(-10,10, 10000)
    y = gaussianpdf(x, 3, 1)
    noise = gaussianpdf(x, -2, .5) 
    data = y + noise
    
    
    
    mu, sigma = opt.curve_fit(gaussianpdf, x, data, p0=[3, .5])
    newGauss = gaussianpdf(x, mu[0], mu[1])
    pl.plot(x, y, color='b')
    pl.plot(x, noise, color='g')
    pl.plot(x, newGauss, color='r')
    pl.show()
    
def makeOneGaussian(u, s):
    bins = 20
    x = np.linspace(-10, 10, 1000, endpoint=True)
    np.random.seed(0)
    pts = np.random.normal(u, s, x.shape)
    
    noise = np.random.normal(0, .5, x.shape)
    
    pl.hist(pts, bins, color='b')
    pl.hist(noise, bins, color='g')
    pl.hist(noise+pts, bins, color='r')
    mu, sigma = st.norm.fit(pts)
    
    gauss = st.norm.pdf(x, mu, sigma)
    pl.plot(x, gauss, color='k')
    #pl.figure(figsize=(3,3), dpi=100)
    #pl.plot(x, gauss)
    #pl.xticks(np.linspace(-20,20,5, endpoint=True))
    #pl.yticks(np.linspace(0, .8, 5, endpoint=True))
    pl.show()
    
if __name__ == "__main__":
		main()