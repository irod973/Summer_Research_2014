import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl

def main():
    #makeOneGaussian(5, 2)
    sim1DData()
    
def gaussianpdf(x, xo, s):
    return np.exp(-(xo-x)**2/(2*s**2)/np.sqrt(2*np.pi))

def sim1DData():
    x = np.linspace(-10,10, 10000)
    y = gaussianpdf(x, 3, .5)
    data = y + .1*np.random.normal(0, .5, len(x))

    mu, sigma = opt.curve_fit(gaussianpdf, x, data, p0=[3, .5])
    
    print mu
    print sigma

def makeOneGaussian(u, s):
    x = np.linspace(-20, 20, 1000, endpoint=True)
    gauss = st.norm(u, s)
    
    pl.figure(figsize=(3,3), dpi=100)
    pl.plot(x, gauss.pdf(x))
    pl.xticks(np.linspace(-20,20,5, endpoint=True))
    pl.yticks(np.linspace(0, .8, 5, endpoint=True))
    pl.show()
    
if __name__ == "__main__":
		main()