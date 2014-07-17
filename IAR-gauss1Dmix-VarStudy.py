import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl

#A study of the behavior of Variance in a 1D Gaussian Mixture (binned)

def main():
        
    MEAN = 5
    SIGMA = 1
    AMP = 60
    
    NOISE_MEAN = 0
    NOISE_SIGMA = 1
    NOISE_AMP = 25
    
    diff = NOISE_MEAN-MEAN
    mean = range(NOISE_MEAN-abs(diff),NOISE_MEAN+abs(diff))
    varFit(mean, SIGMA, NOISE_MEAN, NOISE_SIGMA)
    meanFit(mean, SIGMA, NOISE_MEAN, NOISE_SIGMA)
    #mixFit(mean, SIGMA, NOISE_MEAN, NOISE_SIGMA)

def meanFit(mean, s, nmean, ns):
    du = []
    mu = []
    muErr = []
    
    #Extract the mean and error for each movement of the mean of the first object
    for u in mean:
        stats, statErr = fitSum(u, s, nmean, ns)
        du.append(u-nmean)
        mu.append(abs(stats[0]))
        muErr.append(statErr[0])
    
    #Plot mean vs. mean difference
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(du, mu, yerr=muErr, label = 'From Mixture')
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Absolute Value of Mean of Fit')
    pl.legend(loc=2)
    pl.show()    
    
#Plots the variance of the fit as a function of the difference in means
def varFit(mean, s, nmean, ns):
    du = []
    var = []
    varErr = []
    
    #Extract the variance and error for each movement of the mean of the first object
    for u in mean:
        stats, statErr = fitSum(u, s, nmean, ns)
        du.append(u-nmean)
        var.append(stats[1]**2)
        varErr.append(statErr[1])
    print var    
    #Calculate the analytic variance
    anVar = analyticMixVar(s, ns, du)
    
    #Plot variance vs. mean difference
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(du, var, yerr=varErr, label = 'From Mixture')
    ax.plot(du, anVar, 'k', label = 'Analytic Variance')
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Variance')
    pl.legend(loc=2)
    pl.show()
    
#Formula for the variance of a 1D Gaussian Mixture
def analyticMixVar(s, ns, du, pa=.5):
    pb = 1 - pa
    var = [pa*s**2 + pb*ns**2 + pa*pb*delta**2 for delta in du] 
    return var
    
def gaussianpdf(x, u, s, amp=1):
    return amp*np.exp(-(x-u)**2/(2*s**2))#/(np.sqrt(2*np.pi))
    
def fitSum(u, s, nu, ns, pa=.5):
    samples = 10000
    bins = 150
    pa = .5
    pb = 1 - pa
    dataSamples = pa*samples
    objSamples = pb*samples
    
    #Set up bin space
    xMin = np.mean([u, nu]) - 15*s
    xMax = np.mean([u, nu]) + 15*s
    x = np.linspace(xMin, xMax, bins)
    
    #Draw samples for data and for undetected object
    np.random.seed(0)
    data = np.random.normal(u, s, dataSamples)
    obj = np.random.normal(nu, ns, objSamples)
    binSum = np.concatenate((data, obj))
    
    #Create histograms for data and object
    dataHist, dataEdg = np.histogram(data, x)
    objHist, objEdg = np.histogram(obj, x)
    sumHist, sumEdg = np.histogram(binSum, x)
    dataEdgCent = (dataEdg[:-1] + dataEdg[1:])/2
    sumEdgCent = (sumEdg[:-1] + sumEdg[1:])/2
    
    #Extract the paramaters of the sum fit
    p=[np.mean([u, nu]), s, max(sumHist)]
    sumStats, sumErr = opt.curve_fit(gaussianpdf, sumEdgCent, sumHist, p0=p)
    perr = np.sqrt(np.diag(sumErr))
    
    #If only fit parameters desired, can return here. Sigma is [1]
    return sumStats, perr
    
if __name__ == "__main__":
		main()