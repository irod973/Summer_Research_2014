import numpy as np
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl

#Irving Rodriguez
#
#This file creates a mixture of two 1D Gaussians and fits using one profile. Will do an analytic and binned mixture.
#Summer 2014

def main():
    FLUX_FRACTION = .5
    MEAN = 0
    SIGMA = 3
    AMP = 30
    
    NOISE_MEAN = 5
    NOISE_SIGMA = 3
    NOISE_AMP = 30
    
    #Fits a Gaussian to the sum of two analytic Gaussians
    analyticSumFit(MEAN, SIGMA, AMP, NOISE_MEAN, NOISE_SIGMA, NOISE_AMP)
    
    #Fits a Gaussian to the sum of two Gaussian histograms
    binSumFit(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA, FLUX_FRACTION)

def gaussianpdf(x, u, s, amp=1):
    return amp*np.exp(-(x-u)**2/(2.0*s**2))#/(np.sqrt(2*np.pi))

def analyticSumFit(mean, sig, amp, nmean, nsigma, namp):
    dataWeight = .5
    objWeight = 1 - dataWeight
    
    x = np.linspace(-10,15, 1000)
    #Create two analytic functions
    data = dataWeight*gaussianpdf(x, mean, sig, amp)
    obj = objWeight*gaussianpdf(x, nmean, nsigma, amp)
    dataSum = data + obj
       
    #Fit to the sum of the two functions
    stats, error = opt.curve_fit(gaussianpdf, x, dataSum, p0=[0, 1, 30])
    #Draw fit using returned parameters from curve_fit
    sumFit = gaussianpdf(x, stats[0], stats[1], stats[2])
    
    #Plot
    fig = pl.figure()
    params = [mean, sig, amp, nmean, nsigma, namp]
    paramStr = "Params (mean1, sig1, amp1, mean2, sig2, amp2): " + ', '.join([str(param) for param in params])
    statStr = 'Stats of Sum Fit(mean, sigma): ' + ', '.join([str(round(stat, 2)) for stat in stats])
    titleStr = 'Sum of two 1-D Gaussians\n' + paramStr + '\n' + statStr
    #fig.suptitle(titleStr)
    ax = fig.add_subplot(111)
    ax.plot(x, data, color='b', label = 'Dominant Galaxy', linewidth=3)
    ax.plot(x, obj, color='g', label = 'Contaminant Galaxy', linewidth=3)
    #ax.plot(x, dataSum, color='r', label='Sum')
    ax.plot(x, sumFit,'k--', label = 'Single Galaxy Fit', linewidth=3)
    ax.set_xlim(-10, 15)
    ax.set_ylim(0, 40)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('Flux', fontsize=18)
    pl.legend(loc=9, prop={'size':18})
    pl.show()
    
def binSumFit(mean, sigma, contaminantMean, contaminantSigma, dominantFrac = .5):
    samples = 10000
    bins = 150
    contaminantFrac = 1 - dominantFrac
    dataSamples = dominantFrac*samples
    objSamples = contaminantFrac*samples
    
    #Set up bin space
    xMin = np.mean([mean, contaminantMean]) - (bins/10)*sigma
    xMax = np.mean([mean, contaminantMean]) + (bins/10)*sigma
    x = np.linspace(xMin, xMax, bins)
    
    #Draw samples for data and for meanndetected object
    np.random.seed(0)
    data = np.random.normal(mean, sigma, dataSamples)
    obj = np.random.normal(contaminantMean, contaminantSigma, objSamples)
    binSum = np.concatenate((data, obj))
    
    #Create histograms for data and object
    dataHist, dataEdg = np.histogram(data, x)
    objHist, objEdg = np.histogram(obj, x)
    sumHist, sumEdg = np.histogram(binSum, x)
    sumEdgCent = (sumEdg[:-1] + sumEdg[1:])/2
    
    #Extract the paramaters of the sum fit
    p=[dominantFrac*mean+contaminantFrac*contaminantMean, sigma, max(sumHist)]
    sumStats, sumErr = opt.curve_fit(gaussianpdf, sumEdgCent, sumHist, p0=p)
    #Create fit
    sumFit = gaussianpdf(sumEdgCent, sumStats[0], sumStats[1], sumStats[2])

    #Title Strings
    params = [mean, sigma, contaminantMean, contaminantSigma]
    statsFit = [sumStats[0], sumStats[1], sumStats[2]]
    paramStr = "Params (meanData, sigData, meanObj, sigObj): " + ', '.join([str(param) for param in params])
    statStr = 'Stats of Sum Fit(mean, sigma, amp): ' + ', '.join([str(np.round(stat, 2)) for stat in statsFit])
    titleStr = 'Gaussian Mixture, '+ str(dominantFrac*100) + '% Red with '+str(contaminantFrac*100)+'% Green\n' + paramStr + '\n' + statStr    
 
    #Plotting our results
    fig = pl.figure()
    fig.suptitle(titleStr)
 
    #Plot histograms
    ax = fig.add_subplot(111)
    ax.hist(binSum, sumEdgCent, color='b', label = 'Sum')
    ax.hist(data, sumEdgCent, color='r', label = 'Data Sample')
    ax.hist(obj, sumEdgCent, color='g', label = 'Noise Sample')
    #Plot fits
    ax.plot(sumEdgCent, sumFit, color='k', label = 'Sum Fit')
    ax.set_xlim(sumStats[0]-6*sumStats[1],sumStats[0]+6*sumStats[1])
    pl.legend(loc=2)
    pl.show()
    
if __name__ == "__main__":
		main()