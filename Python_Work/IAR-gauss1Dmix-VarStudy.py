import numpy as np
import scipy.integrate as sint
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl
from matplotlib.widgets import Slider, Button, RadioButtons

#A study of the behavior of Variance in a 1D Gaussian Mixture (binned)
#Will Show 5 Plots:
#1. Variance of Best Fitvs. Separation
#2. Mean of Fit vs. Separation
#3. Variance of Fit vs. Flux Fraction
#4. Gaussian Mixture (2 profiles)
#5. Chi-Squared Colormap

def main():
    #Fixed parameters
    DOMINANT_FLUX_FRACTION = .5
    MEAN = 0
    SIGMA = 1   
    NOISE_MEAN = 2
    NOISE_SIGMA = 1

    #Varying parameters
    sep = abs(MEAN-NOISE_MEAN)
    separationRange = np.linspace(MEAN-abs(sep),MEAN+abs(sep), 21)
    fractionRange = np.linspace(.5, .99, 50)


    #Plot 5 plots on figure
    fig = pl.figure()
    #Fits to mixture of two Gaussians based on parameters specified above
    fitParams, err, binCent, hist = fitMixture(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA, DOMINANT_FLUX_FRACTION, fig, plot=1)
    #Plots chi-squared distribution between mixture and fit
    chi2=chiSq(binCent, hist, fitParams, fig)
    
    #Varies separation and finds variance of fit
    var, analyticVar = varySepFindVariance(MEAN, SIGMA, separationRange, NOISE_SIGMA, DOMINANT_FLUX_FRACTION, fig)
    #Varies separation and finds mean of fit
    separation, fitMean = varySepFindMean(MEAN, SIGMA, separationRange, NOISE_SIGMA, DOMINANT_FLUX_FRACTION, fig)
    #Varies flux fraction and finds variance of fit
    varyFluxFraction(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA, fractionRange, fig)
    
    pl.show()
    
def chiSq(x, mixture, params, fig):
    #Make nxn grid
    n = 50
    chiGrid = np.zeros([n,n])
      
    meanMin = params[0]-params[0]/5
    meanMax = params[0]+params[0]/5
    sigMin = params[1]-params[1]/5
    sigMax = params[1]+params[1]/5
    for i in range(n):
        for j in range(n):
            mean = meanMin + (meanMax-meanMin)*float(i)/(n-1)
            sig = sigMin + (sigMax-sigMin)*float(j)/(n-1)
            chi = 0.0
            for bin in range(len(x)):
                resid = (mixture[bin] - gaussianpdf(x[bin], mean, sig, params[2]))#/np.sqrt(mixture[bin])#Leave in sqrt's for ChiSq, else leastSq fit
                chi += resid**2
            chiGrid[n-1-j,i] = chi

    #Plot chiGrid
    ax = fig.add_subplot(235)
    ax = pl.imshow(chiGrid,extent=[meanMin, meanMax, sigMin, sigMax])
    pl.xlabel('Mean')
    pl.ylabel('Sigma')
    pl.colorbar(ax)

    return chiGrid
        
def varyFluxFraction(mean, sigma, contMean, contSigma, fluxFractions, fig):
    var = []
    varErr = []
    
    #Extract the variance and error for movement through pa
    for frac in fluxFractions:
        fitParams, statErr, x, y = fitMixture(mean, sigma, contMean, contSigma, frac, fig)
        var.append(fitParams[1]**2)
        varErr.append(statErr[1])
        
    #Plot mean vs. mean difference
    ax = fig.add_subplot(233)
    ax.set_title('Variance vs. Flux Fraction')
    ax.errorbar(fluxFractions,var, yerr=varErr)
    ax.set_xlabel('Flux Fraction')
    ax.set_ylabel('Variance')
    return var, varErr

#Finds the mean of the fit for varying separation
def varySepFindMean(mean, sigma, sepRange, contSigma, fluxFraction, fig):
    sep = []
    fitMean = []
    fitMeanErr = []
    
    #Extract the mean and error for each movement of the mean of the first object
    for separation in sepRange:
        fitParams, statErr, x, y = fitMixture(mean, sigma, separation, contSigma, fluxFraction, fig)
        sep.append(separation-mean)
        fitMean.append(abs(fitParams[0]))
        fitMeanErr.append(statErr[0])
    
    #Plot mean vs. mean difference
    ax = fig.add_subplot(232)
    ax.set_title('Mean of Mixture vs. Delta Mean')
    ax.errorbar(sep, fitMean, yerr=fitMeanErr)
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Absolute Value of Mean of Fit')
    ax.set_xlim(min(sep)-1, max(sep)+1)
    return sep, fitMean
    
#Plots the variance of the fit as a function of the difference in means
def varySepFindVariance(mean, sigma, sepRange, contSigma, fluxFrac, fig):
    sep = []
    var = []
    varErr = []
    
    #Extract the variance and error for each movement of the mean of the first object
    for separation in sepRange:
        fitParams, statErr, x, y = fitMixture(mean, sigma, separation, contSigma, fluxFrac, fig)
        sep.append(separation-mean)
        var.append(fitParams[1]**2)
        varErr.append(statErr[1])
    #Calculate the analytic variance
    analyticVar = analyticMixVariance(sigma, contSigma, sep)
    
    #Plot variance vs. mean difference
    ax = fig.add_subplot(231)
    ax.set_title('Variance vs. Delta Mean')
    ax.errorbar(sep, var, yerr=varErr, label = 'From Mixture')
    ax.plot(sep, analyticVar, 'k', label = 'Analytic Variance')
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Variance')
    ax.set_xlim(min(sep)-1, max(sep)+1)
    pl.legend(loc=9, prop={'size':9})
    
    return var, analyticVar
    
#Formula for the variance of a 1D Gaussian Mixture
def analyticMixVariance(domSigma, contSigma, separations, domFrac=.5):
    contFrac = 1 - domFrac
    variance = [domFrac*domSigma**2 + contFrac*contSigma**2 + domFrac*contFrac*sep**2 for sep in separations] 
    return variance
    
def gaussianpdf(x, u, s, amp=1):
    return amp*np.exp(-(x-u)**2/(2*s**2))#/(np.sqrt(2*np.pi))
    
def fitMixture(mean, sigma, contMean, contSigma, fluxFrac, fig, plot=0):
    samples = 100000
    bins = 200
    contFluxFrac = 1 - fluxFrac
    dataSamples = fluxFrac*samples
    objSamples = contFluxFrac*samples
    
    #Set up bin space
    xMin = np.mean([mean, contMean]) - 15*sigma
    xMax = np.mean([mean, contMean]) + 15*sigma
    x = np.linspace(xMin, xMax, bins)
    binSize=(xMax-xMin)/bins
    
    #Draw samples for data and for undetected object
    np.random.seed(0)
    data = np.random.normal(mean, sigma, dataSamples)
    obj = np.random.normal(contMean, contSigma, objSamples)
    binSum = np.concatenate((data, obj))
    
    #Create histograms for data and object
    dataHist, dataEdg = np.histogram(data, x)
    objHist, objEdg = np.histogram(obj, x)
    sumHis, sumEdg = np.histogram(binSum, x)
    sumEdgCent = (sumEdg[:-1] + sumEdg[1:])/2
    sumHist = np.array(sumHis)
    sumHist[sumHist==0] = 1

    #Extract the paramaters of the sum fit
    p=[fluxFrac*mean+contFluxFrac*contMean, np.sqrt(analyticMixVariance(sigma, contSigma, [mean-contMean], fluxFrac)[0]), max(sumHist)]
    sumStats, sumErr = opt.curve_fit(gaussianpdf, sumEdgCent, sumHist, p0=p)#, sigma = np.sqrt(sumHist), absolute_sigma=True)#Remove sigma and absolute sigma if not a ChiSq test
    perr = np.sqrt(np.diag(sumErr))
    #Create fit
    sumFit = gaussianpdf(sumEdgCent, sumStats[0], sumStats[1], sumStats[2])

    if plot==1:
        #Check area of fit vs. area of sum
        sumArea = sum(binSize*sumHist)
        fitArea = sint.quad(gaussianpdf, -np.inf, np.inf,(sumStats[0], sumStats[1], sumStats[2]))
        #Title Strings
        params = [mean, sigma, contMean, contSigma]
        statsFit = [sumStats[0], sumStats[1], sumStats[2]]
        paramStr = "Params (meanData, sigData, meanObj, sigObj): " + ', '.join([str(param) for param in params])
        statStr = 'Stats of Sum Fit(mean, sigma, amp): ' + ', '.join([str(np.round(stat, 2)) for stat in statsFit])
        areaStr = 'Areas (fit, hist wBinSize='+str(binSize)+'): '+str(round(fitArea[0],2))+', '+str(sumArea)
        titleStr = 'Gaussian Mixture, '+ str(fluxFrac*100) + '% Red with '+str(contFluxFrac*100)+'% Green\n' + paramStr + '\n' + statStr + '\n' + areaStr  
        #Plot histograms
        ax = fig.add_subplot(234)
        ax.hist(binSum, sumEdgCent, color='b', label = 'Sum')
        ax.hist(data, sumEdgCent, color='r', label = 'Data Sample')
        ax.hist(obj, sumEdgCent, color='g', label = 'Noise Sample')
        ax.set_title('Mixture Fit')
        ax.text(sumStats[0]+22*sumStats[1], sumStats[2]/2, titleStr)
        #Plot fits
        ax.plot(sumEdgCent, sumFit, color='k', label = 'Sum Fit')
        ax.set_xlim(sumStats[0]-6*sumStats[1],sumStats[0]+6*sumStats[1])
        pl.legend(loc=2, prop={'size':7})
    return sumStats, perr, sumEdgCent, sumHist
    
if __name__ == "__main__":
		main()