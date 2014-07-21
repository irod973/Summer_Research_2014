import numpy as np
import scipy.integrate as sint
from scipy import optimize as opt
from scipy import stats as st
from matplotlib import pyplot as pl
from matplotlib.widgets import Slider, Button, RadioButtons

#A study of the behavior of Variance in a 1D Gaussian Mixture (binned)

def main():
    #When keeping parameters fixed
    PA = .5
    MEAN = 0
    SIGMA = 1   
    NOISE_MEAN = 2
    NOISE_SIGMA = 1
    #When moving through parameter space
    diff = abs(MEAN-NOISE_MEAN)
    nmean = np.linspace(MEAN-abs(diff),MEAN+abs(diff), 21)
    p = np.linspace(.5, .9, 21)

    fig = pl.figure()
    sumStats, err, binCent, hist = fitSum(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA, PA, fig, plot=1)
    chi2=chiSq(binCent, hist, sumStats[2], fig)
    
    var, anVar = varFit(MEAN, SIGMA, nmean, NOISE_SIGMA, PA, fig)
    du, mu = meanFit(MEAN, SIGMA, nmean, NOISE_SIGMA, PA, fig)
    propFit(MEAN, SIGMA, NOISE_MEAN, NOISE_SIGMA, p, fig)
    
    pl.show()
    
    #index1 = np.argmin(chi2, axis=0)
    #index2 = np.argmin(np.argmin(chi2, axis =0), axis=0)
    #Min = chi2[index1][index2]
    #print mini;print Min
    
def chiSq(data, values, amp, fig):
    #Make nxn grid
    n = 40
    chiGrid = np.zeros([n,n])
      
    meanMin = -7
    meanMax = 7
    sigMin = 0.1
    sigMax = 14
    for i in range(n):
        for j in range(n):
            mean = meanMin + (meanMax-meanMin)*float(i)/(n-1)
            sig = sigMin + (sigMax-sigMin)*float(j)/(n-1)
            chi = 0.0
            for l in range(len(data)):
                resid = (values[l] - gaussianpdf(data[l], mean, sig, amp))/np.sqrt(100000)/np.sqrt(values[l])#Leave in sqrt's for ChiSq, else leastSq fit
                chi = chi + resid**2
            chiGrid[n-1-j,i] = chi
            
    #Plot chiGrid
    ax = fig.add_subplot(235)
    ax = pl.imshow(chiGrid,extent=[meanMin, meanMax, sigMin, sigMax])
    pl.xlabel('Mean')
    pl.ylabel('Sigma')
    pl.colorbar(ax)
    
    #Get the index of the first min
    index = np.unravel_index(np.argmin(chiGrid), chiGrid.shape)
    #Plot the location of the bin
    #mean = meanMin+(meanMax-meanMin)*float(index[1])/(n-1)
    #sig = sigMin+(sigMax-sigMin)*float(n-1-index[0])/(n-1)
    #pl.plot(mean, sig, 'wD')
    #pl.xlim(meanMin, meanMax)
    #pl.ylim(sigMin, sigMax)
    #a = np.where(chiGrid.ravel()==chiGrid[index[0]][index[1]])    #Number of mins
    #print 'There are ',len(a),'mins in this distr'
    #print 'The min, ',chi2[index[0]][index[1]],', is at ',index
    
    return chiGrid
        
def propFit(u, s, nu, ns, p, fig):
    var = []
    varErr = []
    
    #Extract the variance and error for movement through pa
    for pa in p:
        stats, statErr, x, y = fitSum(u, s, nu, ns, pa, fig)
        var.append(stats[1]**2)
        varErr.append(statErr[1])
        
    #Plot mean vs. mean difference
    ax = fig.add_subplot(233)
    ax.set_title('Variance vs. pa')
    ax.errorbar(p,var, yerr=varErr)
    ax.set_xlabel('pa')
    ax.set_ylabel('Variance')
    return var, varErr

def meanFit(mean, s, nmean, ns, pa, fig):
    du = []
    mu = []
    muErr = []
    
    #Extract the mean and error for each movement of the mean of the first object
    for u in nmean:
        stats, statErr, x, y = fitSum(mean, s, u, ns, pa, fig)
        du.append(u-mean)
        mu.append(abs(stats[0]))
        muErr.append(statErr[0])
    
    #Plot mean vs. mean difference
    ax = fig.add_subplot(232)
    ax.set_title('Mean of Mixture vs. Delta Mean')
    ax.errorbar(du, mu, yerr=muErr)
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Absolute Value of Mean of Fit')
    ax.set_xlim(min(du)-1, max(du)+1)
    return du, mu
    
#Plots the variance of the fit as a function of the difference in means
def varFit(mean, s, nmean, ns, pa, fig):
    du = []
    var = []
    varErr = []
    
    #Extract the variance and error for each movement of the mean of the first object
    for u in nmean:
        stats, statErr, x, y = fitSum(mean, s, u, ns, pa, fig)
        du.append(u-mean)
        var.append(stats[1]**2)
        varErr.append(statErr[1])
    #Calculate the analytic variance
    anVar = analyticMixVar(s, ns, du)
    
    #Plot variance vs. mean difference
    ax = fig.add_subplot(231)
    ax.set_title('Variance vs. Delta Mean')
    ax.errorbar(du, var, yerr=varErr, label = 'From Mixture')
    ax.plot(du, anVar, 'k', label = 'Analytic Variance')
    ax.set_xlabel('Delta Mean')
    ax.set_ylabel('Variance')
    ax.set_xlim(min(du)-1, max(du)+1)
    pl.legend(loc=9, prop={'size':9})
    
    return var, anVar
    
#Formula for the variance of a 1D Gaussian Mixture
def analyticMixVar(s, ns, du, pa=.5):
    pb = 1 - pa
    var = [pa*s**2 + pb*ns**2 + pa*pb*delta**2 for delta in du] 
    return var
    
def gaussianpdf(x, u, s, amp=1):
    return amp*np.exp(-(x-u)**2/(2*s**2))#/(np.sqrt(2*np.pi))
    
def fitSum(u, s, nu, ns, pa, fig, plot=0):
    samples = 1000000
    bins = 200
    pb = 1 - pa
    dataSamples = pa*samples
    objSamples = pb*samples
    
    #Set up bin space
    xMin = np.mean([u, nu]) - 15*s
    xMax = np.mean([u, nu]) + 15*s
    x = np.linspace(xMin, xMax, bins)
    binSize=(xMax-xMin)/bins
    
    #Draw samples for data and for undetected object
    np.random.seed(0)
    data = np.random.normal(u, s, dataSamples)
    obj = np.random.normal(nu, ns, objSamples)
    binSum = np.concatenate((data, obj))
    
    #Create histograms for data and object
    dataHist, dataEdg = np.histogram(data, x)
    objHist, objEdg = np.histogram(obj, x)
    sumHis, sumEdg = np.histogram(binSum, x)
    sumEdgCent = (sumEdg[:-1] + sumEdg[1:])/2
    sumHist = np.array(sumHis)
    sumHist[sumHist==0] = 1

    #Extract the paramaters of the sum fit
    p=[pa*u+pb*nu, np.sqrt(analyticMixVar(s, ns, [u-nu], pa)[0]), max(sumHist)]
    sumStats, sumErr = opt.curve_fit(gaussianpdf, sumEdgCent, sumHist, p0=p, sigma = np.sqrt(sumHist), absolute_sigma=True)#Remove sigma and absolute sigma if not a ChiSq test
    perr = np.sqrt(np.diag(sumErr))
    #Create fit
    sumFit = gaussianpdf(sumEdgCent, sumStats[0], sumStats[1], sumStats[2])
   
    if plot==1:
        #Check area of fit vs. area of sum
        sumArea = sum(binSize*sumHist)
        fitArea = sint.quad(gaussianpdf, -np.inf, np.inf,(sumStats[0], sumStats[1], sumStats[2]))
        #Title Strings
        params = [u, s, nu, ns]
        statsFit = [sumStats[0], sumStats[1], sumStats[2]]
        paramStr = "Params (meanData, sigData, meanObj, sigObj): " + ', '.join([str(param) for param in params])
        statStr = 'Stats of Sum Fit(mean, sigma, amp): ' + ', '.join([str(np.round(stat, 2)) for stat in statsFit])
        areaStr = 'Areas (fit, hist wBinSize='+str(binSize)+'): '+str(round(fitArea[0],2))+', '+str(sumArea)
        titleStr = 'Gaussian Mixture, '+ str(pa*100) + '% Red with '+str(pb*100)+'% Green\n' + paramStr + '\n' + statStr + '\n' + areaStr  
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