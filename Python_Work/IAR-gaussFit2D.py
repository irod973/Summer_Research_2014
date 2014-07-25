import numpy as np
import scipy as sc
from scipy.stats import kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import astroML.stats as ast

#Irving Rodriguez
#07/14/2014
#2D Gaussian Work, plotting analytic fits to gaussian samples with noise and extracting parameters of fits

def main():
    
    X_MEAN = 4
    Y_MEAN = 3 #Centroid coordinates
    ALPHA = np.pi/5 #Angle between semi-major axis of ellipse and x-axis
    MAJOR = 2
    MINOR = 1 #Semi-major and semi-minor axes of 1-Sigma contour
    
    X_MIN = -10
    X_MAX = 13
    Y_MIN = X_MIN
    Y_MAX = X_MAX
    dx = .05
    dy = dx
    
    u = [X_MEAN, Y_MEAN]
    o = comVarMat(MAJOR, MINOR, ALPHA)
    x, y = np.mgrid[X_MIN:X_MAX:dx, Y_MIN:Y_MAX:dy] #Coordinate system
    
    #Analytic 2D Gaussian
    gauss = gaussianpdf2D((x, y), u[0], u[1], o[0][0], o[0][1], o[1][1]) #Construct smooth gaussian
    #params = [X_MEAN, Y_MEAN, MAJOR, MINOR, ALPHA]
    #titleStr = '2D Analytic Gaussian\nParams (centroid coords, semi-major, semi-minor, alpha) :' + ', '.join([str(round(param,2)) for param in params])
    #plotGaussian((x,y), gauss.reshape(len(x), len(y)), titleStr)
    
    #Fix fitBinsNR so that it outputs one plot instead of one million
    #Fit a single Gaussian to a mixture of a Gaussian sample and a noise sample
    #fitBinsNonRobust(u, o)
    
    #Fit to a sum of 2 analytic Gaussians
    #exStat = fitAnalyticSum((x,y), u, o, gauss) #Add noise to the gaussian, find a gaussian of best fit, extract that gaussian's parameter
    #exNRGauss = gaussianpdf2D((x,y), exStat[0], exStat[1], exStat[2], exStat[3], exStat[4]) #construct the best-fit gaussian
    #params = [X_MEAN, Y_MEAN, o[0][0], o[0][1], o[1][1]]
    #paramsNoise = [0, 8, 2, 1, 2]
    #titleStr = 'Fit on Sum\nOriginal (centroid coords, oX, oXY, oYY): ' + ', '.join([str(round(param,2)) for param in params]) + '\nSecond Gauss (centroid coords, oXX, oXY, oYY): ' + ', '.join([str(round(param,2)) for param in paramsNoise]) + '\nFit Params (centroid coords, oXX, oXY, oYY): ' + ', '.join([str(round(param,2)) for param in exStat])
    #plotGaussian((x,y), exNRGauss.reshape(len(x), len(y)), titleStr)
    
    #Fit a single Gaussian to a mixture of a Gaussian sample and a noise sample
    gaussSample, exRStat = fitBinsRobust(u, o) #Fitting to Gaussian samples using astroML package
    xBar = exRStat[0]
    s = comVarMat(exRStat[1], exRStat[2], exRStat[3])
    exRGauss = gaussianpdf2D((x, y), xBar[0], xBar[1], s[0][0], s[0][1], s[1][1])
    fig = pl.figure()
    params = [X_MEAN, Y_MEAN, o[0][0], o[0][1], o[1][1]]
    paramsNoise = [0, 8, 2, 1, 2]
    titleStr = 'Robust Fit to Noisy Sample (Black = No Noise)\nData Samples (5000 samples, centroid coords, oX, oXY, oYY): ' + ', '.join([str(round(param,2)) for param in params]) + '\nNoisy Samples (500, centroid coords, oXX, oXY, oYY): ' + ', '.join([str(round(param,2)) for param in paramsNoise])
    fig.suptitle(titleStr)
    ax = fig.add_subplot(111)
    ax.contourf(x, y, exRGauss.reshape(len(x), len(y)))
    ax.contour(x, y, gauss.reshape(len(x), len(y)), color=pl.cm.bone)
    ax.plot(gaussSample[0], gaussSample[1], 'k.', alpha=.1)
    pl.show()
    
#Constructs the values of a 2-Dimensional Gaussian distribution for a given coordinate system using the general Gaussian function for two dependent variables
#Note: This function returns a 1D array so that it may be called by scipy.optimize.curve_fit(), which expects a 1D function as its first parameter.
#Input: Coordinates, Centroid Values (2), Second Central Moments (3)
#Output: 2D Gaussian over the given coordinates, flattened to an array 
def gaussianpdf2D((x, y), uX, uY, oXX, oXY, oYY):
    p = oXY/(oXX*oYY)
    zSQ = (x - uX)**2/oXX**2 + (y - uY)**2/oYY**2 - 2*p*(x - uX)*(y - uY)/(oXX*oYY)
    gauss = 1/(2*np.pi*oXX*oYY*np.sqrt(1-p**2)) * np.exp(-zSQ/(2*(1-p**2)))
    return gauss.ravel() #Flattens Gaussian to 1D array
    
def fitBinsRobust(u, o):
    gaussSamples = 5000
    noiseSamples = 500
    
    #Draw random samples
    np.random.seed(1)
    gauss = np.random.multivariate_normal(u, o, gaussSamples)
    noise = np.random.multivariate_normal([0,8], [[2, 1],[1,2]], noiseSamples)
    
    #Add noise and flatten data
    xData = np.concatenate((gauss.ravel()[::2], noise.ravel()[::2]))
    yData = np.concatenate((gauss.ravel()[1::2], noise.ravel()[1::2]))
    data = [xData, yData]
    
    #Non-Robust Stats
    #stats = ast.fit_bivariate_normal(xData, yData)
    
    #Robust stats
    stats = ast.fit_bivariate_normal(xData, yData, robust=True)
   
    return data, stats
      
def fitBinsNonRobust(u, o):
    gaussSamples = 1000
    noiseSamples = 100
    
    np.random.seed(1)
    #Draw random samples from 2D Gaussians
    gaussBin = np.random.multivariate_normal(u, o, gaussSamples)
    noiseBin = np.random.multivariate_normal([0, 8], [[2, 1], [1, 2]], noiseSamples)
    
    #Flatten the samples
    xGauss = gaussBin.ravel()[0::2]
    yGauss = gaussBin.ravel()[1::2]
    xNoise = noiseBin.ravel()[0::2]
    yNoise = noiseBin.ravel()[1::2]
    
    #Add noise samples to Gaussian samples
    noisyGaussX = np.concatenate((xGauss, xNoise))
    noisyGaussY = np.concatenate((yGauss, yNoise))
    
    #Storing our data
    gaussFit = []
    gaussCoords = []
    noiseFit = []
    noiseCoords = [0,0]
    
    gaussians = [(xGauss, yGauss), (noisyGaussX, noisyGaussY)]
    counter = 0
    ex = []
    why = []
    for (xData, yData) in gaussians:
        #Find mins and maxes of gaussian samples
        xMin = xData.min()
        xMax = xData.max()
        yMin = yData.min()
        yMax = yData.max()
        
        #Create evenly spaced coordinates from mins and maxes
        X, Y = np.mgrid[xMin:xMax:200j, yMin:yMax:200j]
        
        #Make stacked vectors for coordinates
        coords = np.vstack([X.ravel(), Y.ravel()])
        
        #Make stacked vectors for drawn samples
        data = np.vstack([xData, yData])
        
        #Estimate pdf from drawn samples (requires stacked vectors)
        density = kde.gaussian_kde(data)
        
        #Fit the Gaussian
        fitGauss = density(coords)
        
        #Get desired coordinates and fit
        if counter == 0:
            gaussFit = fitGauss
            gaussCoords.append(X)
            gaussCoords.append(Y)
            counter += 1
        if counter == 1:
            noiseFit = fitGauss
            noiseCoords[0] = X
            noiseCoords[1] = Y
    
    #Plot samples and fits for noise and no noise   
    fig = pl.figure()
    fig.suptitle('KDE Fit to Gaussian Samples (Non-Robust)')
    
    #First, no noise
    ax1 = fig.add_subplot(211)
    gaussParams = [u[0], u[1], o[0][0], o[0][1], o[1][1]]
    noNoiseTitle = 'No Noise, with Fit (' + str(gaussSamples) + ' samples)\nParams (centroid coords, sigX, sigXY, sigY): ' + ', '.join([str(round(param,2)) for param in gaussParams])
    ax1.set_title(noNoiseTitle)
    ax1.contourf(gaussCoords[0],gaussCoords[1], gaussFit.reshape(len(gaussCoords[0]), len(gaussCoords[1])))
    ax1.plot(xGauss, yGauss, 'k.', alpha = .3)
    
    #Noise added
    ax4 = fig.add_subplot(212)
    noiseParams = [0, 8, 2, 1, 2]
    noiseTitle = 'Noise, with Fit (' + str(noiseSamples) + ' noise samples)\nNoiseParams (centroid coords, sigX, sigXY, sigY): ' + ', '.join([str(round(param,2)) for param in noiseParams])
    ax4.set_title(noiseTitle)
    ax4.contourf(noiseCoords[0],noiseCoords[1], noiseFit.reshape(len(noiseCoords[0]), len(noiseCoords[1])))
    ax4.plot(noisyGaussX, noisyGaussY, 'k.', alpha=.3)
    pl.show()

    #Compare Original to Fit      
        
        #pl.hist2d(xGauss, yGauss, bins=100, range=[[-10,10],[-10,10]], alpha = .5)
        #pl.hist2d(xNoise, yNoise, bins=100, range=[[-10,10],[-10,10]], alpha = .5)
        #pl.hist2d(noisyGaussX, noisyGaussY, bins=100, range=[[-10,10],[-10,10]])
        #pl.show()
        
#Adds a small gaussian to a gaussian distribution, finds the best-fit 2D gaussian, and extracts the best fit's parameters
#Input: Coordinates, centroid matrix, covariance matrix, original distribution
#Output: Array of extracted statistics (centroid of fit (2), second central moments of fit (3))
def fitAnalyticSum((x, y), u, o, gauss):
    noiseGauss = .4*gaussianpdf2D((x,y), -5, -5, 2, 1, 2)
    
    newGauss = gauss + noiseGauss
    
    titleStr = 'Sum of 2 Gaussians'
    plotGaussian((x,y), newGauss.reshape(len(x),len(y)), titleStr)
    
    params = [u[0], u[1], o[0][0], o[0][1], o[1][1]]
    stats, err = curve_fit(gaussianpdf2D, (x,y), newGauss, p0=params)
    return stats
    
#Computes the matrix of second moments using the geometry of the elliptical contour at an angle alpha to the x-axis in real space.
#Input: Semi-Major Axis, Semi-Minor Axis, angle between major axis and x-axis.
#Returns: 2x2 Symmetric Matrix of Second Moments
def comVarMat(major, minor, alpha):
    oXX = (major**2 * (np.cos(alpha))**2 + minor**2 * (np.sin(alpha))**2)**(.5)
    oYY = (major**2 * (np.sin(alpha))**2 + minor**2 * (np.cos(alpha))**2)**(.5)
    oXY = ((major**2 - minor**2)*np.sin(alpha)*np.cos(alpha))**.5
    o = [[oXX, oXY], [oXY, oYY]]
    return o
    
#Constructs the values of a 2-Dimensional Gaussian distribution for a given coordinate system using the multivariate function from scipy.
#Input: Coordinates, 2x1 matrix of centroid coordinates, 2x2 matrix of second moments
#Output: 2D Gaussian over the given coordinates
def make2DGaussian(x, y, u, o):
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    
    gauss = sc.stats.multivariate_normal.pdf(pos, u, o)
    
    return gauss
    
#Plots a given 2D Gaussian and its contours for a given coordinate system.
#Input: Tuple of Coordinates, 2D Gaussian
#Output: 3D Surface Plot and 2D Contour plot
def plotGaussian((x, y), gauss, titleStr=''):
    fig, ax = pl.subplots(1, 2, figsize=(12,6))
    fig.suptitle(titleStr)
    
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].plot_surface(x, y, gauss)
    ax[0].axis('tight')
    
    cax = ax[1].contourf(x, y, gauss)
    ax[0].axis('tight')
    
    pl.colorbar(cax)
    pl.show()
    
if __name__ == "__main__":
    main()