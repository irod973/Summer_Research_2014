import numpy as np
import scipy as sc
from scipy.stats import kde
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import astroML.stats as ast

#Irving Rodriguez
#06/26/2014
#Version 2
#Smooth vs. Noisy Gaussian Plotter
#This function constructs a smooth, 2-Dimensional gaussian, adds noise to it, finds a best-fit gaussian to the new data, and compares the plots of the smooth gaussian vs. the noisy gaussian.
#V2: Implemented seed(1) to noise generator. Upgraded noise generator from 1D 
#Notes: adding noise in 1 direction vs. 2 (multivariate_normal), using seed vs. random, robust vs. non-robust fit


def main():
    
    X_MEAN = 3
    Y_MEAN = 4 #Centroid coordinates
    ALPHA = np.pi/5 #Angle between semi-major axis of ellipse and x-axis
    MAJOR = 2
    MINOR = 1 #Semi-major and semi-minor axes of 1-Sigma contour
    
    X_MIN = -10
    X_MAX = 10
    Y_MIN = X_MIN
    Y_MAX = X_MAX
    dx = .1
    dy = dx
    
    u = [X_MEAN, Y_MEAN]
    o = comVarMat(MAJOR, MINOR, ALPHA)
    x, y = np.mgrid[X_MIN:X_MAX:dx, Y_MIN:Y_MAX:dy] #Coordinate system
   
    gauss = gaussianpdf2D((x, y), u[0], u[1], o[0][0], o[0][1], o[1][1]) #Construct smooth gaussian
    
    fitBins(u, o)
    #exStat = fitSmooth((x,y), u, o, gauss) #Add noise to the gaussian, find a gaussian of best fit, extract that gaussian's parameter
    #exGauss = gaussianpdf2D((x,y), exStat[0], exStat[1], exStat[2], exStat[3], exStat[4]) #construct the best-fit gaussian
    
    exRStat = fitBinsRobust(u, o) #Fitting to Gaussian samples using astroML package
    xBar = exRStat[0]
    s = comVarMat(exRStat[1], exRStat[2], exRStat[3])
    
    exGauss = gaussianpdf2D((x, y), xBar[0], xBar[1], s[0][0], s[0][1], s[1][1])
    
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.contour(x, y, exGauss.reshape(len(x), len(y)), color='k', alpha=.5)
    ax.contour(x, y, gauss.reshape(len(x),len(y)), alpha = .5)
    pl.show()
    #plotGaussian((x, y), gauss.reshape(len(x),len(y))) #plot the smooth gaussian and its contours
    #plotGaussian((x, y), exGauss.reshape(len(x), len(y))) #plot the noisy gaussian and its contours
    
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
    gaussSamples = 20000
    
    #Draw random samples
    np.random.seed(1)
    data = np.random.multivariate_normal(u, o, gaussSamples)
    
    #Flatten data
    xData = data.ravel()[::2]
    yData = data.ravel()[1::2]
    
    #Non-Robust Stats
    #statsNR = ast.fit_bivariate_normal(xData, yData)
    
    statsR = ast.fit_bivariate_normal(xData, yData, robust=True)
   
    return statsR
      
def fitBins(u, o):
    gaussSamples = 5000
    noiseSamples = 400
    
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
    
    gaussians = [(xGauss, yGauss), (noisyGaussX, noisyGaussY)]
    for (noisyGaussX, noisyGaussY) in gaussians:
    
        #Find mins and maxes of gaussian samples
        xMin = noisyGaussX.min()
        xMax = noisyGaussX.max()
        yMin = noisyGaussY.min()
        yMax = noisyGaussY.max()
        
        #Create evenly spaced coordinates from mins and maxes
        X, Y = np.mgrid[xMin:xMax:200j, yMin:yMax:200j]
        
        #Make stacked vectors for coordinates
        coords = np.vstack([X.ravel(), Y.ravel()])
        
        #Make stacked vectors for drawn samples
        data = np.vstack([noisyGaussX, noisyGaussY])
        
        #Estimate pdf from drawn samples
        density = kde.gaussian_kde(data)
        
        #Evaluate Gaussian from coordinates
        estGauss = density(coords)
        
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.contourf(X,Y, estGauss.reshape(len(X), len(Y)))
        ax.plot(noisyGaussX, noisyGaussY, 'k.', alpha = .2)
        plotGaussian((X, Y), estGauss.reshape(len(X), len(Y)))
    
        #pl.hist2d(xGauss, yGauss, bins=100, range=[[-10,10],[-10,10]], alpha = .5)
        #pl.hist2d(xNoise, yNoise, bins=100, range=[[-10,10],[-10,10]], alpha = .5)
        #pl.hist2d(noisyX, noisyY, bins=100, range=[[-10,10],[-10,10]]) 
        
#Adds noise to a gaussian distribution, finds the best-fit 2D gaussian, and extracts the best fit's parameters
#Input: Coordinates, centroid matrix, covariance matrix
#Output: Array of extracted statistics (centroid of fit (2), second central moments of fit (3))
def fitSmooth((x, y), u, o, gauss):
    noiseGauss = gaussianpdf2D((x,y), -5, -5, 2, 1, 2)
    
    newGauss = gauss + .1*noiseGauss
    
    plotGaussian((x,y), newGauss.reshape(len(x),len(y)))
    
    params = [u[0], u[1], o[0][0], o[0][1], o[1][1]]
    stats, err = curve_fit(gaussianpdf2D, (x,y), newGauss, p0=params)
    fitGauss = gaussianpdf2D((x,y), stats[0], stats[1], stats[2], stats[3], stats[4])
    plotGaussian((x, y), fitGauss.reshape(len(x),len(y)))
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
def plotGaussian((x, y), gauss):
    fig, ax = pl.subplots(1, 2, figsize=(12,6))
    
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].plot_surface(x, y, gauss)
    ax[0].axis('tight')
    
    cax = ax[1].contourf(x, y, gauss)
    ax[1].set_xlim([-10,10])
    ax[1].set_ylim([-10,10])
    
    pl.colorbar(cax)
    pl.show()
    
if __name__ == "__main__":
    main()