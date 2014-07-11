import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d.axes3d import Axes3D

#Irving Rodriguez
#06/25/2014
#Version 3
#2D Gauss Plotter
#This function plots a 2-Dimensional Gaussian using the pdf function from scipy and its projection based on 5 parameters:
#Centroid coordiantes (2 parameters)
#Angle between semi-major axis of ellipse and x-axis (1 parameter)
#Semi-major and semi-minor axes of 1-Sigma contour (2 parameters)


def main():       
    X_MEAN = 3
    Y_MEAN = 4 #Centroid coordinates
    ALPHA = np.pi/5 #Angle between semi-major axis of ellipse and x-axis
    MAJOR = 2
    MINOR = 1 #Semi-major and semi-minor axes of 1-Sigma contour
    
    X_MEAN_2 = 5
    Y_MEAN_2 = 0 #Centorid coordinates
    ALPHA_2 = np.pi/5 #Angle between semi-major axis of ellipse and x-axis
    MAJOR_2 = 5
    MINOR_2 = 2 #Semi-major and semi-minor axes of 1-Sigma contour
    
    X_MIN = -10
    X_MAX = 10
    Y_MIN = X_MIN
    Y_MAX = X_MAX
    dx = .1
    dy = dx
    
    u = [X_MEAN, Y_MEAN]
    o = comVarMat(MAJOR, MINOR, ALPHA)
    u2 = [X_MEAN_2, Y_MEAN_2]
    o2 = comVarMat(MAJOR_2, MINOR_2, ALPHA_2)
    
    x, y = np.mgrid[X_MIN:X_MAX:dx, Y_MIN:Y_MAX:dy] #Coordinate syste
    
    gauss1 = make2DGaussian(x, y, u, o)
    gaussianAxes(x,y, gauss1)
    #gauss2 = make2DGaussian(x, y, u2, o2)
    #
    #gauss3 = gauss1 + gauss2
    #
    #fig, ax = gaussianAxes(x, y, gauss3)
    #fig.suptitle('Sum of Two Gaussians')
    #paramStr = '''Parameters (in format muX, muY, oXX, oXY, oYY) : First Gaussian: ''' + str(np.around(u, 2)).strip('[]') + ', ' + str(round(o[0][0], 2)) + ', ' + str(round(o[0][1], 2)) + ', ' + str(round(o[1][1], 2))
    #fig.text(.03,.03,paramStr)
    #print o
    pl.show()
    
#Computes the matrix of second moments using the geometry of the elliptical contour at an angle alpha to the x-axis in real space.
#Input: None
#Returns: 2x2 Symmetric Matrix of Second Moments
def comVarMat(major, minor, alpha):
    oXX = (major**2 * (np.cos(alpha))**2 + minor**2 * (np.sin(alpha))**2)**(.5)
    oYY = (major**2 * (np.sin(alpha))**2 + minor**2 * (np.cos(alpha))**2)**(.5)
    oXY = ((major**2 - minor**2)*np.sin(alpha)*np.cos(alpha))**.5
    o = [[oXX, oXY], [oXY, oYY]]
    return o
    
#Constructs the values of a 2-Dimensional Gaussian distribution for a given coordinate system.
#Input: x and y axes values, 2x1 matrix of centroid coordinates, 2x2 matrix of second moments
#Output: Bivariate Gaussian over the given coordinate system.
def make2DGaussian(x, y, u, o):
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    
    gauss = multivariate_normal.pdf(pos, u, o)

    return gauss
    
#Plots a given 2D Gaussian and its contours for a given coordinate system.
#Input: x and y axes values, 2D Gaussian distribution values
#Output: 3D Surface Plot and 2D Contour plot
def gaussianAxes(x, y, gauss):
    fig, ax = pl.subplots(1, 2, figsize=(12,6))
    
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].plot_surface(x, y, gauss)
    ax[0].axis('tight')
    ax[0].set_title('Surface Plot')
    
    ax[1].contourf(x, y, gauss)
    ax[1].set_xlim([-10,10])
    ax[1].set_ylim([-10,10])    
    ax[1].set_title('Contours')
    
    return fig, ax
    
if __name__ == "__main__":
    main()