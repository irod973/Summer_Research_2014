
import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import gaussianMixtureLib as lib

#Irving Rodriguez
#
#Draws a Gaussian mixture composed of 2 Gaussians, draws a single Gaussian fit, and draws the residual between the mixture and the fit.

#-------------------------------------------------------------------------
#---------------------------------------------------------------------
#-------------------------------------------------------------------------

###############################Define constants
#Galaxy parameters
DOMINANT_FLUX = 1.e6
DOMINANT_HALF_LIGHT_RADIUS = 1  #arcsec
DOMINANT_FLUX_FRACTION = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
DOMINANT_E1 = 0
DOMINANT_E2 = 0
DOMINANT_CENTROID = (0,0)
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION
CONTAMINANT_E1 = 0
CONTAMINANT_E2 = 0
PIXEL_SCALE = .2 #arcsec/pixel
STAMP_SIZE = 100 #pixels
SEPARATION = 5 * PIXEL_SCALE #pixels*pixelscale = "
PHI = 0 #angle between major axis and real x-axis
dx = SEPARATION*np.cos(PHI)
dy = SEPARATION*np.sin(PHI)
PSF=True
SKY=True

#Sky parameters
MEAN_SKY_BACKGROUND = 26.83 #counts/sec/pixel, from red band (David Kirkby notes)
EXPOSURE_TIME = 6900 #sec, from LSST
rng = galsim.BaseDeviate(1)
skyMap = {'meanSky':MEAN_SKY_BACKGROUND, 'expTime':EXPOSURE_TIME, 'rng':rng}

#Initialize parameter objects
domParams = lm.Parameters()
domParams.add('flux', value=DOMINANT_FLUX)
domParams.add('HLR', value=DOMINANT_HALF_LIGHT_RADIUS)
domParams.add('centX', value=DOMINANT_CENTROID[0])
domParams.add('centY', value=DOMINANT_CENTROID[1])
domParams.add('e1', value=DOMINANT_E1)
domParams.add('e2', value=DOMINANT_E2)
domParams.add('fluxFrac', value=DOMINANT_FLUX_FRACTION)

contParams = lm.Parameters()
contParams.add('flux', value=CONTAMINANT_FLUX)
contParams.add('HLR', value=CONTAMINANT_HALF_LIGHT_RADIUS)
contParams.add('centX', value=dx)
contParams.add('centY', value=dy)
contParams.add('e1', value=CONTAMINANT_E1)
contParams.add('e2', value=CONTAMINANT_E2)
contParams.add('fluxFrac', value=CONTAMINANT_FLUX_FRACTION)

#Initialize fit parameters
params = lm.Parameters()
params.add('fitFlux', value=domParams['flux'].value)
params.add('fitHLR', value=domParams['HLR'].value)
params.add('fitCentX', value=domParams['centX'].value)
params.add('fitCentY', value=domParams['centY'].value)
params.add('fite1', value=domParams['e1'].value, min=-1, max=1)
params.add('fite2', value=domParams['e2'].value, min=-1, max=1)

#Create mixture
gals, mixIm = lib.drawMixture(domParams, contParams, skyMap, psf=PSF, sky=SKY)

#Minimize
if PSF==True:
	out = lm.minimize(lib.residualPSF, params, args=[mixIm, skyMap])
else:
	out = lm.minimize(lib.residual, params, args=[mixIm, skyMap])
print 'The minimum Chi-Squared is: ', out.chisqr
lm.report_errors(params)

#Draw best fit
fitIm = lib.drawFit(params, psf=PSF)

###Plots
fig = pl.figure()
domStr = 'Dominant Galaxy: (' + str(domParams['centX'].value) + ', ' + str(domParams['centY'].value) + '), ' + str(DOMINANT_FLUX) + ', ' + str(DOMINANT_HALF_LIGHT_RADIUS) + ', ' + str(DOMINANT_FLUX_FRACTION) + ', 0, 0'
contStr = 'Contaminant Galaxy: (' + str(dx) + ', ' + str(dy) + '), ' + str(CONTAMINANT_FLUX) + ', ' + str(CONTAMINANT_HALF_LIGHT_RADIUS) + ', ' + str(CONTAMINANT_FLUX_FRACTION) + ', 0, 0'
titleStr = 'Parameters (centroid, flux, hlr, flux fraction, e1, e2)\n' + domStr + '\n' + contStr + '\nPixel Scale: ' + str(PIXEL_SCALE) + ' arcsec/pixel'
fig.suptitle(titleStr, fontsize=18)

#Plotting the mixture
ax11 = fig.add_subplot(131)
c1 = ax11.imshow(mixIm.array, origin='lower')
ax11.set_title('Mixture', fontsize = 18)
pl.colorbar(c1, shrink=.5)

#Plotting the fit
ax12 = fig.add_subplot(132)
c2 = ax12.imshow(fitIm.array, origin='lower')
ax12.set_title('Fit', fontsize=18)
pl.colorbar(c2, shrink=.5)

#Plotting the residual
ax13 = fig.add_subplot(133)
c3 = ax13.imshow((fitIm-mixIm).array, origin='lower')
ax13.set_title('Residual', fontsize=18)
pl.colorbar(c3, shrink=.5)

#Get the percentage of flux above sky level
# fluxRange = np.linspace(1, 5.e6, 1000)
# percentThresh = []
# for flux in fluxRange:
# 	domParams['flux'].value = flux
# 	contParams['flux'].value = flux
# 	gals, image = lib.drawMixture(domParams, contParams, skyMap, sky=False)
# 	fluxAbove = fluxAboveThreshold(image, skyMap)
# 	percentThresh.append(fluxAbove/(flux))

# #Plot percent flux above threshold
# fig = pl.figure(2)
# ax21 = fig.add_subplot(111)
# ax21.plot(fluxRange, percentThresh, '.')
# ax21.axhline(y=.95, label='95%')
# ax21.set_title('Percent Flux Above Sky (sum over flux of pixels) vs. Total Mixture Flux')
# ax21.legend(loc=7, prop={'size':12})

#Calculate Signal to Noise Ratio
threshold = .5*(skyMap.get('meanSky')*skyMap.get('expTime'))**.5
mask = mixIm.array>=threshold
weight = mixIm.array
snr = (mixIm.array*mask).sum() / np.sqrt((skyMap.get('meanSky')*skyMap.get('expTime')*mask).sum())
wsnr = (mixIm.array*weight*mask).sum() / np.sqrt((weight*weight*skyMap.get('meanSky')*skyMap.get('expTime')*mask).sum())
print 'Weighted SNR: ', wsnr

#Show mask of pixels above threshold
fig = pl.figure()
ax01 = fig.add_subplot(111)
im01 = ax01.imshow(mask)

pl.show()