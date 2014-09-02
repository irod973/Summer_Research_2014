import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import sersicMixtureLib as lib

#Irving Rodriguez
#
#Draws a mixture of two bulge+disk sersic galaxies, fits a single bulge+disk sersic, and draws the residual between the best fit and the mixture

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

###############################Define constants
#Galaxy parameters
DOMINANT_FLUX = 1.e6
DOMINANT_HALF_LIGHT_RADIUS = 1  #arcsec
DOMINANT_FLUX_FRACTION = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
DOMINANT_E1 = 0
DOMINANT_E2 = 0
DOMINANT_BULGE_INDEX = 4
DOMINANT_DISK_INDEX = 1
DOMINANT_CENTROID = (0,0)
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION
CONTAMINANT_E1 = 0
CONTAMINANT_E2 = 0
CONTAMINANT_BULGE_INDEX = 4
CONTAMINANT_DISK_INDEX = 1
PIXEL_SCALE = .2 #arcsec/pixel
STAMP_SIZE = 100 #pixels
SEPARATION = 5 * PIXEL_SCALE #pixels*pixelscale = "
PHI = 0 #angle between major axis and real x-axis
dx = SEPARATION*np.cos(PHI)
dy = SEPARATION*np.sin(PHI)
PSF=True

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
domParams.add('bulgeNIndex', value=DOMINANT_BULGE_INDEX)
domParams.add('diskNIndex', value=DOMINANT_DISK_INDEX)
domParams.add('fluxFrac', value=DOMINANT_FLUX_FRACTION)

contParams = lm.Parameters()
contParams.add('flux', value=CONTAMINANT_FLUX)
contParams.add('HLR', value=CONTAMINANT_HALF_LIGHT_RADIUS)
contParams.add('centX', value=dx)
contParams.add('centY', value=dy)
contParams.add('e1', value=CONTAMINANT_E1)
contParams.add('e2', value=CONTAMINANT_E2)
contParams.add('bulgeNIndex', value=CONTAMINANT_BULGE_INDEX)
contParams.add('diskNIndex', value=CONTAMINANT_DISK_INDEX)
contParams.add('fluxFrac', value=CONTAMINANT_FLUX_FRACTION)

####################################Draw mixture and fit

#Fit Parameters, will be changed by .minimize unless vary=False
params = lm.Parameters()
params.add('fitDiskFlux', value = 1.e5)
params.add('fitDiskHLR', value = 1.)
params.add('fitCentX', value = 0)
params.add('fitCentY', value = 0)
params.add('fite1', value = 0, min=-1, max=1)
params.add('fite2', value = 0, min=-1, max=1)
params.add('diskNIndex', value = 1, vary=False)
#params.add('bulgeNIndex', value = 4, vary=False)
#params.add('fitBulgeHLR', value = 1.)
#params.add('fitBulgeFlux', value = 1.e5)

#Draw mixture
gals, mixIm = lib.drawMixture(domParams, contParams, skyMap, psf=True, sky=True)
#Print SNR of this Flux
threshold = .5*np.sqrt(26.83*6900)
gals, mixImNoSky = lib.drawMixture(domParams, contParams, skyMap, psf=True)
mask = mixImNoSky.array>=threshold
weight = mixImNoSky.array
wsnr = (mixIm.array*weight*mask).sum() / np.sqrt((weight*weight*26.83*6900*mask).sum())
print 'SNR (weighted) for this flux: ', wsnr

fig=pl.figure()
ax = fig.add_subplot(121)
im = ax.imshow(mask)
ax2 =fig.add_subplot(122)
im2 = ax2.imshow(mixImNoSky.array)
pl.show()

#Minimize residual of mixture and fit with params
if PSF==True:
	out = lm.minimize(lib.residualPSF, params, args=[mixIm])
else:
	out = lm.minimize(lib.residual, params, args=[mixIm])
print 'The minimum least squares is:', out.chisqr
lm.report_errors(params)

#Draw best fit
fitIm = lib.drawFit(params)

#Plot mixture, best fit, and residual
fig = pl.figure()
domStr = 'Dominant Galaxy: (' + str(domParams['centX'].value) + ', ' + str(domParams['centY'].value) + '), ' + str(DOMINANT_FLUX) + ', ' + str(DOMINANT_HALF_LIGHT_RADIUS) + ', ' + str(DOMINANT_FLUX_FRACTION) + ', 0, 0'
contStr = 'Contaminant Galaxy: (' + str(dx) + ', ' + str(dy) + '), ' + str(CONTAMINANT_FLUX) + ', ' + str(CONTAMINANT_HALF_LIGHT_RADIUS) + ', ' + str(CONTAMINANT_FLUX_FRACTION) + ', 0, 0'
fitStr = 'Fit: (' + str(np.around(params['fitCentX'].value, decimals=2)) + ', ' + str(np.around(params['fitCentY'].value, decimals=2)) + '), ' + str(np.around(params['fitDiskFlux'].value, decimals=2)) + ', '  + str(np.around(params['fitDiskHLR'].value, decimals=2)) + ', ' + str(np.around(params['fite1'].value, decimals=2)) + ', ' + str(np.around(params['fite2'].value, decimals=2))
titleStr = 'Parameters (centroid, flux, hlr, flux fraction, e1, e2)\n' + domStr + '\n' + contStr + '\n' + fitStr + '\nPixel Scale: ' + str(PIXEL_SCALE) + ' arcsec/pixel'
fig.suptitle(titleStr, fontsize=18)

ax11 = fig.add_subplot(131)
c11 = ax11.imshow(mixIm.array, origin='lower')
ax11.set_title('Sersic Mixture')
pl.colorbar(c11, shrink=0.5)

ax12 = fig.add_subplot(132)
c12 = ax12.imshow(fitIm.array, origin='lower')
ax12.set_title('Single Sersic Fit')
pl.colorbar(c12, shrink=0.5)

ax13 =  fig.add_subplot(133)
c13 = ax13.imshow((fitIm-mixIm).array, origin='lower')
ax13.set_title('Residual')
pl.colorbar(c13, shrink=0.5)

pl.show()