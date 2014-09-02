import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

#Irving Rodriguez
#
#Draws a mixture of two bulge+disk sersic galaxies, fits a single bulge+disk sersic, and draws the residual between the best fit and the mixture

#-------------------------------------------------------------------------
#Functions--------------------------------------------------------------------
#-------------------------------------------------------------------------

def drawFit(params, pixelScale=.2, stampSize=50, psf=False):
	diskFlux = params['fitDiskFlux'].value
	diskHLR = params['fitDiskHLR'].value
	centX = params['fitCentX'].value
	centY = params['fitCentY'].value
	e1 = params['fite1'].value
	e2 = params['fite2'].value
	diskN = params['diskNIndex'].value
	#bulgeN = params['bulgeNIndex'].value
	#bulgeFlux = params['fitBulgeFlux'].value
	#bulgeHLR = params['fitBulgeHLR'].value

	fit = galsim.Sersic(n=diskN, flux=diskFlux, half_light_radius=diskHLR)
	#fit += galsim.Sersic(n=bulgeN, flux=bulgeFlux, half_light_radius=bulgeHLR)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centX, dy=centY)
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		fit = galsim.Convolve([fit,psf])
	fitImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	fitImage = fit.drawImage(image=fitImage, method='fft')
	return fitImage

def drawMixture(domParams, contParams, pixelScale=.2, stampSize=50, psf=False, sky=False):
	#Unpack parameters
	flux = domParams['flux'].value
	hlr = domParams['HLR'].value
	domBulgeN = domParams['bulgeNIndex'].value
	domDiskN = domParams['diskNIndex'].value
	domFrac = domParams['fluxFraction'].value
	contBulgeN = contParams['bulgeNIndex'].value
	contDiskN = contParams['diskNIndex'].value
	contCentX = contParams['centX'].value
	contCentY = contParams['centY'].value
	contFrac = contParams['fluxFraction'].value

	#Draw mixture
	domGal = galsim.Sersic(n=domDiskN, half_light_radius=hlr, flux=flux)
	#domGal += galsim.Sersic(n=domBulgeN, half_light_radius=hlr, flux=flux)

	contGal = galsim.Sersic(n=contDiskN, half_light_radius=hlr, flux=flux)
	#contGal += galsim.Sersic(n=contBulgeN, half_light_radius=hlr, flux=flux)
	contGal = contGal.shift(dx=contCentX, dy=contCentY)

	mix = domGal*domFrac + contGal*contFrac
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([mix, psf])
	mixImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	mixImage = mix.drawImage(image=mixImage, method='fft')
	if sky==True:
		mixImage.addNoise(galsim.PoissonNoise(rng=galsim.BaseDeviate(1), sky_level=26.83*6900))
	return mixImage

def residual(params, mixIm):
	fitIm = drawFit(params, psf=True)
	return (mixIm - fitIm).array.ravel()

#-------------------------------------------------------------------------
#Body----------------------------------------------------------------------
#-------------------------------------------------------------------------

#Mixture parameters
DOMINANT_FLUX = 8.e5
DOMINANT_HALF_LIGHT_RADIUS = 1. #arcsec
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
BULGE_N = 4.
DISK_N = 1.
DOMINANT_FLUX_FRACTION = .5
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION

PIXEL_SCALE = .2 #arcsec/pixel
SEPARATION = 7.5 * PIXEL_SCALE # pixels*scale=arcsec
PHI = 0. #angle between x-axis and major axes
dx = SEPARATION * np.cos(PHI)
dy = SEPARATION * np.sin(PHI)

#Dominant galaxy parameters
domParams = lm.Parameters()
domParams.add('flux', value = DOMINANT_FLUX)
domParams.add('HLR', value = DOMINANT_HALF_LIGHT_RADIUS)
domParams.add('centX', value = 0)
domParams.add('centY', value = 0)
domParams.add('e1', value = 0, min=-1, max=1)
domParams.add('e2', value = 0, min=-1, max=1)
domParams.add('bulgeNIndex', value = BULGE_N)
domParams.add('diskNIndex', value = DISK_N)
domParams.add('fluxFraction', value = DOMINANT_FLUX_FRACTION)

#Contaminant galaxy parameters
contParams = lm.Parameters()
contParams.add('flux', value = CONTAMINANT_FLUX)
contParams.add('HLR', value = CONTAMINANT_HALF_LIGHT_RADIUS)
contParams.add('centX', value = dx)
contParams.add('centY', value = dy)
contParams.add('e1', value = 0, min=-1, max=1)
contParams.add('e2', value = 0, min=-1, max=1)
contParams.add('bulgeNIndex', value = BULGE_N)
contParams.add('diskNIndex', value = DISK_N)
contParams.add('fluxFraction', value = CONTAMINANT_FLUX_FRACTION)

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
mixIm = drawMixture(domParams, contParams, psf=True, sky=True)
#Print SNR of this Flux
threshold = .5*np.sqrt(26.83*6900)
mixNoSky = drawMixture(domParams, contParams, psf=True)
mask = mixNoSky.array>=threshold
weight = mixNoSky.array
wsnr = (mixIm.array*weight*mask).sum() / np.sqrt((weight*weight*26.83*6900*mask).sum())
print 'SNR (weighted) for this flux: ', wsnr

fig=pl.figure()
ax = fig.add_subplot(121)
im = ax.imshow(mask)
ax2 =fig.add_subplot(122)
im2 = ax2.imshow(mixNoSky.array)
pl.show()

#Minimize residual of mixture and fit with params
out = lm.minimize(residual, params, args=[mixIm])
print 'The minimum least squares is:', out.chisqr
lm.report_errors(params)

#Draw best fit
fitIm = drawFit(params)

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