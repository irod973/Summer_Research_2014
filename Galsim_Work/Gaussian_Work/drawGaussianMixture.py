
import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

#Irving Rodriguez
#
#Draws a Gaussian mixture composed of 2 Gaussians, draws a single Gaussian fit, and draws the residual between the mixture and the fit.


#-------------------------------------------------------------------------
#Functions--------------------------------------------------------------------
#-------------------------------------------------------------------------


def drawFit(params, pixelScale=.2, stampSize=100, psf=False):
	flux = params['fitFlux'].value
	hlr = params['fitHLR'].value
	centX = params['fitCentX'].value
	centY = params['fitCentY'].value
	e1 = params['fite1'].value
	e2 = params['fite2'].value

	fit = galsim.Gaussian(flux=flux, half_light_radius=hlr)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centX, dy=centY)
	if psf == True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		fit = galsim.Convolve([fit, psf])
		fitImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		fitImage = fit.draw(image=fitImage)
		return fitImage
	fitImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	fitImage = fit.draw(image=fitImage)
	return fitImage

#Draw mixture using FFT
def drawMixture(domParams, contParams, skyMap, pixelScale=.2, stampSize=100,  sky=False, psf=False):
	domGal = galsim.Gaussian(flux=domParams['flux'].value*domParams['frac'].value, half_light_radius=domParams['HLR'].value)
	contGal = galsim.Gaussian(flux=contParams['flux'].value*contParams['frac'].value, half_light_radius=contParams['HLR'].value)
	contGal = contGal.shift(dx=contParams['centX'].value, dy=contParams['centY'].value)
	mix =  domGal + contGal
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([mix, psf])
		mixImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		mixImage = mix.drawImage(image=mixImage, method='fft')
		if sky==True:
			mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel times 30 sec exposure time
		gals = [domGal, contGal.original, mix]
		return  gals, mixImage
	mixImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	mixImage = mix.drawImage(image=mixImage, method='fft')
	if sky==True:
		mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel multiplied by full 10-year LSST band exposure time
	gals = [domGal, contGal.original, mix]
	return gals, mixImage

#Compute the chi-squared residual between the parameter image and the mixture image, using errors from the average sky level over a full stack
def residual(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize)
	return (mixIm - fitIm).array.ravel()**2/(skyMap.get('skyLevel')*skyMap.get('expTime') + fitIm.array.ravel())

#Sum of flux from pixels above half the average sky level
def fluxAboveThreshold(image, skyMap):
	fluxAboveThresh = 0
	threshold = .5*(skyMap.get('skyLevel')*skyMap.get('expTime'))**.5
	for elem in image.array.ravel():
		if elem >= threshold:
			fluxAboveThresh += elem
	return fluxAboveThresh


#-------------------------------------------------------------------------
#Body---------------------------------------------------------------------
#-------------------------------------------------------------------------

#Parameters for Mixture
DOMINANT_FLUX = 1.e6
DOMINANT_HALF_LIGHT_RADIUS = 1  #arcsec
DOMINANT_FLUX_FRACTION = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION
PIXEL_SCALE = .2 #arcsec/pixel
STAMP_SIZE = 100 #pixels
SEPARATION = 10 * PIXEL_SCALE #pixels*pixelscale = "
PHI = 0 #angle between major axis and real x-axis
dx = SEPARATION*np.cos(PHI)
dy = SEPARATION*np.sin(PHI)
PSF = True
SKY = True #booleans

#Parameters for sky
SKY_LEVEL = 26.83 #counts/sec/pixel, from red band (David Kirkby notes)
EXPOSURE_TIME = 6900 #sec, from LSST
rng = galsim.BaseDeviate(1)
skyMap = {'skyLevel':SKY_LEVEL, 'expTime':EXPOSURE_TIME, 'rng':rng}


#Mixture Parameters
domParams = lm.Parameters()
domParams.add('flux', value=DOMINANT_FLUX)
domParams.add('HLR', value=DOMINANT_HALF_LIGHT_RADIUS)
domParams.add('centX', value=0)
domParams.add('centY', value=0)
domParams.add('frac', value=DOMINANT_FLUX_FRACTION)

contParams = lm.Parameters()
contParams.add('flux', value=CONTAMINANT_FLUX)
contParams.add('HLR', value=CONTAMINANT_HALF_LIGHT_RADIUS)
contParams.add('centX', value=dx)
contParams.add('centY', value=dy)
contParams.add('frac', value=CONTAMINANT_FLUX_FRACTION)

#If uncertainties for fit parameters are desired, use these values accordingly, else the fitter will not find enough points away from min ChiSquared to estimate uncertainties
params = lm.Parameters()
params.add('fitFlux', value=500000)
params.add('fitHLR', value=2)
params.add('fitCentX', value=2)
params.add('fitCentY', value=0)
params.add('fite1', value=0, min=-1, max=1)
params.add('fite2', value=0, min=-1, max=1)

#If uncertainties for fit parameters are not desired, use these values to fit faster
# params = lm.Parameters()
# params.add('fitFlux', value=mixZeroMom)
# params.add('fitHLR', value=2)
# params.add('fitCentX', value=mixFirstMom[0])
# params.add('fitCentY', value=mixFirstMom[1])
# params.add('fite1', value=e1A)
# params.add('fite2', value=e2A)

#Create mixture
gals, mixIm = drawMixture(domParams, contParams, skyMap, psf=PSF, sky=SKY)
#Minimize
out = lm.minimize(residual, params, args=[mixIm, skyMap])
print 'The minimum Chi-Squared is: ', out.chisqr
lm.report_errors(params)

#Draw best fit
fitIm = drawFit(params, psf=PSF)


#Plots
fig = pl.figure()
domStr = 'Dominant Galaxy: (' + str(domParams['centX'].value) + ', ' + str(domParams['centY'].value) + '), ' + str(DOMINANT_FLUX) + ', ' + str(DOMINANT_HALF_LIGHT_RADIUS) + ', ' + str(DOMINANT_FLUX_FRACTION) + ', 0, 0'
contStr = 'Contaminant Galaxy: (' + str(dx) + ', ' + str(dy) + '), ' + str(CONTAMINANT_FLUX) + ', ' + str(CONTAMINANT_HALF_LIGHT_RADIUS) + ', ' + str(CONTAMINANT_FLUX_FRACTION) + ', 0, 0'
titleStr = 'Parameters (centroid, flux, hlr, flux fraction, e1, e2)\n' + domStr + '\n' + contStr + '\nPixel Scale: ' + str(PIXEL_SCALE) + ' arcsec/pixel'
fig.suptitle(titleStr, fontsize=18)

#Plotting the mixture
ax11 = fig.add_subplot(131)
c1 = ax11.imshow(mixIm.array, origin='lower')
ax11.set_title('Mixture')
pl.colorbar(c1, shrink=.5)

#Plotting the fit
ax12 = fig.add_subplot(132)
c2 = ax12.imshow(fitIm.array, origin='lower')
ax12.set_title('Fit')
pl.colorbar(c2, shrink=.5)

#Plotting the residual
ax13 = fig.add_subplot(133)
c3 = ax13.imshow((fitIm-mixIm).array, origin='lower')
ax13.set_title('Residual')
pl.colorbar(c3, shrink=.5)

#Get the percentage of flux above sky level
fluxRange = np.linspace(1, 5.e6, 1000)
percentThresh = []
for flux in fluxRange:
	domParams['flux'].value = flux
	contParams['flux'].value = flux
	gals, image = drawMixture(domParams, contParams, skyMap, sky=False)
	fluxAbove = fluxAboveThreshold(image, skyMap)
	percentThresh.append(fluxAbove/(flux))

#Plot percent flux above threshold
fig = pl.figure(2)
ax21 = fig.add_subplot(111)
ax21.plot(fluxRange, percentThresh, '.')
ax21.axhline(y=.95, label='95%')
ax21.set_title('Percent Flux Above Sky (pixel sum) vs. Total Mixture Flux')
ax21.legend(loc=7, prop={'size':11})

pl.show()