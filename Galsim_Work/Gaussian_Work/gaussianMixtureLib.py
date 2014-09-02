import galsim
import lmfit as lm
import numpy as np

###
#Draws mixture of two Gaussian galaxies.
def drawMixture(domParams, contParams, skyMap, pixelScale=.2, stampSize=100,  sky=False, psf=False):
	#Unpack parameters from lmfit.Parameters() objects
	dominantFlux = domParams['flux'].value
	dominantHLR = domParams['HLR'].value
	dominant_e1 = domParams['e1'].value
	dominant_e2 = domParams['e2'].value
	dominantCentroid = (domParams['centX'].value, domParams['centY'].value)
	dominantFluxFraction = domParams['fluxFrac'].value

	contaminantFlux = contParams['flux'].value
	contaminantHLR = contParams['HLR'].value
	contaminant_e1 = contParams['e1'].value
	contaminant_e2 = contParams['e2'].value
	contaminantCentroid = (contParams['centX'].value, contParams['centY'].value)
	contaminantFluxFraction = contParams['fluxFrac'].value


	#Create mixture
	domGal = galsim.Gaussian(flux=dominantFlux, half_light_radius=dominantHLR)
	domGal = domGal.shear(e1=dominant_e1, e2=dominant_e2)
	domGal = domGal.shift(dx=dominantCentroid[0], dy=dominantCentroid[1])

	contGal = galsim.Gaussian(flux=contaminantFlux, half_light_radius=contaminantHLR)
	contGal = contGal.shear(e1=contaminant_e1, e2=contaminant_e2)
	contGal = contGal.shift(dx=contaminantCentroid[0], dy=contaminantCentroid[1])	
	
	mix =  dominantFluxFraction*domGal + contaminantFluxFraction*contGal

	#Convolve with psf
	if psf==True:
		big_fft_params = galsim.GSParams(maximum_fft_size=10240*4)
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([mix, psf], gsparams=big_fft_params)
	#Draw image
	mixImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	mixImage = mix.drawImage(image=mixImage, method='fft')
	#Add sky to image
	if sky==True:
		mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('meanSky')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel multiplied by full 10-year LSST band exposure time
	mixComponents = [domGal.original, contGal.original]
	return mixComponents, mixImage

###
#Draws a single Gaussian fit
def drawFit(params, pixelScale=.2, stampSize=100, psf=False):
	#Unpack parameters
	flux = params['fitFlux'].value
	hlr = params['fitHLR'].value
	centroid = (params['fitCentX'].value, params['fitCentY'].value)
	e1 = params['fite1'].value
	e2 = params['fite2'].value

	#Create Gaussian fit
	fit = galsim.Gaussian(flux=flux, half_light_radius=hlr)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centroid[0], dy=centroid[1])
	
	#Convolve with psf
	if psf == True:
		big_fft_params = galsim.GSParams(maximum_fft_size=10240*4)
		psf = galsim.Moffat(beta=3, fwhm=.6)
		fit = galsim.Convolve([fit, psf], gsparams=big_fft_params)
	#Draw image
	fitImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	fitImage = fit.draw(image=fitImage)
	return fitImage

###
#Callable residual function for fitting using lmfit, convolving fit with PSF
#Uses errors from the sky level over a full stack, if desired
def residualPSF(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize, psf=True)
	return (mixIm - fitIm).array.ravel()/((skyMap.get('meanSky')*skyMap.get('expTime') + fitIm.array.ravel())**.5)

###
#Callable residual function for fitting using lmfit, with no PSF
#Uses errors from the sky level over a full stack, if desired
def residual(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize)
	return (mixIm - fitIm).array.ravel()/((skyMap.get('meanSky')*skyMap.get('expTime') + fitIm.array.ravel())**.5)

###
#Collect zeroth, first, and second moments for any number of circular Gaussian components in a Gaussian galaxy mixture
def gaussMixtureMoments(galaxies, gaussParams):
	fractions = [params['fluxFrac'].value for params in gaussParams]
	zeroMoments = [params['flux'].value for params in gaussParams]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in gaussParams]
	secondMoments = [[[gal.getSigma()**2, 0], [0, gal.getSigma()**2]] for gal in galaxies]
	return (fractions, zeroMoments, firstMoments, secondMoments)

###
#Calculates the analytic ellipticity components of a Gaussian mixture with any number of circular Gaussian components
def analyticMixtureElliptictity(fluxFractions, fluxes, centroids, quads):
	#0th Moment of Mixture (weighted fluxes of components)
	mixZeroMom = sum([fluxFrac * flux for fluxFrac, flux in zip(fluxFractions, fluxes)])

	#1st moment of mixture (1/zeroMom times the sum of fraction*flux*component centroid)
	mixCentX = (1/mixZeroMom) * sum([fluxFrac * flux * cent[0] for fluxFrac, flux, cent in zip(fluxFractions, fluxes, centroids)])
	mixCentY = (1/mixZeroMom) * sum([fluxFrac * flux * cent[1] for fluxFrac, flux, cent in zip(fluxFractions, fluxes, centroids)])
	mixFirstMom = (mixCentX, mixCentY)

	#2nd Moments of mixture: 1/zeroMom * weighted sum of comp. 2nd moments + 1/zeroMom * weighted sum of difference between comp. 1st mom and mix 1st mom
	Q00 = (1/mixZeroMom)*sum([fluxFrac*q[0][0] for fluxFrac, q in zip(fluxFractions, quads)]) + (1/mixZeroMom)*sum([fluxFrac*(compFirstMom[0]-mixFirstMom[0])*(compFirstMom[0]-mixFirstMom[0]) for fluxFrac, compFirstMom in zip(fluxFractions, centroids)])
	Q01 = (1/mixZeroMom)*sum([fluxFrac*q[0][1] for fluxFrac, q in zip(fluxFractions, quads)]) + (1/mixZeroMom)*sum([fluxFrac*(compFirstMom[0]-mixFirstMom[0])*(compFirstMom[1]-mixFirstMom[1]) for fluxFrac, compFirstMom in zip(fluxFractions, centroids)])
	Q11 = (1/mixZeroMom)*sum([fluxFrac*q[1][1] for fluxFrac, q in zip(fluxFractions, quads)]) + (1/mixZeroMom)*sum([fluxFrac*(compFirstMom[1]-mixFirstMom[1])*(compFirstMom[1]-mixFirstMom[1]) for fluxFrac, compFirstMom in zip(fluxFractions, centroids)])

	mixSecMom = [[Q00, Q01], [Q01, Q11]]
	
	#Ellipticities
	e1 = (mixSecMom[0][0] - mixSecMom[1][1])/(mixSecMom[0][0] + mixSecMom[1][1])
	e2 = 2*mixSecMom[0][1]/(mixSecMom[0][0]+mixSecMom[1][1])
	return e1, e2, mixZeroMom, mixFirstMom, mixSecMom

###
#Analytic ellipticities for Gaussian mixture
def calcAnalytic(galaxies, galParams):
	#Calculate moments of mixture components
	fracs, zeroMom, firstMom, secMom = gaussMixtureMoments(galaxies, galParams)
	#Get analytic ellipticities
	e1Analytic, e2Analytic, mixZeroMoment, mixFirstMoment, mixSecMoment = analyticMixtureElliptictity(fracs, zeroMom, firstMom, secMom)
	return e1Analytic, e2Analytic, mixZeroMoment, mixFirstMoment, mixSecMoment

###
#Calculates two ellipticities:
#1. Analytic ellipticities for Gaussian mixture
#2. Fit ellipticities for single Gaussian fit, along with uncertainties of these ellipticties from the best fit
def calcEllipticities(domParams, contParams, skyMap, pixelScale=.2, stampSize=100, psf=False, sky=False):
	##Analytic ellipticites
	gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, psf=psf, sky=sky)
	galParams = [domParams, contParams]
	e1Analytic, e2Analytic, mixZeroMom, mixFirstMom, mixSecMom = calcAnalytic(gals, galParams)
	
	##Fit ellipticities
	#Initialize fit parameters
	params = lm.Parameters()
	params.add('fitFlux', value=mixZeroMom)
	params.add('fitHLR', value=domParams['HLR'].value)
	params.add('fitCentX', value=mixFirstMom[0])
	params.add('fitCentY', value=mixFirstMom[1])
	params.add('fite1', value=e1Analytic, min=-1, max=1)
	params.add('fite2', value=e2Analytic, min=-.2, max=.2)

	#Find minimum chi-squared between mixture and fit
	if psf==True:
		out = lm.minimize(residualPSF, params, args=[mixIm, skyMap])
	else:
		out = lm.minimize(residual, params, args=[mixIm, skyMap])

	#Get ellipticities from fit parameters
	e1Fit = out.params['fite1'].value
	e2Fit = out.params['fite2'].value
	#Get errors if relevant
	if sky==True:
		e1err = np.sqrt(np.diag(out.covar))[2]
		e2err = np.sqrt(np.diag(out.covar))[3]
	else:
		e1err = 0
		e2err = 0 #cheating, kind of. the errors are just meaningless and we are not plotting them anyway, but do not want to change return statement

	return (e1Fit, e2Fit, e1err, e2err, e1Analytic, e2Analytic)