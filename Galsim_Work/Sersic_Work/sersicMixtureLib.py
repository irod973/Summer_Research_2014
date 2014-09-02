import galsim
import lmfit as lm
import numpy as np

def drawFit(params, pixelScale=.2, stampSize=50, psf=False):
	diskFlux = params['fitDiskFlux'].value
	diskHLR = params['fitDiskHLR'].value
	centroid= (params['fitCentX'].value, params['fitCentY'].value)
	e1 = params['fite1'].value
	e2 = params['fite2'].value
	diskN = params['diskNIndex'].value
	#bulgeN = params['bulgeNIndex'].value
	#bulgeFlux = params['fitBulgeFlux'].value
	#bulgeHLR = params['fitBulgeHLR'].value

	fit = galsim.Sersic(n=diskN, flux=diskFlux, half_light_radius=diskHLR)
	#fit += galsim.Sersic(n=bulgeN, flux=bulgeFlux, half_light_radius=bulgeHLR)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centroid[0], dy=centroid[1])
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		fit = galsim.Convolve([fit,psf])
	fitImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	fitImage = fit.drawImage(image=fitImage, method='fft')
	return fitImage

def drawMixture(domParams, contParams, skyMap, pixelScale=.2, stampSize=50, psf=False, sky=False):
	#Unpack parameter objects
	dominantFlux = domParams['flux'].value
	dominantHLR = domParams['HLR'].value
	dominant_e1 = domParams['e1'].value
	dominant_e2 = domParams['e2'].value
	dominantCentroid = (domParams['centX'].value, domParams['centY'].value)
	dominantBulgeN = domParams['bulgeNIndex'].value
	dominantDiskN = domParams['diskNIndex'].value
	dominantFluxFraction = domParams['fluxFrac'].value

	contaminantFlux = contParams['flux'].value
	contaminantHLR = contParams['HLR'].value
	contaminant_e1 = contParams['e1'].value
	contaminant_e2 = contParams['e2'].value
	contaminantBulgeN = contParams['bulgeNIndex'].value
	contaminantDiskN = contParams['diskNIndex'].value
	contaminantCentroid = (contParams['centX'].value, contParams['centY'].value)
	contaminantFluxFraction = contParams['fluxFrac'].value


	#Draw mixture
	domGal = galsim.Sersic(n=dominantDiskN, half_light_radius=dominantHLR, flux=dominantFlux)
	#domGal += galsim.Sersic(n=dominantBulgeN, half_light_radius=dominantHLR, flux=dominantFlux)
	domGal = domGal.shear(e1=dominant_e1, e2=dominant_e2)
	domGal = domGal.shift(dx=dominantCentroid[0], dy=dominantCentroid[1])

	contGal = galsim.Sersic(n=contaminantDiskN, half_light_radius=contaminantHLR, flux=contaminantFlux)
	#contGal += galsim.Sersic(n=contaminantBulgeN, half_light_radius=contaminanHLR, flux=contaminantFlux)
	contGal = contGal.shift(dx=contaminantCentroid[0], dy=contaminantCentroid[1])

	gals = [domGal, contGal]

	mix = dominantFluxFraction*domGal + contaminantFluxFraction*contGal
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([mix, psf])
	mixImage = galsim.ImageD(stampSize, stampSize,scale=pixelScale)
	mixImage = mix.drawImage(image=mixImage, method='fft')
	if sky==True:
		mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('meanSky')*skyMap.get('expTime')))
	return gals, mixImage

def residualPSF(params, mixIm):
	fitIm = drawFit(params, psf=True)
	return (mixIm - fitIm).array.ravel()

def residual(params, mixIm):
	fitIm = drawFit(params)
	return (mixIm - fitIm).array.ravel()/fitIm.array.ravel()

def calcEllipticities(domParams, contParams, skyMap, pixelScale=.2, stampSize=50, psf=False, sky=False):
	#Draw the mixture
	gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, psf=psf, sky=sky)

	#Fit Parameters, will be changed by .minimize unless vary=False
	params = lm.Parameters()
	params.add('fitDiskFlux', value = 1.e5)
	params.add('fitDiskHLR', value = 1.)
	params.add('fitCentX', value = 0)
	params.add('fitCentY', value = 0)
	params.add('fite1', value = 0, min=-.9, max=.9)
	params.add('fite2', value= 0, min=-.25, max=.25)
	#params.add('mag', value = .99, max=1, vary=True)
	#params.add('fite2', expr='sqrt(mag**2-fite1**2)')
	params.add('diskNIndex', value = 1, vary=False)
	#params.add('bulgeNIndex', value = 4, vary=False)
	#params.add('fitBulgeHLR', value = 1.)
	#params.add('fitBulgeFlux', value = 1.e5)

	#Find minimum chi-squared between mixture and fit
	if psf==True:
		out = lm.minimize(residualPSF, params, args=[mixIm])
	else:
		out = lm.minimize(residual, params, args=[mixIm])
	#Get ellipticities from fit parameters
	e1Fit = out.params['fite1'].value
	e2Fit = out.params['fite2'].value

	#Get errors, if relevant
	if sky==True:
		e1err = np.sqrt(np.diag(out.covar))[2]
		e2err = np.sqrt(np.diag(out.covar))[3]
	else:
		e1err = 0
		e2err = 0 #cheating, kind of. the errors are meaningless in the sky=False case (unless Poisson from photon shooting) and we are not plotting them anyway, but do not want to change return statement

	return (e1Fit, e2Fit, e1err, e2err)