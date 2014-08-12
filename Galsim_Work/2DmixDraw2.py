
import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

#Irving Rodriguez
#
#This program studies the ellipticity of fitting a single Gaussian to a mixture of two Gaussians.


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
		image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		image = fit.draw(image=image)
		return image
	image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	image = fit.draw(image=image)
	return image

#Draw mixture using FFT
def drawMixture(domParams, contParams, skyMap, pixelScale=.2, stampSize=100,  sky=False, psf=False):
	domGal = galsim.Gaussian(flux=domParams['flux'].value*domParams['frac'].value, half_light_radius=domParams['HLR'].value)
	contGal = galsim.Gaussian(flux=contParams['flux'].value*contParams['frac'].value, half_light_radius=contParams['HLR'].value)
	contGal = contGal.shift(dx=contParams['centX'].value, dy=contParams['centY'].value)
	mix =  domGal + contGal
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([gauss, psf])
		image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		image = mix.drawImage(image=image, method='fft')
		image.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel times 30 sec exposure time
		gals = [domGal, contGal, mix]
		return  gals, image
	image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	image = mix.drawImage(image=image, method='fft')
	if sky==True:
		image.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel multiplied by full 10-year LSST band exposure time
	gals = [domGal, contGal.original, mix]
	return gals, image

def fluxAboveThreshold(image, skyMap):
	fluxAboveThresh = 0
	threshold = .5*(skyMap.get('skyLevel')*skyMap.get('expTime'))**.5
	for elem in image.array.ravel():
		if elem >= threshold:
			fluxAboveThresh += elem
	return fluxAboveThresh

#Compute the chi-squared residual between the parameter image and the mixture image, using errors from the average sky level over a full stack
def residual(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize)
	return (mixIm - fitIm).array.ravel()**2/(skyMap.get('skyLevel')*skyMap.get('expTime') + fitIm.array.ravel())

def mixtureMoments(gals, galParams):
	fractions = [params['frac'].value for params in galParams]
	zeroMoments = [params['flux'].value for params in galParams]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in galParams]
	secondMoments = [[[gal.getSigma()**2, 0], [0, gal.getSigma()**2]] for gal in gals]
	return fractions, zeroMoments, firstMoments, secondMoments

def calcEllipticities(domParams, contParams, skyMap, pixelScale=.2, stampSize=100, sky=False):
	#Draw the mixture
	if sky==True:
		gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, sky=True)
	else:
		gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize)
	
	#Calculate moments of mixture components
	galParams = [domParams, contParams]
	gal = [gals[0], gals[1]]		
	fractions, zeroMom, firstMom, secMom = mixtureMoments(gal, galParams)
	#Get analytic ellipticities
	e1A, e2A, mixZeroMom, mixFirstMom, mixSecMom = analyticMixShear(fractions, zeroMom, firstMom, secMom)

	params = lm.Parameters()
	params.add('fitFlux', value=mixZeroMom)
	params.add('fitHLR', value=2)
	params.add('fitCentX', value=mixFirstMom[0])
	params.add('fitCentY', value=mixFirstMom[1])
	params.add('fite1', value=e1A, min=-1, max=1)
	params.add('fite2', value=e2A, min=-.5, max=.5)

	#Find minimum chi-squared between mixture and fit
	out = lm.minimize(residual, params, args=[mixIm, skyMap])

	#Get ellipticities from fit parameters
	e1Fit = out.params['fite1'].value
	e2Fit = out.params['fite2'].value
	#Get errors
	e1err = np.sqrt(np.diag(out.covar))[2]
	e2err = np.sqrt(np.diag(out.covar))[3]

	return e1Fit, e2Fit, e1err, e2err, e1A, e2A

def analyticMixShear(fluxFractions, fluxes, centroids, quads):

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

	Q = [[Q00, Q01], [Q01, Q11]]
	mixSecMom = Q
	#ellipticities
	e1 = (Q[0][0] - Q[1][1])/(Q[0][0] + Q[1][1])
	e2 = 2*Q[0][1]/(Q[0][0]+Q[1][1])
	return e1, e2, mixZeroMom, mixFirstMom, mixSecMom

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#Parameters for Mixture
domFlux = 1.e6
domHLR = 1.  #arcsec
contFlux = 1.e6
contHLR = domHLR
domFrac = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
pixelScale = .2 #arcsec/pixel
stampSize = 100 #pixels

du = 10 * pixelScale #pixels*pixelscale = "
phi = 0 #angle between major axis and real x-axis
dx = du*np.cos(phi)
dy = du*np.sin(phi)

#Parameters for sky
skyLevel = 26.83 #counts/sec/pixel, from red band (David Kirkby notes)
expTime = 6900 #sec, from LSST
rng = galsim.BaseDeviate(1)
skyMap = {'skyLevel':skyLevel, 'expTime':expTime, 'rng':rng}

domParams = lm.Parameters()
domParams.add('flux', value=domFlux)
domParams.add('HLR', value=domHLR)
domParams.add('centX', value=0)
domParams.add('centY', value=0)
domParams.add('frac', value=domFrac)

contParams = lm.Parameters()
contParams.add('flux', value=contFlux)
contParams.add('HLR', value=contHLR)
contParams.add('centX', value=dx)
contParams.add('centY', value=dy)
contParams.add('frac', value=(1-domFrac))

#Draw a certain mixture, its fit, and residuals
#Create mixture, using FFT draw
gals, mixIm = drawMixture(domParams, contParams, skyMap, sky=True)

#Calculate moments of mixture components
galParams = [domParams, contParams]
gal = [gals[0], gals[1]]		
fractions, zeroMom, firstMom, secMom = mixtureMoments(gal, galParams)
#Get analytic ellipticities
e1A, e2A, mixZeroMom, mixFirstMom, mixSecMom = analyticMixShear(fractions, zeroMom, firstMom, secMom)

params = lm.Parameters()
params.add('fitFlux', value=500000)
params.add('fitHLR', value=2)
params.add('fitCentX', value=2)
params.add('fitCentY', value=0)
params.add('fite1', value=0, min=-1, max=1)
params.add('fite2', value=0, min=-1, max=1)

#If estimated parameter uncertainties are not desired, use these values to fit faster and find the desired chisqr min.
# params = lm.Parameters()
# params.add('fitFlux', value=mixZeroMom)
# params.add('fitHLR', value=2)
# params.add('fitCentX', value=mixFirstMom[0])
# params.add('fitCentY', value=mixFirstMom[1])
# params.add('fite1', value=e1A)
# params.add('fite2', value=e2A)

out = lm.minimize(residual, params, args=[mixIm, skyMap])
print 'The minimum Chi-Squared is: ', out.chisqr
lm.report_errors(params)
fitIm = drawFit(params)

#Get the percentage of flux above sky level
fluxRange = np.linspace(1, 5.e6, 1000)
percentThresh = []
for flux in fluxRange:
	domParams['flux'].value = flux
	contParams['flux'].value = flux
	gals, image = drawMixture(domParams, contParams, skyMap, sky=False)
	fluxAbove = fluxAboveThreshold(image, skyMap)
	percentThresh.append(fluxAbove/(flux))

#Plots
fig = pl.figure()
domStr = 'Dominant Galaxy: ' + str(gals[0].centroid()) + ', '+str(domFlux)+', '+str(domHLR)+', '+str(domFrac)+', 0, 0'
contStr = 'Contaminant Galaxy: (' + str(contParams['centX'].value) + ', ' + str(contParams['centY'].value) + '), '+str(contFlux)+', '+str(contHLR)+', '+str((1-domFrac))+', 0, 0'
titleStr = 'Parameters (centroid, flux, hlr, fraction of flux, e1, e2)\n'+domStr+ '\n'+contStr+'\nPixel Scale: '+ str(pixelScale)+' arcsec/pixel'
fig.suptitle(titleStr, fontsize=12)

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

#Plot percent flux above threshold
fig = pl.figure(2)
ax21 = fig.add_subplot(111)
ax21.plot(fluxRange, percentThresh, '.')
ax21.axhline(y=.95, label='95%')
ax21.set_title('Total Flux vs. Percent Above Sky Level')
ax21.legend(loc=7, prop={'size':11})

pl.show()