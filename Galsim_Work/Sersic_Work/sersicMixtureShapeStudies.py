import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import math
from pprint import pprint

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

def drawMixture(domParams, contParams, skyMap, pixelScale=.2, stampSize=50, psf=False, sky=False):
	#Unpack parameter objects
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

	gals = [domGal, contGal]

	mix = domGal*domFrac + contGal*contFrac
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

def mixtureMoments(gals, galParams):
	fractions = [params['fluxFraction'].value for params in galParams]
	zeroMoments = [gal.getFlux() for gal in gals]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in galParams]
	secondMoments = [[[1, 0], [0, 1]] for gal in gals]
	return fractions, zeroMoments, firstMoments, secondMoments

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
	#Ellipticities
	e1 = (Q[0][0] - Q[1][1])/(Q[0][0] + Q[1][1])
	e2 = 2*Q[0][1]/(Q[0][0]+Q[1][1])
	return e1, e2, mixZeroMom, mixFirstMom, mixSecMom

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

	#Get errors if relevant
	if sky==True:
		e1err = np.sqrt(np.diag(out.covar))[2]
		e2err = np.sqrt(np.diag(out.covar))[3]
		#e1err = params['fite1'].stderr
		#e2err = params['fite2'].stderr
	else:
		e1err = 0
		e2err = 0 #cheating, kind of. the errors are just meaningless and we are not plotting them anyway, but do not want to change return statement

	return (e1Fit, e2Fit, e1err, e2err)

#-------------------------------------------------------------------------
#Body----------------------------------------------------------------------
#-------------------------------------------------------------------------

#Mixture parameters
DOMINANT_FLUX = 5.e5
DOMINANT_HALF_LIGHT_RADIUS = 1. #arcsec
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
BULGE_N = 4.
DISK_N = 1.
DOMINANT_FLUX_FRACTION = .5
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION
PIXEL_SCALE = .2 #arcsec/pixel
SEPARATION = 5. * PIXEL_SCALE # pixels*scale=arcsec
PHI = 0. #angle between x-axis and major axes
dx = SEPARATION * np.cos(PHI)
dy = SEPARATION * np.sin(PHI)

MEAN_SKY_BACKGROUND = 26.83 #counts/sec/pixel, from red band (David Kirkby notes)
EXPOSURE_TIME = 6900 #sec, from LSST
rng = galsim.BaseDeviate(1)
skyMap = {'meanSky':MEAN_SKY_BACKGROUND, 'expTime':EXPOSURE_TIME, 'rng':rng}

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


sepMax = 7.5*PIXEL_SCALE
numSteps = 8
sepRange = np.linspace(-sepMax, sepMax, numSteps)

print 'No sky fits...'
numSteps = 100
sepRange = np.linspace(-sepMax, sepMax, numSteps)
e1Sep, e2Sep = [[] for i in range(2)]
for sep in sepRange:
	print sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	ellipts = calcEllipticities(domParams, contParams, skyMap, psf=True)
	e1Sep.append(ellipts[0])
	e2Sep.append(ellipts[1])
fig = pl.figure()
ax1=fig.add_subplot(111)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.plot(sepRange, e1Sep, 'b', label='Fit e1, no sky', linewidth=3)
ax1.plot(sepRange, e2Sep, 'm', label='Fit e2, no sky', linewidth=3)

print '100 fits with different sky...'
sepRange = np.linspace(-sepMax, sepMax, 8)
e1SepS, e2SepS, e1Std, e2Std, e1MeanStd, e2MeanStd = [[] for i in range(6)]
for sep in sepRange:
	print sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	ellipts = calcEllipticities(domParams, contParams, skyMap, psf=True)
	e1Sep.append(ellipts[0])
	e2Sep.append(ellipts[1])

	#Now, run 100 trials for a single SNR
	N = 100
	trials = range(1, N)
	trialsE1 = []
	trialsE2 = []
	for n in trials:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleSNREllip= calcEllipticities(domParams, contParams, skyMap, psf=True, sky=True)
		trialsE1.append(singleSNREllip[0])
		trialsE2.append(singleSNREllip[1])
	#Mean of 100 trials
	e1SepS.append(np.mean(trialsE1))
	e2SepS.append(np.mean(trialsE2))
	#Standard deviation of 100 trials
	e1Std.append(np.std(trialsE1))
	e2Std.append(np.std(trialsE2))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(trialsE1)/np.sqrt(N))
	e2MeanStd.append(np.std(trialsE2)/np.sqrt(N))

###Plots
#Varying separation

ax1.errorbar(sepRange, e2SepS, yerr=e2Std, label='Fit e2, sky', capthick=2, fmt='.', color='c')
ax1.errorbar(sepRange, e2SepS, yerr=e2MeanStd, fmt='.', capthick=2, color='c')
ax1.errorbar(sepRange, e1SepS, yerr=e1Std, label='Fit e1, sky', fmt='.', capthick=2, color='r')
ax1.errorbar(sepRange, e1SepS, yerr=e1MeanStd, fmt='.', capthick=2, color='r')
ax1.set_title('Shear vs. Separation', fontsize=24)
ax1.set_xlabel('Separation', fontsize=18)
ax1.set_ylabel('Fit Ellipticity', fontsize=18)
ax1.set_xlim(-2, 2)
pl.legend(prop={'size':18}, loc=9)
pl.show()