import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
from pprint import pprint #I use this for debugging. It's nicer than print

#Irving Rodriguez
#
#This program studies the ellipticity of a single Gaussian fit to a mixture of two Gaussians.

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
			mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('meanSky')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel times 30 sec exposure time
		gals = [domGal, contGal.original, mix]
		return  gals, mixImage
	mixImage = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	mixImage = mix.drawImage(image=mixImage, method='fft')
	if sky==True:
		mixImage.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('meanSky')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel multiplied by full 10-year LSST band exposure time
	gals = [domGal, contGal.original, mix]
	return gals, mixImage

#Compute the chi-squared residual between the parameter image and the mixture image, using errors from the average sky level over a full stack
def residual(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize)
	return (mixIm - fitIm).array.ravel()**2/(np.sqrt(skyMap.get('meanSky')*skyMap.get('expTime')) + np.sqrt(fitIm.array.ravel()))**2

def mixtureMoments(gals, galParams):
	fractions = [params['frac'].value for params in galParams]
	zeroMoments = [params['flux'].value for params in galParams]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in galParams]
	secondMoments = [[[gal.getSigma()**2, 0], [0, gal.getSigma()**2]] for gal in gals]
	return fractions, zeroMoments, firstMoments, secondMoments

def calcEllipticities(domParams, contParams, skyMap, pixelScale=.2, stampSize=100, psf=False, sky=False):
	#Draw the mixture
	gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, psf=psf, sky=sky)

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
	params.add('fite2', value=e2A, min=-.2, max=.2)

	#Find minimum chi-squared between mixture and fit
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

	return (e1Fit, e2Fit, e1err, e2err, e1A, e2A)

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
PSF=True

#Parameters for sky
MEAN_SKY_BACKGROUND = 26.83 #counts/sec/pixel, from red band (David Kirkby notes)
EXPOSURE_TIME = 6900 #sec, from LSST
rng = galsim.BaseDeviate(1)
skyMap = {'meanSky':MEAN_SKY_BACKGROUND, 'expTime':EXPOSURE_TIME, 'rng':rng}


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

###Ellipticity studies
#Parameters for ellipticity studies
numSteps = 200
separation = 10 * PIXEL_SCALE #arcsec

###Vary separation in real space
sepRange = np.linspace(-separation, separation, numSteps)
#No Sky
e1Sep, e2Sep, e1SepErr, e2SepErr, e1SepAnalytic, e2SepAnalytic = [[] for i in range(6)]
ellSepList = [e1Sep, e1SepErr, e2Sep, e2SepErr, e1SepAnalytic, e2SepAnalytic]
for sep in sepRange:
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=PSF)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellSepList, valList)]
#With Sky
e1SepS, e2SepS, e1SepErrS, e2SepErrS, e1SepAnalyticS, e2SepAnalyticS = [[] for i in range(6)]
ellSepListS = [e1SepS, e2SepS, e1SepErrS, e2SepErrS, e1SepAnalyticS, e2SepAnalyticS]
for sep in sepRange:
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellSepListS, valList)]
contParams['centX'].value = dx
contParams['centY'].value = dy

###Vary dominant flux fraction
fracRange = np.linspace(.5, .99, numSteps)
#No Sky
e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic = [[] for i in range(6)]
ellFracList = [e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic]
for frac in fracRange:
	domParams['frac'].value = frac
	contParams['frac'].value = 1-frac
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=PSF)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellFracList, valList)]
#With Sky
e1FracS, e2FracS, e1FracErrS, e2FracErrS, e1FracAnalyticS, e2FracAnalyticS = [[] for i in range(6)]
ellFracListS = [e1FracS, e2FracS, e1FracErrS, e2FracErrS, e1FracAnalyticS, e2FracAnalyticS]
for frac in fracRange:
	domParams['frac'].value = frac
	contParams['frac'].value = 1-frac
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellFracListS, valList)]
domParams['frac'].value = DOMINANT_FLUX_FRACTION
contParams['frac'].value = CONTAMINANT_FLUX_FRACTION

###Vary Signal to Noise Ratio
# fluxRange = np.linspace(1.e5, 4.e5, 8)
# snrRange = []
# threshold = .5*(skyMap.get('meanSky')*skyMap.get('expTime'))**.5

# e1Mean, e1Std, e1MeanStd, e2Mean, e2Std, e2MeanStd = [[] for i in range(6)]
# e1SNNoSky = []
# e2SNNoSky = []
# e1SNAnalytic = []
# e2SNAnalytic = []

# for flux in fluxRange:
# 	domParams['flux'].value = flux
# 	contParams['flux'].value = flux

# 	#Image with sky
# 	gals, mixIm = drawMixture(domParams, contParams, skyMap, psf=PSF)
# 	#Calculate SNR
# 	mask = mixIm.array>=threshold
# 	weight = mixIm.array
# 	wsnr = (mixIm.array*weight*mask).sum() / np.sqrt((weight*weight*skyMap.get('meanSky')*skyMap.get('expTime')*mask).sum())
# 	snrRange.append(wsnr)

# 	#No sky
# 	noSkyEllipt = calcEllipticities(domParams, contParams, skyMap, psf=PSF)
# 	e1SNNoSky.append(noSkyEllipt[0])
# 	e2SNNoSky.append(noSkyEllipt[1])

# 	#Sky
# 	skyEllip = calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
# 	e1SNAnalytic.append(skyEllip[4])
# 	e2SNAnalytic.append(skyEllip[5])
# 	#Now, run 100 trials for a single SNR
# 	N = 100
# 	trials = range(1, N)
# 	trialsE1 = []
# 	trialsE2 = []
# 	for n in trials:
# 		rng = galsim.BaseDeviate(n)
# 		skyMap['rng'] = rng
# 		singleSNREllip= calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
# 		trialsE1.append(singleSNREllip[0])
# 		trialsE2.append(singleSNREllip[1])
# 	#Mean of 100 trials
# 	e1Mean.append(np.mean(trialsE1))
# 	e2Mean.append(np.mean(trialsE2))
# 	#Standard deviation of 100 trials
# 	e1Std.append(np.std(trialsE1))
# 	e2Std.append(np.std(trialsE2))
# 	#Standard deviation on the mean, i.e. scatter of a single run
# 	e1MeanStd.append(np.std(trialsE1)/np.sqrt(N))
# 	e2MeanStd.append(np.std(trialsE2)/np.sqrt(N))
# domParams['flux'].value = DOMINANT_FLUX
# contParams['flux'].value = CONTAMINANT_FLUX

###Plot varying parameters
fig=pl.figure()

#Varying separation
ax11 = fig.add_subplot(121)
ax11.errorbar(sepRange, e1SepS, yerr=e1SepErrS, fmt='.', label='Fit e1, Sky', color='b')
ax11.plot(sepRange, e1SepAnalyticS, label='Analytic e1, Sky', color='g', linewidth=3)
ax11.plot(sepRange, e2SepAnalyticS, label='Analytic e2, Sky', color='m', linewidth=3)
ax11.errorbar(sepRange, e2SepS, yerr=e2SepErrS, fmt='.', label='Fit e2, Sky', color='c')
ax11.plot(sepRange, e1Sep, label='Fit e1, No Sky', color='r', linewidth=3)
ax11.set_xlabel('Separation, x-axis (arcsec)')
ax11.set_ylabel('Ellipticity')
ax11.set_title('Shear vs. Object Separation (Flux Fraction = .5)')
ax11.axhline(y=0, color='k')
ax11.axvline(x=0, color='k')
ax11.set_xlim(min(sepRange), max(sepRange))
ax11.legend(prop={'size':11}, loc=9)

#Varying fractional flux
ax12 = fig.add_subplot(122)
ax12.errorbar(fracRange, e1FracS, yerr=e1FracErrS, fmt='.', label='Fit e1, Sky', color='b')
ax12.plot(fracRange, e1FracAnalyticS, label='Analytic e1, Sky', color='g', linewidth=3)
ax12.plot(fracRange, e2FracAnalyticS, label='Analytic e2, Sky', color='m', linewidth=3)
ax12.errorbar(fracRange, e2FracS, yerr=e2FracErrS, fmt='.', label='Fit e2, Sky', color='c')
ax12.plot(fracRange, e1Frac, label='Fit e1, No Sky', color='r', linewidth=3)
ax12.set_xlabel('Dominant Galaxy Flux Fraction')
ax12.set_ylabel('Ellipticity')
ax12.set_title('Shear vs. Flux Fraction (Sep='+ str(dx) + ' arcsec='+str(dx/DOMINANT_HALF_LIGHT_RADIUS)+ ' HLR)')
ax12.legend()

#Varying SNR
# fig=pl.figure(3)
# ax31 = fig.add_subplot(111)
# ax31.axhline(y=0, color='k')
# ax31.axvline(x=0, color='k')
# ax31.errorbar(snrRange, e1Mean, yerr=e1Std, fmt='.', label='Fit e1, Sky', color='b')
# ax31.errorbar(snrRange, e1Mean, yerr=e1MeanStd, fmt='.', color='b')
# ax31.plot(snrRange, e1SNAnalytic, label='Analytic e1, Sky', color='g')
# ax31.plot(snrRange, e2SNAnalytic, label='Analytic e2, Sky', color='m')
# ax31.errorbar(snrRange, e2Mean, yerr=e2Std, fmt='.', label='Fit e2, Sky', color='c')
# ax31.errorbar(snrRange, e2Mean, yerr=e2MeanStd, fmt='.', label='Fit e2, Sky', color='c')
# ax31.plot(snrRange, e1SNNoSky, label='Fit e1, No Sky', color='r')
# ax31.set_xlabel('Signal-to-Noise Ratio (Weighted)')
# ax31.set_ylabel('Ellipticity')
# ax31.set_ylim(-.3, .6)
# ax31.set_xlim(0, 50)
# ax31.set_title('Shear vs. Signal-to-Noise Ratio')
# ax31.legend(prop={'size':11}, loc=7)

#Pull Distributions for sky vs no sky ellipticities from both studies
fig = pl.figure(2)
ax21 = fig.add_subplot(121)
sepSkyNoSkyPull = [(e1sky - e1nosky)/e1Error for e1sky, e1nosky, e1Error in zip(e1SepS, e1Sep, e1SepErrS)]
ax21.plot(sepRange, sepSkyNoSkyPull, '.')
ax21.set_ylim(-max(sepSkyNoSkyPull), max(sepSkyNoSkyPull))
ax21.axvline(x=0, color='k')
ax21.axhline(y=0, color='k')
ax21.set_xlabel('Separation (arcsec)')
ax21.set_ylabel('(e1Sky - e1NoSky)/e1Error')
ax21.set_title('Pull Distribution, vary separation, Sky vs No Sky')

ax22 = fig.add_subplot(122)
fracSkyNoSkyPull = [(e1sky - e1nosky)/e1Error for e1sky, e1nosky, e1Error in zip(e1FracS, e1Frac, e1FracErrS)]
ax22.plot(fracRange, fracSkyNoSkyPull, '.')
ax22.axhline(y=0, color='k')
ax22.set_ylim(-max(fracSkyNoSkyPull), max(fracSkyNoSkyPull))
ax22.set_xlabel('Dominant Galaxy Flux Fraction')
ax22.set_ylabel('(e1Sky - e1NoSky)/e1Error')
ax22.set_title('Pull Distribution, vary fraction, Sky vs No Sky')

pl.show()

#The things below can study noise bias from sky by looking at the pull distributions of the fit ellipticities with and without sky

def gaussResid(params, hist, bin):
	fitGauss = params['amplitude'].value*np.exp(-(bin-params['mean'].value)**2/(2*(params['sigma'].value**2)))
	return (hist-fitGauss)

#Separation pulls
sepRange1 = sepSkyNoSkyPull[0:len(sepSkyNoSkyPull)/4]
sepRange2 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)/4: len(sepSkyNoSkyPull)/2]
sepRange3 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)/2: len(sepSkyNoSkyPull)*3/4]
sepRange4 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)*3/4: len(sepSkyNoSkyPull)]

ranges = [sepRange1, sepRange2, sepRange3, sepRange4]

bins = 15
for sepRange in ranges:
	#Histogram of pulls for this range of separations
	mean = np.mean(sepRange)
	binEdg = np.linspace(mean-5, mean+5, bins)
	binCent = (binEdg[:-1]+binEdg[1:])/2
	hist, edg = np.histogram(sepRange, binEdg)

	#Parameters of Gaussian fit
	gaussParams = lm.Parameters()
	gaussParams.add('mean', value = mean)
	gaussParams.add('sigma', value = 0.5)
	gaussParams.add('amplitude', value = max(hist))

	out = lm.minimize(gaussResid, gaussParams, args=[hist, binCent])
	lm.report_errors(gaussParams)
	gauss = [gaussParams['amplitude'].value*np.exp((-(bin-gaussParams['mean'].value)**2/(2*(gaussParams['sigma'].value)**2))) for bin in binCent]
	
	fig = pl.figure(4)
	ax = fig.add_subplot(111)
	ax.hist(sepRange, binEdg)
	ax.plot(binCent, gauss)
	ax.set_title('Counts vs. Pulls (' + str(numSteps/4) + ' separations, ' + str(bins) + ' bins)')
	ax.set_xlabel('Separation Pulls')
	pl.show()

# Flux Fraction pulls
fracRange1 = fracSkyNoSkyPull[0:len(fracSkyNoSkyPull)/4]
fracRange2 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)/4: len(fracSkyNoSkyPull)/2]
fracRange3 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)/2: len(fracSkyNoSkyPull)*3/4]
fracRange4 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)*3/4: len(fracSkyNoSkyPull)]

fracRanges = [fracRange1, fracRange2, fracRange3, fracRange4]
for fracRange in fracRanges:
	#Histogram of pulls for this range of flux fractions
	mean = np.mean(fracRange)
	binEdg = np.linspace(mean-5, mean+5, bins)
	binCent = (binEdg[:-1]+binEdg[1:])/2
	hist, edg = np.histogram(fracRange, binEdg)

	gaussParams = lm.Parameters()
	gaussParams.add('mean', value = mean)
	gaussParams.add('sigma', value = 0.5)
	gaussParams.add('amplitude', value = max(hist))

	out = lm.minimize(gaussResid, gaussParams, args=[hist, binCent])
	lm.report_errors(gaussParams)
	# print "The arithmetic mean for this range: ", mean
	# print 'Mean: ', gaussParams['mean'].value
	# print 'Sigma: ', gaussParams['sigma'].value
	# print 'Uncertainty on mean (from fit)', gaussParams['sigma'].value/len(fracRange)
	gauss = [gaussParams['amplitude'].value*np.exp((-(bin-gaussParams['mean'].value)**2/(2*(gaussParams['sigma'].value)**2))) for bin in binCent]
	
	fig = pl.figure(4)
	ax = fig.add_subplot(111)
	ax.hist(fracRange, binEdg)
	ax.plot(binCent, gauss)
	ax.set_title('Counts vs. Pulls (' + str(numSteps/4) + ' flux fractions, ' + str(bins) + ' bins)')
	ax.set_xlabel('Flux Fraction Pulls')
	pl.show()
