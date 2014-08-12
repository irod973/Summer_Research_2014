import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
from pprint import pprint

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
		mix = galsim.Convolve([mix, psf])
		image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		image = mix.drawImage(image=image, method='fft')
		# image.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel times 30 sec exposure time
		gals = [domGal, contGal.original, mix]
		return  gals, image
	image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	image = mix.drawImage(image=image, method='fft')
	if sky==True:
		image.addNoise(galsim.PoissonNoise(rng=skyMap.get('rng'), sky_level=skyMap.get('skyLevel')*skyMap.get('expTime'))) #skylevel is counts/sec/pixel multiplied by full 10-year LSST band exposure time
	gals = [domGal, contGal.original, mix]
	return gals, image

#Compute the chi-squared residual between the parameter image and the mixture image, using errors from the average sky level over a full stack
def residual(params, mixIm, skyMap, pixelScale=.2, stampSize=100):
	fitIm = drawFit(params, pixelScale, stampSize)
	return (mixIm - fitIm).array.ravel()**2/(skyMap.get('skyLevel')*skyMap.get('expTime') + fitIm.array.ravel())

#
def mixtureMoments(gals, galParams):
	fractions = [params['frac'].value for params in galParams]
	zeroMoments = [params['flux'].value for params in galParams]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in galParams]
	secondMoments = [[[gal.getSigma()**2, 0], [0, gal.getSigma()**2]] for gal in gals]
	return fractions, zeroMoments, firstMoments, secondMoments

def calcEllipticities(domParams, contParams, skyMap, pixelScale=.2, stampSize=100, sky=False, psf=False):
	#Draw the mixture
	if sky==True:
		gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, sky=True)
	elif psf==True:
		gals, mixIm = drawMixture(domParams, contParams, skyMap, pixelScale, stampSize, psf=True)
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
	#Get errors if relevant
	if sky==True:
		e1err = np.sqrt(np.diag(out.covar))[2]
		e2err = np.sqrt(np.diag(out.covar))[3]
	else:
		e1err = 0
		e2err = 0 #cheating, kind of. the errors are just meaningless and we are not plotting them anyway, but do not want to change return statement

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

du = 10 * pixelScale #arcsec
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

#Ellipticity studies
#Parameters for ellipticity studies
numSteps = 100
separation = 10 * pixelScale #arcsec

#Vary separation in real space
sepRange = np.linspace(-separation, separation, numSteps)
#No PSF
e1Sep, e2Sep, e1SepErr, e2SepErr, e1SepAnalytic, e2SepAnalytic = [[] for i in range(6)]
ellSepList = [e1Sep, e1SepErr, e2Sep, e2SepErr, e1SepAnalytic, e2SepAnalytic]
for sep in sepRange:
	contParams['centX'].value = sep*np.cos(phi)
	contParams['centY'].value = sep*np.sin(phi)
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellSepList, valList)]
#With PSF
e1SepP, e2SepP, e1SepErrP, e2SepErrP, e1SepAnalyticP, e2SepAnalyticP = [[] for i in range(6)]
ellSepListP = [e1SepP, e2SepP, e1SepErrP, e2SepErrP, e1SepAnalyticP, e2SepAnalyticP]
for sep in sepRange:
	contParams['centX'].value = sep*np.cos(phi)
	contParams['centY'].value = sep*np.sin(phi)
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap,psf=True)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellSepListP, valList)]
contParams['centX'].value = dx
contParams['centY'].value = dy

#Vary dominant flux fraction
fracRange = np.linspace(.5, .99, numSteps)
#No PSF
e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic = [[] for i in range(6)]
ellFracList = [e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic]
for frac in fracRange:
	domParams['frac'].value = frac
	contParams['frac'].value = 1-frac
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellFracList, valList)]
#With PSF
e1FracP, e2FracP, e1FracErrP, e2FracErrP, e1FracAnalyticP, e2FracAnalyticP = [[] for i in range(6)]
ellFracListP = [e1FracP, e2FracP, e1FracErrP, e2FracErrP, e1FracAnalyticP, e2FracAnalyticP]
for frac in fracRange:
	domParams['frac'].value = frac
	contParams['frac'].value = 1-frac
	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=True)
	valList = [e1, e2, e1err, e2err, e1A, e2A]
	[lst.append(value) for lst, value in zip(ellFracListP, valList)]
domParams['frac'].value = domFrac
contParams['frac'].value = 1-domFrac

#Vary Signal to Noise Ratio
# signalNoiseRange = np.linspace(10, 40, numSteps)
# fluxRange = [snr*skyMap.get('skyLevel')*skyMap.get('expTime') for snr in signalNoiseRange]
# e1SN, e2SN, e1SNErr, e2SNErr, e1SNAnalytic, e2SNAnalytic = [[] for i in range(6)] #SN means signal to noise
# ellSNList = [e1SN, e2SN, e1SNErr, e2SNErr, e1SNAnalytic, e2SNAnalytic]
# for flux in fluxRange:
# 	domParams['flux'].value = flux
# 	contParams['flux'].value = flux
# 	e1, e2, e1err, e2err, e1A, e2A = calcEllipticities(domParams, contParams, skyMap, psf=True)
# 	valList = [e1, e2, e1err, e2err, e1A, e2A]
# 	[lst.append(value) for lst, value in zip(ellSNList, valList)]
# domParams['flux'].value = domFlux
# contParams['flux'].value = contFlux

#Plots
# domStr = 'Dominant Galaxy: ' + str(gals[0].centroid()) + ', '+str(domFlux)+', '+str(domHLR)+', '+str(domFrac)+', 0, 0'
# contStr = 'Contaminant Galaxy: (' + str(contParams['centX'].value) + ', ' + str(contParams['centY'].value) + '), '+str(contFlux)+', '+str(contHLR)+', '+str((1-domFrac))+', 0, 0'
# titleStr = 'Parameters (centroid, flux, hlr, fraction of flux, e1, e2)\n'+domStr+ '\n'+contStr+'\nPixel Scale: '+ str(pixelScale)+' arcsec/pixel'
# fig1.suptitle(titleStr, fontsize=12)

#Plotting varying parameters
fig=pl.figure()

#Vary separation
ax11 = fig.add_subplot(121)
ax11.errorbar(sepRange, e1SepP, yerr=e1SepErrP, fmt='.', label='Fit e1, PSF')
ax11.plot(sepRange, e1SepAnalyticP, label='Analytic e1, PSF')
ax11.plot(sepRange, e2SepAnalyticP, label='Analytic e2, PSF')
ax11.errorbar(sepRange, e2SepP, yerr=e2SepErrP, fmt='.', label='Fit e2, PSF')
ax11.plot(sepRange, e1Sep, label='Fit e1, No PSF')
ax11.set_xlabel('Separation, x-axis (arcsec)')
ax11.set_ylabel('Ellipticity')
ax11.set_title('Shear vs. Object Separation (Flux Fraction = .5)')
ax11.axhline(y=0, color='k')
ax11.axvline(x=0, color='k')
ax11.set_xlim(min(sepRange), max(sepRange))
ax11.legend(prop={'size':11}, loc=9)

#Vary fractional flux

ax12 = fig.add_subplot(122)
ax12.errorbar(fracRange, e1FracP, yerr=e1FracErrP, fmt='.', label='Fit e1, PSF')
ax12.plot(fracRange, e1FracAnalyticP, label='Analytic e1, PSF')
ax12.plot(fracRange, e2FracAnalyticP, label='Analytic e2, PSF')
ax12.errorbar(fracRange, e2FracP, yerr=e2FracErrP, fmt='.', label='Fit e2, PSF')
ax12.plot(fracRange, e1Frac, label='Fit e1, No PSF')
ax12.set_xlabel('Dominant Galaxy Flux Fraction')
ax12.set_ylabel('Ellipticity')
ax12.set_title('Shear vs. Flux Fraction (Sep='+ str(dx) + ' arcsec='+str(dx/domHLR)+ ' HLR)')
ax12.legend()

#Pull Distributions for sky vs no sky ellipticities from both studies
fig = pl.figure(2)
ax21 = fig.add_subplot(121)
sepPSFNoPSFPull = [(e1PSF - e1noPSF)/e1Error for e1PSF, e1noPSF, e1Error in zip(e1SepP, e1Sep, e1SepErrP)]
ax21.plot(sepRange, sepPSFNoPSFPull, '.')
ax21.set_ylim(-max(sepPSFNoPSFPull), max(sepPSFNoPSFPull))
ax21.axvline(x=0, color='k')
ax21.axhline(y=0, color='k')
ax21.set_xlabel('Separation (arcsec)')
ax21.set_ylabel('(e1PSF - e1NoPSF)/e1Error')
ax21.set_title('Pull Distribution, vary separation, PSF vs No PSF')

ax22 = fig.add_subplot(122)
fracPSFNoPSFPull = [(e1PSF - e1noPSF)/e1Error for e1PSF, e1noPSF, e1Error in zip(e1FracP, e1Frac, e1FracErrP)]
ax22.plot(fracRange, fracPSFNoPSFPull, '.')
ax22.axhline(y=0, color='k')
ax22.set_ylim(-max(fracPSFNoPSFPull), max(fracPSFNoPSFPull))
ax22.set_xlabel('Dominant Galaxy Flux Fraction')
ax22.set_ylabel('(e1PSF - e1NoPSF)/e1Error')
ax22.set_title('Pull Distribution, vary fraction, PSF vs No PSF')

#Plot varying SNR
# fig=pl.figure(3)

# ax31 = fig.add_subplot(111)
# ax31.errorbar(signalNoiseRange, e1SN, yerr=e1SNErr, fmt='.', label='Fit e1')
# ax31.errorbar(signalNoiseRange, e2SN, yerr=e2SNErr, fmt='.', label='Fit e2')
# ax31.plot(signalNoiseRange, e1SNAnalytic, label='Analytic e1')
# ax31.plot(signalNoiseRange, e2SNAnalytic, label='Analytic e2')
# ax31.set_xlabel('Signal-to-Noise Ratio')
# ax31.set_ylabel('Ellipticity')
# ax31.set_title('Shear vs. Signal-to-Noise Ratio')
# ax31.legend(prop={'size':11})

pl.show()

#The things below are to study noise bias from PSF by look at the pull distributions of the fit ellipticities with and without PSF
def gaussResid(params, pullHist, x):
	fitGauss = params['amplitude'].value*np.exp(-(x-params['mean'].value)**2/(2*params['sigma'].value**2))
	return (pullHist-fitGauss)**2/.009

#Separation pulls
# sepRangeM2M1 = sepPSFNoPSFPull[0:len(sepPSFNoPSFPull)/4]
# sepRangeM10 = sepPSFNoPSFPull[len(sepPSFNoPSFPull)/4: len(sepPSFNoPSFPull)/2]
# sepRange01 = sepPSFNoPSFPull[len(sepPSFNoPSFPull)/2: len(sepPSFNoPSFPull)*3/4]
# sepRange12 = sepPSFNoPSFPull[len(sepPSFNoPSFPull)*3/4: len(sepPSFNoPSFPull)]

# ranges = [sepRangeM]

# x = np.linspace(-10, 10, 100)
# for bin in bins:
# 	hist, edg = np.histogram(bin, x)
# 	hist = np.append(hist, 0)

# 	gaussParams = lm.Parameters()
# 	gaussParams.add('mean', value = 0)
# 	gaussParams.add('sigma', value = 1)
# 	gaussParams.add('amplitude', value = max(hist))

# 	out = lm.minimize(gaussResid, gaussParams, args=[hist, x])
# 	print 'Pull Mean: ', gaussParams['mean'].value
# 	print 'Pull Sigma: ', gaussParams['sigma'].value
# 	gauss = [gaussParams['amplitude'].value*np.exp((-(X-gaussParams['mean'].value)**2/(2*gaussParams['sigma'].value)**2)) for X in x]
# 	fig = pl.figure(4)
# 	ax = fig.add_subplot(111)
# 	ax.hist(bin, x)
# 	ax.plot(x, gauss)
# 	ax.set_title('Counts vs. Pulls (1000 separations, 100 bins)')
# 	ax.set_xlabel('Separation Pulls')
# 	pl.show()

#Flux Fraction pulls
# bin1 = fracPSFNoPSFPull[0:len(fracPSFNoPSFPull)/4]
# bin2 = fracPSFNoPSFPull[len(fracPSFNoPSFPull)/4: len(fracPSFNoPSFPull)/2]
# bin3 = fracPSFNoPSFPull[len(fracPSFNoPSFPull)/2: len(fracPSFNoPSFPull)*3/4]
# bin4 = fracPSFNoPSFPull[len(fracPSFNoPSFPull)*3/4: len(fracPSFNoPSFPull)]

# bins = [bin1, bin2, bin3, bin4]
# bins = [fracPSFNoSkyPull]
# x = np.linspace(-20, 20, 200)
# for bin in bins:
# 	hist, edg = np.histogram(bin, x)
# 	hist = np.append(hist, 0)

# 	gaussParams = lm.Parameters()
# 	gaussParams.add('mean', value = 0)
# 	gaussParams.add('sigma', value = 1)
# 	gaussParams.add('amplitude', value = max(hist)/2)

# 	out = lm.minimize(gaussResid, gaussParams, args=[hist, x])
# 	print 'Pull Mean: ', gaussParams['mean'].value
# 	print 'Pull Sigma: ', gaussParams['sigma'].value
# 	gauss = [gaussParams['amplitude'].value*np.exp((-(X-gaussParams['mean'].value)**2/(2*gaussParams['sigma'].value)**2)) for X in x]
# 	fig = pl.figure(4)
# 	ax = fig.add_subplot(111)
# 	ax.hist(bin, x)
# 	ax.plot(x, gauss)
# 	ax.set_title('Counts vs. Pulls (1000 separations, 100 bins)')
# 	ax.set_xlabel('Flux Fraction Pulls')
# 	pl.show()