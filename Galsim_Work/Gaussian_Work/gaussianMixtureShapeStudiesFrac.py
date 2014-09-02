import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
import gaussianMixtureLib as lib
from pprint import pprint #I use this for debugging. It's nicer than print

#Irving Rodriguez
#
#This program studies the ellipticity of a single Gaussian fit to a mixture of two Gaussians.

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

###############################Vary flux fraction for one instance of sky
#Parameters for ellipticity studies
numSteps = 100
fracMin = .5
fracMax = .99

###Vary flux fraction of dominant galaxy
print '#####Varying flux fraction, no sky vs. single iteration of sky...'

fracRange = np.linspace(fracMin, fracMax, numSteps)

#No Sky
e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic = [[] for i in range(6)]
ellFracList = [e1Frac, e2Frac, e1FracErr, e2FracErr, e1FracAnalytic, e2FracAnalytic]

#Lists for no sky
e1, e2, e1Err, e2Err, e1Analytic, e2Analytic = [[] for i in range(6)]
#Lists for sky
e1Sky, e2Sky, e1ErrSky, e2ErrSky, e1AnSky, e2AnSky = [[] for i in range(6)]
elliptNoSky = [e1, e2, e1Err, e2Err, e1Analytic, e2Analytic]
elliptSky = [e1Sky, e2Sky, e1ErrSky, e2ErrSky, e1AnSky, e2AnSky]

for frac in fracRange:
	print "Current flux fraction: ", frac
	domParams['fluxFrac'].value = frac
	contParams['fluxFrac'].value = 1-frac
	elliptList=lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF)
	elliptListSky=lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
	[lst.append(value) for lst, value in zip(elliptNoSky, elliptList)]
	[lst.append(value) for lst, value in zip(elliptSky, elliptListSky)]

###Plot varying flux fraction

fig = pl.figure()
ax1 = fig.add_subplot(111)
ax1.axhline(y=0, color='k')
ax1.errorbar(fracRange, e1Sky, yerr=e1ErrSky, fmt='.', capthick=3, label='Fit e1, Sky', color='r')
ax1.plot(fracRange, e1Analytic, label='Analytic e1', color='g', linewidth=2.5)
ax1.plot(fracRange, e2Analytic, label='Analytic e2', color='m', linewidth=2.5)
ax1.errorbar(fracRange, e2Sky, yerr=e2ErrSky, fmt='.', capthick=3, label='Fit e2, Sky', color='c')
ax1.plot(fracRange, e1, label='Fit e1, No Sky', color='b', linewidth=2.5)
ax1.set_xlabel('Flux Fraction, Dominant Galaxy', fontsize=24)
ax1.set_ylabel('Ellipticity', fontsize=24)
#ax1.set_title('Ellipticity vs. Dominant Flux Fraction', fontsize=24)
ax1.legend(prop={'size':18}, loc=1)
pl.show()

###############################Vary flux fraction, fitting to different instances of sky for fixed fractions

numFracs = 8
skyIterations = 100
print 'Performing ' + str(skyIterations) + ' fits with different seeds for sky rng to ' + str(numFracs) + ' fractions...'

fractions = np.linspace(fracMin, fracMax, numFracs)
e1Mean, e1Std, e1MeanStd, e2Mean, e2Std, e2MeanStd, fracs, e1Pulls = [[] for i in range(8)]

for frac in fractions:
	print "Current fraction: ", frac
	domParams['fluxFrac'].value = frac
	contParams['fluxFrac'].value = 1 - frac

	noSkyEllip = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)

	#Run 'skyIterations' trials for a single fraction
	iterations = range(1, skyIterations)
	e1FromIters = []
	e2FromIters = []
	for n in iterations:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleFracEllip= lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
		e1FromIters.append(singleFracEllip[0])
		e2FromIters.append(singleFracEllip[1])

		fracs.append(frac)
		e1Pulls.append(((singleFracEllip[0]-noSkyEllip[0])/singleFracEllip[2]))

	#Mean of 100 trials
	e1Mean.append(np.mean(e1FromIters))
	e2Mean.append(np.mean(e2FromIters))
	#Standard deviation of 100 trials
	e1Std.append(np.std(e1FromIters))
	e2Std.append(np.std(e2FromIters))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(e1FromIters)/np.sqrt(skyIterations))
	e2MeanStd.append(np.std(e2FromIters)/np.sqrt(skyIterations))

fig=pl.figure()
ax = fig.add_subplot(111)
ax.axhline(y=0, color='k')
ax.plot(fracRange, e1Analytic, label='Analytic e1', color='g', linewidth=2.5)
ax.plot(fracRange, e2Analytic, label='Analytic e2, Sky', color='m', linewidth=2.5)
ax.plot(fracRange, e1, label='Fit e1, No Sky', color='b', linewidth=2.5)
ax.errorbar(fractions, e1Mean, yerr=e1Std, capthick=3, fmt='.', label='Fit e1, Sky', color='r')
ax.errorbar(fractions, e1Mean, yerr=e1MeanStd, capthick=3,fmt='.', color='r')
ax.errorbar(fractions, e2Mean, yerr=e2Std, capthick=3, fmt='.', label='Fit e2, Sky', color='c')
ax.errorbar(fractions, e2Mean, yerr=e2MeanStd, capthick=3, fmt='.', color='c')
ax.set_xlabel('Flux Fraction, Dominant Galaxy', fontsize=18)
ax.set_ylabel('Ellipticity', fontsize=18)
ax.set_title('Ellipticity vs. Flux Fraction', fontsize=24)
ax.legend(prop={'size':18}, loc=1)

fig2 =pl.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(fracs, e1Pulls, 'b.')
ax2.axhline(y=0, color='k')
ax2.set_xlabel('Flux Fractions')
ax2.set_ylabel('(e1Sky - e1NoSky)/e1SkyError')
ax2.set_title('e1 Pull vs. Flux Fraction for 100 fits per fraction')

pl.show()

########The following code is obsolete given that it was looking at the pull distribution for a single instance of sky. Use the above pull distribution instead.
#The things below can study noise bias from sky by looking at the pull distributions of the fit ellipticities with and without sky


# def gaussResid(params, hist, bin):
# 	fitGauss = params['amplitude'].value*np.exp(-(bin-params['mean'].value)**2/(2*(params['sigma'].value**2)))
# 	return (hist-fitGauss)


# Flux Fraction pulls
# fracRange1 = fracSkyNoSkyPull[0:len(fracSkyNoSkyPull)/4]
# fracRange2 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)/4: len(fracSkyNoSkyPull)/2]
# fracRange3 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)/2: len(fracSkyNoSkyPull)*3/4]
# fracRange4 = fracSkyNoSkyPull[len(fracSkyNoSkyPull)*3/4: len(fracSkyNoSkyPull)]

# fracRanges = [fracRange1, fracRange2, fracRange3, fracRange4]
# for fracRange in fracRanges:
# 	#Histogram of pulls for this range of flux fractions
# 	mean = np.mean(fracRange)
# 	binEdg = np.linspace(mean-5, mean+5, bins)
# 	binCent = (binEdg[:-1]+binEdg[1:])/2
# 	hist, edg = np.histogram(fracRange, binEdg)

# 	gaussParams = lm.Parameters()
# 	gaussParams.add('mean', value = mean)
# 	gaussParams.add('sigma', value = 0.5)
# 	gaussParams.add('amplitude', value = max(hist))

# 	out = lm.minimize(gaussResid, gaussParams, args=[hist, binCent])
# 	lm.report_errors(gaussParams)
# 	# print "The arithmetic mean for this range: ", mean
# 	# print 'Mean: ', gaussParams['mean'].value
# 	# print 'Sigma: ', gaussParams['sigma'].value
# 	# print 'Uncertainty on mean (from fit)', gaussParams['sigma'].value/len(fracRange)
# 	gauss = [gaussParams['amplitude'].value*np.exp((-(bin-gaussParams['mean'].value)**2/(2*(gaussParams['sigma'].value)**2))) for bin in binCent]
	
# 	fig = pl.figure(4)
# 	ax = fig.add_subplot(111)
# 	ax.hist(fracRange, binEdg)
# 	ax.plot(binCent, gauss)
# 	ax.set_title('Counts vs. Pulls (' + str(numSteps/4) + ' flux fractions, ' + str(bins) + ' bins)')
# 	ax.set_xlabel('Flux Fraction Pulls')
# 	pl.show()
