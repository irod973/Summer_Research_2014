import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import scipy.optimize as opt
import gaussianMixtureLib as lib
from pprint import pprint

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

###############################Vary separation for one instance of sky
#Parameters for ellipticity studies
numSteps = 100
separation = 10 * PIXEL_SCALE #arcsec

###Vary separation in real space
print '#####Varying separation, no sky vs. single iteration of sky...'

sepRange = np.linspace(-separation, separation, numSteps)

#Lists for no sky
e1, e2, e1Err, e2Err, e1Analytic, e2Analytic = [[] for i in range(6)]
#Lists for sky
e1Sky, e2Sky, e1ErrSky, e2ErrSky, e1AnSky, e2AnSky = [[] for i in range(6)]
elliptNoSky = [e1, e2, e1Err, e2Err, e1Analytic, e2Analytic]
elliptSky = [e1Sky, e2Sky, e1ErrSky, e2ErrSky, e1AnSky, e2AnSky]

#Vary separation
for sep in sepRange:
	print 'Current separation: ', sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	ellipList = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF)
	ellipListSky = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
	[lst.append(value) for lst, value in zip(elliptNoSky, ellipList)]
	[lst.append(value) for lst, value in zip(elliptSky, ellipListSky)]

###Plot varying separation 
fig=pl.figure()

ax1 = fig.add_subplot(111)
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.errorbar(sepRange, e1Sky, yerr=e1ErrSky, fmt='.', capthick=3, label='Fit e1, Sky', color='r')
ax1.plot(sepRange, e1Analytic, label='Analytic e1', color='g', linewidth=2.5)
ax1.plot(sepRange, e2Analytic, label='Analytic e2', color='m', linewidth=2.5)
ax1.errorbar(sepRange, e2Sky, yerr=e2ErrSky, fmt='.', capthick=3, label='Fit e2, Sky', color='c')
ax1.plot(sepRange, e1, label='Fit e1, No Sky', color='b', linewidth=2.5)
ax1.set_xlabel('Separation, x-axis (arcsec)', fontsize=24)
ax1.set_ylabel('Ellipticity', fontsize=24)
#ax1.set_title('Ellipticity vs. Object Separation', fontsize=24)
ax1.set_xlim(min(sepRange), max(sepRange))
ax1.legend(prop={'size':18}, loc=9)

pl.show()

###############################Vary separation, fitting to different instances of sky for fixed separations
numSeps = 8
skyIterations = 100
print '#####Performing ' + str(skyIterations) + ' fits with different seeds for sky rng to ' + str(numSeps) + ' separations...'

separations = np.linspace(-separation, separation, numSeps)
e1Mean, e1Std, e1MeanStd, e2Mean, e2Std, e2MeanStd, seps, e1Pulls = [[] for i in range(8)]

for sep in separations:
	print 'Current separation: ', sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)

	noSkyEllip = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF)

	#Run 'skyIteration' trials for a single separation
	iterations = range(1, skyIterations)
	e1FromIters = []
	e2FromIters = []
	for n in iterations:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleSepEllip= lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
		e1FromIters.append(singleSepEllip[0])
		e2FromIters.append(singleSepEllip[1])

		seps.append(sep)
		e1Pulls.append(((singleSepEllip[0]-noSkyEllip[0])/singleSepEllip[2]))
	#Mean of N trials
	e1Mean.append(np.mean(e1FromIters))
	e2Mean.append(np.mean(e2FromIters))
	#Standard deviation of N trials
	e1Std.append(np.std(e1FromIters))
	e2Std.append(np.std(e2FromIters))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(e1FromIters)/np.sqrt(skyIterations))
	e2MeanStd.append(np.std(e2FromIters)/np.sqrt(skyIterations))

fig=pl.figure()
ax = fig.add_subplot(111)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.plot(sepRange, e1Analytic, label='Analytic e1', color='g', linewidth=2.5)
ax.plot(sepRange, e2Analytic, label='Analytic e2, Sky', color='m', linewidth=2.5)
ax.plot(sepRange, e1, label='Fit e1, No Sky', color='b', linewidth=2.5)
ax.errorbar(separations, e1Mean, yerr=e1Std, capthick=3, fmt='.', label='Fit e1, Sky', color='r')
ax.errorbar(separations, e1Mean, yerr=e1MeanStd, capthick=3,fmt='.', color='r')
ax.errorbar(separations, e2Mean, yerr=e2Std, capthick=3, fmt='.', label='Fit e2, Sky', color='c')
ax.errorbar(separations, e2Mean, yerr=e2MeanStd, capthick=3, fmt='.', color='c')
ax.set_xlabel('Separation (arcsec)', fontsize=18)
ax.set_ylabel('Ellipticity', fontsize=18)
ax.set_xlim(-2.5, 2.5)
ax.set_title('Shear vs. Separation', fontsize=24)
ax.legend(prop={'size':18}, loc=9)

fig2 =pl.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(seps, e1Pulls, 'b.')
ax2.set_xlim(-2.5, 2.5)
ax2.axhline(y=0, color='k')
ax2.set_xlabel('Separation')
ax2.set_ylabel('(e1Sky - e1NoSky)/e1SkyError')
ax2.set_title('e1 Pull vs. Separation for 100 fits per separation')

pl.show()

########The following code is obsolete given that it was looking at the pull distribution for a single instance of sky. Use the above pull distribution instead.
#The things below can study noise bias from sky by looking at the pull distributions of the fit ellipticities with and without sky

# def gaussResid(params, hist, bin):
# 	fitGauss = params['amplitude'].value*np.exp(-(bin-params['mean'].value)**2/(2*(params['sigma'].value**2)))
# 	return (hist-fitGauss)

#Separation pulls
# sepRange1 = sepSkyNoSkyPull[0:len(sepSkyNoSkyPull)/4]
# sepRange2 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)/4: len(sepSkyNoSkyPull)/2]
# sepRange3 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)/2: len(sepSkyNoSkyPull)*3/4]
# sepRange4 = sepSkyNoSkyPull[len(sepSkyNoSkyPull)*3/4: len(sepSkyNoSkyPull)]

# ranges = [sepRange1, sepRange2, sepRange3, sepRange4]

# bins = 15
# for sepRange in ranges:
# 	#Histogram of pulls for this range of separations
# 	mean = np.mean(sepRange)
# 	binEdg = np.linspace(mean-5, mean+5, bins)
# 	binCent = (binEdg[:-1]+binEdg[1:])/2
# 	hist, edg = np.histogram(sepRange, binEdg)

# 	#Parameters of Gaussian fit
# 	gaussParams = lm.Parameters()
# 	gaussParams.add('mean', value = mean)
# 	gaussParams.add('sigma', value = 0.5)
# 	gaussParams.add('amplitude', value = max(hist))

# 	out = lm.minimize(gaussResid, gaussParams, args=[hist, binCent])
# 	lm.report_errors(gaussParams)
# 	gauss = [gaussParams['amplitude'].value*np.exp((-(bin-gaussParams['mean'].value)**2/(2*(gaussParams['sigma'].value)**2))) for bin in binCent]
	
# 	fig = pl.figure(4)
# 	ax = fig.add_subplot(111)
# 	ax.hist(sepRange, binEdg)
# 	ax.plot(binCent, gauss)
# 	ax.set_title('Counts vs. Pulls (' + str(numSteps/4) + ' separations, ' + str(bins) + ' bins)')
# 	ax.set_xlabel('Separation Pulls')
# 	pl.show()