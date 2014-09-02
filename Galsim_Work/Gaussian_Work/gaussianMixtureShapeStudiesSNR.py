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

###Vary Signal to Noise Ratio
numSNRs = 8
fluxMin = 1.e5
fluxMax = 5.e5
skyIterations = 100

fluxRange = np.linspace(fluxMin, fluxMax, numSNRs)
snrRange = []
threshold = .5*(skyMap.get('meanSky')*skyMap.get('expTime'))**.5

e1NoSky, e2NoSky, e1Analytic, e2Analytic = [[] for i in range(4)] #For SNR range
e1Mean, e1Std, e1MeanStd, e2Mean, e2Std, e2MeanStd = [[] for i in range(6)] #For 100 iterations

print '#####Performing ' + str(skyIterations) + ' fits with different seeds for sky rng to ' + str(numSNRs) + ' SNRs...'

for flux in fluxRange:
	domParams['flux'].value = flux
	contParams['flux'].value = flux

	#Calculate SNR
	gals, mixIm = lib.drawMixture(domParams, contParams, skyMap, psf=PSF)
	mask = mixIm.array>=threshold
	weight = mixIm.array
	wsnr = (mixIm.array*weight*mask).sum() / np.sqrt((weight*weight*skyMap.get('meanSky')*skyMap.get('expTime')*mask).sum())
	snrRange.append(wsnr)

	print "Current SNR: ", wsnr

	##Find ellipticities
	#No sky
	noSkyEllipt = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF)
	e1NoSky.append(noSkyEllipt[0])
	e2NoSky.append(noSkyEllipt[1])

	#Sky
	skyEllipt = lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
	e1Analytic.append(skyEllipt[4])
	e2Analytic.append(skyEllipt[5])
	
	#Now, run 100 trials for a single SNR
	iterations = range(1, skyIterations)
	e1FromIters = []
	e2FromIters = []
	for n in iterations:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleSNREllip= lib.calcEllipticities(domParams, contParams, skyMap, psf=PSF, sky=True)
		e1FromIters.append(singleSNREllip[0])
		e2FromIters.append(singleSNREllip[1])
	#Mean of 100 trials
	e1Mean.append(np.mean(e1FromIters))
	e2Mean.append(np.mean(e2FromIters))
	#Standard deviation of 100 trials
	e1Std.append(np.std(e1FromIters))
	e2Std.append(np.std(e2FromIters))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(e1FromIters)/np.sqrt(skyIterations))
	e2MeanStd.append(np.std(e2FromIters)/np.sqrt(skyIterations))

#Plot varying SNR
fig=pl.figure(3)
ax = fig.add_subplot(111)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.errorbar(snrRange, e1Mean, yerr=e1Std, fmt='.', capthick=3, label='Fit e1, Sky', color='r')
ax.errorbar(snrRange, e1Mean, yerr=e1MeanStd, fmt='.', capthick=3, color='r')
ax.plot(snrRange, e1Analytic, label='Analytic e1', color='g', linewidth=3)
ax.plot(snrRange, e2Analytic, label='Analytic e2, Sky', color='m', linewidth=3)
ax.errorbar(snrRange, e2Mean, yerr=e2Std, fmt='.', capthick=2, label='Fit e2, Sky', color='c')
ax.errorbar(snrRange, e2Mean, yerr=e2MeanStd, fmt='.', capthick=2, color='c')
ax.plot(snrRange, e1NoSky, label='Fit e1, No Sky', color='b', linewidth=3)
ax.set_xlabel('Signal-to-Noise Ratio (Weighted)', fontsize=24)
ax.set_ylabel('Ellipticity', fontsize=24)
ax.set_ylim(-.3, .6)
ax.set_xlim(0, 75)
#ax.set_title('Ellipticity vs. Signal-to-Noise Ratio', fontsize=24)
ax.legend(prop={'size':18}, loc=2)

pl.show()