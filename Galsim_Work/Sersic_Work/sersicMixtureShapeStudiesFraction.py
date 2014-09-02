import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl
import math
import sersicMixtureLib as lib
from pprint import pprint

#Irving Rodriguez
#
#Draws a mixture of two bulge+disk sersic galaxies, fits a single bulge+disk sersic, and draws the residual between the best fit and the mixture


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
DOMINANT_BULGE_INDEX = 4
DOMINANT_DISK_INDEX = 1
DOMINANT_CENTROID = (0,0)
CONTAMINANT_FLUX = DOMINANT_FLUX
CONTAMINANT_HALF_LIGHT_RADIUS = DOMINANT_HALF_LIGHT_RADIUS
CONTAMINANT_FLUX_FRACTION = 1 - DOMINANT_FLUX_FRACTION
CONTAMINANT_E1 = 0
CONTAMINANT_E2 = 0
CONTAMINANT_BULGE_INDEX = 4
CONTAMINANT_DISK_INDEX = 1
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
domParams.add('bulgeNIndex', value=DOMINANT_BULGE_INDEX)
domParams.add('diskNIndex', value=DOMINANT_DISK_INDEX)
domParams.add('fluxFrac', value=DOMINANT_FLUX_FRACTION)

contParams = lm.Parameters()
contParams.add('flux', value=CONTAMINANT_FLUX)
contParams.add('HLR', value=CONTAMINANT_HALF_LIGHT_RADIUS)
contParams.add('centX', value=dx)
contParams.add('centY', value=dy)
contParams.add('e1', value=CONTAMINANT_E1)
contParams.add('e2', value=CONTAMINANT_E2)
contParams.add('bulgeNIndex', value=CONTAMINANT_BULGE_INDEX)
contParams.add('diskNIndex', value=CONTAMINANT_DISK_INDEX)
contParams.add('fluxFrac', value=CONTAMINANT_FLUX_FRACTION)


###############################Vary flux fraction for no sky

print '#####Varying flux fraction, no sky...'

numSteps = 100
fracMin = .5
fracMax = .99
fracRange = np.linspace(fracMin, fracMax, numSteps)

e1NoSky, e2NoSky = [[] for i in range(2)]
for frac in fracRange:
	print "Current flux fraction: ", frac
	domParams['fluxFrac'].value = frac
	contParams['fluxFrac'].value = 1 - frac
	ellipt = lib.calcEllipticities(domParams, contParams, skyMap, psf=True)
	e1NoSky.append(ellipt[0])
	e2NoSky.append(ellipt[1])

fig = pl.figure()
ax=fig.add_subplot(111)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.plot(fracRange, e1NoSky, 'b', label='Fit e1, no sky', linewidth=3)
ax.plot(fracRange, e2NoSky, 'm', label='Fit e2, no sky', linewidth=3)


###############################Vary separation, fitting to different instances of sky for fixed separations
numFracs = 8 
skyIterations = 100
e1Mean, e2Mean, e1Std, e2Std, e1MeanStd, e2MeanStd = [[] for i in range(6)]

print '100 fits with different sky...'

fractions = np.linspace(fracMin, fracMax, numFracs)
for frac in fractions:
	print "Current flux fraction: ", frac
	domParams['fluxFrac'].value = frac
	contParams['fluxFrac'].value = 1 - frac

	#Run 'skyIterations' number of trials for a single separation
	iterations = range(1, skyIterations)
	e1FromIter = []
	e2FromIter = []
	for n in iterations:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleFracEllip= lib.calcEllipticities(domParams, contParams, skyMap, psf=True, sky=True)
		e1FromIter.append(singleFracEllip[0])
		e2FromIter.append(singleFracEllip[1])
	#Mean of 100 trials
	e1Mean.append(np.mean(e1FromIter))
	e2Mean.append(np.mean(e2FromIter))
	#Standard deviation of 100 trials
	e1Std.append(np.std(e1FromIter))
	e2Std.append(np.std(e2FromIter))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(e1FromIter)/np.sqrt(skyIterations))
	e2MeanStd.append(np.std(e2FromIter)/np.sqrt(skyIterations))

#Finish plotting

ax.errorbar(fractions, e2Mean, yerr=e2Std, label='Fit e2, sky', capthick=2, fmt='.', color='c')
ax.errorbar(fractions, e2Mean, yerr=e2MeanStd, fmt='.', capthick=2, color='c')
ax.errorbar(fractions, e1Mean, yerr=e1Std, label='Fit e1, sky', fmt='.', capthick=2, color='r')
ax.errorbar(fractions, e1Mean, yerr=e1MeanStd, fmt='.', capthick=2, color='r')
#ax.set_title('Ellipticity vs. Flux Fraction (Sep = 1 arcsec = 1 HLR)', fontsize=24)
ax.set_xlabel('Flux Fraction', fontsize=24)
ax.set_ylabel('Fit Ellipticity', fontsize=24)
ax.set_xlim(.45, 1)
pl.legend(prop={'size':18}, loc=9)
pl.show()
