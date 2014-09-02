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

###############################Vary separation with no sky

sepMax = 7.5*PIXEL_SCALE
numSteps = 100
sepRange = np.linspace(-sepMax, sepMax, numSteps)

print '#####Varying separation, no sky...'

e1NoSky, e2NoSky = [[] for i in range(2)]
for sep in sepRange:
	print "Current Separation: ", sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)
	ellipt = lib.calcEllipticities(domParams, contParams, skyMap, psf=True)
	e1NoSky.append(ellipt[0])
	e2NoSky.append(ellipt[1])

#Plot, but do not show yet
fig = pl.figure()
ax=fig.add_subplot(111)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.plot(sepRange, e1NoSky, 'b', label='Fit e1, no sky', linewidth=3)
ax.plot(sepRange, e2NoSky, 'm', label='Fit e2, no sky', linewidth=3)

###############################Vary separation, fitting to different instances of sky for fixed separations
numSeps = 8
skyIterations = 100
print '#####Performing ' + str(skyIterations) + ' fits with different seeds for sky rng to ' + str(numSeps) + ' separations...'
separations = np.linspace(-sepMax, sepMax, numSeps)
e1Mean, e2Mean, e1Std, e2Std, e1MeanStd, e2MeanStd = [[] for i in range(6)]
for sep in separations:
	print "Current Separation: ", sep
	contParams['centX'].value = sep*np.cos(PHI)
	contParams['centY'].value = sep*np.sin(PHI)

	#Now, run 'skyIterations' number of trials for a single separation
	iterations = range(1, skyIterations)
	e1FromIter = []
	e2FromIter = []
	for n in iterations:
		rng = galsim.BaseDeviate(n)
		skyMap['rng'] = rng
		singleSepEllip= lib.calcEllipticities(domParams, contParams, skyMap, psf=True, sky=True)
		e1FromIter.append(singleSepEllip[0])
		e2FromIter.append(singleSepEllip[1])
	#Mean of 100 trials
	e1Mean.append(np.mean(e1FromIter))
	e2Mean.append(np.mean(e2FromIter))
	#Standard deviation of 100 trials
	e1Std.append(np.std(e1FromIter))
	e2Std.append(np.std(e2FromIter))
	#Standard deviation on the mean
	e1MeanStd.append(np.std(e1FromIter)/np.sqrt(skyIterations))
	e2MeanStd.append(np.std(e2FromIter)/np.sqrt(skyIterations))

###Finish plotting

ax.errorbar(separations, e2Mean, yerr=e2Std, label='Fit e2, sky', capthick=2, fmt='.', color='c')
ax.errorbar(separations, e2Mean, yerr=e2MeanStd, fmt='.', capthick=2, color='c')
ax.errorbar(separations, e1Mean, yerr=e1Std, label='Fit e1, sky', fmt='.', capthick=2, color='r')
ax.errorbar(separations, e1Mean, yerr=e1MeanStd, fmt='.', capthick=2, color='r')
#ax.set_title('Ellipticity vs. Separation', fontsize=24)
ax.set_xlabel('Separation', fontsize=24)
ax.set_ylabel('Fit Ellipticity', fontsize=24)
ax.set_xlim(-2, 2)
pl.legend(prop={'size':18}, loc=9)
pl.show()