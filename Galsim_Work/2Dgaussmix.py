import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

#Irving Rodriguez
#
#This program studies the ellipticity of fitting a single Gaussian to a mixture of two Gaussians. It's tight
#
def gaussMix(domFlux, domSig, unFlux, unSig, ff, du, scale):
	domGal = galsim.Gaussian(flux=domFlux, sigma=domSig)
	unGal = galsim.Gaussian(flux=unFlux, sigma=unSig)
	unGal = unGal.shift(dx=du, dy=0)
	mix = ff * domGal + (1-ff) * unGal
	image = galsim.ImageD(scale=scale)
	image = mix.draw(image=image)
	return image

#Draw mixture using photon shoot method
def gaussMixShoot(domFlux, domSig, unFlux, unSig, ff, dx, dy, seed, scale):
	domGal = galsim.Gaussian(flux=domFlux, half_light_radius=domHLR)
	unGal = galsim.Gaussian(flux=unFlux, half_light_radius=unHLR)
	unGal = unGal.shift(dx=dx, dy=dy)
	mix = ff * domGal + (1-ff) * unGal
	image = galsim.ImageD(64, 64, scale=scale)
	image = mix.drawImage(image=image, rng = seed, method='phot')
	gals = [domGal, unGal, mix]
	return  gals, image

#Compute the least-squares residual between the parameter image and the mixture image 
def leastSqResid(param, mixIm):
	flux = param['fitFlux'].value
	hlr = param['fitHLR'].value
	centX = param['fitCentX'].value
	centY = param['fitCentY'].value
	e1 = param['fite1'].value
	e2 = param['fite2'].value
	fit = galsim.Gaussian(flux=flux, half_light_radius=hlr)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centX, dy=centY)
	image = galsim.ImageD(64, 64, scale=.2)
	image= fit.draw(image=image)
	return (image - mixIm).array.ravel()

#
def fitProfile(result, scale):
	fit = galsim.Gaussian(flux=result.params['fitFlux'].value, half_light_radius=result.params['fitHLR'].value)
	fit = fit.shear(e1=result.params['fite1'].value, e2=result.params['fite2'].value)
	fit = fit.shift(dx=result.params['fitCentX'].value, dy=result.params['fitCentY'].value)
	fitIm = galsim.ImageD(64, 64, scale=scale)
	fitIm = fit.draw(image=fitIm)
	return fit, fitIm

def fitSep(domFlux, domHLR, unFlux, unHLR, ff, shift, dy, seed, pixelScale):
	params = lm.Parameters()
	params.add('fitFlux', value=5.e5)
	params.add('fitHLR', value=2)
	params.add('fitCentX', value=0)
	params.add('fitCentY', value=0)
	params.add('fite1', value=0., min=-1, max=1)
	params.add('fite2', value=0., min=-1, max=1)

	e1 = []
	e2 = []
	e1Err = []
	e2Err = []
	ane1 = []
	ane2 = []
	for dx in shift:
		#Get ellipticities of fit
		gals, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, ff, dx, dy, seed, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(64, 64, scale=.2)
		fitIm = fit.draw(image=fitIm)
		fitMoments = fitIm.FindAdaptiveMom()
		e1.append(fitMoments.observed_shape.e1)
		e2.append(fitMoments.observed_shape.e2)
		e1Err.append(np.sqrt(np.diag(out.covar))[2])
		e2Err.append(np.sqrt(np.diag(out.covar))[3])

		#Get analytic ellipticities
		ae1, ae2 = analyticMixShear(gals, ff, dx)
		ane1.append(ae1)
		ane2.append(ae2)
	return e1, e1Err, e2, e2Err, ane1, ane2

def fitProp(domFlux, domHLR, unFlux, unHLR, f, dx, dy, seed, pixelScale):
	params = lm.Parameters()
	params.add('fitFlux', value=5.e5)
	params.add('fitHLR', value=2)
	params.add('fitCentX', value=0)
	params.add('fitCentY', value=0)
	params.add('fite1', value=0., min=-1, max=1)
	params.add('fite2', value=0., min=-1, max=1)

	e1p = []
	e2p = []
	e1Perr = []
	e2Perr = []
	for ff in f:
		gals, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, ff, dx, dy, seed, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(64, 64, scale=.2)
		fitIm = fit.draw(image=fitIm)
		fitMoments = fitIm.FindAdaptiveMom()
		e1p.append(fitMoments.observed_shape.e1)
		e2p.append(fitMoments.observed_shape.e2)
		e1Perr.append(np.sqrt(np.diag(out.covar))[2])
		e2Perr.append(np.sqrt(np.diag(out.covar))[3])
	return e1p, e1Perr, e2p, e2Perr

def fitSepS(domFlux, domHLR, unFlux, unHLR, ff, shift, seed, pixelScale):
	params = lm.Parameters()
	params.add('fitFlux', value=5.e5)
	params.add('fitHLR', value=2)
	params.add('fitCentX', value=0)
	params.add('fitCentY', value=0)
	params.add('fite1', value=0., min=-1, max=1)
	params.add('fite2', value=0., min=-1, max=1)

	e1s = []
	e2s = []
	e1Serr = []
	e2Serr = []

	for du in shift:
		gals, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, ff, du[0], du[1], seed, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(64, 64, scale=.2)
		fitIm = fit.draw(image=fitIm)        
		fitMoments = fitIm.FindAdaptiveMom()
		e1s.append(fitMoments.observed_shape.e1)
		e2s.append(fitMoments.observed_shape.e2)
		e1Serr.append(np.sqrt(np.diag(out.covar))[2])
		e2Serr.append(np.sqrt(np.diag(out.covar))[3])
	return e1s, e1Serr, e2s, e2Serr

def analyticMixShear(gals, ff, dx):
	domGal = gals[0]
	unGal = gals[1]

	#0th Moments of Components
	fA = domGal.getFlux()
	fB = unGal.getFlux()

	#1st Moments of Components
	uA = (0,0)
	uB = (dx, 0)

	#2nd Moments of Components, Qxy = 0 for circular Gaussian
	domSig = domGal.getSigma()
	QxxA = domSig**2 #I think
	QyyA = QxxA
	QxyA = 0

	unSig = domSig
	QxxB = unSig**2 #I think
	QyyB = QxxB
	QxyB = 0

	#0th Moment of Mixture (flux)
	fM = ff*fA + (1-ff)*fB

	#1st moment of mixture (centroid)
	XM = (1/fM)*((ff*uA[0]*fA) + ((1-ff)*uB[0]*fB)) 
	YM = (1/fM)*((ff*uA[1]*fA) + ((1-ff)*uB[1]*fB))
	uM = (XM, YM)

	#2nd Moments of mixture
	QxxM = (1/fM)*(fA*QxxA+fB*QxxB) + (1/fM)*(fA*(uA[0]-uM[0])*(uA[0]-uM[0])+fB*(uB[0]-uM[0])*(uB[0]-uM[0]))
	QxyM = (1/fM)*(fA*QxyA+fB*QxyB) + (1/fM)*(fA*(uA[0]-uM[0])*(uA[1]-uM[1])+fB*(uB[0]-uM[0])*(uB[1]-uM[1]))
	QyyM = (1/fM)*(fA*QyyA+fB*QyyB) + (1/fM)*(fA*(uA[1]-uM[1])*(uA[1]-uM[1])+fB*(uB[1]-uM[1])*(uB[1]-uM[1]))

	#ellipticities
	e1 = (QxxM - QyyM)/(QxxM + QyyM)
	e2 = 2*QxyM/(QxxM+QxyM)
	return e1, e2

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#Parameters for Mix Plot
domFlux = 2.e6
domHLR = 1  #arcsec
unFlux = 2.e6
unHLR = domHLR
pixelScale = .2 #arcsec/pixel
ff = .5 #Fractional flux of dominant galaxy, so ff(undetected) = 1-ff(dominant)
du = 10. * pixelScale #arcsec
phi = 0 #angle between centroid of contam and real x-axis
dx = du * np.cos(phi)
dy = du * np.sin(phi)

#Parameters for ellipticity studies
numSteps = 50
sep = 15 * pixelScale #arcsec
seed = galsim.BaseDeviate(1)

#Ellipticity studies

#Run from separation along x-axis, with fractional flux defined by ff
shift = np.linspace(-sep, sep, numSteps)
e1, e1Err, e2, e2Err, ane1, ane2 = fitSep(domFlux, domHLR, unFlux, unHLR, ff, shift, dy, seed, pixelScale)

#Run through proportion space, wth separation defined by du
f = np.linspace(.5, .99, numSteps)
e1p, e1Perr, e2p, e2Perr = fitProp(domFlux, domHLR, unFlux, unHLR, f, dx, dy, seed, pixelScale)

#Run from separation along pi/4 axis
shif = [(elem * np.cos(np.pi/4), elem * np.sin(np.pi/4)) for elem in shift]
e1s, e1Serr, e2s, e2Serr = fitSepS(domFlux, domHLR, unFlux, unHLR, ff, shif, seed, pixelScale)


#Create mixture, using drawShoot
gals, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, ff, dx, dy, seed, pixelScale)

params = lm.Parameters()
params.add('fitFlux', value=5.e5)
params.add('fitHLR', value=2)
params.add('fitCentX', value=0)
params.add('fitCentY', value=0)
params.add('fite1', value=0., min=-1, max=1)
params.add('fite2', value=0., min=-1, max=1)

#Fit a single Gaussian
out = lm.minimize(leastSqResid, params, args=[mixIm])

fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
fitIm = galsim.ImageD(64, 64, scale=.2)
fitIm = fit.draw(image=fitIm)

#Plot
fig = pl.figure()

ax1 = fig.add_subplot(231)
c1 = ax1.imshow(mixIm.array, origin='lower')
ax1.set_title('Mixture using .drawShoot')
pl.colorbar(c1)

ax2 = fig.add_subplot(232)
c2 = ax2.imshow(fitIm.array, origin='lower')
ax2.set_title('Fit to the Mixture')
pl.colorbar(c2)

bias = [ei - ai for ei, ai in zip(e1, ane1)]
ax3 = fig.add_subplot(234)
ax3.errorbar(shift, e1, yerr=e1Err, label='Fit e1')
#ax3.errorbar(shift, e2, yerr=e2Err, label='Fit e2')
ax3.plot(shift, ane1, label='Analytic e1')
#ax3.plot(shift, ane2, label='Analytic e2')
ax3.plot(shift, bias, label='Bias')
ax3.set_xlabel('Separation, x-axis (arcsec)')
ax3.set_title('Shear vs. Object Separation (50/50)')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
ax3.set_xlim(min(shift), max(shift))
ax3.legend(prop={'size':9}, loc=9)

ax5 = fig.add_subplot(235)
ax5.errorbar(f, e1p, yerr=e1Perr, label='Fit e1')
ax5.errorbar(f, e2p, yerr=e2Perr, label='Fit e2')
ax5.set_xlabel('dominant proportion')
ax5.set_title('Shear vs. Proportion (sep=2)')
ax5.legend()

ax6 = fig.add_subplot(236)
ax6.errorbar(shift, e1s, yerr=e1Serr, label='Fit e1')
ax6.errorbar(shift, e2s, yerr=e2Serr, label='Fit e2')
ax6.set_xlabel('Separation, pi/4 (arcsec)')
ax6.set_title('Shear vs. Separation (50/50)')
ax6.axhline(y=0, color='k')
ax6.axvline(x=0, color='k')
ax6.set_xlim(min(shift), max(shift))
ax6.legend(prop={'size':9}, loc=9)
ax6.legend()

pl.show()
