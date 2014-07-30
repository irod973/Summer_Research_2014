import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

#Irving Rodriguez
#
#This program studies the ellipticity of fitting a single Gaussian to a mixture of two Gaussians. It's tight

#-------------------------------------------------------------------------
#Functions--------------------------------------------------------------------
#-------------------------------------------------------------------------


def gaussMix(domFlux, domSig, unFlux, unSig, ff, du, scale):
	domGal = galsim.Gaussian(flux=domFlux, sigma=domSig)
	unGal = galsim.Gaussian(flux=unFlux, sigma=unSig)
	unGal = unGal.shift(dx=du, dy=0)
	mix = ff * domGal + (1-ff) * unGal
	image = galsim.ImageD(scale=scale)
	image = mix.draw(image=image)
	return image

#Draw mixture using photon shoot method
def gaussMixShoot(domParams, contParams, rng, pixelScale):
	domGal = galsim.Gaussian(flux=domParams['flux'].value*domParams['frac'].value, half_light_radius=domParams['HLR'].value)
	contGal = galsim.Gaussian(flux=contParams['flux'].value*contParams['frac'].value, half_light_radius=contParams['HLR'].value)
	contGal = contGal.shift(dx=contParams['centX'].value, dy=contParams['centY'].value)
	mix =  domGal + contGal
	# psf = galsim.Moffat(beta=3, fwhm=.6)
	# mix = galsim.Convolve([gauss, psf])
	image = galsim.ImageD(1000, 1000, scale=pixelScale)
	image = mix.drawImage(image=image, rng = rng, method='phot')
	gals = [domGal, contGal, mix]
	return  gals, image

#Compute the least-squares residual between the parameter image and the mixture image 
def leastSqResid(param, mixIm, pixelScale):
	flux = param['fitFlux'].value
	hlr = param['fitHLR'].value
	centX = param['fitCentX'].value
	centY = param['fitCentY'].value
	e1 = param['fite1'].value
	e2 = param['fite2'].value
	fit = galsim.Gaussian(flux=flux, half_light_radius=hlr)
	fit = fit.shear(e1=e1, e2=e2)
	fit = fit.shift(dx=centX, dy=centY)
	# psf = galsim.Moffat(beta=3, fwhm=.6)
	# fit = galsim.Convolve([fit, psf])
	image = galsim.ImageD(1000, 1000, scale=pixelScale)
	image = fit.draw(image=image)
	return (image - mixIm).array.ravel()

def fitProfile(result, scale):
	fit = galsim.Gaussian(flux=result.params['fitFlux'].value, half_light_radius=result.params['fitHLR'].value)
	fit = fit.shear(e1=result.params['fite1'].value, e2=result.params['fite2'].value)
	fit = fit.shift(dx=result.params['fitCentX'].value, dy=result.params['fitCentY'].value)
	fitIm = galsim.ImageD(1000, 1000, scale=scale)
	fitIm = fit.draw(image=fitIm)
	return fit, fitIm

def fitSep(dParams, cParams, rng, pixelScale, shift):
	params = lm.Parameters()
	params.add('fitFlux', value=5.e6)
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
		# FWHM=.6
		cParams['centX'].value = dx
		print cParams['centX'].value
		gals, mixIm = gaussMixShoot(dParams, cParams, rng, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm, pixelScale])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		# psf = galsim.Moffat(beta=3, fwhm=.6)
		# cfit = galsim.Convolve([fit, psf])
		fitIm = galsim.ImageD(1000, 1000, scale=pixelScale)
		fitIm = fit.draw(image=fitIm)
		fitMoments = fitIm.FindAdaptiveMom()
		e1.append(fitMoments.observed_shape.e1)
		e2.append(fitMoments.observed_shape.e2)
		e1Err.append(np.sqrt(np.diag(out.covar))[2])
		e2Err.append(np.sqrt(np.diag(out.covar))[3])

		#Get analytic ellipticities
		ae1, ae2 = analyticMixShear(gals, dParams['frac'].value, dx)
		ane1.append(ae1)
		ane2.append(ae2)
	return e1, e1Err, e2, e2Err, ane1, ane2

def fitProp(domFlux, domHLR, unFlux, unHLR, f, dx, dy, rng, pixelScale):
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
	annie = []
	annie2 = []
	for frac in f:
		galsP, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, frac, dx, dy, rng, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm, pixelScale])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(1000, 1000, scale=pixelScale)
		fitIm = fit.draw(image=fitIm)
		fitMoments = fitIm.FindAdaptiveMom()
		e1p.append(fitMoments.observed_shape.e1)
		e2p.append(fitMoments.observed_shape.e2)
		e1Perr.append(np.sqrt(np.diag(out.covar))[2])
		e2Perr.append(np.sqrt(np.diag(out.covar))[3])

		#Get analytic ellipticities
		ann1, ann2= analyticMixShear(galsP, frac, dx)
		annie.append(ann1)
		annie2.append(ann2)
	return e1p, e1Perr, e2p, e2Perr, annie, annie2

def fitSepS(domFlux, domHLR, unFlux, unHLR, ff, shift, rng, pixelScale):
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
		gals, mixIm = gaussMixShoot(domFlux, domHLR, unFlux, unHLR, ff, du[0], du[1], rng, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm, pixelScale])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(1000, 1000, scale=pixelScale)
		fitIm = fit.draw(image=fitIm)        
		fitMoments = fitIm.FindAdaptiveMom()
		e1s.append(fitMoments.observed_shape.e1)
		e2s.append(fitMoments.observed_shape.e2)
		e1Serr.append(np.sqrt(np.diag(out.covar))[2])
		e2Serr.append(np.sqrt(np.diag(out.covar))[3])
	return e1s, e1Serr, e2s, e2Serr

def analyticMixShear(gals, ff, dx):
	domGal = gals[0]
	contGal = gals[1]

	#0th Moments of Components
	fA = domGal.getFlux()
	fB = contGal.getFlux()

	#1st Moments of Components
	uA = (0,0)
	uB = (dx, 0)

	#2nd Moments of Components, Qxy = 0 for circular Gaussian
	domSig = domGal.getSigma()
	QxxA = domSig**2
	QyyA = QxxA
	QxyA = 0

	contSig = domSig
	QxxB = contSig**2
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
domHLR = 2  #arcsec
contFlux = 2.e6
contHLR = domHLR
pixelScale = .02 #arcsec/pixel
domFrac = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
du = 100. * pixelScale #arcsec
phi = 0 #angle between centroid of contam and real x-axis
dx = du * np.cos(phi)
dy = du * np.sin(phi)

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

#Parameters for ellipticity studies
numSteps = 50
sep = 100 * pixelScale #arcsec
rng = galsim.BaseDeviate(1)

#Ellipticity studies

#Run from separation along x-axis, with fractional flux defined by ff
shift = np.linspace(-sep, sep, numSteps)
e1, e1Err, e2, e2Err, ane1, ane2 = fitSep(domParams, contParams, rng, pixelScale, shift)

# #Run through proportion space, wth separation defined by du
f = np.linspace(.5, .99, numSteps)
#e1p, e1Perr, e2p, e2Perr, annie, annie2 = fitProp(domParams, contParams, rng, pixelScale, f)

#Run from separation along pi/4 axis
shif = [(elem * np.cos(np.pi/4), elem * np.sin(np.pi/4)) for elem in shift]
# e1s, e1Serr, e2s, e2Serr = fitSepS(domParams, contParams, rng, pixelScale, shif)

#Create mixture, using drawShoot
gals, mixIm = gaussMixShoot(domParams, contParams, rng, pixelScale)

params = lm.Parameters()
params.add('fitFlux', value=5.e5)
params.add('fitHLR', value=2)
params.add('fitCentX', value=0)
params.add('fitCentY', value=0)
params.add('fite1', value=0., min=-1, max=1)
params.add('fite2', value=0., min=-1, max=1)

#Fit a single Gaussian
out = lm.minimize(leastSqResid, params, args=[mixIm, pixelScale])

fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
# psf = galsim.Moffat(beta=3, fwhm=.6)
# cfit = galsim.Convolve([fit, psf])
fitIm = galsim.ImageD(1000, 1000, scale=pixelScale)
fitIm = fit.draw(image=fitIm)

#Plots

fig1 = pl.figure()

domStr = 'Dominant Galaxy: ' + str(gals[0].centroid()) + ', '+str(domFlux)+', '+str(domHLR)+', '+str(domFrac)+', 0, 0'
contStr = 'Contaminant Galaxy: ' + str(gals[1].centroid()) + ', '+str(contFlux)+', '+str(contHLR)+', '+str((1-domFrac))+', 0, 0'
titleStr = 'Parameters (centroid, flux, hlr, fraction of flux, e1, e2)\n'+domStr+ '\n'+contStr+'\nPixel Scale: '+ str(pixelScale)+' arcsec/pixel'
fig1.suptitle(titleStr, fontsize=12)

#Plotting the mixture
ax11 = fig1.add_subplot(131)
c1 = ax11.imshow(mixIm.array, origin='lower')
ax11.set_title('Mixture')
pl.colorbar(c1, shrink=.5)

#Plotting the fit
ax12 = fig1.add_subplot(132)
c2 = ax12.imshow(fitIm.array, origin='lower')
ax12.set_title('Fit')
pl.colorbar(c2, shrink=.5)

#Plotting the residual
ax13 = fig1.add_subplot(133)
c3 = ax13.imshow((fitIm-mixIm).array, origin='lower')
ax13.set_title('Residual')
pl.colorbar(c3, shrink=.5)



fig2=pl.figure(2)

#Plotting varying separation
# anFitDiff = [ei - ai for ei, ai in zip(e1, ane1)]
# ax21 = fig2.add_subplot(121)
# ax21.plot(shift, e1, label='Fit e1')
# ax21.plot(shift, ane1, label='Analytic e1')
# ax21.plot(shift, anFitDiff, label='Difference')
# ax21.plot(shift, ane2, label='Analytic e2')
# ax21.plot(shift, e2, label='Fit e2')
# ax21.set_xlabel('Separation, x-axis (arcsec)')
# ax21.set_ylabel('e1')
# ax21.set_title('Shear vs. Object Separation (Flux Fraction = .5)')
# ax21.axhline(y=0, color='k')
# ax21.axvline(x=0, color='k')
# ax21.set_xlim(min(shift), max(shift))
# ax21.legend(prop={'size':11}, loc=9)

#Plotting varying fractional flux
# ax22 = fig2.add_subplot(122)
# ax22.plot(f, e1p, label='Fit e1')
# ax22.plot(f, e2p, label='Fit e2')
# # ax5.plot(f, annie)
# # ax5.plot(f, annie2)
# ax22.set_xlabel('Dominant Galaxy Flux Fraction')
# ax22.set_ylabel('e1')
# ax22.set_title('Shear vs. Flux Fraction (Sep=2 arcsec=2 HLR)')
# ax22.legend()

#Plotting varying separation along the pi/4 axis
# ax6 = fig2.add_subplot(236)
# ax6.errorbar(shift, e1s, yerr=e1Serr, label='Fit e1')
# ax6.errorbar(shift, e2s, yerr=e2Serr, label='Fit e2')
# ax6.set_xlabel('Separation, pi/4 (arcsec)')
# ax6.set_title('Shear vs. Separation (50/50)')
# ax6.axhline(y=0, color='k')
# ax6.axvline(x=0, color='k')
# ax6.set_xlim(min(shift), max(shift))
# ax6.legend(prop={'size':9}, loc=9)
# ax6.legend()

pl.show()