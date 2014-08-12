import galsim
import lmfit as lm
import numpy as np
import matplotlib.pyplot as pl

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
def drawMixture(domParams, contParams, pixelScale=.2, stampSize=100, psf=False):
	domGal = galsim.Gaussian(flux=domParams['flux'].value*domParams['frac'].value, half_light_radius=domParams['HLR'].value)
	contGal = galsim.Gaussian(flux=contParams['flux'].value*contParams['frac'].value, half_light_radius=contParams['HLR'].value)
	contGal = contGal.shift(dx=contParams['centX'].value, dy=contParams['centY'].value)
	mix =  domGal + contGal
	if psf==True:
		psf = galsim.Moffat(beta=3, fwhm=.6)
		mix = galsim.Convolve([gauss, psf])
		image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
		image = mix.drawImage(image=image, method='fft')
		gals = [domGal, contGal, mix]
		return  gals, image
	image = galsim.ImageD(stampSize, stampSize, scale=pixelScale)
	image = mix.drawImage(image=image, method='fft')
	gals = [domGal, contGal.original, mix]
	return  gals, image

#Compute the least-squares residual between the parameter image and the mixture image 
def leastSqResid(params, mixIm, pixelScale=.2, stampSize=100):
	image = drawFit(params, pixelScale, stampSize)
	return (image - mixIm).array.ravel()**2

def mixtureMoments(gals, galParams):
	fractions = [params['frac'].value for params in galParams]
	zeroMoments = [params['flux'].value for params in galParams]
	firstMoments = [(params['centX'].value, params['centY'].value) for params in galParams]
	secondMoments = [[[gal.getSigma()**2, 0], [0, gal.getSigma()**2]] for gal in gals]
	return fractions, zeroMoments, firstMoments, secondMoments

def varySeparation(domParams, contParams, shift, pixelScale=.2, stampSize=100):
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

		contParams['centX'].value = dx
		gals, mixIm = drawMixture(domParams, contParams, pixelScale, stampSize)
		#Find minimum of least-squares between mixture and fit
		out = lm.minimize(leastSqResid, params, args=[mixIm])
		
		#Get ellipticity from fit parameters
		e1.append(out.params['fite1'].value)
		e2.append(out.params['fite2'].value)

		#Calculate moments of mixture components
		galParams = [domParams, contParams]
		gal = [gals[0], gals[1]]		
		fractions, zeroMom, firstMom, secMom = mixtureMoments(gal, galParams)

		#Get analytic ellipticities
		ae1, ae2 = analyticMixShear(fractions, zeroMom, firstMom, secMom)
		ane1.append(ae1)
		ane2.append(ae2)

	return e1, e1Err, e2, e2Err, ane1, ane2

def varyFracFlux(domParams, contParams, fracRange, pixelScale=.2, stampSize=100):
	params = lm.Parameters()
	params.add('fitFlux', value=5.e5)
	params.add('fitHLR', value=2)
	params.add('fitCentX', value=0)
	params.add('fitCentY', value=0)
	params.add('fite1', value=0., min=-1, max=1)
	params.add('fite2', value=0., min=-1, max=1)

	dParams = domParams
	cParams = contParams

	e1p = []
	e2p = []
	e1Perr = []
	e2Perr = []
	annie = []
	annie2 = []

	for frac in fracRange:
		
		dParams['frac'].value = frac
		cParams['frac'].value = 1-frac
		gals, mixIm = drawMixture(domParams, contParams, pixelScale, stampSize)

		#Find minimum of least-squares between mixture and fit
		out = lm.minimize(leastSqResid, params, args=[mixIm])

		#Get ellipticity from fit parameters
		e1p.append(out.params['fite1'].value)
		e2p.append(out.params['fite2'].value)

		#Calculate moments of mixture components
		galParams = [domParams, contParams]
		gal = [gals[0], gals[1]]		
		fractions, zeroMom, firstMom, secMom = mixtureMoments(gal, galParams)

		#Get analytic ellipticities
		ae1p, ae2p = analyticMixShear(fractions, zeroMom, firstMom, secMom)
		annie.append(ae1p)
		annie2.append(ae2p)

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
		gals, mixIm = drawMixture(domFlux, domHLR, unFlux, unHLR, ff, du[0], du[1], rng, pixelScale)
		out = lm.minimize(leastSqResid, params, args=[mixIm, pixelScale])
		fit = galsim.Gaussian(flux=out.params['fitFlux'].value, half_light_radius=out.params['fitHLR'].value)
		fit = fit.shear(e1=out.params['fite1'].value, e2=out.params['fite2'].value)
		fit = fit.shift(dx=out.params['fitCentX'].value, dy=out.params['fitCentY'].value)
		fitIm = galsim.ImageD(100, 100, scale=pixelScale)
		fitIm = fit.draw(image=fitIm)        
		fitMoments = fitIm.FindAdaptiveMom()
		e1s.append(fitMoments.observed_shape.e1)
		e2s.append(fitMoments.observed_shape.e2)
		e1Serr.append(np.sqrt(np.diag(out.covar))[2])
		e2Serr.append(np.sqrt(np.diag(out.covar))[3])
	return e1s, e1Serr, e2s, e2Serr

def analyticMixShear(fluxFractions, fluxes, centroids, quads):
	#Calculate mixture moments from components, use moments to return ellipticity

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

	#ellipticities
	e1 = (Q[0][0] - Q[1][1])/(Q[0][0] + Q[1][1])
	e2 = 2*Q[0][1]/(Q[0][0]+Q[1][1])
	return e1, e2

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#Parameters for Mixture
domFlux = 2.e6
domHLR = 1  #arcsec
contFlux = 2.e6
contHLR = domHLR
domFrac = .5 #Fractional flux of dominant galaxy, so contFrac = 1-domFrac
pixelScale = .2 #arcsec/pixel
stampSize = 100 #pixels

du = 10 * pixelScale #arcsec
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


#Ellipticity studies
#Parameters for ellipticity studies
numSteps = 100
sep = 10 * pixelScale #arcsec

#Vary separation along x-axis in real
shift = np.linspace(-sep, sep, numSteps)
e1, e1Err, e2, e2Err, ane1, ane2 = varySeparation(domParams, contParams, shift)

#Vary dominant flux
contParams['centX'].value = dx
fracRange = np.linspace(.5, .99, numSteps)
e1p, e1Perr, e2p, e2Perr, annie, annie2 = varyFracFlux(domParams, contParams,fracRange)

#Run from separation along pi/4 axis
shif = [(elem * np.cos(np.pi/4), elem * np.sin(np.pi/4)) for elem in shift]
# e1s, e1Serr, e2s, e2Serr = fitSepS(domParams, contParams, rng, pixelScale, shif)

#Draw a certain mixture, its fit, and residuals
#Create mixture, using drawShoot
domParams['frac'].value = domFrac
contParams['frac'].value = 1-domFrac
gals, mixIm = drawMixture(domParams, contParams)

params = lm.Parameters()
params.add('fitFlux', value=5.e5)
params.add('fitHLR', value=2)
params.add('fitCentX', value=0)
params.add('fitCentY', value=0)
params.add('fite1', value=0., min=-1, max=1)
params.add('fite2', value=0., min=-1, max=1)

#Fit a single Gaussian
out = lm.minimize(leastSqResid, params, args=[mixIm])
fitIm = drawFit(params)
lm.report_errors(params)

#Plots
fig1 = pl.figure()

domStr = 'Dominant Galaxy: ' + str(gals[0].centroid()) + ', '+str(domFlux)+', '+str(domHLR)+', '+str(domFrac)+', 0, 0'
contStr = 'Contaminant Galaxy: (' + str(contParams['centX'].value) + ', ' + str(contParams['centY'].value) + '), '+str(contFlux)+', '+str(contHLR)+', '+str((1-domFrac))+', 0, 0'
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


#Plotting varying parameters
fig2=pl.figure(2)

anFitDiff = [ei - ai for ei, ai in zip(e1, ane1)]
ax21 = fig2.add_subplot(121)
ax21.plot(shift, e1, label='Fit e1')
ax21.plot(shift, ane1, label='Analytic e1')
ax21.plot(shift, anFitDiff, label='Difference')
ax21.plot(shift, ane2, label='Analytic e2')
ax21.plot(shift, e2, label='Fit e2')
ax21.set_xlabel('Separation, x-axis (arcsec)')
ax21.set_ylabel('e1')
ax21.set_title('Shear vs. Object Separation (Flux Fraction = .5)')
ax21.axhline(y=0, color='k')
ax21.axvline(x=0, color='k')
ax21.set_xlim(min(shift), max(shift))
ax21.legend(prop={'size':11}, loc=9)

#Vary fractional flux
ax22 = fig2.add_subplot(122)
ax22.plot(fracRange, e1p, label='Fit e1')
ax22.plot(fracRange, e2p,label='Fit e2')
ax22.plot(fracRange, annie, label='Analytic e1')
ax22.plot(fracRange, annie2, label='Analytic e2')
ax22.set_xlabel('Dominant Galaxy Flux Fraction')
ax22.set_ylabel('e1')
ax22.set_title('Shear vs. Flux Fraction (Sep='+ str(dx) + ' arcsec='+str(domHLR)+ ' HLR)')
ax22.legend()

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