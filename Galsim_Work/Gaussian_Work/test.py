import galsim

def drawMixture():
	domGal = galsim.Gaussian(flux=100000, half_light_radius=1)
	psf = galsim.Moffat(beta=3, fwhm=.6)
	convolve = galsim.Convolve([domGal, psf])
	print convolve

drawMixture()