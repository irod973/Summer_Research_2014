import galsim
import pyfits as pf
import numpy as np

#Note sigmas are in arcsecs

gal_flux = 1.e5
gal_sigma = 3
psf_sigma = 1 
pixelscale = .2 #arsec/pixel

#Make my galaxy
gal = galsim.Gaussian(flux = gal_flux, sigma = gal_sigma)

#Make my psf

psf = galsim.Gaussian(flux = 1.0, sigma = psf_sigma)

#Make my galaxy profile

galProf = galsim.Convolve([gal, psf])

#Make an image

galImage = gal.drawImage(scale = pixelscale)
convolveImage = galProf.drawImage(scale = pixelscale)

print 'The added flux to the convolved image, presumably from the specified pixel scale, is: ' + str(convolveImage.added_flux)

galImage.addNoise(galsim.GaussianNoise(sigma=30)) #adding noise to our non-convolved galaxy
#Let's make a FITS file from our image

galImage.write('demo1gal.fits')
convolveImage.write('demo1convolvegal.fits')
#pf.writeto('IrvingsFITSFile', image) #Says image should be a nparray...

