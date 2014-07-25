import galsim

galFlux = 1e5
galR0 = 2.5
g1 = .1
g2 = .2
psfBeta = 5
psfRe = 1.0
pixelScale = .2 #arcsec/pixel
skyLevel = 2.5e3 #counts/arcsec^2

#Create profile
gal = galsim.Exponential(flux=galFlux, scale_radius=galR0)

#Shear profile
gal = gal.shear(g1=g1, g2=g2)

#Make psf profile
psf = galsim.Moffat(beta=psfBeta, flux=1, half_light_radius=psfRe)

#Convolve
result = galsim.Convolve([gal, psf])
image = result.drawImage(scale=pixelScale)

#Add Poisson noise. PoissonNoise class needs skylevel in each pixel. Note dimensional analysis on skylevel and pixelscale.
randomSeed = 0
rng = galsim.BaseDeviate(randomSeed)

skyLevelPixel = skyLevel * pixelScale**2
noise = galsim.PoissonNoise(rng, sky_level=skyLevelPixel)
image.addNoise(noise)

#Write to file
image.write('demo2fits.fits')

#Word.
