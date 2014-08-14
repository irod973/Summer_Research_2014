import lmfit
import galsim
import numpy as np
import matplotlib.pyplot as plt

def drawShoot_galaxy(flux, hlr, e1, e2):
    gal = galsim.Sersic(half_light_radius=hlr, flux=flux, n=2.0)
    gal = gal.shear(e1=e1, e2=e2)
    image = galsim.ImageD(32, 32, scale=0.2)
    image = gal.drawShoot(image=image)
    return image

def draw_galaxy(flux, hlr, e1, e2):
    gal = galsim.Gaussian(half_light_radius=hlr, flux=flux)
    gal = gal.shear(e1=e1, e2=e2)
    image = galsim.ImageD(32, 32, scale=0.2)
    image = gal.draw(image=image)
    return image

def resid(param, target_image):
    flux = param['flux'].value
    hlr = param['hlr'].value
    e1 = param['e1'].value
    e2 = param['e2'].value
    gal = galsim.Gaussian(half_light_radius=hlr, flux=flux)
    gal = gal.shear(e1=e1, e2=e2)
    image = galsim.ImageD(32, 32, scale=0.2)
    image = gal.draw(image=image)
    return (image-target_image).array.ravel()
    
def resid2(param, target_image):
    flux_1 = param['flux_1'].value
    hlr_1 = param['hlr_1'].value
    e1_1 = param['e1_1'].value
    e2_1 = param['e2_1'].value

    flux_2 = param['flux_2'].value
    hlr_2 = param['hlr_2'].value
    e1_2 = param['e1_2'].value
    e2_2 = param['e2_2'].value

    gal_1 = galsim.Gaussian(half_light_radius=hlr_1, flux=flux_1)
    gal_1 = gal_1.shear(e1=e1_1, e2=e2_1)

    gal_2 = galsim.Gaussian(half_light_radius=hlr_2, flux=flux_2)
    gal_2 = gal_2.shear(e1=e1_2, e2=e2_2)

    gal = gal_1 + gal_2

    image = galsim.ImageD(32, 32, scale=0.2)
    image = gal.draw(image=image)
    return (image-target_image).array.ravel()
im = drawShoot_galaxy(40000, 1.0, 0.2, 0.3)

parameters = lmfit.Parameters()
parameters.add('flux', value=30000)
parameters.add('hlr', value=1.2, min=0.0)
parameters.add('e1', value=0.0, min=-1.0, max=1.0)
parameters.add('e2', value=0.0, min=-1.0, max=1.0)

result = lmfit.minimize(resid, parameters, args=[im])
best_fit = draw_galaxy(result.params['flux'].value,
                       result.params['hlr'].value,
                       result.params['e1'].value,
                       result.params['e2'].value)

lmfit.report_errors(result.params)

hsm = im.FindAdaptiveMom()
print hsm.observed_shape.e1
print hsm.observed_shape.e2

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(im.array)
ax2 = fig.add_subplot(132)
ax2.imshow(best_fit.array)
ax3 = fig.add_subplot(133)
blah = ax3.imshow((im-best_fit).array)
plt.colorbar(blah)
plt.show()
