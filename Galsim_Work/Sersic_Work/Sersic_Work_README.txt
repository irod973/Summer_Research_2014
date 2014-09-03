This directory simulates and studies undetected galaxy blends as mixtures of 2 2D Sersic profiles. Shapes of the 'undetected blend' are studied by using an identical profile (a single one) to fit to the mixture and extracting the parameters of the best fit. 

sersicMixtureLib.py - Contains all functions used in this directory.

drawSersicMixture.py - Draws a mixture of 2 galaxies and the best single-galaxy fit to the mixture, as well as the residual between the fit and the mixture. 
The galaxies can be specified by any Sersic index that Galsim will allow (0.3<n<6.5).

sersicMixtureShapeStudiesSep.py - Studies the ellipticity of the best fit as separation between the two galaxies varies.

sersicMixtureShapeStudiesFraction.py - Studies the ellipticity of the best fit as the flux fraction of the dominant galaxy varies.

sersicMixtureShapeStudies.py - This file contains the backbone of the 2 ShapeStudies files. It is older than the 2 files, however, and all of its functionality is in those files, so there is no need to use it.