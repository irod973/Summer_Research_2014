This directory simulates and studies undetected galaxy blends as mixtures of 2 2D Gaussian profiles. Shapes of the 'undetected blend' are studied by using a single Gaussian profile to fit to the mixture and extracting the parameters of the best fit. 

gaussianMixtureLib.py - This contains all functions used in the directory. 

drawGaussianMixture.py - Draws the 2D mixture with specified parameters, along with the best fit and the residual between the fit and the mixture. Can also study SNR of mixture iwth sky, if desired.

gaussianMixtureShapeStudiesSep.py - Studies the ellipticity of the best fit as separation between the two galaxies varies.

gaussianMixtureShapeStudiesFrac.py - Studies the ellipticity of the best fit as the flux fraction of the dominant galaxy varies.

gaussianMixtureShapeStudiesSNR.py - Studies the ellipticity of the best fit as the Signal-To-Noise Ratio of the whole mixture varies.

gaussianMixtureShapeStudies.py - This file contains the backbone of the 3 ShapeStudies files. It is older than the 3 files, however, and all of its functionality is in those files, so there is no need to use it.