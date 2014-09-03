This directory contains files that look at 1D and 2D mixtures of two Gaussians and the best fit to each mixture using a single Gaussian. These files are pure python and do not use Galsim.

IAR-gauss1Dfit.py - Plots two 1D mixtures and their fit: a mixture of two analytic Gaussians and another of two Gaussian histograms.

IAR-gauss1Dmix-VarStudy - Creates a binned mixture and studies the second central moment of a single Gaussian fit to that mixture. Looks at variance (equivalent to scm) vs. separation and variance vs. flux fraction.

The following files are obsolete and were not used for results during this research. They also may not work:
IAR-gauss1D.py - Draws a 1D mixture.  
IAR-gauss2D.py - Draws a 2D mixture, both binned and analytic.
IAR-gaussFit2D.py - Draws a 2D mixture and tries to fit to it using kernel density estimators and other methods