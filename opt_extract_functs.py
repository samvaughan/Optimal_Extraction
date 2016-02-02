from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial.chebyshev import chebfit,chebval






def wavelegth_fitter(wavelength_fits, cube, var, aperture_mask, lamdas, order, aperture_indices, plot=False, verbose=False):

    """
    Fit polynomials in the wavelength direction of each pixel in an aperture. This makes the P(x,lamda) values used in Horne 1986
    Returns a cube of model values, called mcube

    Wavelength_fits is an cube of un-normalised values, initially all 0s
    cube is the datacube in question
    var is the variance cube
    aperture mask is a 3D mask of True and False. Good values=True, Bad ones =False
    lamdas is an array of lamda values along the wavelength axis, formed from CRPIX,CRVAL,CDELT in the fits header
    order is the order of fitting
    aperture indices are indices of the cube corresponding to our circular aperture, in an Nx2 array. This is looped over.
    plot=True shows a plot of the polynomials (both before and after normalising spatially)
    verbose=True prints more to the terminal

    """

    #Lamdas must be scaled to between -1 and 1 for Chebyshev fitting
    min_val=lamdas.min()
    max_val=lamdas.max()
    scaled_l=np.array([2*(lamda-min_val)/(max_val-min_val) -1 for lamda in lamdas])

    for n, (i, j) in enumerate(zip(aperture_indices[0], aperture_indices[1])):

        """Take each spaxel and fit a polynomial along the wavelength direction, each pixel weighted by one over its variance"""
        if verbose==True:
            print("Fitting Pixel {}".format(n))

        weights=1.0/var[:, i, j]

        #Fit the Chebyshev coefficients
        coefficients=chebfit(scaled_l, cube[:, i, j], order, w=weights)

        #Make a polynomial from these
        polynomial=chebval(scaled_l, coefficients)
        #Ensure all values are positve
        polynomial[polynomial<0]=0


        wavelength_fits[:, i, j]=polynomial

    if plot==True:
        for i, j in zip(aperture_indices[0], aperture_indices[1]):
            plt.plot(lamdas, wavelength_fits[:, :])

        plt.show()

    if verbose==True:
        print("Normalising Spatially")
    #Normalise spatially
    spatial_norm=np.sum(np.sum(wavelength_fits, axis=2), axis=1)
    mcube=wavelength_fits/spatial_norm[:, np.newaxis, np.newaxis]



    if plot==True:
        for i, j in zip(aperture_indices[0], aperture_indices[1]):

            plt.plot(lamdas, mcube[:,i, j])

        plt.show()

    return mcube





def extract(cube):

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    print("extracting standard spectrum\n\n")
    #extract usual spectrum
    spectrum=np.sum(np.sum(cube, axis=2), axis=1)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
    return spectrum






def sigma_clipping(cube, var, mcube, spec, aperture_mask, sigma, aperture_indices, verbose=False):

    """
    Make a cube of residuals, via residuals=(aperture_mask)*{cube-spec[:, np.newaxis, np.newaxis]*mcube}**2/var, ie see how far away each pixel is from the model spectrum at that pixel.
    If a pixel has a residual greater than sigma**2, flip the mask at that pixel so that it is later ignored from the wavelength fitting and optimal extraction


    Returns the 3D mask
    """


    print("masking outliers")

    #Forming the residuals
    arg=cube-spec[:, np.newaxis, np.newaxis]*mcube
    residuals=(aperture_mask)*np.power(arg, 2)/var

    #If verbose is true, show which pixels are masked
    if verbose==True:
        for i, (residual, cubeval, mcubeval, specval, varval) in enumerate(zip(residuals[:, 27, 47], cube[:, 27, 47], mcube[:, 27, 47], spec[:], var[:, 27, 47])) :
            if residual>sigma**2:
                print("At {}, 27, 47\n{}: cube is {}, model is {}, specval is {}, var is {}\n".format(i, residual, cubeval, mcubeval, specval, varval))



    #Find the outliers
    outliers=np.where(residuals>sigma**2)
    cntr=len(residuals[outliers])


    print("There are {} outliers".format(cntr))
    #Flip the mask
    aperture_mask[outliers]=~aperture_mask[outliers]



    return (aperture_mask, cntr)



def variance(mcube, var, spec, rdnoise, gain):
    """Updates the variance cube according to step 6 in Horne 1986"""
    print("Updating the Variance Cube")

    #print("Are there any nonfinite values in Var? {}".format(np.any(~np.isfinite(var))))
    #print("Are there any nonfinite values in MCube? {}".format(np.any(~np.isfinite(mcube))))
    variance=(spec[:, np.newaxis, np.newaxis]*mcube/gain)+rdnoise**2

    return variance
