#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os.path
import argparse


from opt_extract_functs import *


parser = argparse.ArgumentParser(description='Optimally extract a spectrum from two SWIFT datacubes: a Science cube and a Variance cube')

parser.add_argument('objcube', type=str, help='The filename of the Object Cube')

parser.add_argument('varcube', type=str, help='The filename of the Variance Cube')


parser.add_argument('r', type=int, help="Radius of Aperture to extract")

parser.add_argument('outfile', help='The output filename for the spectrum')

parser.add_argument('--v', action='store_true', help='Verbose')

parser.add_argument('--show', action='store_true', help='Show final plot before saving')

parser.add_argument('--x', nargs='?', const=46, default=46, help='X coordinate of centre of aperture. Default is 48', type=int)
parser.add_argument('--y', nargs='?', const=27, default=27, help='Y coordinate of centre of aperture. Default is 28', type=int)
parser.add_argument('--w', nargs='?', type=str, help='A cube of weights for each pixel, to be used for the optimal extraction, saved as a .npy file')


args = parser.parse_args()

objcube=args.objcube
variancecube=args.varcube
r=args.r

outfile=args.outfile
var_outfile="{}_var.fits".format(outfile[:-5])

weights=args.w

verbose=False
if args.v:
    verbose=True

show_final=False
if args.show:
    show_final=True



#Check if output files exist:
if os.path.isfile(outfile):
    raise IOError("File {} already exists!".format(outfile))
elif os.path.isfile(var_outfile):
    raise IOError("File {} already exists!".format(outfile))


#For Garret's data, a=65, b=25
if args.x:
    a=args.x
if args.y:
    b=args.y







"""Making the masked data cube and variance cubes"""
#opening all the data
o=fits.open(objcube)
cube=o[0].data




v=fits.open(variancecube)
var=v[0].data



#bgnd=fits.open(bckgrndcube)
#background=bgnd[0].data

d1,d2,d3=np.shape(cube)

print("\n\nShape of cube is {}, {}, {}".format(d1, d2, d3))
print("Extracting an Aperture at {}, {}, radius {}\n\n".format(a, b, r))






#Making the circular Aperture
y,x = np.ogrid[-b:d2-b, -a:d3-a]
mask = x*x + y*y <= r*r

#Making a 3d masked cube, rather than just a masked slice
#True=Good, False=Bad
aperture_mask = np.repeat(mask[np.newaxis, :, :], d1, axis=0)
cube[~aperture_mask]=0
var[~aperture_mask]=np.inf

if verbose==True:
    plt.imshow(cube[2000, :, :])
    plt.colorbar()
    plt.title("Wavelength Slice 2000 of cube")
    plt.show()



#Making the model cube, which is blank at first
blank_cube=np.zeros([d1, d2, d3])


#FIXME Check these values or add options?
gain=0.98
rdnoise=4.5
sigma=10
l_order=200

#Wavelength array creation
cdelt=o[0].header["CDELT3"]
crval=o[0].header["CRVAL3"]
crpix=o[0].header["CRPIX3"]

print("CDELT is {}, CRVAL is {}, CRPIX is {}".format(cdelt, crval, crpix))
lamdas=np.array([(l-crpix)*cdelt + crval for l in range(d1)])

#Get the indices of the aperture spaxels at each wavelength
aperture_indices = np.where(aperture_mask[2000,:,:] == True)

if verbose:
    print(aperture_indices)


N=len(aperture_indices[0])


sum_spec=np.sum(np.sum(aperture_mask*cube, axis=2), axis=1)
#plt.plot(lamdas, sum_spec)
#plt.show()
spec=extract(cube)


var=var+rdnoise**2

cube_bad=np.where(~np.isfinite(cube))
cube[cube_bad]=0
cube[cube<0]=0

varbad=np.where(~np.isfinite(var))
var[varbad]=np.inf

cntr=1.0
c=1
#Use the cube of pixel weights, if given
if weights:
    mcube=np.load(weights)


while cntr !=0:
#for c in range(5):
    print("Wavelength Iteration {}".format(c))

    if not weights:
        print("\n\nCONSTRUCTING SPATIAL PROFILE\n")
        mcube=wavelegth_fitter(blank_cube, cube, var, aperture_mask, lamdas, l_order, aperture_indices, plot=False, verbose=verbose)
    else:
        print("\n\nUSING WEIGHTS FROM FILE {}".format(weights))

    print("\n\nREVISING VARIANCE ESTIMATES\n")

    var=variance(mcube, var, spec, rdnoise, gain)
    varbad=np.where(~np.isfinite(var))
    var[varbad]=np.inf
    print("\n\nMASKING COSMICS/BAD PIXELS\n")
    aperture_mask, cntr=sigma_clipping(cube, var, mcube, spec, aperture_mask, sigma, aperture_indices, verbose=verbose)
    var[~aperture_mask]=np.inf
    cube[~aperture_mask]=0


    print("\n\nEXTRACTING OPTIMAL SPECTRUM\n")
    spec=np.sum(np.sum(aperture_mask*cube*mcube/var, axis=2), axis=1)/np.sum(np.sum(aperture_mask*mcube*mcube/var, axis=2), axis=1)
    vspec=np.sum(np.sum(aperture_mask*mcube, axis=2), axis=1)/np.sum(np.sum(aperture_mask*mcube*mcube/var , axis=2), axis=1)
    c+=1






#Save the wavlenegth fits in case we need to extract a sky.
if not weights:
    mcube_outname="{}_weights.npy".format(outfile[:-5])
    np.save(mcube_outname, mcube)


if show_final==True:
    f, axarr=plt.subplots(3, sharex=True)
    axarr[0].set_title("Optimal Extract (Blue) and Normal (red)")
    axarr[0].plot(lamdas, spec, c="b")
    axarr[0].plot(lamdas, sum_spec, c="r")
    axarr[1].plot(lamdas, spec/sum_spec)
    axarr[1].set_title("Optimal Extraction / Normal Extraction")

    axarr[2].plot(lamdas, spec/np.power(vspec, 0.5))
    axarr[2].set_title("Signal to Noise Ratio")

    plt.figure()
    plt.plot(lamdas, spec, c="b")

    plt.figure()
    plt.plot(lamdas, sum_spec, c="r")

    plt.show()



"""
Write the spectrum and variance to a fits file. Update the header with useful values from the original cube
"""

header_template = o[0].header
hdu = fits.PrimaryHDU(spec, header_template)



hdu.header['CDELT1']=cdelt
hdu.header['CD1_1']=cdelt
hdu.header['CD1_2']=cdelt
hdu.header["CRVAL1"]=crval
hdu.header["CRPIX1"]=crpix


hdu.writeto(outfile)

hdu2=fits.PrimaryHDU(vspec, header_template)

hdu2.header['CDELT1']=cdelt
hdu2.header['CD1_1']=cdelt
hdu2.header['CD1_2']=cdelt
hdu2.header["CRVAL1"]=crval
hdu2.header["CRPIX1"]=crpix

hdu2.writeto(var_outfile)
