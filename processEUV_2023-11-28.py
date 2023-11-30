#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUV Spectroscopy Processing 

@author: Aidan Klemmer & Derek Sutherland & Stephan Fuelling
"""
#import sys
import os
#from time import sleep
#import argparse
from astropy.io import fits 
#from astropy.modeling import models 
import astropy.units as u
from astroquery.nist import Nist 
from specutils import Spectrum1D #, SpectralRegion
#from specutils.manipulation import noise_region_uncertainty
#from specutils.fitting import find_lines_threshold, find_lines_derivative
from specutils.fitting import find_lines_derivative
from PIL import Image, ImageOps
import numpy as np;
import matplotlib.pyplot as plt;
from matplotlib import colors
#from scipy.optimize import curve_fit
#from scipy.signal import find_peaks, peak_widths, medfilt, fftconvolve
from scipy.signal import fftconvolve
from scipy import ndimage
#from scipy.interpolate import interp1d
from skimage.measure import profile_line
import warnings

#import stistools

# File path *.fit file for a particular shot number shotnum 
path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_11_14_22\\'
# File path *.fit file for a particular dark frame
darkframe_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_11_14_22\\'
# Dark frame file name
dark_pic_name = r'221114008.fit'

shotinfo_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\'

shotinfo_name = r'EUV_refdata_11_22.txt'

save_results_path = 'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_CCD\\'

'''
Potential future feature: Add argument parser to accept absolute paths, shot numbers, etc... from user rather than being hard coded in the code. 

parser = argparse.ArgumentParser(description='Processing EUV data.')
'''
# Get Wavelength Functions 
def getWavelength(position):
    '''
    function getWavelength: The function getWavelength provides the wavelengths including both MCP offset and tilt. But, note it currently does not include a grating offset (\Delta z) or tilt (\delta) which could be added later, if needed. 

    '''
    position = position/1000 # Position of CCD along Rowland circle [in]
    Ro=0.4994 # Rowland circle radius [m] (~ 1 m diameter)
    alpha=1.495704 # Angle of incidence of light [radians]
    delta=0.0 # Correction for grating tilt [radians]
    Dz=0.0  # Correction for grating offset [m]
    x00=np.array([0.006297119,0.006100908,-0.003381397,0.000561561,-0.003819074,0.006,-0.003819074,-0.003819074,-0.003819074,-0.003819074]) # [inches]
    gam=np.array([-0.030718329,-0.036544725,-0.036209325,-0.042160846,-0.045871025,-0.004877428,-0.004877428,-0.004877428,-0.030718329,-0.030718329]) # MCP tilt correction [radians]
    
    # For each CCD position, calculate the MCP offset to the MCP position [x0 from x00 array] and the tilt of the MCP [gam from gam array] to feed into later calculations 
    if position <= 6.25:
        x0 = x00[0]
        gam = gam[0]
    elif position <= 6.75:
        x0 = x00[1]
        gam = gam[1]
    elif position <= 7.5:
        x0 = x00[2]
        gam = gam[2]
    elif position <= 8.125:
        x0 = x00[3]
        gam = gam[3]
    elif position <= 8.75:
        x0 = x00[4]
        gam = gam[4]
    elif position <= 9.375:
        x0 = x00[5]
        gam = gam[5]
    elif position <= 9.5:
        x0 = x00[6]
        gam = gam[6]
    elif position <= 10:
        x0 = x00[7]
        gam = gam[7]
    elif position <= 10.5:
        x0 = x00[8]
        gam = gam[8]
    elif position <= 11.5:
        x0 = x00[9]
        gam = gam[9]
    else:
        print('Invalid CCD position')
    
    # Calculating modified diffraction angle with MCP offset factor x0 (beta = arccos(x2/2R)) with unit conversion (eq. 1.6)
    beta=np.arccos(((position+x0)*0.0254)/(2*(Ro))) 
    M=3.411292384 # Image reduction factor [unitless]
    S0=528 # Approximate center of the MCP [pixels]
    pixel=np.linspace(0,1071, 1072)   
    pixel_size = 13*10**-6 # Pixel size of CCD [m]
    DY=0.001990838 # MCP radial offset off of Rowland circle [m]
    d=1666.666667 # Diffraction grating spacing [nm]
    
    # Initialize an empty array that will contain the combined corrections for MCP offset and tilt 
    wave = [] 
    # Loop over each pixel to calculate total correction factor
    for i,pix in enumerate(pixel):
        # First term in Eq. 1.18 within (sin{arctan[...]-\delta} without a \Delta z or \delta (no vertical grading offset or tilt?)
        term1 = np.sin(alpha-delta)
        # Second term in Eq. 1.18 within sin{arctan[...]+\delta+arctan[...]} Note: s_1^'' = M*(S0-pix)*13*10**-6
        term2 = np.sin(np.arctan(2*Ro*np.cos(beta)*np.sin(beta)/(2*Ro*np.cos(beta)**2 + Dz))+delta+np.arctan((M*(S0-pix)*pixel_size*np.cos(beta-gam)/np.cos(beta)+DY*np.tan(beta))/(2*Ro+DY-M*(S0-pix)*pixel_size*np.sin(beta-gam)/np.cos(beta)))) 
        # Multiplication by diffraction grating spacing [nm] to complete RHS of Eq. 1.18 
        wave += [d*(term1 - term2)] 
         
    return wave

#    term1 = np.sin(alpha) 
#    term2 = np.sin(beta+np.arctan((M*(S0-pix)*13*10**-6*np.cos(beta-gam)+DY*np.sin(beta))/((2*Ro+DY)*np.cos(beta)-M*(S0-pix)*13*10**-6*np.sin(beta-gam))))


# Get Shot Information 
def getShotInfo(shotnum):
    '''
    function getShotInfo: The function getShotInfo is assuming a particular number of columns in the data file that could be pulled from the text file directly rather than being hard-coded in. This form is likely just fine for most use cases, but it should be noted that if additional microchannel plates beyond three are used at some point, this function will need to be modified to include the additional pulse timing and widths for additional MCP channels. 
    '''
    # extract CCD position, pulser timings and widths (exposure duration) for each shot
    shotinfo_filepath = shotinfo_path + shotinfo_name

    shotinfo_data = np.loadtxt(shotinfo_filepath, usecols=(0,1,2,3,4,5,6,7)) # Load data column wise and create an array 

    # assign extracted data to particular quantities that will be returned as function output  
    shot_pos = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][1] # CCD position for a given input shotnumber 
    p1_timing = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][2] # MCP Strip #1 Pulse Timing (microseconds)
    p1_width = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][3] # MCP Strip #1 Pulse Width (microseconds)
    p2_timing = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][4] # MCP Strip #2 Pulse Timing (microseconds)
    p2_width = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][5] # MCP Strip #2 Pulse Timing (microseconds)
    p3_timing = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][6] # MCP Strip #3 Pulse Timing (microseconds)
    p3_width = shotinfo_data[np.where(shotinfo_data[:,0] == shotnum)[0][0]][7] # MCP Strip #3 Pulse Timing (microseconds)

    return shot_pos,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width

def shiftImage(image, imageHeight, imageWidth):
    # UNR in plane curvature correction 
    # converted from .java
    
#    r1=4056.57
#    r2=5000
#    r3=4500
    y0=576.6049
    r_he=6000.0
    
    y = 0
    while y < imageHeight:
        dx = r_he * ( 1.0 - np.cos(np.sin((y - y0) / r_he)) )
        ddx = dx - int(dx)
        x = imageWidth-int(dx)-2
        
        while x > 2:
            get1 = image.getpixel((x - 1, y))
            get2 = image.getpixel((x, y))
            val = int(ddx * get1 + (1.0 - ddx) * get2)
            if val<0: val = 0
            image.putpixel((x + int(dx), y), val)
            x -= 1
        y += 1
        
    return image

# Process Image 
def preProcessImage(shotnum):
    '''
    function processImage: The function processImage imports the raw data from the spectrometer and performs some image processing. This includes an in plane curvature correction, and background substraction, and gaussian convolution 
    '''
    # File path *.fit file for a particular shot number shotnum 
    filepath = path + '%s.fit' %(shotnum)
    # Split file path into root name and extension ext
    name, ext = os.path.splitext(filepath)
    # Use astropy package astropy.io.fits to open .fits file 
    img = fits.open(filepath)
    hdr = fits.getheader(filepath)
    ccdTemp = hdr['CCD-TEMP']
    

    # Printing image info to terminal 
    #print(img.info())
    #Extracting relevant information from image 
    data = img[0].data
    width = data.shape[1]
    height = data.shape[0]

    euv_arr = np.array(data, dtype=np.int16)
    
    # subract darkframe --- Always subtract a 'fresh' background, it should be taken right aftter a FuZE-Q shot
    darkframe_filepath = darkframe_path + dark_pic_name
    dark_img = fits.open(darkframe_filepath)
    dark_img_data = dark_img[0].data
    dark_img = Image.fromarray(dark_img_data.reshape((height, width)), "I;16")
    dark_arr = np.array(dark_img, dtype=np.int16)
    euv_sub_arr = euv_arr - dark_arr
   
    # eliminate negative pixel values
    for x in range(width):
        for y in range(height):
            if euv_sub_arr[y][x] < 0: euv_sub_arr[y][x] = 0
    
    # flip up & down --- Do this after the background subtraction!
    euv_sub_flip_img = ImageOps.flip(Image.fromarray(euv_sub_arr.reshape((height, width)), "I;16"))
    """
    # Shift image --- Shift image _after_ background subtraction
    image_shifted = shiftImage(output_flipped, height, width)
    # flip image left & right
    image_shifted = ImageOps.mirror(image_shifted) 
    """
    
    # Convert back to an array
    euv_sub_flip_arr = np.array(euv_sub_flip_img, dtype=np.int16)
    
    # convolute with gaussian spread function
    gaussian_filepath = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\Exponential2048-55.fits'
    gaussian_img = fits.open(gaussian_filepath)
    gaussian_img_data = gaussian_img[0].data
    gaussian_img_data = np.array(gaussian_img_data, dtype=np.int16)
    euv_conv_arr = fftconvolve(euv_sub_flip_arr, gaussian_img_data, mode='same')
    euv_conv_img = np.array(Image.fromarray(euv_conv_arr).resize(size=(1072, 1027)))
    return euv_sub_flip_arr, euv_conv_img, height, width, hdr, ccdTemp

def subtractConvolutedImage (img1, img2, height, width):
    #  The function 'subtractBackground()' computes a scaling factor for the blurred (convoluted) image by averaging the pixel values
    #  in the 'dark gap' between the strips for each x value (i.e. each image column) and use this scaling factor to scale the
    #  convoluted image column by column.
    
    #r1 = 23
    r1 = 11 # this brings the dark gaps to the center of the gaps
    r2 = r1+148
    r3 = r2+148
    r4 = r3+148
    r5 = r4+148
    r6 = r5+148
    r7 = r6+148
    y1 = [r1,r2,r3,r4,r5,r6,r7]
    idx1 = 0
    idx2 = 0
    xw = width

    img3 = np.copy(img1)
    
    num_strips=6
    yrange = 4
    xrange = 6

    for ix in range(xw):
        for istrip in range(0,num_strips):
            #loop through 3 strips, starting vertical pixel location stored in array y1[iy]
            zfactor1 = float(0.0)
            zfactor2 = float(0.0)
            zloop = float(0.0)
            for dy in range(-yrange,yrange): # average over three pixels in the y-direction
                for dx in range(-xrange,xrange): # average over 7 pixels in the y-direction: this is in the 'dead-space' between the strips
                    if ((ix+dx >= 0) and (ix+dx < xw)):
                        zloop += 1.0
                        idx1 = ix + dx
                        idx2 = ix + dx 
                        idy1 = (y1[istrip] + dy)
                        idy2 = (y1[istrip+1] + dy)
                        zfactor1 += (float(img3[idy1][idx1]) / float(img2[idy1][idx1]))
                        zfactor2 += (float(img3[idy2][idx2]) / float(img2[idy2][idx2]))

            zfactor1 = zfactor1 / float(zloop)
            zfactor2 = zfactor2 / float(zloop)
            dz = (zfactor2-zfactor1)/float(y1[istrip+1]-y1[istrip])
    
            for iy in range(y1[istrip], y1[istrip+1]): 
                dy1 = (iy - y1[istrip])
                img1[iy][ix] = img3[iy][ix] - int((zfactor1 + dz*float(dy1))*float(img2[iy][ix]))
                if (img1[iy][ix]<0):
                    img1[iy][ix] = 0

    img1 = ndimage.median_filter(img1, size=2) # small median filter to smooth and improve contrast
 
    # Shift image --- Shift image _after_ background subtraction
    euv_sub_flip_conv_img = shiftImage(Image.fromarray(img1.reshape((height, width)), "I;16"), height, width)
    # flip image left & right
    euv_sub_rot_conv_img = ImageOps.mirror(euv_sub_flip_conv_img) 

    euv_sub_rot_conv_arr = np.array(euv_sub_rot_conv_img, dtype=np.int16)
  
    return euv_sub_rot_conv_arr

# Generate Lineouts 
def getLineouts(processed_image, shotnum):
    '''
    function getLineouts: The function getLineouts accepts a processed image and generates 1-D line charts showing intensity vs. wavelength from each of the three MCP stripes currently being used in the EUV spectrometer. 
    '''
    # Definition the MCP channel width and locations [pixels]
    width_strip = 110 
    r1 = 176
    r2 = r1+148
    r3 = r2+148
    
    # Using the scikit-image process_line function to return the intensity profile of the image measured along a scan line 
    line1 = profile_line(processed_image, (r1+74, 0), (r1+74, 1071), linewidth=width_strip, mode='constant')
    line2 = profile_line(processed_image, (r2+74, 0), (r2+74, 1071), linewidth=width_strip, mode='constant')
    line3 = profile_line(processed_image, (r3+74, 0), (r3+74, 1071), linewidth=width_strip, mode='constant')
    
    
    
    
       
    # Implement moving average to smooth lineouts slightly over 5 bin window
    i=0
    mov_avg1 = []
    mov_avg2 = []
    mov_avg3 = []
    
    window_size = 4
    while i < len(line1) - window_size + 1:
        window_average1 = np.sum(line1[i:i+window_size])/window_size
        window_average2 = np.sum(line2[i:i+window_size])/window_size
        window_average3 = np.sum(line3[i:i+window_size])/window_size
        mov_avg1.append(window_average1)
        mov_avg2.append(window_average2)
        mov_avg3.append(window_average3)
        i += 1
        
    fig1, (ax1 , ax2) = plt.subplots(2,1,figsize=(16,9),sharey=True)
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=16)
    major_ticks = np.arange(round(wavelength_data[0])-0.5,round(wavelength_data[1071])+0.5,1)
    minor_ticks = np.arange(round(wavelength_data[0])-0.5,round(wavelength_data[1071])+0.5,0.25)
    
    # Plot unsmoothed data
    ax1.plot(wavelength_data[0:1072], line1, color='blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width),linewidth=0.7)
    ax1.plot(wavelength_data[0:1072], line2, color='red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width),linewidth=0.7)
    ax1.plot(wavelength_data[0:1072], line3, color='green', label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width),linewidth=0.7)
    ax1.legend(loc='upper right',fontsize=8)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(alpha=0.8, which = 'major')
    ax1.grid(alpha=0.3, which = 'minor')
    ax1.set_title("Unsmoothed")


    # Plot smoothed data 
    ax2.plot(wavelength_data[2:1071], mov_avg1, color='blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width),linewidth=0.7)
    ax2.plot(wavelength_data[2:1071], mov_avg2, color='red',label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width),linewidth=0.7)
    ax2.plot(wavelength_data[2:1071], mov_avg3,color='green', label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width),linewidth=0.7)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Counts [AU]')
    ax2.legend(loc='upper right',fontsize=8)
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(alpha=0.8, which = 'major')
    ax2.grid(alpha=0.3, which = 'minor')
    ax2.set_title('Smoothed')
    #fig1.set_dpi(1000)

    # Save Figure 
    plt.savefig(save_results_path + 'Lineouts_%s.png' %(shotnum))
    
    return line1, line2, line3, mov_avg1, mov_avg2, mov_avg3

# Find spectral lines from EUV data (using threshold method)
def get_lines_threshold(shotnum,line1,line2,line3,mov_avg1,mov_avg2,mov_avg3, wavelength_data):
    # Create Spectrum1D objects for each lineout
    '''
    # Note: Creating Spectrum 1D objects using unsmoothed spectra 

    # find spectral lines in unsmoothed spectra 
    line1_spectrum = Spectrum1D(flux = line1*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    
    line2_spectrum = Spectrum1D(flux = line2*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    
    line3_spectrum = Spectrum1D(flux = line3*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further manipulation
    '''

    # Note: Creating Spectrum1D objects using smoothed spectra 

    line1_spectrum = Spectrum1D(flux = mov_avg1*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further 
    
    line2_spectrum = Spectrum1D(flux = mov_avg2*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further 

    line3_spectrum = Spectrum1D(flux = mov_avg3*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further manipulation

    # Trim/cut unphysical parts of the spectrum 

    #line1 = [n if n > 0 else 0 for n in line1]
    #line2 = [n if n > 0 else 0 for n in line2]
    #line3 = [n if n > 0 else 0 for n in line3]

    # Plot Spectrum1D objects to ensure everything look right 

    fig2, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(16,9))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=14)
    ax1.plot(line1_spectrum.spectral_axis,line1_spectrum.flux,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    ax1.set(xlabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax1.set_ylim(bottom = 0, top = max(mov_avg1)*1.2)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(line2_spectrum.spectral_axis,line2_spectrum.flux,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    ax2.set(xlabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax2.set_ylim(bottom = 0, top = max(mov_avg2)*1.2)
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.spectral_axis,line3_spectrum.flux,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    ax3.set(xlabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax3.set_ylim(bottom = 0, top = max(mov_avg3)*1.2)
    ax3.legend()
    ax3.grid(True)
    """
     # Find spectral lines using threshold method 
    '''
    Note: The noise_region below will be likely unique to the spectrum and especially the wavelength range being looked into. It will be hard to automate this part since it relies on spotting a clear "noise" region in the spectrum by the user. The values below have been chosen as being a "noise" region (i.e. no discernable lines roughly in the middle of the spectrum window between ~ 25.6 - 42.5 nm).
    '''
    noise_region = SpectralRegion(34.09 * u.nm, 34.82 * u.nm) # Define noise spectral region for generating uncertainity 

    '''
    Note: The line*_flux_spectrum below will be likely unique to the spectrum and especially the wavelength range being looked into. It will be hard to automate this part since it relies defining a threshold for the derivative line finding algorithm. The values below have been chosen to fit discernable lines as best as possible, though should always be checked by the operator that it looks reasonable and especially that no high intensity lines are being missed.
    '''
    line1_noise_factor = 50 # Define threshold noise factor
    line2_noise_factor = 40 # Define threshold noise factor
    line3_noise_factor = 40 # Define threshold noise factor

    line1_spectrum = noise_region_uncertainty(line1_spectrum,noise_region)
    line2_spectrum = noise_region_uncertainty(line2_spectrum,noise_region)
    line3_spectrum = noise_region_uncertainty(line3_spectrum,noise_region)

    with warnings.catch_warnings(): #Ignore warnings 
        warnings.simplefilter('ignore')
        line1_lines = find_lines_threshold(line1_spectrum,noise_factor=line1_noise_factor)
        line2_lines = find_lines_threshold(line2_spectrum,noise_factor=line2_noise_factor)
        line3_lines = find_lines_threshold(line3_spectrum,noise_factor=line3_noise_factor)

    # Print the emission spectral lines found using derivative line finding
    #print(line1_lines[line1_lines['line_type'] == 'emission'])
    #print(line2_lines[line2_lines['line_type'] == 'emission'])
    #print(line3_lines[line3_lines['line_type'] == 'emission'])

    # Plot the emission spectrum with found spectal lines to assess performance 
    fig3, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,9))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=14)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    for iter in np.arange(0,len(line1_lines[line1_lines['line_type'] == 'emission']),1):
        ax1.axhline(line1_lines[line1_lines['line_type'] == 'emission'][iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax1.set(xlabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax1.set_ylim(bottom = 0, top = max(line1)*1.2)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    
    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    for iter in np.arange(0,len(line2_lines[line2_lines['line_type'] == 'emission']),1):
        ax2.axhline(line2_lines[line2_lines['line_type'] == 'emission'][iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax2.set(xlabel = 'Wavelength ({})'.format(line2_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax2.set_ylim(bottom = 0, top = max(line2)*1.2)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    for iter in np.arange(0,len(line3_lines[line3_lines['line_type'] == 'emission']),1):
        ax3.axvline(line3_lines[line3_lines['line_type'] == 'emission'][iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax3.set(xlabel = 'Wavelength ({})'.format(line3_spectrum.spectral_axis.unit), ylabel = 'Intensity [Arb.]')
    ax3.set_ylim(bottom = 0, top = max(line3)*1.2)
    ax3.legend(loc='upper right')
    ax3.grid(True)
    """
    line1_lines = ""
    line2_lines = ""
    line3_lines = ""
    return line1_lines,line2_lines,line3_lines 

# Find spectral lines from EUV data (using derivative method)
def get_lines_derivative(shotnum,line1,line2,line3,mov_avg1,mov_avg2,mov_avg3,wavelength_data):
    # Create Spectrum1D objects for each lineout
    '''
    # Note: Creating Spectrum 1D objects using unsmoothed spectra 

    # find spectral lines in unsmoothed spectra 
    line1_spectrum = Spectrum1D(flux = line1*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    
    line2_spectrum = Spectrum1D(flux = line2*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    
    line3_spectrum = Spectrum1D(flux = line3*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further manipulation
    '''
    # Note: Creating Spectrum1D objects using smoothed spectra 
    
    line1_spectrum = Spectrum1D(flux = mov_avg1*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further 
    
    line2_spectrum = Spectrum1D(flux = mov_avg2*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further 

    line3_spectrum = Spectrum1D(flux = mov_avg3*u.dimensionless_unscaled, spectral_axis=wavelength_data[2:1071]*u.nm) # Create Spectrum1D object for further manipulation

    # Trim/cut unphysical parts of the spectrum 

    #line1 = [n if n > 0 else 0 for n in line1]
    #line2 = [n if n > 0 else 0 for n in line2]
    #line3 = [n if n > 0 else 0 for n in line3]

    # Plot Spectrum1D objects to ensure everything look right 
    fig2, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,9))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=14)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    ax1.set(ylabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = 0, right = max(mov_avg1)*1.2)
    ax1.legend(loc='upper right',fontsize=8)
    ax1.grid(True)

    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    ax2.set(ylabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = 0, right = max(mov_avg2)*1.2)
    ax2.legend(loc='upper right',fontsize=8)
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    ax3.set(ylabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = 0, right = max(mov_avg3)*1.2)
    ax3.legend(loc='upper right',fontsize=8)
    ax3.grid(True)

    # Find spectral lines using derivative method 
    '''
    Note: The line#_flux_spectrum below will be likely unique to the spectrum and especially the wavelength range being looked into. It will be hard to automate this part since it relies defining a threshold for the derivative line finding algorithm. The values below have been chosen to fit discernable lines as best as possible, though should always be checked by the operator that it looks reasonable and especially that no high intensity lines are being missed.
    '''
    line1_flux_threshold = 0 # Define derivative flux threshold
    line2_flux_threshold = 0 # Define derivative flux threshold
    line3_flux_threshold = 0 # Define derivative flux threshold

    with warnings.catch_warnings(): #Ignore warnings 
        warnings.simplefilter('ignore')
        line1_lines = find_lines_derivative(line1_spectrum,flux_threshold=line1_flux_threshold)
        line2_lines = find_lines_derivative(line2_spectrum,flux_threshold=line2_flux_threshold)
        line3_lines = find_lines_derivative(line3_spectrum,flux_threshold=line3_flux_threshold)

    # Restrict to only identified 'emission' lines
    line1_lines = line1_lines[line1_lines['line_type'] == 'emission']
    line2_lines = line2_lines[line2_lines['line_type'] == 'emission']
    line3_lines = line3_lines[line3_lines['line_type'] == 'emission']

    # Print the emission spectral lines found using derivative line finding
    #print(line1_lines)
    #print(line2_lines)
    #print(line3_lines)

    # Plot the emission spectrum with found spectal lines to assess performance 
    fig3, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,9))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=14)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    for iter in np.arange(0,len(line1_lines),1):
        ax1.axhline(line1_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax1.set(ylabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = 0, right = max(line1)*1.2)
    ax1.legend(loc='upper right',fontsize=8)
    ax1.grid(True)

    
    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    for iter in np.arange(0,len(line2_lines),1):
        ax2.axhline(line2_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax2.set(ylabel = 'Wavelength ({})'.format(line2_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = 0, right = max(line2)*1.2)
    ax2.legend(loc='upper right',fontsize=8)
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    for iter in np.arange(0,len(line3_lines),1):
        ax3.axhline(line3_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax3.set(ylabel = 'Wavelength ({})'.format(line3_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = 0, right = max(line3)*1.2)
    ax3.legend(loc='upper right',fontsize=8)
    ax3.grid(True)

    return line1_lines,line2_lines,line3_lines,line1_spectrum,line2_spectrum,line3_spectrum 

# Query NIST atomic spectral line database 
def queryNist(wavelength_data):
    '''
    function queryNist: The function queryNist queries the NIST atomic spectra database within a particular wavelength range of interest for the EUV spectrometer system. This current extracts all spectral information within a wavelength range for carbon, oxygen, silicon, and tungsten impurities and saves them to respective astropy table objects for further manipulation. 
    '''
    # Carbon lines 
    tableC = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'C', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Oxygen lines 
    tableO = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'O', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Silicon lines 
    tableSi = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'Si', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Tungsten lines 
    tableW = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'W', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Copper lines 
    tableCu = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'Cu', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Chromium lines 
    tableCr = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'Cr', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Zirconium lines 
    tableZr = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'Zr', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Boron lines 
    tableB = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'B', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Nitrogen lines 
    tableN = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'N', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    # Aluminum lines 
    tableAl = Nist.query(min(wavelength_data) * u.nm, max(wavelength_data) * u.nm, linename = 'Al', energy_level_unit = 'eV', wavelength_type = 'vacuum')

    return tableC, tableO, tableSi, tableW, tableCu, tableCr, tableZr, tableB, tableN, tableAl

# Match EUV spectral lines to NIST database lines 
def linematch(line1_lines,line2_lines,line3_lines,tableZ): 
    '''
    function linematch: The function linematch takes the identified lines from the EUV spectrum and attempts to match them to spectral lines within a relevant wavelength range from NIST, and selects the best fit if multiple lines match within a particular wavelength range. This is a bit sloppy as written so should be cleaned up later. 
    '''
    # Line matching for MCP Strip #1 
    # For each identified line in the EUV spectrum 
    line1=[]
    # Set the wavelength threshold for a line being considered a match
    lambda_thres = 0.06

    for row in np.arange(len(line1_lines)-1):
        # Print the identifed line in the EUV spectrum
        #print("Attempting to match line with wavelength %s" %(line1_lines[line1_lines['line_type'] == 'emission'][row][0]))

        tmp = [] # Empty list to store matched lines
        # For each spectral line in the specified wavelength range from NIST 
        for nist_row in np.arange(len(tableZ)):
            # Check if there is an observed wavelength. If so, proceed. 
            if np.ma.is_masked(tableZ[nist_row][1]) == False:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match and store 
                if np.absolute(tableZ[nist_row][1] - line1_lines[line1_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
                    #print('Its a match!!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][1]))

                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][1],np.absolute(tableZ[nist_row][1] - line1_lines[line1_lines['line_type'] == 'emission'][row][0].value)])
            
            
            # If there's not an observed wavelength, use Ritz wavelength instead 
            else:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store
                if np.absolute(tableZ[nist_row][2]- line1_lines[line1_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
#                    print('Its a match!')
#                    print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][2],np.absolute(tableZ[nist_row][2] - line1_lines[line1_lines['line_type'] == 'emission'][row][0].value)])
            
        
        # Remove duplicate entires from identified lines after going through all NIST spectral lines to match a given EUV line 

        tmp = np.asarray(tmp) # Convert to a numpy array 
        tmp = np.unique(tmp,axis=0) # Remove duplicate entries 
        
        # If more than one line matched, select the one that is the best fit 
        if len(tmp) > 0: # If the number of matched lines is more than 0
            tmp_dif = tmp[:,2] # Extract the difference between EUV and NIST 
            ind_min_dif = np.argmin(tmp_dif) # Extract the index of the minimum
            #print(tmp[ind_min_dif])
            tmp = tmp[ind_min_dif] # Set best matched line as output 
        
        tmp = tmp.tolist() # Convert back to list 

        #print('Printing Matched Line for EUV Line %s ' %(line1_lines[line1_lines['line_type'] == 'emission'][row][0]))
        line1.append(tmp) # Append to matched spectral lines array 
    line1 = [i for i in line1 if i != []] # trim out lines that were not matched 

    # Line matching for MCP Strip #2 
    # For each identified line in the EUV spectrum 
    line2 = []
    for row in np.arange(len(line2_lines)-1):
        # Print the identifed line in the EUV spectrum
        # print("Attempting to match line with wavelength %s" %(line2_lines[line2_lines['line_type'] == 'emission'][row][0]))

        tmp = [] # Empty list to store matched lines
        # For each spectral line in the specified wavelength range from NIST 
        for nist_row in np.arange(len(tableZ)):
            # Check if there is an observed wavelength, if there is, use it.
            if np.ma.is_masked(tableZ[nist_row][1]) == False:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store
                if np.absolute(tableZ[nist_row][1] - line2_lines[line2_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
                    #print('Its a match!!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][1]))

                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][1],np.absolute(tableZ[nist_row][1] - line2_lines[line2_lines['line_type'] == 'emission'][row][0].value)])
            
            
            # If there's not an observed wavelength, use Ritz wavelength instead 
            else:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store
                if np.absolute(tableZ[nist_row][2]- line2_lines[line2_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
#                    print('Its a match!')
#                    print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][2],np.absolute(tableZ[nist_row][2] - line2_lines[line2_lines['line_type'] == 'emission'][row][0].value)])
            
        
        # Remove duplicate entires from identified lines after going through all NIST spectral lines to match a given EUV line 

        tmp = np.asarray(tmp) # Convert to a numpy array 
        tmp = np.unique(tmp,axis=0) # Remove duplicate entries 
        
        # If more than one line matched, select the one that is the best fit 
        if len(tmp) > 0: # If the number of matched lines is more than 0
            tmp_dif = tmp[:,2] # Extract the difference between EUV and NIST 
            ind_min_dif = np.argmin(tmp_dif) # Extract the index of the minimum
            #print(tmp[ind_min_dif])
            tmp = tmp[ind_min_dif] # Set best matched line as output 
        
        tmp = tmp.tolist() # Convert back to list 

        #print('Printing Matched Line for EUV Line %s ' %(line2_lines[line2_lines['line_type'] == 'emission'][row][0]))
        line2.append(tmp) # Append to matched spectral lines array 
    line2 = [i for i in line2 if i != []] # trim out lines that were not matched 

    # Line matching for MCP Strip #3 
    # For each identified line in the EUV spectrum 
    line3=[]
    for row in np.arange(len(line3_lines)-1):
        # Print the identifed line in the EUV spectrum
        #print("Attempting to match line with wavelength %s" %(line3_lines[line3_lines['line_type'] == 'emission'][row][0]))

        tmp = [] # Empty list to store matched lines
        # For each spectral line in the specified wavelength range from NIST 
        for nist_row in np.arange(len(tableZ)):
            # Check if there is an observed wavelength, if there is, use it.
            if np.ma.is_masked(tableZ[nist_row][1]) == False:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store 
                if np.absolute(tableZ[nist_row][1] - line3_lines[line3_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
                    #print('Its a match!!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][1]))

                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][1],np.absolute(tableZ[nist_row][1] - line3_lines[line3_lines['line_type'] == 'emission'][row][0].value)])
            
            
            # If there's not an observed wavelength, use Ritz wavelength instead 
            else:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store
                if np.absolute(tableZ[nist_row][2]- line3_lines[line3_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
#                    print('Its a match!')
#                    print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][2],np.absolute(tableZ[nist_row][2] - line3_lines[line3_lines['line_type'] == 'emission'][row][0].value)])
            
        
        # Remove duplicate entires from identified lines after going through all NIST spectral lines to match a given EUV line 

        tmp = np.asarray(tmp) # Convert to a numpy array 
        tmp = np.unique(tmp,axis=0) # Remove duplicate entries 
        
        # If more than one line matched, select the one that is the best fit 
        if len(tmp) > 0: # If the number of matched lines is more than 0
            tmp_dif = tmp[:,2] # Extract the difference between EUV and NIST 
            ind_min_dif = np.argmin(tmp_dif) # Extract the index of the minimum
            #print(tmp[ind_min_dif])
            tmp = tmp[ind_min_dif] # Set best matched line as output 
        
        tmp = tmp.tolist() # Convert back to list 

        #print('Printing Matched Line for EUV Line %s ' %(line3_lines[line3_lines['line_type'] == 'emission'][row][0]))
        line3.append(tmp) # Append to matched spectral lines array 
    line3 = [i for i in line3 if i != []] # trim out lines that were not matched 

    return line1, line2, line3, lambda_thres

def plot_results(line1_store,line2_store,line3_store,line1_spectrum,line2_spectrum,line3_spectrum,tableZ):
    # Invert this later for easier reading of the spectral information. 
     # Plot the emission spectrum with found spectal lines to assess performance 
    fig4, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,9))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=14)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    for element in line1_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                 ax1.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                 ax1.text(-190,float(element[iter][1])+0.05,'%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax1.set(ylabel = 'Wavelength ({})'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = -200, right = max(line1_spectrum.flux)*1.2)
    ax1.legend(loc='upper right',fontsize=8)
    ax1.grid(True)

    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    for element in line2_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                 ax2.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                 ax2.text(-480,float(element[iter][1])+0.05,'%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax2.set(ylabel = 'Wavelength ({})'.format(line2_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = -500, right = max(line2_spectrum.flux)*1.2)
    ax2.legend(loc='upper right',fontsize=8)
    ax2.grid(True)

    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green', label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    for element in line3_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                 ax3.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                 ax3.text(-480,float(element[iter][1])+0.05, '%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax3.set(ylabel = 'Wavelength ({})'.format(line3_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = -500, right = max(line3_spectrum.flux)*1.2)
    ax3.legend(loc='upper right',fontsize=8)
    ax3.grid(True)

## End of functions
#----------------------
'''
Multi-shot plotting capability for now. Should instead read the EUV.....txt to pull out the shot numbers that have at least one of the MCP settings as not NaN. This is fine for having independent control of the shot numbers in question for now. 
'''
#shotlist_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/shot_numbers.txt'
shotlist_filepath = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\shot_numbers.txt'
#shotlist_filepath = r'/home/daszap/projects/EUV/EUV_Code/shot_numbers.txt'
shotlist = np.loadtxt(shotlist_filepath,dtype=int)

for shotnum in np.nditer(shotlist):

    ## Run processing functions

    # get shot info function returns CCD position and pulser timings and widths (durations of exposure)
    shot_pos,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width = getShotInfo(shotnum)

    # get wavelength function returns array of wavelengths [nm] for pixel values 0-1072
    wavelength_data = getWavelength(shot_pos)

    # pre-process image function returns the euv-dark and euv-dark-convoluted images plus the fits header of the raw euv image
    euv_sub_flip_arr, euv_conv_img, height, width, hdr, ccdTemp = preProcessImage(shotnum)
    
    processed_image = subtractConvolutedImage (euv_sub_flip_arr, euv_conv_img, height, width)
    #
#    """
    # extact lineouts of MCP strips 
    line1, line2, line3, mov_avg1, mov_avg2, mov_avg3 = getLineouts(processed_image, shotnum)
    '''
    Note: Either the get_lines_threshold or get_lines_derivative should be used, but not both at the same time. The former is a method for extracting lines in a spectrum using a noise/thresholding approach, whereas the latter is using a thresholding based on the derivative of the spectrum. Use whichever produces the best looking identification. 

    Currently, I think the derivative method performs a lot better than the simple thresholding method. 
    '''
    # identify lines using threshold method 
    #line1_lines,line2_lines,line3_lines = get_lines_threshold(shotnum,line1,line2,line3,mov_avg1,mov_avg2,mov_avg3,wavelength_data)

    # identify lines using derivative method 
    line1_lines,line2_lines,line3_lines, line1_spectrum,line2_spectrum,line3_spectrum = get_lines_derivative(shotnum, line1,line2,line3,mov_avg1,mov_avg2,mov_avg3,wavelength_data)
    #"""
    # Query NIST database for various elements within wavelength range (currently set to carbon, oxygen, and silicon, tungsten, and copper)
    tableC, tableO, tableSi, tableW, tableCu, tableCr, tableZr, tableB, tableN, tableAl = queryNist(wavelength_data)

    # Assemble combined table of spectral lines from various elements of interest
    # Uncomment below for the nine lightest elements 
    #tableZ = [tableB,tableC,tableN,tableO,tableAl,tableSi,tableCr,tableCu,tableZr]
    # Uncomment below for the six lightest elements being looked at
    # tableZ = [tableB,tableC,tableN,tableO,tableAl,tableSi]
    # Uncomment below for four lightest elements being looked at
    #tableZ = [tableB,tableC,tableN,tableO]
    # Uncomment below for just carbon and oxygen lines 
    # tableZ = [tableC,tableO]
    tableZ = [ tableSi, tableO]

    # Match EUV spectral lines for each MCP strip to NIST database lines 
    line1_store = []
    line2_store = []
    line3_store = []
    # For each element spectral line database, find matching lines 
    for element in tableZ:
       line1, line2, line3, lambda_thres = linematch(line1_lines,line2_lines,line3_lines,element)

       line1_store.append(line1)
       line2_store.append(line2)
       line3_store.append(line3)
    """
    print('Line Matches for MCP Strip #1 to within %s nm precision' %(lambda_thres))
    print(line1_store)
    print(len(line1_store))
    print('Line Matches for MCP Strip #2 to within %s nm precision' %(lambda_thres))
    print(line2_store)
    print(len(line2_store))
    print('Line Matches for MCP Strip #3 to within %s nm precision' %(lambda_thres))
    print(line3_store)
    print(len(line3_store))
    """
    plot_results(line1_store,line2_store,line3_store,line1_spectrum,line2_spectrum,line3_spectrum,tableZ)

    #----------------------

    ### CCD image plotting options

    # a colormap and a normalization instance
#    cmap = plt.cm.bone
    cmap = plt.cm.gray
    norm = colors.LogNorm(clip=True)

    # map the normalized data to colors
    # image is now RGBA (512x512x4) 
    image_log = cmap(norm(processed_image))
    
    ## Plotting and figure handling

    fig5, ax2 = plt.subplots()
#    ax2.imshow(image_log, origin='lower',cmap='bone') # show processed image - log scale
    ax2.imshow(processed_image, origin='lower',cmap='gray')
#    ax2.imshow(processed_image, origin='lower',cmap='bone')

    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.title('EUV Processed Image FuZE-Q Shot %s' %(shotnum),fontsize=10)
    fig5.set_dpi(1000)

    # save image
    plt.imsave(save_results_path + 'CCD_%s.png' %(shotnum), processed_image, cmap='gray')
    fits.writeto(save_results_path + 'CCD_%s.fits' %(shotnum), processed_image, hdr, overwrite = True)
    print('CCD Temperature: %0.2f °C' % ccdTemp)
    
#    stistools.ocrreject.ocrreject('G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_CCD\\CCD_%s.fits' %(shotnum),'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_CCD\\CCD_%s_ocr.fits' %(shotnum), verbose=True)
    
    plt.show() 





