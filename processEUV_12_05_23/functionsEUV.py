"""
EUV Spectroscopy Processing 
@author: Aidan Klemmer & Stephan Fuelling & Derek Sutherland
aidan@nevada.unr.edu
12/5/23
University of Nevada, Reno
and
Zap Energy Inc

File 3
"""

import os
from astropy.io import fits 
import astropy.units as u
from astroquery.nist import Nist 
from specutils import Spectrum1D,  SpectralRegion
from specutils.manipulation import noise_region_uncertainty
from specutils.fitting import find_lines_derivative,find_lines_threshold
from PIL import Image, ImageOps
import numpy as np
from scipy.signal import fftconvolve
from scipy import ndimage
from skimage.measure import profile_line
import warnings
import matplotlib.pyplot as plt;


## import data from file 4
import calibrationParameters as calParams
import pathLocations as pathLoc        

## convert position 
# Get Wavelength Functions 
def getWavelength(position):
    '''
    function getWavelength: The function getWavelength provides the wavelengths including both MCP offset and tilt. But, note it currently does not include a grating offset (\Delta z) or tilt (\delta) which could be added later, if needed. 

    '''     
    
    # run get calibration function
    x0, gam, M = calParams.getCal(position)
    
    position = position/1000 # Position of CCD along Rowland circle [in]
    Ro=0.4994 # Rowland circle radius [m] (~ 1 m diameter)
    alpha=1.495704 # Angle of incidence of light [radians]
    
    delta=0.0 # Correction for grating tilt [radians]
    Dz=0.0  # Correction for grating offset [m]

    # Calculating modified diffraction angle with MCP offset factor x0 (beta = arccos(x2/2R)) with unit conversion (eq. 1.6)
    beta=np.arccos(((position+x0)*0.0254)/(2*(Ro))) 
    #M=3.411292384 # Image reduction factor [unitless]
    #M=3 # Image reduction factor [unitless]
    S0=536 # Approximate center of the MCP [pixels]
    pixel=np.linspace(0,1071, 1072)   
    pixel_size = 13*10**-6 # Pixel size of CCD [m]
    DY=0.001990838 # MCP radial offset off of Rowland circle [m]
    #DY=0.000425
    d=1666.666667 # Diffraction grating spacing [nm]
    
    # Initialize an empty array that will contain the combined corrections for MCP offset and tilt 
    wave = [] 
    # Loop over each pixel to calculate total correction factor
    for i,pix in enumerate(pixel):
        term1 = np.sin(alpha)
        term2 = np.sin(beta + np.arctan((M * (pix-S0) * pixel_size * np.cos(beta-gam) + DY*np.sin(beta)) / ((2*Ro+DY) * np.cos(beta) - M*(pix-S0) *pixel_size * np.sin(beta-gam))))   
        wave += [d*(term1 - term2)] 
    #print(wave)

    return wave

# Get Shot Information 
def getShotInfo(shotnum):
    '''
    function getShotInfo: The function getShotInfo is assuming a particular number of columns in the data file that could be pulled from the text file directly rather than being hard-coded in. This form is likely just fine for most use cases, but it should be noted that if additional microchannel plates beyond three are used at some point, this function will need to be modified to include the additional pulse timing and widths for additional MCP channels. 
    '''
    # extract CCD position, pulser timings and widths (exposure duration) for each shot
    #shotinfo_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_refdata_11_22.txt' # Path to text file containing desired information
    #shotinfo_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/230315/EUV_refdata_23_03_15.txt' # Path to text file containing desired information
    #shotinfo_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_refdata_11_22.txt' # Path to text file containing desired information
    #shotinfo_filepath = r'/home/daszap/projects/EUV/EUV_Code/EUV_refdata_11_22.txt' # Path to text file containing desired information

    # extract CCD position, pulser timings and widths (exposure duration) for each shot
    shotinfo_filepath = pathLoc.shotinfo_path + pathLoc.shotinfo_name

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

def removeHotPixels(image, imageHeight, imageWidth, threshold):
    for y in range(0, imageHeight-1):
        p0 = image.getpixel ((0,y))
        p1 = image.getpixel ((1,y))
        for x in range (8, imageWidth-10):          
            p2 = image.getpixel ((x+2,y))
            if ((p0) < (p1 - threshold)) and ((p2) < (p1 - threshold)):
                print(x,y, p0, p1, p2)
                p1 = int((p0+p2)/2)
                image.putpixel((x+1,y), p1)
            p0 = p1
            p1 = p2
    return image

# shift image function - used in processImage()
def shiftImage(image, imageHeight, imageWidth):
    # UNR in plane curvature correction 
    # converted from .java
    
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
def processImage(shotnum):
    import numpy as np
    
    '''
    function processImage: The function processImage imports the raw data from the spectrometer and performs some image processing. This includes an in plane curvature correction, and background substraction, and gaussian convolution 
    '''
    # File path *.fit file for a particular shot number shotnum 
    #filepath = r'/Users/daszap/Projects/EUV/EUV_Data/221114/%s.fit' %(shotnum)
    #filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/221/%s.fit' %(shotnum)
    #filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/230315/%s.fit' %(shotnum)
    #filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/EUV_11_11_22/%s.fit' %(shotnum)
    #filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/231127/%s.fit' %(shotnum)          
    #filepath = r'/home/daszap/projects/EUV/EUV_Data/221111/%s.fit' %(shotnum)
    
    # File path *.fit file for a particular shot number shotnum 
    filepath = pathLoc.path + '%s.fit' %(shotnum)

    
    # Split file path into root name and extension ext
    name, ext = os.path.splitext(filepath)
    # Use astropy package astropy.io.fits to open .fits file 
    img = fits.open(filepath)
    hdr = fits.getheader(filepath)
    ccdTemp = hdr['CCD-TEMP']

    #Extracting relevant information from image 
    data = img[0].data
    width = data.shape[1]
    height = data.shape[0]

    euv_arr = np.array(data, dtype=np.int16)
    
    # subract darkframe --- Always subtract a 'fresh' background, it should be taken right aftter a FuZE-Q shot
    #darkframe_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/Dark_Frames/5sec_EUV_background_new_0deg.fit'
    #darkframe_filepath = r'/home/daszap/projects/EUV/EUV_Data/Dark_Frames/5sec_EUV_background_new_0deg.fit'
    # subract darkframe --- Always subtract a 'fresh' background, it should be taken right aftter a FuZE-Q shot
    darkframe_filepath = pathLoc.darkframe_path + pathLoc.dark_pic_name
    dark_img = fits.open(darkframe_filepath)
    dark_img_data = dark_img[0].data
    dark_img = Image.fromarray(dark_img_data.reshape((height, width)), "I;16")
    dark_arr = np.array(dark_img, dtype=np.int16)
    euv_sub_arr = euv_arr - dark_arr
    
    fig5, ax2 = plt.subplots()
    ax2.imshow(euv_sub_arr, origin='lower',cmap='gray')
  

   
    # eliminate negative pixel values
    for x in range(width):
        for y in range(height):
            if euv_sub_arr[y][x] < 0: euv_sub_arr[y][x] = 0
    
    # flip up & down --- Do this after the background subtraction!
    euv_sub_flip_img = ImageOps.flip(Image.fromarray(euv_sub_arr.reshape((height, width)), "I;16"))
    #eliminate hot pixels
    euv_sub_flip_img = removeHotPixels(euv_sub_flip_img, height, width, 500)
    
    # Convert back to an array
    euv_sub_flip_arr = np.array(euv_sub_flip_img, dtype=np.int16)
    
    # convolute with gaussian spread function
    #gaussian_filepath = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\Exponential2048-55.fits'
    #gaussian_filepath = r'/Applications/processEUV/Exponential2048-55.fits'
    gaussian_filepath = pathLoc.gaussian_path + pathLoc.gaussian_name
    #gaussian_filepath = r'/home/daszap/projects/EUV/EUV_Code/Exponential2048-55.fits'
    gaussian_img = fits.open(gaussian_filepath)
    gaussian_img_data = gaussian_img[0].data
    gaussian_img_data = np.array(gaussian_img_data, dtype=np.int16)
    euv_conv_arr = fftconvolve(euv_sub_flip_arr, gaussian_img_data, mode='same')
    euv_conv_arr2 = np.array(Image.fromarray(euv_conv_arr).resize(size=(1072, 1027)))

    fig8, ax2 = plt.subplots()
    ax2.imshow(euv_conv_arr2, origin='lower',cmap='gray')

    #  The section computes a scaling factor for the blurred (convoluted) image by averaging the pixel values
    #  in the 'dark gap' between the strips for each x value (i.e. each image column) and use this scaling factor to scale the
    #  convoluted image column by column.

    # define image variables - helpful for troubleshooting
    img1 = euv_sub_flip_arr
    img2 = euv_conv_arr2
    
    #r1 = 23 # historical value
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

    # median filter smoothing - optional (use if needed)
    #img1 = ndimage.median_filter(img1, size=2) # small median filter to smooth and improve contrast
 
    # Shift image --- Shift image _after_ background subtraction
    euv_sub_flip_conv_img = shiftImage(Image.fromarray(img1.reshape((height, width)), "I;16"), height, width)
    # flip image left & right
    euv_sub_rot_conv_img = ImageOps.mirror(euv_sub_flip_conv_img) 

    euv_sub_rot_conv_arr = np.array(euv_sub_rot_conv_img, dtype=np.int16)

    fig8, ax2 = plt.subplots()
    ax2.imshow(euv_sub_rot_conv_arr, origin='lower',cmap='gray')
  
    return euv_sub_rot_conv_arr


# Generate Lineouts 
def getLineouts(processed_image):
    '''
    function getLineouts: The function getLineouts accepts a processed image and generates 1-D line charts showing intensity vs. wavelength from each of the three MCP stripes currently being used in the EUV spectrometer. 
    '''
    # Definition the MCP channel width and locations [pixels]
    width_strip = 60
    r1 = 141
    r2 = r1+148
    r3 = r2+148
    
    # Using the scikit-image process_line function to return the intensity profile of the image measured along a scan line 
    line1 = profile_line(processed_image, (r1+74, 0), (r1+74, 1071), linewidth=width_strip, mode='constant')
    line2 = profile_line(processed_image, (r2+74, 0), (r2+74, 1071), linewidth=width_strip, mode='constant')
    line3 = profile_line(processed_image, (r3+74, 0), (r3+74, 1071), linewidth=width_strip, mode='constant')

    return line1, line2, line3


# Find spectral lines from EUV data (using derivative method)
def get_lines(shotnum,line1,line2,line3,wavelength_data):
    
    # Trim/cut unphysical parts of the spectrum 
    line1 = [n if n > 0 else 0 for n in line1]
    line2 = [n if n > 0 else 0 for n in line2]
    line3 = [n if n > 0 else 0 for n in line3]
    
    # Create Spectrum1D objects for each lineout
    # find spectral lines in spectra 
    line1_spectrum = Spectrum1D(flux = line1*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    line2_spectrum = Spectrum1D(flux = line2*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further 
    line3_spectrum = Spectrum1D(flux = line3*u.dimensionless_unscaled, spectral_axis=wavelength_data*u.nm) # Create Spectrum1D object for further manipulation

    # Find spectral lines using derivative method 
    '''
    Note: The line#_flux_spectrum below will be likely unique to the spectrum and especially the wavelength range being looked into. It will be hard to automate this part since it relies defining a threshold for the derivative line finding algorithm. The values below have been chosen to fit discernable lines as best as possible, though should always be checked by the operator that it looks reasonable and especially that no high intensity lines are being missed.
    '''
    # define wavelength region of noise without and spectral peaks
    start_noise = 15.5 
    end_noise = 16.5
    
    # specutils noise
    noise_region = SpectralRegion(start_noise*u.nm, end_noise*u.nm)
    line1_spectrum = noise_region_uncertainty(line1_spectrum, noise_region)
    line2_spectrum = noise_region_uncertainty(line2_spectrum, noise_region)
    line3_spectrum = noise_region_uncertainty(line3_spectrum, noise_region)
    
    
    with warnings.catch_warnings(): #Ignore warnings 
        warnings.simplefilter('ignore')
        line1_lines = find_lines_derivative(line1_spectrum,flux_threshold=0.25*np.max(line1_spectrum.flux))
        line2_lines = find_lines_derivative(line2_spectrum,flux_threshold=0.25*np.max(line2_spectrum.flux))
        line3_lines = find_lines_derivative(line3_spectrum,flux_threshold=0.25*np.max(line3_spectrum.flux))

    # Restrict to only identified 'emission' lines
    line1_lines = line1_lines[line1_lines['line_type'] == 'emission']
    line2_lines = line2_lines[line2_lines['line_type'] == 'emission']
    line3_lines = line3_lines[line3_lines['line_type'] == 'emission']

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
    
    # Set the wavelength threshold for a line being considered a match
    lambda_thres = 0.05
    

    # Line matching for MCP Strip #1 
    # For each identified line in the EUV spectrum 
    line1=[]

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
                    #print('Its a match!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
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
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match and store
                if np.absolute(tableZ[nist_row][1] - line2_lines[line2_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
                    #print('Its a match!!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][1]))

                    tmp.append([tableZ[nist_row][0],tableZ[nist_row][1],np.absolute(tableZ[nist_row][1] - line2_lines[line2_lines['line_type'] == 'emission'][row][0].value)])
            
            
            # If there's not an observed wavelength, use Ritz wavelength instead 
            else:
                # If the difference between the EUV line and NIST line is less than the wavelength threshold lambda_thres, consider it a match adnd store
                if np.absolute(tableZ[nist_row][2]- line2_lines[line2_lines['line_type'] == 'emission'][row][0].value) < lambda_thres:
                    #print('Its a match!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
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
                    #print('Its a match!')
                    #print('%s %s nm' %(tableZ[nist_row][0],tableZ[nist_row][2]))
                   
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






