"""
EUV Spectroscopy Processing 
@author: Aidan Klemmer & Stephan Fuelling & Derek Sutherland
aidan@nevada.unr.edu
12/5/23
University of Nevada, Reno
and
Zap Energy Inc

File 1
"""

import numpy as np

# import functions from files 2 and 3
import functionsEUV_SF as funcEUV
import plotEUV as pltEUV
import pathLocations as pathLoc


shotlist_filepath = pathLoc.shotlist_path + pathLoc.shotlist_name
shotlist = np.loadtxt(shotlist_filepath,dtype=int)


for shotnum in np.nditer(shotlist):
    
    
    #---
    ## Run processing functions
    # these functions are the core data processing needed to produce spectral lineouts with a wavelength calibration
    
    # Core functions
    # get shot info function returns CCD position and pulser timings and widths (durations of exposure)
    shot_pos,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width = funcEUV.getShotInfo(shotnum)

    # get wavelength function returns array of wavelengths [nm] for pixel values 0-1072
    wavelength_data = funcEUV.getWavelength(shot_pos)

    # process image function returns the corrected/processed image 
    processed_image = funcEUV.processImage(shotnum)

    # extact lineouts of MCP strips 
    strip1, strip2, strip3 = funcEUV.getLineouts(processed_image)
    
    #-
    ## automated line matching/identification
    
    # Note: Either the get_lines_threshold or get_lines_derivative should be used, but not both at the same time. The former is a method for extracting lines in a spectrum using a noise/thresholding approach, whereas the latter is using a thresholding based on the derivative of the spectrum. Use whichever produces the best looking identification. 
    # Currently, I think the derivative method performs a lot better than the simple thresholding method. 
    
    # identify lines using threshold method 
    #line1_lines,line2_lines,line3_lines = get_lines_threshold(shotnum,line1,line2,line3,mov_avg1,mov_avg2,mov_avg3,wavelength_data)
    
    # identify lines using derivative method 
    #line1_lines,line2_lines,line3_lines, line1_spectrum,line2_spectrum,line3_spectrum = funcEUV.get_lines(shotnum, strip1,strip2,strip3,wavelength_data)


    # Query NIST database for various elements within wavelength range (currently set to carbon, oxygen, and silicon, tungsten, and copper)
    #tableC, tableO, tableSi, tableW, tableCu, tableCr, tableZr, tableB, tableN, tableAl = funcEUV.queryNist(wavelength_data)
    
    
    # Assemble combined table of spectral lines from various elements of interest

    # Uncomment below for the nine lightest elements 
    #tableZ = [tableB,tableC,tableN,tableO,tableAl,tableSi,tableCr,tableCu,tableZr]
    # Uncomment below for the six lightest elements being looked at
    #tableZ = [tableB,tableC,tableN,tableO,tableAl,tableSi]
    # Uncomment below for four lightest elements being looked at
    #tableZ = [tableB,tableC,tableN,tableO]
    # Uncomment below for just carbon and oxygen lines 
    #tableZ = [tableO, tableC]
    #tableZ = [tableO]
    
    # Match EUV spectral lines for each MCP strip to NIST database lines 
    line1_store = []
    line2_store = []
    line3_store = []
    # For each element spectral line database, find matching lines 
    '''
    for element in tableZ:
       line1, line2, line3, lambda_thres = funcEUV.linematch(line1_lines,line2_lines,line3_lines,element)
       line1_store.append(line1)
       line2_store.append(line2)
       line3_store.append(line3)
    '''
    #-
    #---
    

    #---
    ## Data plotting functions    
    # these functions are plotting functions to visualize the data

    pltEUV.plot_lineouts_raw(shotnum,wavelength_data, strip1,strip2,strip3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width)
    
    #pltEUV.plot_lineouts(shotnum,line1_spectrum,line2_spectrum,line3_spectrum,strip1,strip2,strip3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width)
    
    #pltEUV.plot_line_ident(shotnum,line1_lines,line2_lines,line3_lines, line1_spectrum,line2_spectrum,line3_spectrum,strip1,strip2,strip3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width)
    
    #pltEUV.plot_line_lineouts(shotnum,line1_store,line2_store,line3_store,line1_spectrum,line2_spectrum,line3_spectrum,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width)
  
    # early "start" - needs work
    #pltEUV.plot_linefit(strip1,strip2,strip3, wavelength_data)
    
    #---
    
    
    
    
    
    