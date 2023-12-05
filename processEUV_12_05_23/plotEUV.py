"""
EUV Spectroscopy Processing 
@author: Aidan Klemmer & Stephan Fuelling & Derek Sutherland
aidan@nevada.unr.edu
12/5/23
University of Nevada, Reno
and
Zap Energy Inc

File 4
"""


## plotting functions for EUV

import numpy as np
import matplotlib.pyplot as plt;
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def plot_lineouts_raw(shotnum,wavelength_data, line1,line2,line3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width):

    fig1, (ax1) = plt.subplots(1,1,figsize=(12,6),sharey=True)
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=12)
    major_ticks = np.arange(round(wavelength_data[0])-0.5,round(wavelength_data[1071])+0.5,1)
    minor_ticks = np.arange(round(wavelength_data[0])-0.5,round(wavelength_data[1071])+0.5,0.25)
    
    # Plot unsmoothed data
    ax1.plot(wavelength_data, line1[0:1072], color='blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width),linewidth=0.7)
    ax1.plot(wavelength_data, line2[0:1072], color='red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width),linewidth=0.7)
    ax1.plot(wavelength_data, line3[0:1072], color='green', label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width),linewidth=0.7)

    ax1.legend(loc='upper right',fontsize=8)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(alpha=0.8, which = 'major')
    ax1.grid(alpha=0.3, which = 'minor')
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Intensity [Arb.]')

    fig1.set_dpi(800)


def plot_lineouts(shotnum,line1_spectrum,line2_spectrum,line3_spectrum,line1,line2,line3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width):
# Plot Spectrum1D objects to ensure everything look right 
    fig2, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,6))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=10)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    ax1.set(ylabel = 'Wavelength [{}]'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = 0, right = max(line1_spectrum.flux)*1.2)
    ax1.legend(loc='upper right',fontsize=6)
    ax1.grid(True)

    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    ax2.set(ylabel = 'Wavelength [{}]'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = 0, right = max(line2_spectrum.flux)*1.2)
    ax2.legend(loc='upper right',fontsize=6)
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    ax3.set(ylabel = 'Wavelength [{}]'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = 0, right = max(line3_spectrum.flux)*1.2)
    ax3.legend(loc='upper right',fontsize=6)
    ax3.grid(True)
    fig2.set_dpi(800)


def plot_line_ident(shotnum,line1_lines,line2_lines,line3_lines, line1_spectrum,line2_spectrum,line3_spectrum,line1,line2,line3,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width):
# Plot the emission spectrum with found spectal lines to assess performance 
    fig3, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,6))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=10)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    for iter in np.arange(0,len(line1_lines),1):
        ax1.axhline(line1_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax1.set(ylabel = 'Wavelength [{}]'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = 0, right = max(line1_spectrum.flux)*1.2)
    ax1.legend(loc='upper right',fontsize=6)
    ax1.grid(True)

    
    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    for iter in np.arange(0,len(line2_lines),1):
        ax2.axhline(line2_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax2.set(ylabel = 'Wavelength [{}]'.format(line2_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = 0, right = max(line2_spectrum.flux)*1.2)
    ax2.legend(loc='upper right',fontsize=6)
    ax2.grid(True)
    
    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green',label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    for iter in np.arange(0,len(line3_lines),1):
        ax3.axhline(line3_lines[iter][0].value,linestyle = '--', linewidth = 0.9,color = 'black')
    ax3.set(ylabel = 'Wavelength [{}]'.format(line3_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = 0, right = max(line3_spectrum.flux)*1.2)
    ax3.legend(loc='upper right',fontsize=6)
    ax3.grid(True)
    fig3.set_dpi(800)


def plot_line_lineouts(shotnum,line1_store,line2_store,line3_store,line1_spectrum,line2_spectrum,line3_spectrum,p1_timing,p1_width,p2_timing,p2_width,p3_timing,p3_width):
    # Invert this later for easier reading of the spectral information. 
     # Plot the emission spectrum with found spectal lines to assess performance 
    fig4, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,6))
    plt.suptitle('EUV Spectra Lineouts FuZE-Q Shot %s' %(shotnum),fontsize=10)
    ax1.plot(line1_spectrum.flux,line1_spectrum.spectral_axis,linewidth = 0.9, color = 'blue', label='MCP Strip #1: @ %.1f us with %.1f us exposure' %(p1_timing, p1_width))
    for element in line1_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                ax1.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                ax1.text(0.5*np.max(line1_spectrum.flux),float(element[iter][1])+0.05,'%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax1.set(ylabel = 'Wavelength [{}]'.format(line1_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax1.set_xlim(left = 0, right = max(line1_spectrum.flux)*1.2)
    ax1.legend(loc='upper right',fontsize=6)
    ax1.grid(True)

    ax2.plot(line2_spectrum.flux,line2_spectrum.spectral_axis,linewidth = 0.9, color = 'red', label='MCP Strip #2: @ %.1f us with %.1f us exposure' %(p2_timing, p2_width))
    for element in line2_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                ax2.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                ax2.text(0.5*np.max(line2_spectrum.flux),float(element[iter][1])+0.05,'%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax2.set(ylabel = 'Wavelength [{}]'.format(line2_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax2.set_xlim(left = 0, right = max(line2_spectrum.flux)*1.2)
    ax2.legend(loc='upper right',fontsize=6)
    ax2.grid(True)

    ax3.plot(line3_spectrum.flux,line3_spectrum.spectral_axis,linewidth = 0.9, color = 'green', label='MCP Strip #3: @ %.1f us with %.1f us exposure' %(p3_timing, p3_width))
    for element in line3_store:
        if len(element) != 0:
            for iter in np.arange(0,len(element),1):
                ax3.axhline(float(element[iter][1]),linestyle = '--', linewidth = 0.9,color = 'black')
                ax3.text(0.5*np.max(line3_spectrum.flux),float(element[iter][1])+0.05, '%s -- %s nm' %(element[iter][0],element[iter][1]),rotation='horizontal',fontsize=8)
    ax3.set(ylabel = 'Wavelength [{}]'.format(line3_spectrum.spectral_axis.unit), xlabel = 'Intensity [Arb.]')
    ax3.set_xlim(left = 0, right = max(line3_spectrum.flux)*1.2)
    ax3.legend(loc='upper right',fontsize=6)
    ax3.grid(True)
    fig4.set_dpi(800)



def plot_linefit(line1, line2, line3, wavelength_data):
    
    
    # Function to fit the peaks using a Gaussian shape
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
    # Function to fit the peaks using a Lorentzian shape
    def lorentzian(x, A, mu, gamma):
        return A / (1 + ((x - mu) / gamma)**2)
    
    # Function to fit the peaks using a Voigt profile
    def voigt(x, A, mu, sigma, gamma):
        return A * np.real(np.exp(-(x - mu + 1j*gamma)**2 / (2 * sigma**2)))
    
    # Find the peaks in the spectra
    peaks1, _ = find_peaks(line1, height=250)
    peaks2, _ = find_peaks(line2, height=3000)
    peaks3, _ = find_peaks(line3, height=1500)
    
    # Plot the original spectra
    fig5, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(5,7))

    
    ax1.plot(wavelength_data, line1[0:1072], color='black', label='MCP Strip #1')
    ax2.plot(wavelength_data, line2[0:1072], color='black', label='MCP Strip #2')
    ax3.plot(wavelength_data, line3[0:1072], color='black', label='MCP Strip #3')
    
    # Fit the strip 1 peaks
    for peak1 in peaks1:
        start = max(0, peak1 - 50)
        end = min(len(wavelength_data), peak1 + 50)
        xdata = wavelength_data[start:end]
        ydata = line1[start:end]
        
        #fit gaussian
        popt_gaus1, _ = curve_fit(gaussian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak1], 0.1])
        #print(*popt_gaus1)
        ax1.plot(xdata, gaussian(xdata, *popt_gaus1), 'b-', label='Gaussian Fit - Strip #1')
        #fit lorentzian
        popt_lor1, _ = curve_fit(lorentzian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak1], 0.1])
        ax1.plot(xdata, lorentzian(xdata, *popt_lor1), 'c-', label='Lorentzian Fit - Strip #1')
        #fit voigt
        #popt_voigt1, _ = curve_fit(voigt, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak1], 10, 10])
        #ax1.plot(xdata, voigt(xdata, *popt_voigt1), 'c-', label='Voigt Fit - Strip #1')
        
    
    # Fit the strip 2 peaks
    for peak2 in peaks2:
        start = max(0, peak2 - 50)
        end = min(len(wavelength_data), peak2 + 50)
        xdata = wavelength_data[start:end]
        ydata = line2[start:end]
        
        #fit gaussian
        popt_gaus2, _ = curve_fit(gaussian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak2], 0.1])
        ax2.plot(xdata, gaussian(xdata, *popt_gaus2), 'b-', label='Gaussian Fit - Strip #2')
        #fit lorentzian
        
        popt_lor2, _ = curve_fit(lorentzian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak2], 0.1])
        ax2.plot(xdata, lorentzian(xdata, *popt_lor2), 'c-', label='Lorentzian Fit - Strip #2')
        #fit voigt
        #popt_voigt2, _ = curve_fit(voigt, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak2], 10, 10])
        #ax2.plot(xdata, voigt(xdata, *popt_voigt2), 'c-', label='Voigt Fit - Strip #2')
        
    
    # Fit the strip 3 peaks
    for peak3 in peaks3:
        start = max(0, peak3 - 50)
        end = min(len(wavelength_data), peak3 + 50)
        xdata = wavelength_data[start:end]
        ydata = line3[start:end]

        #fit gaussian
        popt_gaus3, _ = curve_fit(gaussian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak3], 0.1])
        ax3.plot(xdata, gaussian(xdata, *popt_gaus3), 'b-', label='Gaussian Fit - Strip #3')
        
        #fit lorentzian
        popt_lor3, _ = curve_fit(lorentzian, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak3], 0.1])
        ax3.plot(xdata, lorentzian(xdata, *popt_lor3), 'c-', label='Lorentzian Fit - Strip #3')
        #fit voigt
        #popt_voigt3, _ = curve_fit(voigt, xdata, ydata, p0=[np.max(ydata), wavelength_data[peak3], 10, 10])
        #ax3.plot(xdata, voigt(xdata, *popt_voigt3), 'c-', label='Voigt Fit - Strip #3')
        
    #gaus_fwhm_1 = popt_gaus1[2]*2.355 #sigma to fwhm
    #gaus_fwhm_2 = popt_gaus2[2]*2.355
    #gaus_fwhm_3 = popt_gaus3[2]*2.355
    
    #ax1.text(57.5, 9000, 'FWHM Gaussian fit: %.3f nm' %(gaus_fwhm_1), fontsize=7)
    #ax2.text(57.5, 10000, 'FWHM Gaussian fit: %.3f nm' %(gaus_fwhm_2), fontsize=7)
    #ax3.text(57.5, 6000, 'FWHM Gaussian fit: %.3f nm' %(-gaus_fwhm_3), fontsize=7)

    ax1.set_ylabel('Counts [arb. units]', fontsize=8)
    ax2.set_ylabel('Counts [arb. units]', fontsize=8)
    ax3.set_ylabel('Counts [arb. units]', fontsize=8)
    ax3.set_xlabel('Wavelength [nm]', fontsize=8)

    ax1.legend(loc=2,fontsize=6)
    ax2.legend(loc=2,fontsize=6)
    ax3.legend(loc=2,fontsize=6)
    
    #ax1.set_xlim(57, 59)
    #ax2.set_xlim(57, 59)
    #ax3.set_xlim(57, 59)
    
    fig5.set_dpi(800)


    