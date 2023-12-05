"""
EUV Spectroscopy Processing 
@author: Aidan Klemmer & Stephan Fuelling & Derek Sutherland
aidan@nevada.unr.edu
12/5/23
University of Nevada, Reno
and
Zap Energy Inc

File 5
"""

# Calibration parameters for EUV spectrometer


def getCal(position):
    
    M = 3.028 # magnification parameter
      
    # historical data
    '''
    x00=np.array([0.006297119,0.006100908,-0.003381397,0.000561561,-0.003819074,0.006,-0.003819074,-0.003819074,-0.003819074,-0.003819074]) # [inches]
    gam=np.array([-0.030718329,-0.036544725,-0.036209325,-0.042160846,-0.045871025,-0.004877428,-0.004877428,-0.004877428,-0.030718329,-0.030718329]) # MCP tilt correction [radians]
    #x00=np.array([-0.0780,-0.0827,-0.0503, -0.0387,-0.0464,-0.0464,-0.0464,-0.0464,-0.0464,-0.0464]) # [inches]
    #gam=np.array([-0.0133, -0.0226, -0.0205, -0.0180, -0.0260, -0.0167, -0.0167, -0.0167, -0.0167, -0.0167]) # MCP tilt correction [radians]
    '''
    
    ## New calibration params for Nov 2022 dataset
    
    # this calibration may not be accurate for future datasets
    # user input will be needed at the start. future work should include a more complete dataset.
    
    if position <= 5700:
        x0 = 0.0003886778952847507
        gam = 3.131299620076861

    if position <= 6250:
        x0 = -0.00952914839115815
        gam = 3.1288837223576014

    if position <= 6750:
        x0 = -0.015315644802472139
        gam = 3.1281513111137467

    if position <= 7250:  
        x0 = -0.021444329156998954
        gam = 3.126856626019882

    if position <= 8125:
        x0 = -0.027755952208282926
        gam = 3.1247473486009514
           
    if position <= 8800:
        x0 = -0.03365557547564084
        gam = 3.122319881203102   
        
    return x0, gam, M
        
        
        
    
