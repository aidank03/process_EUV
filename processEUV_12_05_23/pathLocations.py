"""
EUV Spectroscopy Processing 
@author: Aidan Klemmer & Stephan Fuelling & Derek Sutherland
aidan@nevada.unr.edu
12/5/23
University of Nevada, Reno
and
Zap Energy Inc

File 2
"""

# File path *.fit file for a particular shot number shotnum 
path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_11_27_23\\'
#path = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/231127/'       

# File path *.fit file for a particular dark frame
darkframe_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_11_27_23\\'
#darkframe_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/Dark_Frames/

# Dark frame file name
dark_pic_name = r'231127013.fit'
#dark_pic_name = r'5sec_EUV_background_new_0deg.fit'

# Gaussian spread function path
gaussian_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\'
#gaussian_path = r'/Applications/processEUV/'
gaussian_name = r'Exponential2048-55.fits'

# Shot information
shotinfo_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\'
#shotinfo_path = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/' # Path to text file containing desired information
#shotinfo_path = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/230315/' # Path to text file containing desired information
#shotinfo_path = r'/home/daszap/projects/EUV/EUV_Code/' # Path to text file containing desired information
shotinfo_name = r'EUV_refdata_11_23.txt'
#shotinfo_name = r'EUV_refdata_11_22.txt'
#shotinfo_name = r'EUV_refdata_23_03_15.txt'


#shotinfo_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_refdata_11_22.txt' # Path to text file containing desired information
#shotinfo_filepath = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/EUV_Data/230315/EUV_refdata_23_03_15.txt' # Path to text file containing desired information
#shotinfo_filepath = r'/home/daszap/projects/EUV/EUV_Code/EUV_refdata_11_22.txt' # Path to text file containing desired information


save_results_path = 'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\EUV_CCD\\'

## filepath to shot number list .txt file
# future work should point this towards EUV datasets in data pipeline
shotlist_path = r'G:\\Zap Energy\\EXT-UNR\\EUV Spectroscopy\\EUV Data\\'
#shotlist_path = r'/Users/Aidanklemmer/Desktop/HAWK/ZEI/'
#shotlist_filepath = r'/home/daszap/projects/EUV/EUV_Code/'

shotlist_name = r'shot_numbers.txt'




