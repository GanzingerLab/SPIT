# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:40:42 2025

@author: castrolinares
"""
from spit import settings 
from spit.SPIT import SPIT_Run, SPIT_Dataset, localize_tiff_run, localize_tiff_dataset

class RegistrationSettings:
    def __init__(self):
        self.registration_folder = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\regis' #Folder containing the H-matrices and crop coordinates needed for the alignment of the channels
        self.verticalROI = [0, 1100] #Specify the heigth of the channels that you want to use. I am pretty sure that if we set it larger, it still does it correctly. 
        self.to_keep = [0, None] #number of frames to procees. 
        #[0, None] means all frames, if you want to change it, set the specific number ([0:200] would be the first 200 frames 
        #while [300:None] would be from frame 301 until the end)
        
        self.ch_width_annapurna = 640 #The width of the channels in pixels. 
        self.x_coords_annapurna = { # Define start and end X coordinate for each channel. 
            '638nm': (18, 18 + self.ch_width_annapurna),
            '561nm': (691, 691 + self.ch_width_annapurna),
            '488nm': (1370, 1370 + self.ch_width_annapurna),
            '405nm': (1370, 1370 + self.ch_width_annapurna),
        }
        
        self.ch_width_K2 = 560 #The width of the channels in pixels. 
        self.x_coords_K2 = { # Define start and end X coordinate for each channel.
            '638nm': (60, 60 + self.ch_width_K2),
            '561nm': (740, 740 + self.ch_width_K2),
            '488nm': (1415, 1415 + self.ch_width_K2),
            '405nm': (1415, 1415 + self.ch_width_K2),
        }
       
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.channels = ['m_ch', 'l_ch'] # list with: l_ch = left, m_ch = middle, r_ch=right
        self.microscope = 'K2'  #or ANNAPURNA
        #ANNAPURNA
        self.ch_width_annapurna_tiff = 640   #The width of the channels in pixels. 
        self.x_coords_annapurna_tiff = {     # Define start and end site for each channel. 
            'l_ch': (18, 18 + self.ch_width_annapurna),  
            'm_ch': (691, 691 + self.ch_width_annapurna),  
            'r_ch': (1370, 1370 + self.ch_width_annapurna)
        }  
        
        #K2
        self.ch_width_K2_tiff = 560   
        self.x_coords_K2_tiff = { #K2
            'l_ch': (60, 60 + self.ch_width_K2),  
            'm_ch': (740, 740 + self.ch_width_K2),  
            'r_ch': (1415, 1415 + self.ch_width_K2)
        }
        ######################################################
    
class LocalizationSettings:
    def __init__(self):
        self.box = 7 #same as in picasso localize
        self.gradient405 = 300 #same as in picasso localize
        self.gradient488 = 1000 #same as in picasso localize
        self.gradient561 = 330 #same as in picasso localize
        self.gradient638 = 1000 #same as in picasso localize
        
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.gradient_l = 500    #same as in picasso localize
        self.gradient_m = 500   #same as in picasso localize 
        self.gradient_r = 350   #same as in picasso localize
        ######################################################
        
        #'com' for stuff that moves too fast and does not look like a gaussian spot, 'lq' for gaussian spots. 
        self.fit_method = 'lq' 
        self.camera_info = {
            'Gain': 1,   #Set the gain of the microscope. 
            'Baseline': 100, #Baseline is the average dark camera count
            'Sensitivity': 0.6,  #Sensitivity is the conversion factor (electrons per analog-to-digital (A/D) count)
            'qe': 0.9  #In Picasso qe (quantum efficiency) is not used anymore. It is left for bacward compatibility. 
        }
        self.fit_method = 'lq' #'com' for stuff that moves too fast and does not look like a gaussian spot, 'lq' for gaussian spots. 
        self.skip = 'not_track'
        self.suffix = '' #sufix for the name of the file, if necessary.
        self.transform = False #Do non-affine corrections of the localized spots if you have multiple channels.
        self.plot = True 

class LinkSettings:
    def __init__(self):
        self.coloc = False #Bool --> are you using a colocalization file or not? 
        self.dt = None  #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        self.quick = False #use the quick version? 200px squared from the center and only 500 frames. 
        self.roi = True #Do you want to filter per ROI? 
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.fil_len = 20 #filter the length of the tracks. Tracks shorter than this ammount of frames will be filtered
        self.fil_diff = 0.0002 #Filter for immobile particles. Tracks with a diffusion coefficient smaller than this number will be filtered
        self.tracker = 'trackpy' #Tracker algorithm to use: trackpy or swift. After talking with Chris, swift is very complicated and the focus of the developers is not 
        # really tracking, but diffusion rates. So, swift is not implemented, and I am not sure if it will. 
        self.memory = 1 #max number of frames from which a particle can disappear 
        self.search = 15#max search range for trackpy linking in px 

class ColocTracksSettings:
    def __init__(self):
        self.ch0 = '561' #reference channel
        self.ch1 = '638' #to compare with ch0 
        self.th = 300 #maximum distanc considered
        self.min_overlapped_frames = 5 #minimum amunt of frames in which the spots of the tracks have to be closer than the threshold distance set in th in a row.
        self.min_len_track = 5  #minumum length of a track to be considered for colocalization
        self.suffix = '' #sufix for the name of the file, if necessary.
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 

class ColocLocsSettings:
    def __init__(self):
        self.ch0 = '561'#'561nm'  
        self.ch1 = '638' #'638nm' 
        
        #####USED ONLY FOR AFFINE TRANSFORM OF TIFF FILES#####
        self.ch0_tiffs = 'l_ch'#'561nm'  
        self.ch1_tiffs = 'm_ch' #'638nm'  
        ######################################################
        
        self.th = 250 #Threshold distance to consider colocalization in nm. Default by Chris: 250
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        
settings = settings.Settings(RegistrationSettings, LocalizationSettings, LinkSettings, ColocTracksSettings, ColocLocsSettings)
#%%
#In this new version of SPIT, please run SPIT commands under an if __name__ == "__main__": 
                    #It is veri important to avoid problems with multiprocessing (during linking). If you, in the same script start analyzing
                    #other stuff from the ouput, then you can stop using the clause.
#This new versio uses object oriented programming to share data and simplify the analysis process. There are four types of objects:
    #SPIT_Run --> used to run SPIT in a single run folder.
    #SPIT_Dataset --> used to run SPIT in all the folders within a folder (a dataset). All the data will then be analized with the same 
    #parameters, as one should do to be consitent. 
    #localize_tiff_run --> to process snapshots in tif format from K2 or Annapurna. 
    #localize_tiff_datset --> same idea as SPIT_Dataset, but for snapshots in tif format from K2 or Annapurna. 
#To initialize one of this you do (first check the setting and run the setting cell, you will pass the settings to the object):
if __name__ == "__main__":
    #spit_run = SPIT_Run(folder to analyze, settings, folder to save the output)
    #it is the same for localize_tiff_run. Example:
    spit_run = (r'D:\Data\test_error_result_files\chi\Run00002', settings, r'D:\Data\test_error_result_files')
    #spit_dataset = (folder to analyze, settings)
    #it is the same for localize_tiff_dataset. Example:
    spit_datset = SPIT_Dataset(r'D:\Data\test_error_result_files', settings)
    #then you can run different command which names I hope are self-explanatory:
    # for SPIT_Run:
    spit_run.affine_transform()
    spit_run.localize()
    spit_run.roi() #to use this one, after affine_transform(), please open the images with imageJ and using the freehand tool, 
        #draw ROIs and saved them with name with the format: roiX.roi where X starts at 0 and goes up (to as many ROIs as you have (minus one))
    spit_run.link()
    spit_run.coloc_tracks()
    spit_run.coloc_spots()
    spit_run.full_analysis_noROI(mode = 'tracks')  #Does the full analysis without considering ROIs colocalizing . Colocalizes full tracks. 
        #if mode = 'spots' it will colocalize first the spots and then link (it will run link twice, once for the separated channels and once for the coloc channel)
    spit_run.full_analysis_ROI(mode = 'tracks') #does the same as the above but uses ROIs. For that, you first have to run .affine_transform()
        #then make the ROIs, and then run this command (which will start by localize)
    
    
    #SPIT_Datset has more or less the same commands, but it does each function per folder. 
    spit_datset.affine_transform() #this will give all the separated channels in the output folder (which will be just inside of the 
        #folder to analyze) for each of the subfolders containing .raw files. 
    spit_datset.localize() #same as above but with colocalize. 
    spit_datset.roi() #to use this one, after affine_transform(), please open the images with imageJ and using the freehand tool, 
        #draw ROIs and saved them with name with the format: roiX.roi where X starts at 0 and goes up (to as many ROIs as you have (minus one))
    spit_datset.link()
    spit_datset.coloc_tracks()
    spit_datset.coloc_spots()
    spit_datset.SPIT_noROI(mode = 'tracks')  #Does the full analysis without considering ROIs colocalizing . Colocalizes full tracks.
        #if mode = 'spots' it will colocalize first the spots and then link (it will run link twice, once for the separated channels and once for the coloc channel)
    spit_datset.SPIT_ROI(mode = 'tracks') #does the same as the above but uses ROIs. For that, you first have to run .affine_transform()
        #then make the ROIs, and then run this command (which will start by localize)
    
    
    #Similarly, localize_tiff_run and localize_tiff_dataset have the functions (both have the same function names, but one does it for 
    #one folder, the other for the whole datset):
    #.affine_transform()
    #.localize()
    #.roi()
    #.colocalize() --> colocalizes spots
    #.full_analysis_noROI()
    #.full_analysis_ROI()


#%% EXAMPLE
if __name__ == "__main__":
    test = SPIT_Dataset(r'D:\Data\test_error_result_files', settings)
    test.affine_transform()
    #I make the ROIs files and then
    test.SPIT_ROI()  #simply like this uses mode = 'tracks' as default. 
#%% EXAMPLE 2
if __name__ == "__main__":
    test2 = localize_tiff_dataset(r'D:\Data\test_error_result_files\snaps', settings)
    test2.full_analysis_noROI()
#%%EXAMPLE 3
# I am testing gradient values, so I run it in a single folder
if __name__ == "__main__":
    test3 = SPIT_Run(r'D:\Data\test_error_result_files\Run00007', settings, r'D:\Data\test_error_result_files')
    test3.affine_transform()
#now that I have the separated channels, I can make a new cell:
#%%
#and under the __name__ == "__main__" clause again, I can try different gradient values:
if __name__ == "__main__":
    test3.localize()  #this will do it with the settings I have above. 
#%%
# But I can use this objects to more easily try differnt things, like modifying settings like the gardient values used to detect spots:
if __name__ == "__main__":
    original561 = settings.localization_settings.gradient561
    original638 = settings.localization_settings.gradient638 
    for i in range(300, 800, 100):
        settings.localization_settings.gradient561 = i
        settings.localization_settings.gradient638 = i  
        settings.localization_settings.suffix = f"-gradient-{i}"
        test4 = SPIT_Run(r'D:\Data\test_error_result_files\Run00007', settings, r'D:\Data\test_error_result_files')
        test4.affine_transform()
        test4.localize()
    settings.localization_settings.gradient561 = original561 #I fo this so the values match what was set on the written setting
    settings.localization_settings.gradient638 = original638
        #if you try to continue with the analysis for all this output, SPIT only will do it for the first folders it finds, so you have to 
        #reorder or check which one you prefer from these files. 
#%%
if __name__ == "__main__":
    test4 = SPIT_Run(r'D:\Data\test_error_result_files\Run00007', settings, r'D:\Data\test_error_result_files')
    test4.coloc_spots()
    