# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:40:42 2025

@author: castrolinares
"""
from spit import settings 
from spit.SPIT import SPIT_Run, SPIT_Dataset

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
        
        self.ch_width_K2 = 560
        self.x_coords_K2 = {
            '638nm': (60, 60 + self.ch_width_K2),
            '561nm': (740, 740 + self.ch_width_K2),
            '488nm': (1415, 1415 + self.ch_width_K2),
            '405nm': (1415, 1415 + self.ch_width_K2),
        }
    
class LocalizationSettings:
    def __init__(self):
        self.box = 7
        self.gradient405 = 300
        self.gradient488 = 1000
        self.gradient561 = 330
        self.gradient638 = 1000
        self.camera_info = {
            'Gain': 1,
            'Baseline': 100,
            'Sensitivity': 0.6,
            'qe': 0.9
        }
        self.fit_method = 'lq'
        self.skip = 'not_track'
        self.suffix = ''
        self.transform = False
        self.plot = True 

class LinkSettings:
    def __init__(self):
        self.coloc = False
        self.dt = None
        self.quick = False
        self.roi = True
        self.suffix = ''
        self.fil_len = 20
        self.fil_diff = 0.0002
        self.tracker = 'trackpy'
        self.memory = 1
        self.search = 15

class ColocTracksSettings:
    def __init__(self):
        self.ch0 = '488'
        self.ch1 = '638'
        self.th = 300
        self.min_overlapped_frames = 5
        self.min_len_track = 5
        self.suffix = ''
        self.dt = None

class ColocLocsSettings:
    def __init__(self):
        self.ch0 = '488'#'561nm'  
        self.ch1 = '638' #'638nm'  
        self.th = 250 #Threshold distance to consider colocalization in nm. Default by Chris: 250
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        
settings = settings.Settings(RegistrationSettings, LocalizationSettings, LinkSettings, ColocTracksSettings, ColocLocsSettings)
#%%
if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.freeze_support()  # only needed if you're ever freezing to .exe
    a = SPIT_Run(r'D:\Data\test_error_result_files\chi\Run00002', settings, r'D:\Data\test_error_result_files')
    # a.affine_transform()
    # a.localize()
    # locs = a.locs
    # a.roi()
    # locs_roi = a.locs_roi
    a.link()
    # tracks_coloc = a.tracks
    # tracks_coloc_stats = a.tracks_stats
    # a.coloc_tracks()
    # a.full_analysis_noROI()
#%%
if __name__ == "__main__":
    b = SPIT_Dataset(r'D:\Data\test_error_result_files', settings)
    b.affine_transform()