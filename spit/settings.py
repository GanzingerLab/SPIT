# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 12:05:03 2025

@author: castrolinares
"""


import os
import shutil
import csv
import json
import traceback
from glob import glob
from multiprocessing import freeze_support
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import imageio
import yaml
from tqdm import tqdm


import largestinteriorrectangle as lir
from picasso.io import load_movie, save_info
from picasso.localize import (
    get_spots,
    identify_async,
    identifications_from_futures,
)
import picasso.gausslq as gausslq
import picasso.avgroi as avgroi


from spit import tools
from spit import localize
from spit import linking as link
from spit import table
from spit import plot_diffusion
from spit import colocalize as coloc



import os
import numpy as np



class Settings:
    def __init__(self, registration_cls, localization_cls, link_cls, coloc_tracks_cls, coloc_locs_cls):
        self.registration_settings = registration_cls()
        self.localization_settings = localization_cls()
        self.link_settings = link_cls()
        self.coloc_tracks_settings = coloc_tracks_cls()
        self.coloc_spots_settings = coloc_locs_cls()
        self.microscope = self.registration_settings.microscope
    def load_H_left_annapurna(self):
        path = os.path.join(self.registration_settings.registration_folder, 'H_left_Ann.csv')
        return np.loadtxt(path, delimiter=",", dtype=float)
    
    def load_H_right_annapurna(self):
        path = os.path.join(self.registration_settings.registration_folder, 'H_right_Ann.csv')
        return np.loadtxt(path, delimiter=",", dtype=float)
    
    def load_crop_annapurna(self):
        path = os.path.join(self.registration_settings.registration_folder, 'crop_A_1100.csv')
        crop_loc = np.loadtxt(path, delimiter=",", dtype=int)
        return [crop_loc[0], crop_loc[1]], [crop_loc[2], crop_loc[3]]
    
    def load_H_left_K2(self):
        path = os.path.join(self.registration_settings.registration_folder, 'H_left_K2.csv')
        return np.loadtxt(path, delimiter=",", dtype=float)
    
    def load_H_right_K2(self):
        path = os.path.join(self.registration_settings.registration_folder, 'H_right_K2.csv')
        return np.loadtxt(path, delimiter=",", dtype=float)
    
    def load_crop_K2(self):
        path = os.path.join(self.registration_settings.registration_folder, 'K2-BIVOUAC_lims.csv')
        crop_loc = np.loadtxt(path, delimiter=",", dtype=int)
        return [crop_loc[0], crop_loc[1]], [crop_loc[2], crop_loc[3]]
    def gradient(self, filename):
        if '488nm' in filename:
            return self.localization_settings.gradient488
        elif '561nm' in filename:
            return self.localization_settings.gradient561
        elif '638nm' in filename:
            return self.localization_settings.gradient638
        elif '405nm' in filename:
            return self.localization_settings.gradient405
    
    def get_naclib(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        result_txt  = tools.read_result_file('\\'.join(file.split('\\')[:-1])+'\\'+file.split('\\')[-2]+'_result.txt') #this opens the results.txt file to check the microscope used. 
        root = os.path.dirname(__file__)
        paramfiles_path = os.path.join(root, "Registration_folder/") #This sets the folder where the naclib coefficients are. 
        #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_Ann.csv'))
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_K2.csv'))
    def get_px2um(self, file):
        result_txt  = tools.read_result_file('\\'.join(file.split('\\')[:-1]) + '\\' + 
                                             [element for element in file.split('\\') if element.startswith('Run')][0] + '_result.txt') #this opens the results.txt file to check the microscope used. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 0.09
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 0.108
    def get_px2nm(self, file):
        result_txt  = tools.read_result_file('\\'.join(file.split('\\')[:-1]) + '\\' + 
                                             [element for element in file.split('\\') if element.startswith('Run')][0] + '_result.txt') #this opens the results.txt file to check the microscope used. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
    
    def gradient_tiffs(self, filename): #function to get the correct gradient later on the code. 
        if 'l_ch.tif' in filename:
            return self.localization_settings.gradient_l
        elif 'm_ch.tif' in filename:
            return self.localization_settings.gradient_m
        elif 'r_ch.tif' in filename:
            return self.localization_settings.gradient_r
    
    def get_naclib_tiffs(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        root = os.path.dirname(__file__)
        paramfiles_path = os.path.join(root, "Registration_folder/") #This sets the folder where the naclib coefficients are. 
        #It should be in a folder called paramfile inside the folder where the script is located. 
        if self.microscope == 'ANNAPURNA': 
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_Ann.csv'))
        elif self.microscope == 'K2-BIVOUAC':
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_K2.csv'))
    
    def get_px2um_tiffs(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
               #It should be in a folder called paramfile inside the folder where the script is located. 
        if self.microscope == 'ANNAPURNA': 
            return 0.09
        elif self.microscope == 'K2':
            return 0.108
    def get_px2nm_tiffs(self): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        if self.microscope == 'ANNAPURNA': 
            return 90.16
        elif self.microscope == 'K2':
            return 108
    
