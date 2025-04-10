# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:28:54 2025

@author: castrolinares
"""

import numpy as np
# import registration_functions as reg
import os
import imageio
import yaml as _yaml
from glob import glob
import shutil
import csv
from matplotlib import pyplot as plt
# from spit import tools
import tifffile as tiff
from scipy.ndimage import affine_transform

###
#used new package to detect the largest rectangle after alginment (removing edges) that speeds up the process. Only needed for module 2. 
import largestinteriorrectangle as lir


#Setings for the alignment. Please check everything is correct. 
class Settings:
    def __init__(self):
        #General
        self.registration_folder = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\regis' #Folder containing the H-matrices and crop coordinates
        #needed for the alignment of the channels
        self.channels = ['m_ch', 'l_ch'] # list with: l_ch = left, m_ch = middle, r_ch=right
        self.verticalROI = [0, 1100]  #number of frames to procees. [0, None] means all frames, if you want to change it, set the specific number ([0:200] would be the first 
        #200 frames while [300:None] would be from frame 301 until the end)
        
        #ANNAPURNA
        self.ch_width_annapurna = 640   #The width of the channels in pixels. 
        self.x_coords_annapurna = {     # Define start and end site for each channel. 
            'l_ch': (18, 18 + self.ch_width_annapurna),  
            'm_ch': (691, 691 + self.ch_width_annapurna),  
            'r_ch': (1370, 1370 + self.ch_width_annapurna)
        }  
        
        #K2
        self.ch_width_K2 = 560   
        self.x_coords_K2 = { #K2
            'l_ch': (60, 60 + self.ch_width_K2),  
            'm_ch': (740, 740 + self.ch_width_K2),  
            'r_ch': (1415, 1415 + self.ch_width_K2)
        }
        self.microscope = 'K2'
    #load registration files: affine-correction matrices and cropping coordinates. Folder is the registration set above. The name after that should be the file location. 
    #Annapurna
    def load_H_left_annapurna(self):
        H_left_path = os.path.join(self.registration_folder, 'H_left_Ann.csv')
        Hl = np.loadtxt(H_left_path, delimiter=",", dtype=float)
        return Hl
    def load_H_right_annapurna(self):
        H_right_path = os.path.join(self.registration_folder, 'H_right_Ann.csv')
        Hr = np.loadtxt(H_right_path, delimiter=",", dtype=float)
        return Hr
    def load_crop_annapurna(self):
        crop_path = os.path.join(self.registration_folder,'crop_A_1100.csv')
        crop_loc = np.loadtxt(crop_path, delimiter=",", dtype=int)
        xlim = [crop_loc[0], crop_loc[1]]
        ylim = [crop_loc[2], crop_loc[3]]
        return xlim, ylim
    
    #K2
    def load_H_left_K2(self):
        H_left_path = os.path.join(self.registration_folder, 'H_left_K2.csv')
        Hl = np.loadtxt(H_left_path, delimiter=",", dtype=float)
        return Hl
    def load_H_right_K2(self):
        H_right_path = os.path.join(self.registration_folder, 'H_right_K2.csv')
        Hr = np.loadtxt(H_right_path, delimiter=",", dtype=float)
        return Hr
    def load_crop_K2(self):
        crop_path = os.path.join(registration_folder,'K2-BIVOUAC_lims.csv')
        crop_loc = np.loadtxt(crop_path, delimiter=",", dtype=int)
        xlim = [crop_loc[0], crop_loc[1]]
        ylim = [crop_loc[2], crop_loc[3]]
        return xlim, ylim

def load_tif(file):
    '''
    Load .tif file.
    '''
    ######
    with tiff.TiffFile(file) as ff:
        data = ff.asarray()
    return data  
def h_affine_transform(image, H):
    """
    Apply an affine transformation.
    """
    return affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))  

#Folder of the images you want to align:      
directory_path = r'D:\Data\Megan\snaps'
pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 

settings = Settings() #initialize the settings defined above.
registration_folder  = settings.registration_folder
verticalROI = settings.verticalROI
for fol in directory_names: #go to each folder and: 
    print(fol)
    save_folder = os.path.join('\\'.join(directory_path.split('\\')[:-1]), 'output', fol.replace(directory_path, '')[1:]) #define save folder. 
    if not os.path.exists(save_folder): #create the save folder if it does not exist. 
        os.makedirs(save_folder)
    #check whether Annapurna or K2 was used and initialize the neceesary variables depening on that
    if settings.microscope == 'ANNAPURNA': 
        x_coords = settings.x_coords_annapurna
        Hl  = settings.load_H_left_annapurna()
        Hr = settings.load_H_right_annapurna()
        xlim, ylim = settings.load_crop_annapurna()
    elif settings.microscope == 'K2':
        x_coords = settings.x_coords_K2
        Hl  = settings.load_H_left_K2()
        Hr = settings.load_H_right_K2()
        xlim, ylim = settings.load_crop_K2()
    #check the imaging mode used: sequence or record (a.k.a VCR). 
    pattern = settings.channels #get the lasers that you used. 
    file_names =  glob(fol + '/**/**.tif', recursive=True)#open the raw file
    for file_name in file_names:
        d = load_tif(file_name)
        for ch in pattern: #for each laser used
            image = np.copy(d[verticalROI[0]:verticalROI[1], x_coords[ch][0]:x_coords[ch][1]]) #crop the specific channel
            if ch == 'r_ch':#if the laser used is 405 or 488, use the right H matrix to correct. 
                im = h_affine_transform(image, Hr)
            elif ch == 'l_ch':#if the laser used is the 638. used the left H matrix to correct. 
                im = h_affine_transform(image, Hl)
            elif ch == 'm_ch':#If none of them have been used (561 laser has been used), do not modify the image
                im = np.copy(image)
            cropped_im  = im[ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.uint16) #crop the image in the proper cropping coordinates (after the correction). 
            save = os.path.join(save_folder, file_name.split('\\')[-1].split('.')[0]+'_'+ch+'.tif') #save the image as .tif
            imageio.mimwrite(save, [cropped_im])
    print('Finished with files in', fol.replace(directory_path, '')[1:])
print('########Finished########')