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
from spit import tools
###
#used new package to detect the largest rectangle after alginment (removing edges) that speeds up the process. Only needed for module 2. 
import largestinteriorrectangle as lir

#needed functions. I will add them some day to SPIT.tools
def read_result_file(file):
    with open(file, 'r') as resultTxt:
        resultLines = resultTxt.readlines()
    resultdict = {}
    for line in resultLines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            if key not in resultdict:
                resultdict[key] = value.strip()
    return resultdict


def get_pattern(result):
    pattern_dict = {}
    for i in result.keys():
        if 'Pattern' in i:
            pat = i.replace("tern", "")
            pattern_dict[pat] = result[i].split(',')
    # found_string_list = [i.split(':')[1].split(',') for i in resultLines if 'Pattern' in i]
    return pattern_dict

def get_VCR_pattern(result):
    VCR_dict = {'405nm': False, '488nm': False, '561nm': False, '638nm': False}
    for i in result: 
        if 'Laser' in i and 'ON' in result[i]:
            VCR_dict[i[6:11]] = True


#Setings for the alignment. Please check everything is correct. 
class Settings:
    def __init__(self):
        #General
        self.registration_folder = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\regis' #Folder containing the H-matrices and crop coordinates
        #needed for the alignment of the channels
        self.verticalROI = [0, 1100] #Specify the heigth of the channels that you want to use. I am pretty sure that if we set it larger, it still does it correctly. 
        self.to_keep = [0, 200]  #number of frames to procees. [0, None] means all frames, if you want to change it, set the specific number ([0:200] would be the first 
        #200 frames while [300:None] would be from frame 301 until the end)
        
        #ANNAPURNA
        self.ch_width_annapurna = 640   #The width of the channels in pixels. 
        self.x_coords_annapurna = {     # Define start and end site for each channel. 
            '638nm': (18, 18 + self.ch_width_annapurna),  
            '561nm': (691, 691 + self.ch_width_annapurna),  
            '488nm': (1370, 1370 + self.ch_width_annapurna), 
            '405nm': (1370, 1370 + self.ch_width_annapurna)  
        }  
        
        #K2
        self.ch_width_K2 = 560  #682  #The width of the channels in pixels. 
        # self.x_coords_K2 = { #K2
        #     '638nm': (0, 0 + self.ch_width_K2),  
        #     '561nm': (0 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2),  
        #     '488nm': (0 + self.ch_width_K2 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2 + self.ch_width_K2), 
        #     '405nm': (0 + self.ch_width_K2 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2 + self.ch_width_K2)  
        # }
        #Or, if using 560 
        self.x_coords_K2 = { #K2
            '638nm': (60, 60 + self.ch_width_K2),  
            '561nm': (740, 740 + self.ch_width_K2),  
            '488nm': (1415, 1415 + self.ch_width_K2), 
            '405nm': (1415, 1415 + self.ch_width_K2)
        }
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
    
    #K2
    def load_H_left_K2(self):
        H_left_path = os.path.join(self.registration_folder, 'H_left_K2.csv')
        Hl = np.loadtxt(H_left_path, delimiter=",", dtype=float)
        return Hl
    def load_H_right_K2(self):
        H_right_path = os.path.join(self.registration_folder, 'H_right_K2.csv')
        Hr = np.loadtxt(H_right_path, delimiter=",", dtype=float)
        return Hr

    
# Obtain cropping dimensions. Give direction to a folder containing a sequence (Pat01, Pat02, Pat03...) where all three channels have been imaged. 
#Use the same as you used for obtaining the affine matrices. 
directory_path = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\example data\from_chi\Run00002'


pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders.
directory_names = list(set(os.path.dirname(file) for file in pathsRaw))#makes a lost with the direction to each folder containing .raw files. 

settings = Settings() #initialize the settings defined above.
registration_folder  = settings.registration_folder
verticalROI = settings.verticalROI

for fol in directory_names: #go to each folder and: 
    result_file = fol+'\\' +fol.split("\\")[-1]+'_result.txt'#get direction result.txt file
    result_txt=read_result_file(result_file) #get a dictionary with the information in the result.txt file. 
    #check whether Annapurna or K2 was used and initialize the neceesary variables depening on that
    if result_txt['Computer'] == 'ANNAPURNA':
        x_coords = settings.x_coords_annapurna
        Hl  = settings.load_H_left_annapurna()
        Hr = settings.load_H_right_annapurna()
    elif result_txt['Computer'] == 'K2-BIVOUAC':
        x_coords = settings.x_coords_K2
        Hl  = settings.load_H_left_K2()
        Hr = settings.load_H_right_K2()
    #check the imaging mode used: sequence or record (a.k.a VCR). 
    if result_txt['Mode'] == 'Sequence':#if you used sequence for that run
        pattern = get_pattern(result_txt)#get the specific patterns that you used. 
        for pat, ch in pattern.items():  #and for each pattern
            file_name = fol+'\\' +fol.split("\\")[-1]+'_'+pat+'.raw' #open the raw file
            d, inf = tools.load_raw(file_name)
            for i in ch:#for each channel that has been used in that specific pattern
                image = d[:, verticalROI[0]:verticalROI[1], x_coords[i.strip()][0]:x_coords[i.strip()][1]] #Crop the image in the specific x_coordinates 
                if i.strip() in ['405nm', '488nm']: #if the laser used is 405 or 488, use the right H matrix to correct. 
                    im_r = np.array(list(map(lambda img: tools.h_affine_transform(img, Hr), image)))
                elif i.strip() in ['638nm'] :#if the laser used is the 638. used the left H matrix to correct. 
                    im_l = np.array(list(map(lambda img: tools.h_affine_transform(img, Hl), image)))
                else:#If none of them have been used (561 laser), do not modify the image
                    im_m = np.copy(image) 
        locs = lir.lir(np.array(im_l[0], "bool")) #find the largest possible rectangle for the left channel and save the coordinates for x and y. 
        ylim_l = [locs[1], locs[3]]
        xlim_l = [locs[0], locs[2]]
        locs_r = lir.lir(np.array(im_r[0], "bool")) #find the largest possible rectangle for the right channel and save the coordinates for x and y. 
        ylim_r = [locs_r[1], locs_r[3]]
        xlim_r = [locs_r[0], locs_r[2]]
        xlim = [max(xlim_l[0], xlim_r[0]), min(xlim_l[1], xlim_r[1])] #get the cropping coordinates combining both squares --> largest recatngle with both right and left channel 
        #have pixels (not empy). 
        ylim = [max(ylim_l[0], ylim_r[0]), min(ylim_l[1], ylim_r[1])]
        lims = xlim+ylim #put the x and y coordinates in a single list. 
        save_folder = os.path.join(directory_path, 'output', fol.replace(directory_path, '')[1:]) #define the save folder and create it if it does not exist. 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save = os.path.join(save_folder, result_txt['Computer']+'_lims.csv') #save the coordinates in a csv file that module 1 can use (please copy it to your registration folder
        #and correct, if necessary, the name on the settings)
        with open(save, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(lims)
        print(xlim, ylim)
print('########Finished########')
print("File is in", save)