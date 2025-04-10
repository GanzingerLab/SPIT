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
        crop_path = os.path.join(self.registration_folder,'K2-BIVOUAC_lims.csv')
        crop_loc = np.loadtxt(crop_path, delimiter=",", dtype=int)
        xlim = [crop_loc[0], crop_loc[1]]
        ylim = [crop_loc[2], crop_loc[3]]
        return xlim, ylim

def main(): 
    directory_path = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\example data\from_chi'
    pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
    directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
    for path in directory_names:
        if os.path.isdir(path):
            preprocess(path, directory_path)
    print('########Finished########')
def preprocess(fol, directory_path):            
    settings = Settings() #initialize the settings defined above.
    registration_folder  = settings.registration_folder
    verticalROI = settings.verticalROI
    to_keep = settings.to_keep
    
    result_file = fol+'\\' +fol.split("\\")[-1]+'_result.txt'  #get direction result.txt file
    datalog_file = fol+'\\' +fol.split("\\")[-1]+'_datalog.txt' #get direction of datalog.txt file. 
    result_txt=tools.read_result_file(result_file) #get a dictionary with the information in the result.txt file. 
    save_folder = os.path.join(directory_path, 'output', fol.replace(directory_path, '')[1:]) #define save folder. 
    if not os.path.exists(save_folder): #create the save folder if it does not exist. 
        os.makedirs(save_folder)
    shutil.copy(result_file, save_folder) #copy result.txt and datalog.txt files into the save folder. 
    shutil.copy(datalog_file, save_folder)
    #check whether Annapurna or K2 was used and initialize the neceesary variables depening on that
    if result_txt['Computer'] == 'ANNAPURNA': 
        x_coords = settings.x_coords_annapurna
        Hl  = settings.load_H_left_annapurna()
        Hr = settings.load_H_right_annapurna()
        xlim, ylim = settings.load_crop_annapurna()
    elif result_txt['Computer'] == 'K2-BIVOUAC':
        x_coords = settings.x_coords_K2
        Hl  = settings.load_H_left_K2()
        Hr = settings.load_H_right_K2()
        xlim, ylim = settings.load_crop_K2()
    #check the imaging mode used: sequence or record (a.k.a VCR). 
    if result_txt['Mode'] == 'Sequence': #if you used sequence for that run
        pattern = tools.get_pattern(result_txt) #get the specific patterns that you used. 
        for pat, ch in pattern.items():  #and for each pattern
            file_name = fol+'\\' +fol.split("\\")[-1]+'_'+pat+'.raw' #open the raw file
            d, inf = tools.load_raw(file_name)
            for i in ch: #for each channel that ahs been used in that specific pattern
                image = d[to_keep[0]:to_keep[1], verticalROI[0]:verticalROI[1], x_coords[i.strip()][0]:x_coords[i.strip()][1]] #Crop the image in the specific x_coordinates to use
                if i.strip() in ['405nm', '488nm']: #if the laser used is 405 or 488, use the right H matrix to correct. 
                    im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hr), image)))
                elif i.strip() in ['638nm']: #if the laser used is the 638. used the left H matrix to correct. 
                    im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hl), image)))
                else: #If none of them have been used (561 laser), do not modify the image
                    im = np.copy(image)
                cropped_im  = im[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.uint16) #crop the image in the proper cropping coordinates (after the correction). 
                with open(os.path.join(save_folder,pat+"_"+i.strip()+'.yaml'), "w") as file: #save the .yaml file
                    _yaml.dump_all(inf, file, default_flow_style=False)
                save = os.path.join(save_folder, pat+"_"+i.strip()+'.tif') #save the image as .tif
                imageio.mimwrite(save, cropped_im)    
    elif result_txt['Mode'] == 'VCR': #if you used record for that run
        pattern = tools.get_VCR_pattern(result_txt) #get the lasers that you used. 
        file_name = fol+'\\' +fol.split("\\")[-1]+'_'+'record'+'.raw'#open the raw file
        d, inf = tools.load_raw(file_name)
        for ch, presence in pattern.items(): #for each laser used
            if presence: 
                image = d[to_keep[0]:to_keep[1], verticalROI[0]:verticalROI[1], x_coords[ch][0]:x_coords[ch][1]] #crop the specific channel
                if ch in ['405nm', '488nm']:#if the laser used is 405 or 488, use the right H matrix to correct. 
                    im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hr), image)))
                elif ch in ['638nm']:#if the laser used is the 638. used the left H matrix to correct. 
                    im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hl), image)))
                else:#If none of them have been used (561 laser has been used), do not modify the image
                    im = np.copy(image)
                cropped_im  = im[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.uint16) #crop the image in the proper cropping coordinates (after the correction). 
                with open(os.path.join(save_folder, ch+'.yaml'), "w") as file:#save the .yaml file
                    _yaml.dump_all(inf, file, default_flow_style=False)
                save = os.path.join(save_folder, ch+'.tif') #save the image as .tif
                imageio.mimwrite(save, cropped_im)
    print('Finished with files in', fol.replace(directory_path, '')[1:])
if __name__ == '__main__':
    main()