# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:28:54 2025

@author: castrolinares
"""

import numpy as np
import os
import imageio
import yaml as _yaml
from glob import glob
import shutil
from spit import tools

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


class Settings:
    def __init__(self):
        #General
        self.verticalROI = [0, 1100]
        self.to_keep = [0, 200]
        
        #ANNAPURNA
        self.ch_width_annapurna = 640   #The width of the channels in pixels. 
        self.x_coords_annapurna = {     # Define start and end site for each channel. 
            '638nm': (18, 18 + self.ch_width_annapurna),  
            '561nm': (691, 691 + self.ch_width_annapurna),  
            '488nm': (1370, 1370 + self.ch_width_annapurna), 
            '405nm': (1370, 1370 + self.ch_width_annapurna)  
        }  
        
        #K2
        self.ch_width_K2 = 682    #560   #The width of the channels in pixels. 
        self.x_coords_K2 = { #K2
            '638nm': (0, 0 + self.ch_width_K2),  
            '561nm': (0 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2),  
            '488nm': (0 + self.ch_width_K2 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2 + self.ch_width_K2), 
            '405nm': (0 + self.ch_width_K2 + self.ch_width_K2, 0 + self.ch_width_K2 + self.ch_width_K2 + self.ch_width_K2)  
        }
        #Or, if using 560 x_coords = { #K2
        #     'ch1': (60, 60 + self.ch_width_K2),  
        #     'ch2': (740, 740 + self.ch_width_K2),  
        #     'ch3': (1415, 1415 + self.ch_width_K2)  
        # }
        
    
        
directory_path = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\example data\test_batch_correction3'



pathsRaw = glob(directory_path + '/**/**.raw', recursive=True)
directory_names = list(set(os.path.dirname(file) for file in pathsRaw))
settings = Settings()
verticalROI = settings.verticalROI
to_keep = settings.to_keep

for fol in directory_names:
    result_file = fol+'\\' +fol.split("\\")[-1]+'_result.txt'
    datalog_file = fol+'\\' +fol.split("\\")[-1]+'_datalog.txt'
    result_txt=read_result_file(result_file)
    save_folder = os.path.join(directory_path, 'output','split', fol.replace(directory_path, '')[1:])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    shutil.copy(result_file, save_folder)
    shutil.copy(datalog_file, save_folder)
    if result_txt['Computer'] == 'ANNAPURNA':
        x_coords = settings.x_coords_annapurna
    elif result_txt['Computer'] == 'K2-BIVOUAC':
        x_coords = settings.x_coords_K2
    if result_txt['Mode'] == 'Sequence':
        pattern = get_pattern(result_txt)
        for pat, ch in pattern.items(): 
            file_name = fol+'\\' +fol.split("\\")[-1]+'_'+pat+'.raw'
            d, inf = tools.load_raw(file_name)
            for i in ch:
                image = d[to_keep[0]:to_keep[1], verticalROI[0]:verticalROI[1], x_coords[i.strip()][0]:x_coords[i.strip()][1]]
                im = np.copy(image)
                with open(os.path.join(save_folder, pat+"_"+i.strip()+'_unaligned.yaml'), "w") as file:
                    _yaml.dump_all(inf, file, default_flow_style=False)
                save = os.path.join(save_folder, pat+"_"+i.strip()+'_unaligned.tif')
                imageio.mimwrite(save, im)    
    elif result_txt['Mode'] == 'VCR':
        pattern = get_VCR_pattern(result_txt)
        file_name = fol+'\\' +fol.split("\\")[-1]+'_'+'record'+'.raw'
        d, inf = tools.load_raw(file_name)
        for ch, presence in pattern.items():
            if presence: 
                image = d[to_keep[0]:to_keep[1], verticalROI[0]:verticalROI[1], x_coords[ch][0]:x_coords[ch][1]]
                im = np.copy(image)
                with open(os.path.join(save_folder, ch+'_unaligned.yaml'), "w") as file:
                    _yaml.dump_all(inf, file, default_flow_style=False)
                save = os.path.join(save_folder, ch+'_unaligned.tif')
                imageio.mimwrite(save, im)
    print('Finished with files in', fol.replace(directory_path, '')[1:])
print('########Finished########')