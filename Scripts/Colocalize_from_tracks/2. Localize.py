# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:54:26 2025

@author: castrolinares
"""
import os
from glob import glob
import traceback
import pandas as pd
from picasso.io import load_movie, save_info
from picasso.localize import (
    get_spots,
    identify_async,
    identifications_from_futures,
)
import picasso.gausslq as gausslq
import picasso.avgroi as avgroi
import numpy as np
from spit import localize
from spit import tools
from multiprocessing import freeze_support

class Settings:
    def __init__(self):
        self.box = 7
        
        self.gradient405 = 300  
        self.gradient488 = 700 
        self.gradient561 = 330
        self.gradient638 = 350
         
        self.camera_info = {}
        #Set the gain of the microscope. 
        self.camera_info['Gain'] = 1
        #Baseline is the average dark camera count
        self.camera_info['Baseline'] = 100
        #Sensitivity is the conversion factor (electrons per analog-to-digital (A/D) count)
        self.camera_info['Sensitivity'] = 0.6
        #In Picasso qe (quantum efficiency) is not used anymore. It is left for backward compatibility. 
        self.camera_info['qe'] = 0.9
        #'lq' for circular particles. 'com' for wierdly shaped particles.      
        self.fit_method = 'lq' 
        self.skip = 'not_track'
        self.suffix = ''
        self.transform = False  #Do non-affine corrections of the localized spots if you have multiple channels.
        self.plot = True
    def gradient(self, filename): #function to get the correct gradient later on the code. 
        if '488nm' in filename:
            return self.gradient488
        elif '561nm' in filename:
            return self.gradient561
        elif '638nm' in filename:
            return self.gradient638
        elif '405nm' in filename:
            return self.gradient405
    def get_naclib(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        result_txt  = tools.read_result_file('\\'.join(file.split('\\')[:-1])+'\\'+file.split('\\')[-2]+'_result.txt') #this opens the results.txt file to check the microscope used. 
        root = os.path.dirname(__file__)
        paramfiles_path = os.path.join(root, "Registration_folder/") #This sets the folder where the naclib coefficients are. 
        #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_Ann.csv'))
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return pd.read_csv(os.path.join(paramfiles_path, 'naclib_coefficients_K2.csv'))
    def get_px2um(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        result_txt  = tools.read_result_file('\\'.join(file.split('\\')[:-1]) + '\\' + 
                                             [element for element in file.split('\\') if element.startswith('Run')][0] + '_result.txt') #this opens the results.txt file to check the microscope used. 
        #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 0.09
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 0.108



def main(): 
    directory_path = r'D:\Data\Chi_data\first data\output2\Run00002'
    pathstif = glob(directory_path + '/**/**.tif', recursive=True)
    paths_im = list(set(os.path.dirname(file) for file in pathstif))
    for path in paths_im:
        print(path)
        if 'cluster_analysis' in path:
            continue
        elif os.path.isdir(path):
            localizee(path)
    
def localizee(folder): 
    # try:
     settings = Settings()
     transformInfo = 'False' 
     #Actually not needed, because you can only add folders, based on a function in def main: 
     if os.path.isdir(folder): 
        print('Analyzing directory', folder)
        pathsTif = glob(folder + '/*.tif', recursive=True)
        paths = pathsTif
     # subdirectories = list({os.path.dirname(file_path) for file_path in paths})
        print(f'A total of {len(paths)} files detected...')
        print('--------------------------------------------------------')
     else:
         print(f'{folder} is not a folder')
         
     # If any of the folders does not contain tif or raw images, it will be skipped and the folder will be saved in the following list:
     skippedPaths = []  
     if paths: 
         movieList = []
         filelist = []
         for i, path in enumerate(paths):
             print(path)
             if settings.skip in path or 'cluster_analysis' in path:
                 skippedPaths.append(path)
                 break
             
             filelist.append(path)
             movie, info = load_movie(path)
             movieList.append(movie)
             area = info[0]['Width']*info[0]['Height']*settings.get_px2um(path)*settings.get_px2um(path)
             gradient = settings.gradient(path)
             print(f'Localizing file {path}')
             print('--------------------------------------------------------')
             print('gradient:', settings.gradient(path))
             
             #Localize spots in the images based on the chosen fit-method
             current, futures = identify_async(movie, gradient, settings.box)
             ids = identifications_from_futures(futures)     
             if settings.fit_method == 'lq':
                 spots = get_spots(movie, ids, settings.box, settings.camera_info)
                 theta = gausslq.fit_spots_parallel(spots, asynch=False)
                 locs = gausslq.locs_from_fits(ids, theta, settings.box, settings.camera_info['Gain'])
             elif settings.fit_method == 'com':
                 spots = get_spots(movie, ids, settings.box, settings.camera_info)
                 theta = avgroi.fit_spots_parallel(spots, asynch=False)
                 locs = avgroi.locs_from_fits(ids, theta, settings.box, settings.camera_info['Gain'])
             else:
                 print('This should never happen... Please, set a proper method: com for moving particles, lq for moving stuff')
             #save the localizations in a dataframe        
             df_locs = pd.DataFrame(locs)
             # Compatibility with Swift
             df_locs = df_locs.rename(columns={'frame': 't', 'photons': 'intensity'})
 
             # adding localization precision, nearest neighbor, change photons, add cell_id column
             df_locs['loc_precision'] = df_locs[['lpx', 'lpy']].mean(axis=1)
             df_locs['nearest_neighbor'] = localize.get_nearest_neighbor(df_locs)
             df_locs['cell_id'] = 0

             # Non-affine correction only makes sense if we are dealing with two/three channel data. If you do not have these or want to update them, 
             #use get_non-affine_coefs.py. 
             if settings.transform:
                 #open non-affine coefficients. 
                 naclibCoefficients = settings.get_naclib(path)
                 #transform localizations based on the coefficients assigned to channel 2 (488nm or 405nm channel)
                 if '488nm' in path or '405nm' in path:
                     df_locs, dataset = localize.transform_locs(df_locs,
                                                                naclibCoefficients,
                                                                channel=2,
                                                                fig_size=list(movie[0].shape[::-1]))
                     transformInfo = 'true, based on '+str(dataset)
                #transform localizations based on the coefficients assigned to channel 0 (638nm channel)
                 elif '638nm' in path:
                     df_locs, dataset = localize.transform_locs(df_locs,
                                                                naclibCoefficients,
                                                                channel=0,
                                                                fig_size=list(movie[0].shape[::-1]))
                     transformInfo = 'true, based on '+str(dataset)
                 #do not modify 531nm channel, since it is the reference channel.
                 else:
                     transformInfo = 'false, reference channel'
            #update info (.yaml)            
             localize_info = {
                 'Generated by': 'Picasso Localize',
                 'Box Size': settings.box,
                 'Min. Net Gradient': gradient,
                 'Color correction': transformInfo,
                 'Area': float(area),
                 'Fit method': settings.fit_method
             }
             info[0]["Byte Order"] = "<" #I manually checked with https://hexed.it/ that the tif files are still saved as little-endian
             infoNew = info.copy()
             infoNew.append(localize_info)
             #get saving folder
             base, ext = os.path.splitext(path)
  
             pathChannel = base
 
             pathOutput = pathChannel + settings.suffix + '_locs.csv'
             #save localizations and ifnromation
             df_locs.to_csv(pathOutput, index=False)
             save_info(os.path.splitext(pathOutput)[0]+'.yaml', infoNew)
             
             #plot, if asked for, the summary plots 
             if settings.plot:
                 plotPath = tools.getOutputpath(
                     pathOutput, 'plots', keepFilename=True)
                 localize.plot_loc_stats(df_locs, plotPath)

            
             print(f'File saved to {pathOutput}')
             print('                                                        ')
             
    # except:
    #     # "There are no files in this subfolder, rise error"
    #     skippedPaths.append(folder)
    #     print('Skipping...\n')
    #     print('--------------------------------------------------------')
    
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

if __name__ == '__main__':
    freeze_support()
    main()