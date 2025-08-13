# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:36:43 2025

@author: castrolinares
"""

import os
import shutil
import csv
import json
import traceback
from glob import glob
from multiprocessing import freeze_support
from natsort import natsorted
import yaml as _yaml
import re
# import pickle
import numpy as np
import pandas as pd
import imageio
import yaml
from tqdm import tqdm
import tifffile as tiff
from scipy.ndimage import affine_transform
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
from spit import plot_coloc

import os
import numpy as np

class SPIT_Run:
    def __init__(self, folder, settings, output_folder = None):
        self.folder = folder
        self.settings = settings
        if output_folder is None:
            self.output_folder = folder
        else:
            self.output_folder = output_folder
        self.image_folder = os.path.join(self.output_folder, 'output', self.folder.replace(self.output_folder, '')[1:])
    def affine_transform(self):            
        verticalROI = self.settings.registration_settings.verticalROI
        to_keep = self.settings.registration_settings.to_keep
        
        base_name = os.path.basename(self.folder)
        result_file = self.folder+'\\' +self.folder.split("\\")[-1]+'_result.txt'  #get direction result.txt file
        datalog_file = self.folder+'\\' +self.folder.split("\\")[-1]+'_datalog.txt' #get direction of datalog.txt file. 
        used_fallback = False
        if not (os.path.exists(result_file) and os.path.exists(datalog_file)):
            try:
                result_file, datalog_file = self._find_alternative_result_file(self.folder)
                used_fallback = True
            except FileNotFoundError as e:
                print(f"[Skip] {self.folder} — {e}")
                return
        result_txt=tools.read_result_file(result_file) #get a dictionary with the information in the result.txt file. 
         #define save folder. 
        if not os.path.exists(self.image_folder): #create the save folder if it does not exist. 
            os.makedirs(self.image_folder)
        if used_fallback:
            # Rename fallback files to match current folder name
            new_result_file = os.path.join(self.image_folder, base_name + '_result.txt')
            new_datalog_file = os.path.join(self.image_folder, base_name + '_datalog.txt')
            shutil.copy(result_file, new_result_file)
            shutil.copy(datalog_file, new_datalog_file)
            fallback_info_path = os.path.join(self.image_folder, 'FALLBACK_INFO.txt')
            with open(fallback_info_path, 'w') as f:
                f.write("Fallback result/datalog files were used.\n")
                f.write(f"Source folder: {os.path.dirname(result_file)}\n")
                f.write(f"Copied to: {self.image_folder}\n")
        else:
            # Keep original file names
            shutil.copy(result_file, self.image_folder)
            shutil.copy(datalog_file, self.image_folder)
        #check whether Annapurna or K2 was used and initialize the neceesary variables depening on that
        if result_txt['Computer'] == 'ANNAPURNA': 
            x_coords = self.settings.registration_settings.x_coords_annapurna
            Hl  = self.settings.load_H_left_annapurna()
            Hr = self.settings.load_H_right_annapurna()
            xlim, ylim = self.settings.load_crop_annapurna()
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            x_coords = self.settings.registration_settings.x_coords_K2
            Hl  = self.settings.load_H_left_K2()
            Hr = self.settings.load_H_right_K2()
            xlim, ylim = self.settings.load_crop_K2()
        #check the imaging mode used: sequence or record (a.k.a VCR). 
        if result_txt['Mode'] == 'Sequence': #if you used sequence for that run
            pattern = tools.get_pattern(result_txt) #get the specific patterns that you used.
            self.split_ch = {}
            for pat, ch in pattern.items():  #and for each pattern
                file_name = os.path.join(self.folder, f"{os.path.basename(self.folder)}_{pat}.raw") #open the raw file
                d, inf = tools.load_raw(file_name)
                for i in ch: #for each channel that ahs been used in that specific pattern
                    ch = i.strip()
                    image = d[to_keep[0]:to_keep[1], verticalROI[0]:verticalROI[1], x_coords[ch][0]:x_coords[ch][1]] #Crop the image in the specific x_coordinates to use
                    if ch in ['405nm', '488nm']: #if the laser used is 405 or 488, use the right H matrix to correct. 
                        im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hr), image)))
                    elif ch in ['638nm']: #if the laser used is the 638. used the left H matrix to correct. 
                        im = np.array(list(map(lambda img: tools.h_affine_transform(img, Hl), image)))
                    else: #If none of them have been used (561 laser), do not modify the image
                        im = np.copy(image)
                    cropped_im  = im[:, ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.uint16) #crop the image in the proper cropping coordinates (after the correction). 
                    self.split_ch[ch] = np.copy(cropped_im)
                    yaml_path = os.path.join(self.image_folder, f"{pat}_{ch}.yaml")
                    with open(yaml_path, "w") as file: #save the .yaml file
                        _yaml.dump_all(inf, file, default_flow_style=False)
                    save = os.path.join(self.image_folder, f"{pat}_{ch}.tif") #save the image as .tif
                    imageio.mimwrite(save, cropped_im)    
        elif result_txt['Mode'] == 'VCR': #if you used record for that run
            pattern = tools.get_VCR_pattern(result_txt) #get the lasers that you used. 
            file_name = self.folder+'\\' +self.folder.split("\\")[-1]+'_'+'record'+'.raw'#open the raw file
            d, inf = tools.load_raw(file_name)
            self.split_ch = {}
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
                    self.split_ch[ch] = np.copy(cropped_im)
                    with open(os.path.join(self.image_folder, ch+'.yaml'), "w") as file:#save the .yaml file
                        _yaml.dump_all(inf, file, default_flow_style=False)
                    save = os.path.join(self.image_folder, ch+'.tif') #save the image as .tif
                    imageio.mimwrite(save, cropped_im)
        print('Finished with files in', self.folder.replace(self.output_folder, '')[1:])
    def localize(self): 
        try:
         transformInfo = 'False' 
         #Actually not needed, because you can only add folders, based on a function in def main: 
         if os.path.isdir(self.image_folder): 
            print('Analyzing directory', self.image_folder)
            pathsTif = glob(self.image_folder + '/*.tif', recursive=True)
            paths = pathsTif
         # subdirectories = list({os.path.dirname(file_path) for file_path in paths})
            print(f'A total of {len(paths)} files detected...')
            print('--------------------------------------------------------')
         else:
             print(f'{self.folder} is not a folder')
             
         # If any of the folders does not contain tif or raw images, it will be skipped and the folder will be saved in the following list:
         skippedPaths = []  
         if paths: 
             movieList = []
             filelist = []
             self.locs = {}
             for i, path in enumerate(paths):
                 # print(path)
                 if self.settings.localization_settings.skip in path or 'cluster_analysis' in path:
                     skippedPaths.append(path)
                     break
                 
                 filelist.append(path)
                 movie, info = load_movie(path)
                 movieList.append(movie)
                 area = info[0]['Width']*info[0]['Height']*self.settings.get_px2um(path)*self.settings.get_px2um(path)
                 gradient = self.settings.gradient(path)
                 print(f'Localizing file {path}')
                 print('--------------------------------------------------------')
                 print('gradient:', self.settings.gradient(path))
                 
                 #Localize spots in the images based on the chosen fit-method
                 current, futures = identify_async(movie, gradient, self.settings.localization_settings.box)
                 ids = identifications_from_futures(futures)     
                 box = self.settings.localization_settings.box
                 camera_info = self.settings.localization_settings.camera_info
                 if self.settings.localization_settings.fit_method == 'lq':
                     spots = get_spots(movie, ids, box, camera_info)
                     theta = gausslq.fit_spots_parallel(spots, asynch=False)
                     locs = gausslq.locs_from_fits(ids, theta, box, camera_info['Gain'])
                 elif self.settings.localization_settings.fit_method == 'com':
                     spots = get_spots(movie, ids, box,camera_info)
                     theta = avgroi.fit_spots_parallel(spots, asynch=False)
                     locs = avgroi.locs_from_fits(ids, theta, box, camera_info['Gain'])
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
                 if self.settings.localization_settings.transform:
                     #open non-affine coefficients. 
                     naclibCoefficients = self.settings.get_naclib(path)
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
                     'Box Size': self.settings.localization_settings.box,
                     'Min. Net Gradient': gradient,
                     'Color correction': transformInfo,
                     'Area': float(area),
                     'Fit method': self.settings.localization_settings.fit_method
                 }
                 info[0]["Byte Order"] = "<" #I manually checked with https://hexed.it/ that the tif files are still saved as little-endian
                 infoNew = info.copy()
                 infoNew.append(localize_info)
                 #get saving folder
                 base, ext = os.path.splitext(path)
      
                 pathChannel = base
     
                 pathOutput = pathChannel + self.settings.localization_settings.suffix + '_locs.csv'
                 #save localizations and ifnromation
                 df_locs.to_csv(pathOutput, index=False)
                 save_info(os.path.splitext(pathOutput)[0]+'.yaml', infoNew)
                 ch = pathOutput.split('\\')[-1].split('_')[-2]
                 self.locs[ch] = df_locs
                 
                 #plot, if asked for, the summary plots 
                 if self.settings.localization_settings.plot:
                     plotPath = tools.getOutputpath(
                         pathOutput, 'plots', keepFilename=True)
                     localize.plot_loc_stats(df_locs, plotPath)

                
                 print(f'File saved to {pathOutput}')
                 print('                                                        ')
                 
        except:
            # "There are no files in this subfolder, rise error"
            skippedPaths.append(self.folder)
            print('Skipping...\n')
            print('--------------------------------------------------------')
    def roi(self):
        '''Restrict localizations to ROIs'''
        # print(self.image_folder)
        # format filepaths
        if os.path.isdir(self.image_folder):
            print('Analyzing directory...')
            paths = glob(self.image_folder + '/*nm_locs.csv')

        # initialize placeholders
        skippedPaths = []

        # print all kept paths
        for path in paths:
            print(path)
        print(f'A total of {len(paths)} files detected...')
        print('--------------------------------------------------------')
        self.locs_roi = {}
        # main loop
        for idx, path in tqdm(enumerate(paths), desc='Saving new loc-files...', total=len(paths)):
            print('--------------------------------------------------------')
            print(f'Running file {path}')
            try:
                (df_locs, info) = tools.load_locs(path)
                # Look for ROI paths
                pathsROI = natsorted(glob(os.path.dirname(path) + '/*.roi', recursive=False))
                print(f'Found {len(pathsROI)} ROI.')

                dict_roi = {'cell_id': [], 'path': [], 'contour': [],
                            'area': [], 'roi_mask': [], 'centroid': []}
                
                # this stuff needs to go into tools
                df_locs = df_locs.drop('cell_id', axis=1)
                for idx, roi_path in enumerate(pathsROI):
                    roi_contour = tools.get_roi_contour(roi_path)
                    dict_roi['cell_id'].append(re.search(r'roi(\d+)\.roi$', roi_path))
                    dict_roi['path'].append(roi_path)
                    dict_roi['contour'].append(roi_contour)
                    dict_roi['area'].append(tools.get_roi_area(roi_contour))
                    dict_roi['roi_mask'].append(
                        tools.get_roi_mask(df_locs, roi_contour))
                    dict_roi['centroid'].append(
                        tools.get_roi_centroid(roi_contour))

                df_roi = pd.DataFrame(dict_roi)
                df_locsM = pd.concat([df_locs[roi_mask] for roi_mask in df_roi.roi_mask], keys=list(
                    np.arange((df_roi.cell_id.size))))

                df_locsM.index = df_locsM.index.set_names(['cell_id', None])
                df_locsM = df_locsM.reset_index(level=0)
                df_locsM = df_locsM.sort_values(['cell_id', 't'])
                df_locsM = df_locsM.drop_duplicates(subset=['x', 'y'])  # if ROIs overlap
                df_locs = df_locsM
                # get right output paths
                pathOutput = path.replace('locs.csv', 'roi_locs.csv')
                df_locs.to_csv(pathOutput, index=False)
                ch = pathOutput.split('\\')[-1].split('_')[-3]
                self.locs_roi[ch] = df_locs
                roi_info = {'Cell ROIs': str(df_roi.cell_id.unique())}
                infoNew = info.copy()
                infoNew.append(roi_info)
                save_info(os.path.splitext(pathOutput)[0] + '.yaml', infoNew)
            except Exception:
                skippedPaths.append(path)

                print('--------------------------------------------------------')
                print(f'Path {path} could not be analyzed. Skipping...\n')
                traceback.print_exc()

        print('                                                        ')
        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Skipped paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}\n')
    def link(self):
        try:
            if os.path.isdir(self.image_folder):
                if self.settings.link_settings.coloc:
                    paths = glob(self.image_folder + '/*colocs.csv', recursive=True)
                else:
                    paths = glob(self.image_folder + '/*_locs.csv', recursive=True)
                    
            skippedPaths = []
            quick = self.settings.link_settings.quick
            if self.settings.link_settings.coloc:
                self.tracks_coloc = {}
                self.tracks_coloc_stats = {}
            else:
                self.tracks = {}
                self.tracks_stats = {}
            # main loop
            for idx, path in tqdm(enumerate(paths), desc='Linking localizations...', total=len(paths)):
                try:
                    if self.settings.link_settings.roi and 'roi' not in path:
                        skip = path.split('\\')[-1]
                        print(f"\n\n Skipping {skip} bacause it is not filtered by roi and within Settings self.roi == {self.settings.link_settings.roi}\n")
                        continue
                    if not self.settings.link_settings.roi and 'roi' in path:
                        skip = path.split('\\')[-1]
                        print(f"\n\n Skipping {skip} bacause it is filtered by roi and within Settings self.roi == {self.settings.link_settings.roi}\n")
                        continue
                    (df_locs, info) = tools.load_locs(path)
                    if not self.settings.link_settings.coloc:
                        # fix locIDs before they get mixed up by linking
                        df_locs = df_locs.rename_axis('locID').reset_index()
            #         # retrieve exposure time
                    resultPath = '\\'.join(path.split('\\')[:-1]) + '\\' + [element for element in path.split('\\') if element.startswith('Run')][0] + '_result.txt'
                    if self.settings.link_settings.dt is not None:
                        dt = self.settings.link_settings.dt
                    else: 
                        resultTxt = open(resultPath, 'r')
                        resultLines = resultTxt.readlines()
                        if tools.find_string(resultLines, 'Interval'): 
                            interval = tools.find_string(resultLines, 'Interval').split(":")[-1].strip()
                            if interval.split(" ")[-1] == 'sec':
                                dt = 1.0 * int(float(interval.split(" ")[0]))
                            elif interval.split(" ")[-1] == 'ms':
                                dt = 0.001 * int(float(interval.split(" ")[0]))
                        else:
                            dtStr = tools.find_string(
                                resultLines, 'Camera Exposure')[17:-1]
                            dt = 0.001 * int(float((''.join(c for c in dtStr if (c.isdigit() or c == '.')))))
            
                    if 'roi' in path:
                        roi_boolean = True
                    else:
                        roi_boolean = False
            #         # Select 200px^2 center FOV and first 500 frames
                    if quick:
                        print(info[0])
                        img_size = info[0]['Height']  # get image size
                        roi_width = 100
                        if not roi_boolean:  # avoiding clash with ROIs-only limit frames
                            df_locs = df_locs[(df_locs.x > (img_size/2-roi_width))
                                              & (df_locs.x < (img_size/2+roi_width))]
                            df_locs = df_locs[(df_locs.y > (img_size/2-roi_width))
                                              & (df_locs.y < (img_size/2+roi_width))]
                        df_locs = df_locs[df_locs.t <= 500]
                        quick = '_quick'
                    else:
                        quick = ''
            
                    if roi_boolean:
                        # Look for ROI paths
                        pathsROI = glob(os.path.dirname(path) +
                                        '/*.roi', recursive=False)
                        print(f'Adding {len(pathsROI)} ROI infos.')
            
                        dict_roi = {'cell_id': [], 'path': [], 'contour': [],
                                    'area': [], 'roi_mask': [], 'centroid': []}
                        # this stuff needs to go into tools
                        for idx, roi_path in enumerate(pathsROI):
                            roi_contour = tools.get_roi_contour(roi_path)
                            dict_roi['cell_id'].append(re.search(r'roi(\d+)\.roi$', roi_path))
                            dict_roi['path'].append(roi_path)
                            dict_roi['contour'].append(roi_contour)
                            dict_roi['area'].append(tools.get_roi_area(roi_contour))
                            dict_roi['roi_mask'].append(
                                tools.get_roi_mask(df_locs, roi_contour))
                            dict_roi['centroid'].append(
                                tools.get_roi_centroid(roi_contour))
            
                        df_roi = pd.DataFrame(dict_roi)
            ################################################################################################
            #         # save ('quick'-cropped) locs in nm and plot stats
                    px2nm = self.settings.get_px2nm(resultPath)
                    df_locs_nm = tools.df_convert2nm(df_locs, px2nm)
                    path_nm = os.path.splitext(path)[0]+quick+self.settings.link_settings.suffix+'_nm.csv'
                    df_locs_nm.to_csv(path_nm, index=False)
                    path_plots_loc = tools.getOutputpath(path_nm, 'plots', keepFilename=True)
            
                    # tau_bleach = plot_diffusion.plot_loc_stats(df_locs_nm, path_plots_loc, dt=dt)
            
                    # prepare rest of the paths
                    path_output = os.path.splitext(path_nm)[0] +'_'+ self.settings.link_settings.tracker
                    path_plots = tools.getOutputpath(path_nm, 'plots', keepFilename=True) +'_'+ self.settings.link_settings.tracker
            
                    # Choose tracking algorithmus
                    if self.settings.link_settings.tracker == 'trackpy':
                        # print('Using trackpy.\n')
                        # export parameters to yaml
                        with open(os.path.splitext(path_nm)[0] +'_'+ self.settings.link_settings.tracker+ '.yaml', 'w') as f:
                            yaml.dump(vars(self.settings.link_settings), f)
            
                        df_tracksTP = link.link_locs_trackpy(df_locs, search=self.settings.link_settings.search, memory=self.settings.link_settings.memory)
            
                        # # linked file is saved with pixel-corrected coordinates and
                        # # swiftGUI compatible columns, and unique track.ids
                        df_tracks = tools.df_convert2nm(df_tracksTP, px2nm)
                        df_tracks['seg.id'] = df_tracksTP['track.id']
                        if 'roi' in path:
                            df_tracks = tools.get_unique_trackIDs(df_tracks)
                        df_tracks.to_csv(path_output + '.csv', index=False)
                        ch = path_output.split('\\')[-1].split('_')[1]
                    #If I ever adapt it, swift goes here:
                    
            
            #         # Analysis and Plotting
            
                    print('Calculating and plotting particle-wise diffusion analysis...\n')
                    df_stats = link.get_particle_stats(df_tracks,
                                                       dt=dt,
                                                       particle='track.id',
                                                       t='t')
            
            #         # adding ROI stats to track stat file
                    if roi_boolean:
                        df_stats = df_stats.merge(
                            df_roi[['path', 'contour', 'area', 'centroid', 'cell_id']], on='cell_id', how='left')
                    # Save dataframe with track statistics (unfiltered)
                    if os.path.isfile(path_output + '_stats.hdf'):
                        os.remove(path_output + '_stats.hdf')  # force overwriting
                    df_stats.to_hdf(path_output + '_stats.hdf',
                                    key='df_stats', mode='w')
                    
                    if self.settings.link_settings.coloc:
                        self.tracks_coloc[ch] = df_tracks
                        self.tracks_coloc_stats[ch] = df_stats
                    else:
                        self.tracks[ch] = df_tracks
                        self.tracks_stats[ch] = df_stats
                    
                # Filter short tracks and immobile particles
                    if not self.settings.link_settings.coloc:
                        df_statsF = link.filter_df(df_stats, filter_length=self.settings.link_settings.fil_len, filter_D=self.settings.link_settings.fil_diff)
                        plot_diffusion.plot_track_stats(df_tracks, df_stats, df_statsF, path_plots, dt=dt, px2nm = px2nm)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    print('I do not think you have many tracks... OR self.search is too large')
                    skippedPaths.append(path)
                    continue
        except Exception as e:
            print('Error')
        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Analysis failed on paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}')
    def coloc_tracks(self):
        settings = self.settings.coloc_tracks_settings
        # format paths according to a specified ending, e.g. "488nm_locs.csv"
        ch0 = settings.ch0
        ch1 = settings.ch1

        # getting all filenames of the first channel, later look for corresponding second channel files
        if os.path.isdir(self.image_folder):
            print('Analyzing directory...')
            pathsCh0 = glob(self.image_folder + f'//**//*{ch0}*_locs_nm_trackpy.csv', recursive=True)
            print(f'Found {len(pathsCh0)} files for channel 0...')
            # for path in pathsCh0:
            #     print(path)
        else:
            raise FileNotFoundError('Directory not found')
        print('--------------------------------------------------------')
        skippedPaths = []
        # main loop
        for idx, pathCh0 in tqdm(enumerate(pathsCh0), desc='Looking for colocalizations...'):
            print(pathCh0)
            try:
                dirname = os.path.dirname(pathCh0)
                if ch1 == None:
                    raise FileNotFoundError('Second channel not declared.')
                    print('--------------------------------------------------------')
                else:
                    pathCh1 = glob(dirname + f'/**{ch1}*_locs_nm_trackpy.csv')[idx]
                    # read in the linked files
                    df_locs_ch0 = pd.read_csv(pathCh0)
                    df_locs_ch1 = pd.read_csv(pathCh1)
                    # pixels to nm

        #             # get colocalizations
                    df_colocs, coloc_stats = coloc.coloc_tracks(df_locs_ch0, df_locs_ch1,leng = settings.min_len_track, max_distance=settings.th, n = settings.min_overlapped_frames)
                        
        #             # get right output paths
                    pathOutput = os.path.splitext(pathCh0)[0][:] + settings.suffix

                    print('Saving colocalizations...')
                    df_colocs.to_csv(pathOutput + '_colocsTracks.csv', index=False)
                    if 'roi' in pathCh0:
                        # Look for ROI paths
                        pathsROI = glob(os.path.dirname(pathCh0) +
                                        '/*.roi', recursive=False)
                        print(f'Adding {len(pathsROI)} ROI infos.')
            
                        dict_roi = {'cell_id': [], 'path': [], 'contour': [],
                                    'area': [], 'centroid': []}
                        # this stuff needs to go into tools
                        for idx, roi_path in enumerate(pathsROI):
                            roi_contour = tools.get_roi_contour(roi_path)
                            dict_roi['cell_id'].append(idx)
                            dict_roi['path'].append(roi_path)
                            dict_roi['contour'].append(roi_contour)
                            dict_roi['area'].append(tools.get_roi_area(roi_contour))
                            dict_roi['centroid'].append(
                                tools.get_roi_centroid(roi_contour))
            
                        df_roi = pd.DataFrame(dict_roi)
                        coloc_stats = coloc_stats.merge(
                            df_roi[['path', 'contour', 'area', 'centroid', 'cell_id']], on='cell_id', how='left')

                    info_file = "\\".join(pathCh0.split('\\')[:-1]) +"\\"+ pathCh0.split('\\')[-1].split('.')[0]+'.yaml'
                    # export parameters to yaml
                    infoNew = tools.load_info(info_file)
                    infoNew.append(vars(settings))
                    
                    save_info(pathOutput + '_colocsTracks.yaml', infoNew)
                    
                    if os.path.isfile(pathOutput + '_colocsTracks_stats.hdf'):
                        os.remove(pathOutput + '_colocsTracks_stats.hdf')  # force overwriting
                    coloc_stats.to_hdf(pathOutput + '_colocsTracks_stats.hdf',
                                    key='df_stats', mode='w')
                    self.coloc_tracks = df_colocs
                    self.coloc_tracks_stats = coloc_stats
            except Exception:
                skippedPaths.append(pathCh0)
                print('--------------------------------------------------------')
                print(f'Path {pathCh0} could not be analyzed. Skipping...\n')
                traceback.print_exc()

            

        print('                                                        ')

        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Skipped paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}\n')
    def coloc_spots(self):
        settings = self.settings.coloc_spots_settings
        # format paths according to a specified ending, e.g. "488nm_locs.csv"
        ch0 = settings.ch0
        ch1 = settings.ch1
        # getting all filenames of the first channel, later look for corresponding second channel files
        if os.path.isdir(self.image_folder):
            print('Analyzing directory...')
            pathsCh0 = glob(self.image_folder + f'//**//*{ch0}*_locs.csv', recursive=True)
            print(f'Found {len(pathsCh0)} files for channel 0...')
            # for path in pathsCh0:
            #     print(path)
        else:
            raise FileNotFoundError('Directory not found')
        print('--------------------------------------------------------')
        skippedPaths = []
        # main loop
        for idx, pathCh0 in tqdm(enumerate(pathsCh0), desc='Looking for colocalizations...'):
            print(pathCh0)
            try:
                dirname = os.path.dirname(pathCh0)
                if ch1 == None:
                    raise FileNotFoundError('Second channel not declared.')
                    print('--------------------------------------------------------')
                else:
                    pathCh1 = glob(dirname + f'/**{ch1}*_locs.csv')[idx]
                    # print(f'\nFound a second channel for file {idx}.')
                    resultPath  = '\\'.join(pathCh0.split('\\')[:-1]) + '\\' + [element for element in pathCh0.split('\\') if element.startswith('Run')][0] + '_result.txt'
                    if not settings.dt == None:
                        dt = settings.dt
                    else:
                        resultTxt = open(resultPath, 'r')
                        resultLines = resultTxt.readlines()
                        if tools.find_string(resultLines, 'Interval'): 
                            interval = tools.find_string(resultLines, 'Interval').split(":")[-1].strip()
                            if interval.split(" ")[-1] == 'sec':
                                dt = int(float(interval.split(" ")[0]))
                            elif interval.split(" ")[-1] == 'ms':
                                dt = 0.001 * int(float(interval.split(" ")[0]))
                        else:
                            dtStr = tools.find_string(
                                resultLines, 'Camera Exposure')[17:-1]
                            dt = 0.001 * int(float((''.join(c for c in dtStr if (c.isdigit() or c == '.')))))

                    # read in the linked files
                    df_locs_ch0 = pd.read_csv(pathCh0)
                    df_locs_ch1 = pd.read_csv(pathCh1)
                    # pixels to nm
                    px2nm = self.settings.get_px2nm(pathCh0)
                    df_locs_ch0 = tools.df_convert2nm(df_locs_ch0, px2nm)
                    df_locs_ch1 = tools.df_convert2nm(df_locs_ch1, px2nm)

        #             # get colocalizations
                    df_colocs = coloc.colocalize_from_locs(df_locs_ch0, df_locs_ch1, threshold_dist=settings.th)

        #             # get right output paths
                    pathOutput = os.path.splitext(pathCh0)[0][:-5] + settings.suffix
                    pathPlots = tools.getOutputpath(pathCh0, 'plots', keepFilename=True)[:-9] + settings.suffix

                    print('Saving colocalizations...')
                    df_colocs_px = tools.df_convert2px(df_colocs, px2nm)
                    df_colocs_px.to_csv(pathOutput + '_colocs.csv', index=False)
                    print('Calculating and plotting colocalization analysis.')
                    if not df_colocs.empty:
                        plot_coloc.plot_coloc_stats(df_locs_ch0, df_locs_ch1, df_colocs,
                                                    threshold=settings.th,
                                                    path=pathPlots, dt=dt, roll_param=5)
                    info_file = "\\".join(pathCh0.split('\\')[:-1]) +"\\"+ pathCh0.split('\\')[-1].split('.')[0]+'.yaml'
                    # export parameters to yaml
                    infoNew = tools.load_info(info_file)
                    infoNew.append(vars(settings))
                    
                    save_info(pathOutput + '_colocs.yaml', infoNew)
                    self.coloc_spots = df_colocs_px

            except Exception:
                skippedPaths.append(pathCh0)
                print('--------------------------------------------------------')
                print(f'Path {pathCh0} could not be analyzed. Skipping...\n')
                traceback.print_exc()

            

        print('                                                        ')

        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Skipped paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}\n')
    def full_analysis_noROI(self, mode = 'tracks'):
        original_roi = self.settings.link_settings.roi
        self.settings.link_settings.roi = False
        self.affine_transform()
        self.localize()
        if mode == 'tracks':
            original_coloc = self.settings.link_settings.coloc
            if original_coloc:
                print('LinkingSettings - coloc was True, changing to False for mode = "Tracks"')
                self.settings.link_settings.coloc = False
            self.link()
            self.coloc_tracks()
            if original_coloc:
                self.settings.link_settings.coloc = original_coloc
        elif mode == 'spots':
            self.coloc_spots()
            self.settings.link_settings.coloc = False
            self.localize()
            self.settings.link_settings.coloc = True
            self.localize
        self.settings.link_settings.roi = original_roi
    def full_analysis_ROI(self, mode = 'tracks'):
        pathsROI = glob(self.image_folder + '/*.roi', recursive=False)
        if not pathsROI:
            print('Be aware that for this mode you first need to do the .affine_transform and then manually draw ROIs with imageJ freehand tool, and save them as roiX.roi where X is a number starting by 0 and going up to as many ROIs there are')
        else:
            original_roi = self.settings.link_settings.roi
            self.settings.link_settings.roi = True
            self.localize()
            self.roi()
            if mode == 'tracks':
                self.link()
                self.coloc_tracks()
            elif mode == 'spots':
                self.coloc_spots()
                self.settings.link_settings.coloc = False
                self.link()
                self.settings.link_settings.coloc = True
                self.link()
            self.settings.link_settings.roi = original_roi
    def _find_alternative_result_file(self, folder):
        """
        Search one level up from `folder` for any folder containing both *_result.txt and *_datalog.txt files.
        """
        parent = os.path.dirname(folder)

        # Look for all *_result.txt files one level down from the parent directory
        result_files = glob(os.path.join(parent, '*', '*_result.txt'))

        for result_file in result_files:
            sib_folder = os.path.dirname(result_file)
            base = os.path.basename(sib_folder)
            datalog_file = os.path.join(sib_folder, base + '_datalog.txt')
            
            if os.path.exists(datalog_file):
                print(f"[Fallback] Found result files in: {sib_folder} for {folder}")
                return result_file, datalog_file

        raise FileNotFoundError("No valid *_result.txt found within the same parent folder. Copy one yourself.")
class SPIT_Dataset:
    def __init__(self, folder, settings):
        self.folder = folder
        self.settings = settings
        self.output_folder = os.path.join(folder, 'output')
    def affine_transform(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                # if "cont" in path:
                    to_process = SPIT_Run(path, self.settings, directory_path)
                    to_process.affine_transform()
        print('########Finished########')
    def localize(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                # if "cont" in path:
                    to_process = SPIT_Run(path, self.settings, directory_path)
                    to_process.localize()
        print('########Finished########')
    def roi(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                to_process = SPIT_Run(path, self.settings, directory_path)
                # if "cont" in path:
                # if any('roi_locs.csv' in fname for fname in os.listdir(to_process.image_folder)):
                #     print(f'skipped {path}')
                #     continue  # Skip roi() if such a file exists
                to_process.roi()

        print('########Finished########')
    def link(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                to_process = SPIT_Run(path, self.settings, directory_path)
                to_process.link()
        print('########Finished########')
    def coloc_tracks(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                to_process = SPIT_Run(path, self.settings, directory_path)
                to_process.coloc_tracks()
        print('########Finished########')
    def coloc_spots(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                print(path)
                to_process = SPIT_Run(path, self.settings, directory_path)
                to_process.coloc_spots()
        print('########Finished########')
    def SPIT_noROI(self, mode = 'tracks'):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                to_process = SPIT_Run(path, self.settings, directory_path)
                to_process.full_analysis_noROI(mode = mode)
        print('########Finished########')
    def SPIT_ROI(self, mode = 'tracks'):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.raw', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path):
                print(f"Analyszing {path}")
                to_process = SPIT_Run(path, self.settings, directory_path)
                to_process.full_analysis_ROI(mode = mode)
        print('########Finished########')
class localize_tiff_run:
    def __init__(self, folder, settings, output_folder = None):
        self.folder = folder
        self.settings = settings
        if output_folder is None:
            self.output_folder = folder
        else:
            self.output_folder = output_folder
        self.image_folder = os.path.join(self.output_folder, 'output', self.folder.replace(self.output_folder, '')[1:])
    def affine_transform(self):
        settings2 = self.settings.registration_settings
        # registration_folder  = settings2.registration_folder
        verticalROI = settings2.verticalROI
        if not os.path.exists(self.image_folder): #create the save folder if it does not exist. 
            os.makedirs(self.image_folder)
        #check whether Annapurna or K2 was used and initialize the neceesary variables depening on that
        if settings2.microscope == 'ANNAPURNA': 
            x_coords = settings2.x_coords_annapurna_tiff
            Hl  = self.settings.load_H_left_annapurna()
            Hr = self.settings.load_H_right_annapurna()
            xlim, ylim = self.settings.load_crop_annapurna()
        elif settings2.microscope == 'K2':
            x_coords = settings2.x_coords_K2_tiff
            Hl  = self.settings.load_H_left_K2()
            Hr = self.settings.load_H_right_K2()
            xlim, ylim = self.settings.load_crop_K2()
        #check the imaging mode used: sequence or record (a.k.a VCR). 
        pattern = settings2.channels #get the lasers that you used. 
        file_names =  glob(self.folder + '/**/**.tif', recursive=True)#open the raw file
        for file_name in file_names:
            d = self._load_tif(file_name)
            for ch in pattern: #for each laser used
                image = np.copy(d[verticalROI[0]:verticalROI[1], x_coords[ch][0]:x_coords[ch][1]]) #crop the specific channel
                if ch == 'r_ch':#if the laser used is 405 or 488, use the right H matrix to correct. 
                    im = self._h_affine_transform(image, Hr)
                elif ch == 'l_ch':#if the laser used is the 638. used the left H matrix to correct. 
                    im = self._h_affine_transform(image, Hl)
                elif ch == 'm_ch':#If none of them have been used (561 laser has been used), do not modify the image
                    im = np.copy(image)
                cropped_im  = im[ylim[0]:ylim[1], xlim[0]:xlim[1]].astype(np.uint16) #crop the image in the proper cropping coordinates (after the correction). 
                save = os.path.join(self.image_folder, file_name.split('\\')[-1].split('.')[0]+'_'+ch+'.tif') #save the image as .tif
                imageio.mimwrite(save, [cropped_im])
        
        print('########Finished########')
    def localize(self):
        transformInfo = 'False' 
        #Actually not needed, because you can only add folders, based on a function in def main: 
        if os.path.isdir(self.image_folder): 
           print('Analyzing directory', self.image_folder)
           pathsTif = glob(self.image_folder + '/*.tif', recursive=True)
           paths = pathsTif
        # subdirectories = list({os.path.dirname(file_path) for file_path in paths})
           print(f'A total of {len(paths)} files detected...')
           print('--------------------------------------------------------')
        else:
            print(f'{self.image_folder} is not a folder')
            
        # If any of the folders does not contain tif or raw images, it will be skipped and the folder will be saved in the following list:
        skippedPaths = []  
        if paths: 
            movieList = []
            filelist = []
            for i, path in enumerate(paths):
                filelist.append(path)
                movie, info = load_movie(path)
                movieList.append(movie)
                area = info[0]['Width']*info[0]['Height']*self.settings.get_px2um_tiffs()*self.settings.get_px2um_tiffs()
                gradient = self.settings.gradient_tiffs(path)
                print(path)
                print(f'Localizing file {path}')
                print('--------------------------------------------------------')
                print('gradient:', self.settings.gradient_tiffs(path))
                
                #Localize spots in the images based on the chosen fit-method
                current, futures = identify_async(movie, gradient, self.settings.localization_settings.box)
                ids = identifications_from_futures(futures)     
                box = self.settings.localization_settings.box
                camera_info = self.settings.localization_settings.camera_info
                if self.settings.localization_settings.fit_method == 'lq':
                    spots = get_spots(movie, ids, box, camera_info)
                    theta = gausslq.fit_spots_parallel(spots, asynch=False)
                    locs = gausslq.locs_from_fits(ids, theta, box, camera_info['Gain'])
                elif self.settings.localization_settings.fit_method == 'com':
                    spots = get_spots(movie, ids, box,camera_info)
                    theta = avgroi.fit_spots_parallel(spots, asynch=False)
                    locs = avgroi.locs_from_fits(ids, theta, box, camera_info['Gain'])
                else:
                    print('This should never happen... Please, set a proper method: com for moving particles, lq for moving stuff')
                #save the localizations in a dataframe        
                df_locs = pd.DataFrame(locs)
                # Compatibility with Swift
                df_locs = df_locs.rename(columns={'frame': 't', 'photons': 'intensity'})
    
                # adding localization precision, nearest neighbor, change photons, add cell_id column
                df_locs['loc_precision'] = df_locs[['lpx', 'lpy']].mean(axis=1)
                df_locs['cell_id'] = 0

                # Non-affine correction only makes sense if we are dealing with two/three channel data. If you do not have these or want to update them, 
                #use get_non-affine_coefs.py. 
                if self.settings.localization_settings.transform:
                    #open non-affine coefficients. 
                    naclibCoefficients = self.settings.get_naclib_tiffs(path)
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
                    'Box Size': self.settings.localization_settings.box,
                    'Min. Net Gradient': gradient,
                    'Color correction': transformInfo,
                    'Area': float(area),
                    'Fit method': self.settings.localization_settings.fit_method
                }
                info[0]["Byte Order"] = "<" #I manually checked with https://hexed.it/ that the tif files are still saved as little-endian
                infoNew = info.copy()
                infoNew.append(localize_info)
                #get saving folder
                base, ext = os.path.splitext(path)
     
                pathChannel = base
    
                pathOutput = pathChannel + self.settings.localization_settings.suffix + '_locs.csv'
                #save localizations and ifnromation
                df_locs.to_csv(pathOutput, index=False)
                save_info(os.path.splitext(pathOutput)[0]+'.yaml', infoNew)
        
                print(f'File saved to {pathOutput}')
                print('                                                        ')
        else: 
           # "There are no files in this subfolder, rise error"
           skippedPaths.append(self.folder)
           print('Skipping...\n')
           print('--------------------------------------------------------')
    def roi(self):
        '''Restrict localizations to ROIs'''

        # format filepaths
        if os.path.isdir(self.image_folder):
            print('Analyzing directory...')
            paths = glob(self.image_folder + '/*ch_locs.csv')

        # initialize placeholders
        skippedPaths = []

        # print all kept paths
        for path in paths:
            print(path)
        print(f'A total of {len(paths)} files detected...')
        print('--------------------------------------------------------')
        self.locs_roi = {}
        # main loop
        for idx, path in tqdm(enumerate(paths), desc='Saving new loc-files...', total=len(paths)):
            print('--------------------------------------------------------')
            print(f'Running file {path}')
            try:
                (df_locs, info) = tools.load_locs(path)
                # Look for ROI paths
                pathsROI = natsorted(glob(os.path.dirname(path) + '/*.roi', recursive=False))
                print(f'Found {len(pathsROI)} ROI.')

                dict_roi = {'cell_id': [], 'path': [], 'contour': [],
                            'area': [], 'roi_mask': [], 'centroid': []}
                
                # this stuff needs to go into tools
                df_locs = df_locs.drop('cell_id', axis=1)
                for idx, roi_path in enumerate(pathsROI):
                    roi_contour = tools.get_roi_contour(roi_path)
                    dict_roi['cell_id'].append(re.search(r'roi(\d+)\.roi$', roi_path))
                    dict_roi['path'].append(roi_path)
                    dict_roi['contour'].append(roi_contour)
                    dict_roi['area'].append(tools.get_roi_area(roi_contour))
                    dict_roi['roi_mask'].append(
                        tools.get_roi_mask(df_locs, roi_contour))
                    dict_roi['centroid'].append(
                        tools.get_roi_centroid(roi_contour))

                df_roi = pd.DataFrame(dict_roi)
                df_locsM = pd.concat([df_locs[roi_mask] for roi_mask in df_roi.roi_mask], keys=list(
                    np.arange((df_roi.cell_id.size))))

                df_locsM.index = df_locsM.index.set_names(['cell_id', None])
                df_locsM = df_locsM.reset_index(level=0)
                df_locsM = df_locsM.sort_values(['cell_id', 't'])
                df_locsM = df_locsM.drop_duplicates(subset=['x', 'y'])  # if ROIs overlap
                df_locs = df_locsM
                # get right output paths
                pathOutput = path.replace('locs.csv', 'roi_locs.csv')
                df_locs.to_csv(pathOutput, index=False)
                ch = pathOutput.split('\\')[-1].split('_')[-3]
                self.locs_roi[ch] = df_locs
                roi_info = {'Cell ROIs': str(df_roi.cell_id.unique())}
                infoNew = info.copy()
                infoNew.append(roi_info)
                save_info(os.path.splitext(pathOutput)[0] + '.yaml', infoNew)
            except Exception:
                skippedPaths.append(path)

                print('--------------------------------------------------------')
                print(f'Path {path} could not be analyzed. Skipping...\n')
                traceback.print_exc()

        print('                                                        ')
        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Skipped paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}\n')
    def colocalize(self):
        settings = self.settings.coloc_spots_settings
        # format paths according to a specified ending, e.g. "488nm_locs.csv"
        ch0 = settings.ch0_tiffs
        ch1 = settings.ch1_tiffs

        # getting all filenames of the first channel, later look for corresponding second channel files
        if os.path.isdir(self.image_folder):
            print('Analyzing directory...')
            pathsCh0 = glob(self.image_folder + f'//**//*{ch0}*_locs.csv', recursive=True)
            print(f'Found {len(pathsCh0)} files for channel 0...')
            # for path in pathsCh0:
            #     print(path)
        else:
            raise FileNotFoundError('Directory not found')
        print('--------------------------------------------------------')
        skippedPaths = []
        # main loop
        for idx, pathCh0 in tqdm(enumerate(pathsCh0), desc='Looking for colocalizations...'):
            print(pathCh0)
            try:
                dirname = os.path.dirname(pathCh0)
                if ch1 == None:
                    raise FileNotFoundError('Second channel not declared.')
                    print('--------------------------------------------------------')
                else:
                    pathCh1 = glob(dirname + f'/**{ch1}*_locs.csv')[idx]
                    # print(f'\nFound a second channel for file {idx}.')
                    print(pathCh0)
                    print(pathCh1)
                    # read in the linked files
                    df_locs_ch0 = pd.read_csv(pathCh0)
                    df_locs_ch1 = pd.read_csv(pathCh1)
                    # pixels to nm
                    px2nm = self.settings.get_px2nm_tiffs()
                    df_locs_ch0 = tools.df_convert2nm(df_locs_ch0, px2nm)
                    df_locs_ch1 = tools.df_convert2nm(df_locs_ch1, px2nm)

        #             # get colocalizations
                    df_colocs = coloc.colocalize_from_locs(df_locs_ch0, df_locs_ch1, threshold_dist=settings.th)

        #             # get right output paths
                    pathOutput = os.path.splitext(pathCh0)[0][:-5] + settings.suffix
                    # pathPlots = tools.getOutputpath(pathCh0, 'plots', keepFilename=True)[:-9] + settings.suffix

                    print('Saving colocalizations...')
                    df_colocs_px = tools.df_convert2px(df_colocs, px2nm)
                    df_colocs_px.to_csv(pathOutput + '_colocs.csv', index=False)
                    print('Calculating and plotting colocalization analysis.')

                    info_file = "\\".join(pathCh0.split('\\')[:-1]) +"\\"+ pathCh0.split('\\')[-1].split('.')[0]+'.yaml'
                    # export parameters to yaml
                    infoNew = tools.load_info(info_file)
                    infoNew.append(vars(settings))
                    
                    save_info(pathOutput + '_colocs.yaml', infoNew)

            except Exception:
                skippedPaths.append(pathCh0)
                print('--------------------------------------------------------')
                print(f'Path {pathCh0} could not be analyzed. Skipping...\n')
                traceback.print_exc()

            

        print('                                                        ')

        print('--------------------------------------------------------')
        print('/////////////////////FINISHED//////////////////////////')
        print('--------------------------------------------------------')
        if skippedPaths:
            print('Skipped paths:')
            for skippedPath in skippedPaths:
                print(f'\n{skippedPath}\n')
    def full_analysis_noROI(self, mode = 'tracks'):
        original_roi = self.settings.link_settings.roi
        self.settings.link_settings.roi = False
        self.affine_transform()
        self.localize()
        self.colocalize()
        self.settings.link_settings.roi = original_roi
    def full_analysis_ROI(self, mode = 'tracks'):
        pathsROI = glob(self.image_folder + '/*.roi', recursive=False)
        if not pathsROI:
            print('Be aware that for this mode you first need to do the .affine_transform and then manually draw ROIs with imageJ freehand tool, and save them as roiX.roi where X is a number starting by 0 and going up to as many ROIs there are')
        else:
            original_roi = self.settings.link_settings.roi
            self.settings.link_settings.roi = True
            self.localize()
            self.roi()
            self.colocalize()
            self.settings.link_settings.roi = original_roi
    def _load_tif(self, file):
        '''
        Load .tif file.
        '''
        ######
        with tiff.TiffFile(file) as ff:
            data = ff.asarray()
        return data  
    def _h_affine_transform(self, image, H):
        """
        Apply an affine transformation.
        """
        return affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))  
        
        
class localize_tiff_dataset:
    def __init__(self, folder, settings):
        self.folder = folder
        self.settings = settings
    def affine_transform(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.affine_transform()
        print('########Finished########')
    def localize(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.localize()
        print('########Finished########')
    def roi(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                print(path)
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.roi()
        print('########Finished########')
    def colocalize(self):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                print(path)
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.colocalize()
        print('########Finished########')
    def full_analysis_noROI(self, mode = 'tracks'):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.full_analysis_noROI(mode = mode)
        print('########Finished########')
    def full_analysis_ROI(self, mode = 'tracks'):
        directory_path = self.folder
        pathsRaw = glob(directory_path + '/**/**.tif', recursive=True) #Check for each .row file in the folder and subfolders. 
        directory_names = list(set(os.path.dirname(file) for file in pathsRaw)) #makes a lost with the direction to each folder containing .raw files. 
        for path in directory_names:
            if os.path.isdir(path) and 'output' not in path:
                print(f"Analyszing {path}")
                to_process = localize_tiff_run(path, self.settings, directory_path)
                to_process.full_analysis_ROI(mode = mode)
        print('########Finished########')
        self.output_folder = os.path.join(self.folder, 'output')