# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:23:45 2025

@author: castrolinares
"""
import os
import shutil
import traceback
import yaml
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from glob import glob
from spit import linking as link
from spit import tools
from spit import table
from spit import plot_diffusion
from multiprocessing import freeze_support


class Settings:
    def __init__(self):
        self.coloc = False  #Bool --> are you using a colocalization file or not? 
        self.dt = None  #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
        self.quick = False  #use the quick version? 200px squared from the center and only 500 frames. 
        self.roi = True #Do you want to filter per ROI? 
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.fil_len = 20 #filter the length of the tracks. Tracks shorter than this ammount of frames will be filtered
        self.fil_diff = 0.0002 #Filter for immobile particles. Tracks with a diffusion coefficient smaller than this number will be filtered
        self.tracker = 'trackpy' #Tracker algorithm to use: trackpy or swift. After talking with Chris, swift is very complicated and the focus of the developers is not 
        # really tracking, but the diffusion rates. So, swift is not implemented, and I am not sure if it will. 
    #################Tracking params############
        self.memory = 1 #max number of frames from which a particle can disappear 
        self.search = 15 #max search range for trackpy linking in px 
    def get_px2nm(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        result_txt  = read_result_file(file) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108
def main(): 
    directory_path = r'D:\Data\Tom\Test_2'
    # directory_path = r'C:\Users\castrolinares\Data analysis\SPIT_G\Raquel_6Feb2024\example data\GCL002_Sample_from_yesterday\output\after_adding_dil2\Run00010'
    pathscsv = glob(directory_path + '/**/**.csv', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    print(paths_locs)
    for image in paths_locs:
        if os.path.isdir(image):
            linkk(image)


def linkk(folder):
    try:
        settings = Settings()    
        if os.path.isdir(folder):
            if settings.coloc:
                paths = glob(folder + '/*colocs.csv', recursive=True)
            else:
                paths = glob(folder + '/*_locs.csv', recursive=True)
                
        skippedPaths = []
        quick = settings.quick
    
        # main loop
        for idx, path in tqdm(enumerate(paths), desc='Linking localizations...', total=len(paths)):
            try:
                if settings.roi and 'roi' not in path:
                    skip = path.split('\\')[-1]
                    print(f"\n\n Skipping {skip} bacause it is not filtered by roi and within Settings self.roi == {settings.roi}\n")
                    continue
                if not settings.roi and 'roi' in path:
                    skip = path.split('\\')[-1]
                    print(f"\n\n Skipping {skip} bacause it is filtered by roi and within Settings self.roi == {settings.roi}\n")
                    continue
                (df_locs, info) = tools.load_locs(path)
                if not settings.coloc:
                    # fix locIDs before they get mixed up by linking
                    df_locs = df_locs.rename_axis('locID').reset_index()
        #         # retrieve exposure time
                resultPath = '\\'.join(path.split('\\')[:-1]) + '\\' + [element for element in path.split('\\') if element.startswith('Run')][0] + '_result.txt'
                if not settings.dt == None:
                    dt = settings.dt
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
                        dict_roi['cell_id'].append(idx)
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
                px2nm = settings.get_px2nm(resultPath)
                df_locs_nm = tools.df_convert2nm(df_locs, px2nm)
                path_nm = os.path.splitext(path)[0]+quick+settings.suffix+'_nm.csv'
                df_locs_nm.to_csv(path_nm, index=False)
                path_plots_loc = tools.getOutputpath(path_nm, 'plots', keepFilename=True)
        
                # tau_bleach = plot_diffusion.plot_loc_stats(df_locs_nm, path_plots_loc, dt=dt)
        
                # prepare rest of the paths
                path_output = os.path.splitext(path_nm)[0] +'_'+ settings.tracker
                path_plots = tools.getOutputpath(path_nm, 'plots', keepFilename=True) +'_'+ settings.tracker
        
                # Choose tracking algorithmus
                if settings.tracker == 'trackpy':
                    # print('Using trackpy.\n')
                    # export parameters to yaml
                    with open(os.path.splitext(path_nm)[0] +'_'+ settings.tracker + '.yaml', 'w') as f:
                        yaml.dump(vars(settings), f)
        
                    df_tracksTP = link.link_locs_trackpy(df_locs, search=settings.search, memory=settings.memory)
        
                    # # linked file is saved with pixel-corrected coordinates and
                    # # swiftGUI compatible columns, and unique track.ids
                    df_tracks = tools.df_convert2nm(df_tracksTP, px2nm)
                    df_tracks['seg.id'] = df_tracksTP['track.id']
                    if 'roi' in path:
                        df_tracks = tools.get_unique_trackIDs(df_tracks)
                    df_tracks.to_csv(path_output + '.csv', index=False)
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
        
            # Filter short tracks and immobile particles
                if not settings.coloc:
                    df_statsF = link.filter_df(df_stats, filter_length=settings.fil_len, filter_D=settings.fil_diff)
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
