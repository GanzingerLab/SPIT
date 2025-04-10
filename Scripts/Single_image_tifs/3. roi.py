# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 15:13:47 2025

@author: castrolinares
"""
import os
from glob import glob
import traceback
import pandas as pd
from tqdm import tqdm
from spit import tools
import numpy as np
from picasso.io import save_info

from multiprocessing import freeze_support


def main(): 
    directory_path = r'D:\Data\Chi_data\20250401 for Gerard\CART3_FMC63_LoExp\Run00001_timelapse\output'
    pathscsv = glob(directory_path + '/**/**.csv', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    for image in paths_locs:
        if os.path.isdir(image):
            roi(image)


def roi(files):
    '''Restrict localizations to ROIs'''

    # format filepaths
    if os.path.isdir(files):
        print('Analyzing directory...')
        paths = glob(files + '/*nm_locs.csv')

    # initialize placeholders
    skippedPaths = []

    # print all kept paths
    for path in paths:
        print(path)
    print(f'A total of {len(paths)} files detected...')
    print('--------------------------------------------------------')

    # main loop
    for idx, path in tqdm(enumerate(paths), desc='Saving new loc-files...', total=len(paths)):
        print('--------------------------------------------------------')
        print(f'Running file {path}')
        try:
            (df_locs, info) = tools.load_locs(path)
            # Look for ROI paths
            pathsROI = glob(os.path.dirname(path) + '/*.roi', recursive=False)
            print(f'Found {len(pathsROI)} ROI.')

            dict_roi = {'cell_id': [], 'path': [], 'contour': [],
                        'area': [], 'roi_mask': [], 'centroid': []}
            
            # this stuff needs to go into tools
            df_locs = df_locs.drop('cell_id', axis=1)
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
            
            
            
if __name__ == '__main__':
    freeze_support()
    main()