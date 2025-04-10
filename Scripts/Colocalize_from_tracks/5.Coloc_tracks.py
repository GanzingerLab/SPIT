import os
from glob import glob
import traceback
from spit import colocalize as coloc
from spit import tools
import yaml
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support
from picasso.io import save_info

class Settings:
    def __init__(self):
        self.ch0 = '638'#'561nm'  
        self.ch1 = '488' #'638nm'  
        self.th = 300 #Threshold distance to consider colocalization in nm. Default by Chris: 250
        self.min_overlapped_frames = 5 #minimum amunt of frames in which the spots of the tracks have to be closer than the threshold distance set in th in a row.
        self.min_len_track = 5 #minimum length of a track (in frames) to consider it for the analysis.
        self.suffix = '' #sufix for the name of the file, if necessary. 
        self.dt = None #specify the exposure time (in seconds!) or None. If None, the script will look for it in the _result.txt file. 
    def get_px2nm(self, file): #if self.transform = True, this will get the correct naclib coefficients (Annapurna VS K2)
        result_txt  = tools.read_result_file(file) #this opens the results.txt file to check the microscope used. 
                #It should be in a folder called paramfile inside the folder where the script is located. 
        if result_txt['Computer'] == 'ANNAPURNA': 
            return 90.16
        elif result_txt['Computer'] == 'K2-BIVOUAC':
            return 108

def main(): 
    directory_path = r'D:\Data\Chi_data\first data\output2\Run00002'
    pathscsv = glob(directory_path + '/**/**.csv', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    for image in paths_locs:
        if os.path.isdir(image):
            colocalize_tracks(image)


def colocalize_tracks(dirpath):
    settings = Settings()
    # format paths according to a specified ending, e.g. "488nm_locs.csv"
    ch0 = settings.ch0
    ch1 = settings.ch1

    # getting all filenames of the first channel, later look for corresponding second channel files
    if os.path.isdir(dirpath):
        print('Analyzing directory...')
        pathsCh0 = glob(dirpath + f'//**//*{ch0}*_locs_nm_trackpy.csv', recursive=True)
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
                # print(f'\nFound a second channel for file {idx}.')
                if not settings.dt == None:
                    dt = settings.dt
                else:
                    resultPath = os.path.join(os.path.dirname(pathCh0), pathCh0.split('\\')[-2]+'_result.txt')
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


if __name__ == '__main__':
    freeze_support()
    main()
