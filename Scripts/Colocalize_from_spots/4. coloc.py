import os
from glob import glob
import traceback
from spit import colocalize as coloc
from spit import plot_coloc
from spit import tools
from spit import table
import yaml
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support
from picasso.io import save_info

class Settings:
    def __init__(self):
        self.ch0 = '561'#'561nm'  
        self.ch1 = '638' #'638nm'  
        self.th = 250 #Threshold distance to consider colocalization in nm. Default by Chris: 250
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
    directory_path = r'D:\Data\Tom\Test_2'
    pathscsv = glob(directory_path + '/**/**.csv', recursive=True)
    paths_locs = list(set(os.path.dirname(file) for file in pathscsv))
    for image in paths_locs:
        if os.path.isdir(image):
            colocalizee(image)


def colocalizee(dirpath):
    settings = Settings()
    # format paths according to a specified ending, e.g. "488nm_locs.csv"
    ch0 = settings.ch0
    ch1 = settings.ch1

    # getting all filenames of the first channel, later look for corresponding second channel files
    if os.path.isdir(dirpath):
        print('Analyzing directory...')
        pathsCh0 = glob(dirpath + f'//**//*{ch0}*_locs.csv', recursive=True)
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
                px2nm = settings.get_px2nm(resultPath)
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

                plot_coloc.plot_coloc_stats(df_locs_ch0, df_locs_ch1, df_colocs,
                                            threshold=settings.th,
                                            path=pathPlots, dt=dt, roll_param=5)
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


if __name__ == '__main__':
    freeze_support()
    main()