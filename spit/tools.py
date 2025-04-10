from glob import glob
import cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.cm import get_cmap
from read_roi import read_roi_file
from matplotlib.path import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml
import picasso.io as io
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from scipy.ndimage import affine_transform
from scipy.optimize import curve_fit
import spit.tools as tools
# %%path tools


def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


def find_string(string_list, string):
    # define function which finds a given word in a list of strings and returns the last one
    found_string_list = [i for i in string_list if string in i]
    return found_string_list[len(found_string_list)-1]


def getExperimentName(path):
    """
    Given a data filepath or Run directory, it returns the experiment folder.
    Example: ...\Data\20211201 Jurkat 19 single dye\Run00035\Run00035_record.raw
            --> '20211201 Jurkat 19 single dye'

    """
    if os.path.isdir(path):
        experimentName = os.path.split(os.path.dirname(path))[1]
    elif os.path.isfile(path):
        experimentName = os.path.split(os.path.split(os.path.dirname(path))[0])[1]
    else:
        print('File or folder does not exist.')
    return experimentName


def getRunName(path):
    """
    Given a data filepath or Run directory, it returns the experiment folder.
    Example: ...\Data\20211201 Jurkat 19 single dye\Run00035\Run00035_record.raw
            --> 'Run00035'

    """
    if os.path.isdir(path):
        runName = os.path.split(path)[1]
    elif os.path.isfile(path):
        runName = os.path.split(os.path.split(path)[0])[1]
    else:
        print('File or folder does not exist.')
    return runName


def getOutputpath(path, foldername, keepFilename=False):
    """
    Takes input path and adds a folder in the path after the last folder. KeepFilename True if there is more than 
         one file processed (different channels for example)

    Example: Y:\04 DNA paint tracking\Data\20211201 Jurkat 19 single dye\Run00035\Run00035_record.raw
            --> '...\Data\20211201 Jurkat 19 single dye\Run00035\test\20211201 Jurkat 19 single dye_Run00035'
        with keepFilename= True:
            --> '...\Data\20211201 Jurkat 19 single dye\Run00035\test\20211201 Jurkat 19 single dye_Run00035_record'


    Parameters
    ----------
    path : string
        File or directory path of a Run folder or a file in a Run folder
    foldername : string
        Folder that should be inserted before the filename
    keepFilename: bool
        Keeps filename without extension if True.

    Returns
    -------
    outputpath: string
        Path of the output folder

    """
    experimentName = getExperimentName(path)
    if os.path.isdir(path):
        outputpath = os.path.join(path, foldername)
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
    if os.path.isfile(path):
        if keepFilename == True:
            outputpath = os.path.join(os.path.dirname(
                path), foldername, experimentName+'_'+os.path.splitext(os.path.split(path)[1])[0])
        else:
            outputpath = os.path.join(os.path.dirname(
                path), foldername, experimentName+'_'+os.path.split(os.path.dirname(path))[1])

        if not os.path.exists(os.path.dirname(outputpath)):
            os.mkdir(os.path.dirname(outputpath))

    return outputpath


def get_dt(path):
    resultPath = os.path.join(os.path.dirname(
        path), tools.getRunName(path))+'_result.txt'
    resultTxt = open(resultPath, 'r')
    resultLines = resultTxt.readlines()
    dtStr = find_string(resultLines, 'Camera Exposure')[17:-1]
    dt = 0.001*int(float((''.join(c for c in dtStr if (c.isdigit() or c == '.')))))
    return dt


def get_duration(path):
    resultPath = os.path.join(os.path.dirname(
        path), tools.getRunName(path))+'_result.txt'
    resultTxt = open(resultPath, 'r')
    resultLines = resultTxt.readlines()
    framesStr = tools.find_string(resultLines, 'Record Length')[17:-1]
    frames = int((''.join(c for c in framesStr if (c.isdigit()))))
    return frames
# %%File loading and saving and splitting


def df_convert2nm(df, px2nm=108):
    df = df.copy(deep=True)
    df.x = px2nm * df.x
    df.y = px2nm * df.y
    if 'sx' in df.columns:
        df.sx = px2nm * df.sx
        df.sy = px2nm * df.sy
    if 'lpx' in df.columns:
        df.lpx = px2nm * df.lpx
        df.lpy = px2nm * df.lpy
    if 'nearest_neighbor' in df.columns:
        df['nearest_neighbor'] = px2nm * df['nearest_neighbor']
    if 'loc_precision' in df.columns:
        df['loc_precision'] = px2nm * df['loc_precision']
    return df


def df_convert2px(df, px2nm=108):
    df = df.copy(deep=True)
    df.x = df.x / px2nm
    df.y = df.y / px2nm
    if 'sx' in df.columns:
        df.sx = df.sx / px2nm
        df.sy = df.sy / px2nm
    if 'lpx' in df.columns:
        df.lpx = df.lpx / px2nm
        df.lpy = df.lpy / px2nm
    if 'nearest_neighbors' in df.columns:
        df['nearest_neighbors'] = df['nearest_neighbors'] / px2nm
    if 'loc_precision' in df.columns:
        df['loc_precision'] = df['loc_precision'] / px2nm

    return df


def load_raw(path, prompt_info=None):
    info = load_info(path)
    dtype = np.dtype(info[0]["Data Type"])
    shape = (info[0]["Frames"], info[0]["Height"], info[0]["Width"])
    movie = np.memmap(path, dtype, "r", shape=shape)
    if info[0]["Byte Order"] != "<":
        movie = movie.byteswap()
        info[0]["Byte Order"] = "<"
    return movie, info


def load_info(path, qt_parent=None):
    path_base, path_extension = os.path.splitext(path)
    filename = path_base + ".yaml"
    with open(filename, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    return info


def load_locs(file, trackID='track.id', t='t'):
    """
    Load (simulated) loc files into dataframes and change dataframe.columns
    from swift-compatible to trackpy compatible and for better codability.
    """
    locs = pd.read_csv(file)
    info = io.load_info(file)
    if trackID in locs.columns:
        locs.rename(columns={trackID: 'particle'}, inplace=True)
        locs.rename(columns={t: 'frame'}, inplace=True)
    return (locs, info)


# def load_csv(file):
#     """
#     Load (simulated) linked loc files into dataframes and change dataframe.columns
#     from swift-compatible to a more general format.
#     """
#     linked = pd.read_csv(file)
#     if 'track.id' in linked.columns:
#         linked.rename(columns={'track.id': 'particle'}, inplace=True)
#     if 't' in linked.columns:
#         linked.rename(columns={'t': 'frame'}, inplace=True)
#     return linked


def dump_info(path, info):
    """
    Save info file as .yaml in path directory.
    """
    path_base, path_extension = os.path.splitext(path)
    filename = path_base + ".yaml"
    with open(filename, "w+") as newfile:
        newfile.write(yaml.dump(info))


def dump_csv(df, path, filename):
    """Save (simulated) dataframes as csv and change dataframe.columns 
    as swift-compatible.
    """
    filepath = os.path.splitext(path)
    if 'particle' in df.columns:
        dfDump = df.rename(columns={'particle': 'track.id'})
        dfDump = dfDump.rename(columns={'frame': 't'})
        dfDump.to_csv(os.path.join(filepath[0], filename), index=False)
    else:
        df.to_csv(os.path.join(filepath[0], filename), index=False)


def load_tif(file):
    '''
    Load .tif file.
    '''
    ######
    with tiff.TiffFile(file) as ff:
        data = ff.asarray()
    return data


def split_movie(path, transform):
    """
    Split a three-channel sequence of images (raw/tiff) in 3 single sequences. Assume a width of 2048px.

    Parameters
    ----------
    path : String
        Path to the file to load.
    transform : Bool
        To indicate if a correction must be applied on the movie.

    Returns
    -------
    movieSplit : List of array
        Contains 3 different arrays for the 3 different channels.
    infoSplit : List
        Info file. 

    """
    # load movie and info
    # movie, info = tools.load_raw(path)
    movie, info = io.load_movie(path)
    infoSplit = info
    infoSplit[0]['Width'] = 682

    movieSplit = np.array_split(movie[:, :, 1:2047], 3, axis=2)

    if transform == True:
        movieSplit, infoSplit = transform_movie(movieSplit, infoSplit)

    return movieSplit, infoSplit


def transform_movie(movieSplit, infoSplit):
    """
    Using the affine transformation matrix to correct the splitted movie inputs.

    Parameters
    ----------
    movieSplit : List of array
        Contains 3 different arrays for the 3 different channels.
    infoSplit : List
        Info file.

    Returns
    -------
    movieSplitCorrCrop : List of array
        Contains 3 different arrays for the 3 different channels, with the affine transformation applied.
    infoSplitCorrCrop : List
        Info file.

    """
    # load matrix and info
    root = __file__
    root = root.replace("split.py", "paramfiles/")
    H_lm = np.load(os.path.join(root, 'H_lm.npy'))
    H_rm = np.load(os.path.join(root, 'H_rm.npy'))
    infoH = tools.load_info(os.path.join(root, 'H.yaml'))
    shape = movieSplit[0].shape
    movieSplitCorr = [np.zeros((shape[0], shape[1], shape[2]),
                               dtype=np.uint16) for _ in range(3)]

    # middle channel doesn't need to be transformed
    movieSplitCorr[1] = movieSplit[1]

    for frame in tqdm(range(shape[0])):
        movieSplitCorr[0][frame] = h_affine_transform(movieSplit[0][frame], H_lm)
        movieSplitCorr[2][frame] = h_affine_transform(movieSplit[2][frame], H_rm)

    # define biggest overlapping rectangle for final cropping
    xlimA = infoH[0]['Biggest Rectangle xA']
    xlimB = infoH[0]['Biggest Rectangle xB']
    ylimA = infoH[0]['Biggest Rectangle yA']
    ylimB = infoH[0]['Biggest Rectangle yB']

    movieSplitCorrCrop = []
    for i in range(3):
        # cropping to biggest overlapping rectangle
        movieSplitCorrCrop.append(movieSplitCorr[i][:, ylimA:ylimB, xlimA:xlimB])

    # updating info file (same for all three channels)
    infoSplitCorrCrop = [infoSplit[0].copy()]
    infoSplitCorrCrop[0]['Alignment'] = 'Affine transformation from ' + root
    infoSplitCorrCrop[0]['Biggest Rectangle xA'] = int(xlimA)
    infoSplitCorrCrop[0]['Biggest Rectangle xB'] = int(xlimB)
    infoSplitCorrCrop[0]['Biggest Rectangle yA'] = int(ylimA)
    infoSplitCorrCrop[0]['Biggest Rectangle yB'] = int(ylimB)
    infoSplitCorrCrop[0]['Width'] = movieSplitCorrCrop[0].shape[2]
    infoSplitCorrCrop[0]['Height'] = movieSplitCorrCrop[0].shape[1]

    return movieSplitCorrCrop, infoSplitCorrCrop


def h_affine_transform(image, H):
    """
    Apply an affine transformation.

    Parameters
    ----------
    image : Array
        Image to correct.
    H : Array
        Transformation matrix.

    Returns
    -------
    Array
        Corrected image.

    """
    """ Transforms the image with the affine transformation matrix H.
    References:
    https://stackoverflow.com/questions/27546081/determining-a-homogeneous-
    affine-transformation-matrix-from-six-points-in-3d-usi/27547597#27547597
    http://elonen.iki.fi/code/misc-notes/affine-fit/
    """
    return affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))


def write_fig_info(path, **kwargs):
    with open(f'{path}_info.txt', 'w') as f:
        for k, v in kwargs.items():
            f.write(str(k) + ': ' + str(v) + '\n\n')

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
    return VCR_dict

# %% ROI tools


def get_roi_contour(path_roi, trimmed=False):
    roi = read_roi_file(path_roi)
    for key in roi:
        roi_name = key
    roi_x = roi[roi_name]['x']
    roi_y = roi[roi_name]['y']

    # vertices of the cropping polygon
    xc = np.array(roi_x)
    yc = np.array(roi_y)

    if trimmed:
        xmin = int(xc.min())
        ymin = int(yc.min())
        xc = xc-xmin
        yc = yc-ymin
    contour = np.vstack((xc, yc)).T
    contour_filled = np.append(contour, [contour[0]], axis=0)
    return contour_filled


def get_roi_centroid(contour):
    M = cv2.moments(contour.astype(int))
    cX = (M["m10"] / M["m00"])
    cY = (M["m01"] / M["m00"])
    centroid = np.array([[cX, cY]])
    return centroid


def get_roi_area(contour):
    # calculate area from roi contour, in px*px
    t = 0
    for count in range(len(contour)-1):
        y = contour[count+1][1] + contour[count][1]
        x = contour[count+1][0] - contour[count][0]
        z = y * x
        t += z
    return abs(t/2.0)


def get_roi_mask(locs, contour):
    # construct a Path from the vertices
    pth = Path(contour, closed=False)

    # mask the localizations that don't fall within the contour
    mask = pth.contains_points(locs[['x', 'y']].to_numpy())

    return mask


# %% Equations

# analytic MSD expressions
'''
.. _michalet:
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.041914
.. _manzo:
    https://iopscience.iop.org/article/10.1088/0034-4885/78/12/124601
'''


def msd_free(tau, a, b=0):
    '''
    MSD fitting eq. for simple brownian diffusion taking localization precision into account: ``msd=a*tau+b``
    According to: Xavier Michalet, Physical Review E, 82, 2010 (michalet_)
    '''
    msd = a*tau+b
    return msd


def msd_anomal(tau, a, b):
    '''
    MSD fitting eq. for anomalous diffusion: ``msd=a*tau**b``
    According to: Carlo Manzo, Report on Progress in Physics, 78, 2015 (manzo_)
    '''
    msd = a*tau**b
    return msd

# jump distance distribution equations

# pdf fit for 2 species: mobile, immobile


def pdf_jd(r, dt, d_coef1, a1, d_coef2, a2):
    return (a1*r/(2*d_coef1*dt)) * np.exp(-(r*r) / (4*d_coef1*dt)) + (a2*r/(2*d_coef2*dt)) * np.exp(-(r*r) / (4*d_coef2*dt))

# pdf fit for 1 species:


def pdf_jd_sub(r, dt, d_coef, a):
    return (a*r/(2*d_coef*dt)) * np.exp(-(r*r) / (4*d_coef*dt))

# exponential decay with or without offset


def exp_single(x, a, b, c=None):
    '''
    Exponential decay function: ``y=a*np.exp(-x/b)+c``
    '''
    if c:
        y = a*np.exp(-x/b)+c
    else:
        y = a*np.exp(-x/b)
    return y


# %% Fit tools
def fit_decay(values, offset=False):
    """
    Fit exponential decay with or without offset.
    Args:
      values : Series
        Values that exhibit exponential decay trend.

    Returns:   
       popt : Array
           Fit coefficients (a,b,c optional)

    """
    # Fitting
    x = values.index
    y = values
    N = len(y)
    try:
        if offset:
            # Init start values
            p0 = [np.median(y), N, y.iloc[-1]]
            popt, pcov = curve_fit(exp_single, x, y, p0=p0)
        else:
            # Init start values
            p0 = [np.median(y), N]
            popt, pcov = curve_fit(exp_single, x, y, p0=p0)
    except:
        popt = np.full(2, np.nan)

    return popt


# %% Dataframe tools


def scrape_data(path, matchstring=''):
    paths = glob(path + '/**/*'+matchstring, recursive=True)
    keptpaths = []
    for path in paths:
        if matchstring in os.path.basename(path):
            keptpaths.append(path)
    return keptpaths


def get_unique_trackIDs(df, group="cell_id"):
    # highest track_id per group, cumulated over groups and shifted one group down
    track_id_max = df.groupby([group])["track.id"].max() + 1
    track_id_cum = track_id_max.cumsum().shift(1, fill_value=0)

    df["track.id"] = df["track.id"] + \
        np.array(track_id_cum)[df[group].factorize()[0]]
    df["track.id"] = df["track.id"].astype(int)

    return df


def fill_track_gaps(df_tracks):
    df_tracks = df_tracks.reset_index(drop=True)
    df_tracks_filled = df_tracks.set_index(['track.id', 't']).unstack()

    df_tracks_filled.x = df_tracks_filled.x.ffill(
        axis=1).where(df_tracks_filled.x.bfill(axis=1).notna())

    df_tracks_filled.y = df_tracks_filled.y.ffill(
        axis=1).where(df_tracks_filled.y.bfill(axis=1).notna())

    df_tracks_filled = df_tracks_filled.stack().reset_index()
    return df_tracks_filled


def join_segments(df_tracks, id_list):
    df_tracks.loc[df_tracks['track.id'].isin(id_list), 'track.id'] = id_list[0]
    # drop duplicates in case segments overlap
    df_tracks = df_tracks.drop_duplicates(['track.id', 't'])

    # correct new loc_count
    df_tracks.loc[df_tracks['track.id'] == id_list[0], 'loc_count'] = df_tracks.loc[df_tracks['track.id']
                                                                                    == id_list[0]].shape[0]
    return df_tracks

# %% Misc: Colors, prefixes


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def float2SI(number):
    units = {0: ' ',
             1: 'K',  2: 'M',  3: 'G',  4: 'T',  5: 'P',  6: 'E',  7: 'Z',  8: 'Y',  9: 'R',  10: 'Q',
             -1: 'm', -2: 'u', -3: 'n', -4: 'p', -5: 'f', -6: 'a', -7: 'z', -8: 'y', -9: 'r', -10: 'q'
             }

    mantissa, exponent = f"{number:e}".split("e")
    unitRange = int(exponent)//3
    unit = units.get(unitRange, None)
    unitValue = float(mantissa)*10**(int(exponent) % 3)
    return f"{unitValue:.0f} {unit}" if unit else f"{number:.5e}"


def power_to_wcm2(laser, power):
    illuminated_area = np.pi*(0.01*0.01*221.184/2)**2  # illuminated area in cm2
    laserdict = {405: 97, 488: 162, 561: 163*0.68,
                 638: 114}  # max power after fiber in mW
    mwcm2 = 0.01*power*laserdict[laser]/illuminated_area
    return mwcm2/1000
