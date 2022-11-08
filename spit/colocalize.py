import numpy as np
import pandas as pd
from tqdm import tqdm
from spit import localize
from sklearn import preprocessing
from scipy.spatial import distance as dist


def colocalize_from_locs(locsCh0, locsCh1, threshold_dist):
    # concatenate channels for cdist analysis
    locsCh0['channel'] = 0
    locsCh1['channel'] = 1

    # create localization ID columns from indices for both files
    locsCh0 = locsCh0.rename_axis('locID0').reset_index()
    locsCh1 = locsCh1.rename_axis('locID1').reset_index()

    locsAll = pd.concat([locsCh0, locsCh1], ignore_index=True)

    # get colocalizations

    # initialize dictionary
    d = {'t': [], 'locID0': [], 'locID1': [], 'dist': [], 'loc0x': [], 'loc0y': [],
         'loc1x': [], 'loc1y': [], 'loc_precision0': [], 'loc_precision1': [], 'intensity0': [], 'intensity1': [],
         'net_gradient0': [], 'net_gradient1': [], 'bg0': [], 'bg1': []}

    # main loop linked_all.frame.max()
    for i in tqdm(range(int(locsAll['t'].max())+1)):
        # to avoid dealing with the whole dataframe for all operations, only take rows of current frame
        locsAllCurrent = locsAll[locsAll.t == i]

        # caclulate pairwise distances between different channels frame by frame and store in matrix for frame [i]
        distanceMatrix = dist.cdist(locsAllCurrent[(locsAllCurrent['channel'] == 0)][['x', 'y']],
                                    locsAllCurrent[(locsAllCurrent['channel'] == 1)][[
                                        'x', 'y']],
                                    metric='euclidean')

        # threshold distances into binary matrix
        try:
            binaryMatrix = 1 - \
                preprocessing.binarize(
                    distanceMatrix, threshold=threshold_dist, copy=True)
        except:
            binaryMatrix = np.zeros(distanceMatrix.shape)
            # print(f'No (co-)localizations in frame {i}')

        # return non-zero indices from binary matrix
        nonzeroCh0, nonzeroCh1 = np.nonzero(binaryMatrix)

        # return list of distances that are below the treshhold
        distList = distanceMatrix[nonzeroCh0, nonzeroCh1]

        # extract list of binary matrix indices of localizations present in ith frame
        LocsIndices0 = locsAllCurrent.locID0[(
            locsAllCurrent['channel'] == 0)].to_numpy()
        LocsIndices1 = locsAllCurrent.locID1[(
            locsAllCurrent['channel'] == 1)].to_numpy()

        # reduce previous list to only indices with colocalization events in ith frame
        colocIndicesList0 = LocsIndices0[nonzeroCh0].astype(int).tolist()
        colocIndicesList1 = LocsIndices1[nonzeroCh1].astype(int).tolist()

        # add to dictionary
        d['t'].extend(np.full(len(colocIndicesList0), i))
        d['locID0'].extend(colocIndicesList0)
        d['locID1'].extend(colocIndicesList1)
        d['dist'].extend(distList)
        d['loc0x'].extend(locsCh0.iloc[colocIndicesList0].x)
        d['loc0y'].extend(locsCh0.iloc[colocIndicesList0].y)
        d['loc1x'].extend(locsCh1.iloc[colocIndicesList1].x)
        d['loc1y'].extend(locsCh1.iloc[colocIndicesList1].y)
        d['loc_precision0'].extend(locsCh0.iloc[colocIndicesList0].loc_precision)
        d['loc_precision1'].extend(locsCh1.iloc[colocIndicesList1].loc_precision)
        d['intensity0'].extend(locsCh0.iloc[colocIndicesList0].intensity)
        d['intensity1'].extend(locsCh1.iloc[colocIndicesList1].intensity)
        d['net_gradient0'].extend(locsCh0.iloc[colocIndicesList0].net_gradient)
        d['net_gradient1'].extend(locsCh1.iloc[colocIndicesList1].net_gradient)
        d['bg0'].extend(locsCh0.iloc[colocIndicesList0].bg)
        d['bg1'].extend(locsCh1.iloc[colocIndicesList1].bg)

    colocEvents = pd.DataFrame.from_dict(d)

    # clean-up duplicates
    # -> if one particle colocalizes with two or more from the other channel: keep the closer one
    colocEvents = colocEvents.groupby('locID1').min().reset_index()
    colocEvents = colocEvents.groupby('locID0').min().reset_index()

    colocEvents['colocID'] = range(1, len(colocEvents['t']) + 1)
    colocEvents['x'] = colocEvents[['loc0x', 'loc1x']].mean(axis=1)
    colocEvents['y'] = colocEvents[['loc0y', 'loc1y']].mean(axis=1)
    colocEvents['loc_precision'] = colocEvents[[
        'loc_precision0', 'loc_precision1']].mean(axis=1)
    colocEvents['intensity'] = colocEvents[['intensity0', 'intensity1']].sum(axis=1)
    colocEvents['net_gradient'] = colocEvents[[
        'net_gradient0', 'net_gradient1']].sum(axis=1)
    colocEvents['bg'] = colocEvents[['bg0', 'bg1']].sum(axis=1)
    colocEvents = colocEvents.drop(columns=['loc_precision0', 'loc_precision1', 'intensity0',
                                            'intensity1', 'net_gradient0', 'net_gradient1', 'bg0', 'bg1'])
    colocEvents['nearest_neighbor'] = localize.get_nearest_neighbor(colocEvents)

    return colocEvents
