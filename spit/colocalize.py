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
         'net_gradient0': [], 'net_gradient1': [], 'bg0': [], 'bg1': [], 'cell_id': []}
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
                preprocessing.binarize(distanceMatrix, threshold=threshold_dist, copy=True)
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
        d['cell_id'].extend(locsCh0.iloc[colocIndicesList0].cell_id)

    colocEvents = pd.DataFrame.from_dict(d)

    # clean-up duplicates
    # -> if one particle colocalizes with two or more from the other channel: keep the closer one
    colocEvents = colocEvents.groupby('locID1').min().reset_index()
    colocEvents = colocEvents.groupby('locID0').min().reset_index()

    colocEvents['colocID'] = range(1, len(colocEvents['t']) + 1)
    colocEvents['x'] = colocEvents['loc0x']
    colocEvents['y'] = colocEvents['loc0y']
    colocEvents['loc_precision'] = colocEvents[[
        'loc_precision0', 'loc_precision1']].mean(axis=1)
    colocEvents['intensity'] = colocEvents[['intensity0', 'intensity1']].sum(axis=1)
    colocEvents['net_gradient'] = colocEvents[[
        'net_gradient0', 'net_gradient1']].sum(axis=1)
    colocEvents['bg'] = colocEvents[['bg0', 'bg1']].sum(axis=1)
    colocEvents = colocEvents.drop(columns=['loc_precision0', 'loc_precision1', 'intensity0',
                                            'intensity1', 'net_gradient0', 'net_gradient1', 'bg0', 'bg1'])
    if len(list(set(d['t']))) > 1:
         colocEvents['nearest_neighbor'] = localize.get_nearest_neighbor(colocEvents)
    cell_id = colocEvents.pop('cell_id')
    colocEvents['cell_id'] = cell_id

    return colocEvents

#%%
def coloc_tracks(df_488, df_638, leng = 10, max_distance = 250, n = 3):
    matched_tracks = []
    for t1 in tqdm(df_488[df_488.loc_count >=leng]['track.id'].unique()):
        if df_488[df_488['track.id'] == t1].shape[0] >= leng:
            track1 = df_488[df_488['track.id'] == t1]
            # Calculate the bounding box for track1
            min_x1, max_x1 = track1['x'].min() - 100, track1['x'].max() + 100
            min_y1, max_y1 = track1['y'].min() - 100, track1['y'].max() + 100
            
            # Filter tracks in df_638 based on the bounding box
            filtered_tracks = df_638[(df_638['x'] >= min_x1) & (df_638['x'] <= max_x1) & (df_638['y'] >= min_y1) & (df_638['y'] <= max_y1)]
            
            for t2 in filtered_tracks[filtered_tracks.loc_count >= leng]['track.id'].unique():
                if df_638[df_638['track.id'] == t2].shape[0] >= leng:
                    track2 = df_638[df_638['track.id'] == t2]
                    common_frames = track1[track1['t'].isin(track2['t'])]
                    coords1 = common_frames[['x', 'y']].values
                    coords2 = track2[track2['t'].isin(common_frames['t'])][['x', 'y']].values
                    # Calculate distances using np.linalg.norm
                    distances = np.linalg.norm(coords1 - coords2, axis=1)
                    colocalized_frames = common_frames[distances <= max_distance]
                    colocalized_times = colocalized_frames['t'].values

                    # Check for n consecutive distances <= max_distance
                    for i in range(len(distances) - (n-1)):
                        if all(distances[i:i+n] <= max_distance): #possibility to skip? as in memory in trackpy? Consider time to check distances. 
                            matched_tracks.append([t1, 
                                                   t2, 
                                                   colocalized_times, 
                                                   common_frames['t'].values, 
                                                   len(colocalized_times)/len(common_frames['t'].values)*100,
                                                   np.sum(distances <= max_distance),
                                                   distances, 
                                                   np.average(distances[distances <= max_distance]), 
                                                   track1[['x','y']], 
                                                   track2[['x','y']]])
                            break
    if matched_tracks:
        matched_df = pd.DataFrame(matched_tracks, columns = ['track.id0', 
                                                             'track.id1', 
                                                             'overlap_t', 
                                                             'coexist_s',
                                                             '%col_time' ,
                                                             '#frames_coloc', 
                                                             'distances', 
                                                             'average_distance', 
                                                             'track0', 
                                                             'track1' ])
        print('\n')
        
        
        repeated_id0 = matched_df['track.id0'].value_counts()
        repeated_id0 = repeated_id0[repeated_id0 > 1].index
        
        repeated_id1 = matched_df['track.id1'].value_counts()
        repeated_id1 = repeated_id1[repeated_id1 > 1].index
        to_remove = []
        for i in repeated_id0:
            rows = matched_df[matched_df['track.id0'] == i]
            indices = rows.index.tolist()
            # print(indices)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    if any(element in matched_df.loc[indices[i]].overlap_t for element in matched_df.loc[indices[j]].overlap_t):
                        idx = matched_df.loc[[indices[i], indices[j]]].average_distance.idxmax()
                        if int(idx) in matched_df.index:
                            to_remove.append(idx)
                            # print(matched_df.loc[to_remove])
                            # matched_df = matched_df.drop(a)
       
        
        
        for i in repeated_id1:
            rows = matched_df[matched_df['track.id1'] == i]
            indices = rows.index.tolist()
            # print(indices)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    if any(element in matched_df.loc[indices[i]].overlap_t for element in matched_df.loc[indices[j]].overlap_t):
                        idx = matched_df.loc[[indices[i], indices[j]]].average_distance.idxmax()
                        if int(idx) in matched_df.index:
                            to_remove.append(idx)
        
        # Initialize an empty list to store the concatenated data
        concatenated_data = []
        matched_df = matched_df.drop(to_remove)
        matched_df['colocID'] = range(len(matched_df))
        matched_df = matched_df[['colocID', 'track.id0','track.id1','overlap_t', 'coexist_s','%col_time' ,'#frames_coloc', 'distances', 'average_distance', 'track0', 'track1' ]]
        
        # Iterate through the matched_df DataFrame
        for index, row in matched_df.iterrows():
            coloc_id = row['colocID']
            track_id0 = row['track.id0']
            track_id1 = row['track.id1']
            
            # Extract the localization data for track.id0
            track0_localizations = df_488[df_488['track.id'] == track_id0].copy()
            track0_localizations.drop(['sx', 'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'loc_precision', 'nearest_neighbor', 'seg.id'], axis=1, inplace=True)
            track0_localizations = track0_localizations.add_suffix('_0')
            
            # Extract the localization data for track.id1
            track1_localizations = df_638[df_638['track.id'] == track_id1].copy()
            track1_localizations.drop(['sx', 'sy', 'bg', 'lpx', 'lpy', 'ellipticity', 'net_gradient', 'loc_precision', 'nearest_neighbor', 'seg.id'], axis=1, inplace=True)
            track1_localizations = track1_localizations.add_suffix('_1')
            
            # Identify the unique time points where either track has data
            combined_times = pd.Series(list(set(track0_localizations['t_0']).union(set(track1_localizations['t_1'])))).sort_values().reset_index(drop=True)
            combined_times.name = 't'
            
            # Merge the localization data based on the combined time points
            merged = pd.merge(combined_times.to_frame(), track0_localizations, left_on='t', right_on='t_0', how='left')
            merged = pd.merge(merged, track1_localizations, left_on='t', right_on='t_1', how='left')
            
            # Drop the original time columns and add importanbt new columns 
            merged.drop(['t_0', 't_1'], axis=1, inplace=True)
            merged['colocID'] = coloc_id
            merged['cell_id'] = merged[['cell_id_0', 'cell_id_1']].mean(axis=1)
            merged.drop(['cell_id_0', 'cell_id_1'], axis = 1, inplace = True)
            merged['distance'] = merged.apply(lambda row: np.linalg.norm([row['x_0'] - row['x_1'], row['y_0'] - row['y_1']]) if pd.notna(row['x_0']) and pd.notna(row['x_1']) and pd.notna(row['y_0']) and pd.notna(row['y_1']) else np.NAN, axis=1)
            merged['x'] = merged.apply(lambda row: row[['x_0', 'x_1']].mean() if pd.notna(row['x_0']) and pd.notna(row['x_1']) and row['distance'] <= max_distance else np.NAN, axis=1)
            merged['y'] = merged.apply(lambda row: row[['y_0', 'y_1']].mean() if pd.notna(row['y_0']) and pd.notna(row['y_1']) and row['distance'] <= max_distance else np.NAN, axis=1)
        
            
            # Append the merged data to the list
            concatenated_data.append(merged)
        
        # Concatenate all the data into a single DataFrame
        final_df = pd.concat(concatenated_data, ignore_index=True)
        new_order = ['colocID', 'track.id_0','track.id_1', 't', 'locID_0','locID_1','x', 'y','distance', 'x_0', 'y_0', 'x_1', 'y_1', 'intensity_0', 'intensity_1', 'cell_id', 'loc_count_0', 'loc_count_1']
        
        # Reorder the columns
        final_df = final_df[new_order]
        return final_df, matched_df
    else:
        print("\n No colocs")
        return None, None
