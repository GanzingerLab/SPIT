# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------SPIT Command Line Interface----------------------------------------------

# %% table
'''
Create a wiki-compatible table from K2 result.txt files.
'''


def _table(args):
    from spit import table as table

    # check if file directory is specified
    if args.path is None:
        raise FileNotFoundError('No directory specified')
    else:
        path = args.path
    table.createTable(path)

# %% localize


def _localize(args):
    '''Localizes spots using Picasso'''
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
    from spit import localize
    from spit import tools

    # check if file directory is specified
    if args.files is None:
        raise FileNotFoundError('No directory specified')
    else:
        files = args.files

    if not [x for x in (args.gradient, args.gradient0, args.gradient1, args.gradient2) if x is not None]:
        raise AttributeError('You need to set a gradient value.')

    # print all parameters
    ws = '  '
    print('Localize - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        print(f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')

    # format filepaths
    if os.path.isdir(files):
        print('Analyzing directory...')
        if args.recursive == True:
            pathsRaw = glob(files + '/**/**.raw', recursive=True)
            pathsTif = glob(files + '/**/**.tif', recursive=True)
            paths = pathsRaw + pathsTif
        else:
            pathsRaw = glob(files + '/*.raw', recursive=True)
            pathsTif = glob(files + '/*.tif', recursive=True)
            paths = pathsRaw + pathsTif
    elif os.path.isfile(files):
        paths = glob(files)

    # remove filepaths that do match string
    if args.avoidstring:
        keptpaths = []
        for path in paths:
            if args.avoidstring not in path:
                keptpaths.append(path)
        paths = keptpaths

    # remove filepaths that do not match string
    if args.matchstring:
        keptpaths = []
        for path in paths:
            if args.matchstring in path:
                keptpaths.append(path)
        paths = keptpaths

    skippedPaths = []
    # print all kept paths
    for path in paths:
        print(path)
    print(f'A total of {len(paths)} files detected...')
    print('--------------------------------------------------------')

    # main operations
    if paths:
        box = args.box_side_length
        min_net_gradient = args.gradient
        camera_info = {}
        camera_info['baseline'] = args.baseline
        camera_info['sensitivity'] = args.sensitivity
        camera_info['gain'] = args.gain
        camera_info['qe'] = args.qe
        transformInfo = False
        px2um = 0.108

        for i, path in enumerate(paths):
            try:
                movie, info = load_movie(path)
                # figure out how to treat the movie files
                # if no overall gradient is given, split channels and treat them separately
                # movie now is one channel, either by splitting the input file or the input file itself is just one channel
                if args.gradient == None:
                    movieList, info = tools.split_movie(path, transform=False)
                    min_net_gradient = [args.gradient0,
                                        args.gradient1, args.gradient2]
                    area = info[0]['Width']*info[0]['Height']*px2um*px2um
                # if overall gradient is given, movie is localized as a whole
                else:
                    movieList, info = [movie], info
                    min_net_gradient = [args.gradient]
                    area = info[0]['Width']*info[0]['Height']*px2um*px2um

                # go through the list that was created based on the CLI information given
                for j in range(len(movieList)):
                    movie = movieList[j]

                    if min_net_gradient[j] == None:
                        continue
                    if args.gradient != None:
                        print(f'Localizing file {path}')
                    else:
                        print(f'Localizing channel {j} of {path}')
                    print('--------------------------------------------------------')

                    current, futures = identify_async(
                        movie, min_net_gradient[j], box)
                    ids = identifications_from_futures(futures)
                    # choosing fit-method (com for tracking, lq for stationary stuff)
                    if args.fit_method == 'lq':
                        spots = get_spots(movie, ids, box, camera_info)
                        theta = gausslq.fit_spots_parallel(spots, asynch=False)
                        locs = gausslq.locs_from_fits(
                            ids, theta, box, args.gain)
                    elif args.fit_method == 'com':
                        spots = get_spots(movie, ids, box, camera_info)
                        theta = avgroi.fit_spots_parallel(spots, asynch=False)
                        locs = avgroi.locs_from_fits(
                            ids, theta, box, args.gain)
                    else:
                        print('This should never happen...')

                    df_locs = pd.DataFrame(locs)

                    # Compatibility with Swift
                    df_locs = df_locs.rename(
                        columns={'frame': 't', 'photons': 'intensity'})

                    # adding localization precision, nearest neighbor, change photons, add cell_id column
                    df_locs['loc_precision'] = df_locs[[
                        'lpx', 'lpy']].mean(axis=1)
                    df_locs['nearest_neighbor'] = localize.get_nearest_neighbor(df_locs)
                    df_locs['cell_id'] = 0
                    # correction only makes sense if we are dealing with two/three channel data
                    if args.transform and args.gradient == None:
                        root = __file__
                        root = root.replace("__main__.py", "paramfiles/")
                        naclibCoefficients = pd.read_csv(
                            os.path.join(root, 'naclib_coefficients.csv'))

                        if j == 0:
                            df_locs, dataset = localize.transform_locs(df_locs,
                                                                       naclibCoefficients,
                                                                       channel=j,
                                                                       fig_size=(682, 682))
                            transformInfo = 'true, based on '+str(dataset)
                        if j == 2:
                            df_locs, dataset = localize.transform_locs(df_locs,
                                                                       naclibCoefficients,
                                                                       channel=j,
                                                                       fig_size=(682, 682))
                            transformInfo = 'true, based on '+str(dataset)
                        if j == 1:
                            transformInfo = 'false, reference channel'

                    localize_info = {
                        'Generated by': 'Picasso Localize',
                        'Box Size': box,
                        'Min. Net Gradient': min_net_gradient[j],
                        'Color correction': transformInfo,
                        'Area': float(area),
                        'Fit method': args.fit_method
                    }
                    infoNew = info.copy()
                    infoNew.append(localize_info)

                    base, ext = os.path.splitext(path)

                    if args.gradient != None:
                        pathChannel = base
                    else:
                        pathChannel = base + '_ch' + str(j)

                    pathOutput = pathChannel + args.suffix + '_locs.csv'

                    df_locs.to_csv(pathOutput, index=False)
                    save_info(os.path.splitext(pathOutput)[0]+'.yaml', infoNew)

                    if args.plot == True:
                        # centersInit = [1100,2200] #for combFit
                        plotPath = tools.getOutputpath(
                            pathOutput, 'plots', keepFilename=True)
                        localize.plot_loc_stats(df_locs, plotPath)

                print(f'File saved to {pathOutput}')
                print('                                                        ')
            except Exception:
                skippedPaths.append(path)

                print('--------------------------------------------------------')
                print(f'Path {path} could not be analyzed. Skipping...\n')
                traceback.print_exc()

    else:
        print('Error. No files found.')
        raise FileNotFoundError
    print('--------------------------------------------------------')
    print('/////////////////////FINISHED//////////////////////////')
    print('--------------------------------------------------------')
    if skippedPaths:
        print('Skipped paths:')
        for skippedPath in skippedPaths:
            print(f'\n{skippedPath}\n')


# %% transform


def _transform(args):
    import os
    from glob import glob
    import traceback
    from spit import localize
    import yaml
    import pandas as pd
    from tqdm import tqdm
    from picasso.io import save_info
    from spit import tools

    # print params
    ws = '  '
    print('Transform - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        print(f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')

    # format paths according to a specified ending, e.g. "ch2_locs.csv"
    dirpath = args.files
    ch0 = args.channel0
    ch1 = args.channel1  # reference channel
    ch2 = args.channel2

    # getting all filenames of the first channel, later look for corresponding channel files
    if os.path.isdir(dirpath):
        print('Analyzing directory...')
        if args.recursive == True:
            pathsCh0 = glob(dirpath + f'//**//*{ch0}*_locs.csv', recursive=True)
        else:
            pathsCh0 = glob(dirpath + f'//*{ch0}*_locs.csv', recursive=False)

        print(f'Found {len(pathsCh0)} files for channel 0...')

        # remove filepaths that do match string
        if args.avoidstring:
            keptpaths = []
            for path in pathsCh0:
                if args.avoidstring not in path:
                    keptpaths.append(path)
            pathsCh0 = keptpaths

        # remove filepaths that do not match string
        if args.matchstring:
            keptpaths = []
            for path in pathsCh0:
                if args.matchstring in path:
                    keptpaths.append(path)
            pathsCh0 = keptpaths

        for path in pathsCh0:
            print(path)
    else:
        raise FileNotFoundError('Directory not found')
    print('--------------------------------------------------------')
    skippedPaths = []
    # main loop

    root = __file__
    root = root.replace("__main__.py", "paramfiles/")
    naclibCoefficients_path = os.path.join(root, 'naclib_coefficients.csv')
    naclibCoefficients = pd.read_csv(naclibCoefficients_path)
    transform_info = {
        'Transformation parameters': naclibCoefficients_path}

    for idx, pathCh0 in tqdm(enumerate(pathsCh0), desc='Looking for other channels...'):
        try:
            # read in the file
            df_locs_ch0_original, info_ch0 = tools.load_locs(pathCh0)
            df_locs_ch0, dataset = localize.transform_locs(df_locs_ch0_original,
                                                           naclibCoefficients,
                                                           channel=0,
                                                           fig_size=(682, 682))

            # get right output paths
            pathOutput = os.path.splitext(
                pathCh0)[0][:-9] + args.suffix + '_transform'

            infoNew = info_ch0.copy()
            infoNew.append(transform_info)
            save_info(pathOutput + '_ch0_locs.yaml', infoNew)

            print('\nSaving transformed localizations...')
            df_locs_ch0.to_csv(pathOutput + '_ch0_locs.csv', index=False)

            dirname = os.path.dirname(pathCh0)

            pathCh1 = glob(pathCh0.replace('ch0', 'ch1'))
            pathCh2 = glob(pathCh0.replace('ch0', 'ch2'))

            if len(pathCh1) == 1:
                print(f'\nFound the reference channel for file {idx}.')
                df_locs_ch1, info_ch1 = tools.load_locs(pathCh1[0])

                # get right output paths
                pathOutput = os.path.splitext(
                    pathCh1[0])[0][:-9] + args.suffix + '_transform'
                infoNew = info_ch1.copy()
                infoNew.append(transform_info)
                save_info(pathOutput + '_ch1_locs.yaml', infoNew)

                print('Saving untransformed localizations...')
                df_locs_ch1.to_csv(pathOutput + '_ch1_locs.csv', index=False)

            else:
                print(f'\nFound no reference channel for file {idx}.')

            if len(pathCh2) == 1:
                print(f'\nFound the third channel for file {idx}.')
                df_locs_ch2_original, info_ch2 = tools.load_locs(pathCh2[0])
                df_locs_ch2, dataset = localize.transform_locs(df_locs_ch2_original,
                                                               naclibCoefficients,
                                                               channel=2,
                                                               fig_size=(682, 682))
                # get right output paths
                pathOutput = os.path.splitext(
                    pathCh2[0])[0][:-9] + args.suffix + '_transform'

                infoNew = info_ch2.copy()
                infoNew.append(transform_info)
                save_info(pathOutput + '_ch2_locs.yaml', infoNew)

                print('Saving transformed localizations...')
                df_locs_ch2.to_csv(pathOutput + '_ch2_locs.csv', index=False)
            else:
                print(f'\nFound no third channel for file {idx}.')

                print('--------------------------------------------------------')

        except Exception:
            skippedPaths.append(pathCh0)
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

# %% roi


def _roi(args):
    '''Restrict localizations to ROIs'''
    import os
    from glob import glob
    import traceback
    import pandas as pd
    from tqdm import tqdm
    from spit import tools
    import numpy as np
    from picasso.io import save_info

    # check if file directory is specified
    if args.files is None:
        raise FileNotFoundError('No directory specified')
    else:
        files = args.files

    # print params
    ws = '  '
    print('ROI - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        if getattr(args, element) is not None:
            print(
                f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')
    # format filepaths
    if os.path.isdir(files):
        print('Analyzing directory...')
        if args.recursive == True:
            paths = glob(files + '/**/**_locs.csv', recursive=True)
        else:
            paths = glob(files + '/*_locs.csv')

    elif os.path.isfile(files):
        paths = glob(files)
    # remove filepaths that do match string
    if args.avoidstring:
        keptpaths = []
        for path in paths:
            if args.avoidstring not in path:
                keptpaths.append(path)
        paths = keptpaths

    # remove filepaths that do not match string
    if args.matchstring:
        keptpaths = []
        for path in paths:
            if args.matchstring in path:
                keptpaths.append(path)
        paths = keptpaths

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
            df_locsM = df_locsM.drop_duplicates(
                subset=['x', 'y'])  # if ROIs overlap
            df_locs = df_locsM

            # get right output paths
            pathOutput = path.replace('ch', args.suffix+'roi_ch')
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
# %% link


def _link(args):
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

    # check if file directory is specified
    if args.files is None:
        raise FileNotFoundError('No directory specified')
    else:
        files = args.files

    # print params
    ws = '  '
    print('Linking - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        if getattr(args, element) is not None:
            print(
                f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')

    # format filepaths
    if os.path.isdir(files):
        print('Analyzing directory...')
        if args.recursive == True:
            if args.coloc:
                paths = glob(files + '/**/**colocs.csv', recursive=True)
            else:
                paths = glob(files + '/**/**_locs.csv', recursive=True)
        else:
            if args.coloc:
                paths = glob(files + '/*colocs.csv')
            else:
                paths = glob(files + '/*_locs.csv')
    elif os.path.isfile(files):
        paths = glob(files)

    # remove filepaths that do match string
    if args.avoidstring:
        keptpaths = []
        for path in paths:
            if args.avoidstring not in path:
                keptpaths.append(path)
        paths = keptpaths

    # remove filepaths that do not match string
    if args.matchstring:
        keptpaths = []
        for path in paths:
            if args.matchstring in path:
                keptpaths.append(path)
        paths = keptpaths

    # initialize placeholders
    skippedPaths = []
    quick = ''

    # print all kept paths
    for path in paths:
        print(path)
    print(f'A total of {len(paths)} files detected...')
    print('--------------------------------------------------------')

    # main loop
    px2nm = args.pixels
    for idx, path in tqdm(enumerate(paths), desc='Linking localizations...', total=len(paths)):
        try:
            print('--------------------------------------------------------')
            print(f'Running file {path}')
            (df_locs, info) = tools.load_locs(path)
            if not args.coloc:
                # fix locIDs before they get mixed up by linking
                df_locs = df_locs.rename_axis('locID').reset_index()
            # retrieve exposure time
            if not args.dt == None:
                dt = args.dt
            else:
                resultPath = os.path.join(os.path.dirname(
                    path), tools.getRunName(path))+'_result.txt'
                resultTxt = open(resultPath, 'r')
                resultLines = resultTxt.readlines()
                dtStr = tools.find_string(
                    resultLines, 'Camera Exposure')[17:-1]
                dt = 0.001 * \
                    int(float((''.join(c for c in dtStr if (c.isdigit() or c == '.')))))

            if 'roi' in path:
                roi_boolean = True
            else:
                roi_boolean = False
            # Select 200px^2 center FOV and first 500 frames
            if args.quick:
                img_size = info[0]['Height']  # get image size
                roi_width = 100
                if not roi_boolean:  # avoiding clash with ROIs-only limit frames
                    df_locs = df_locs[(df_locs.x > (img_size/2-roi_width))
                                      & (df_locs.x < (img_size/2+roi_width))]
                    df_locs = df_locs[(df_locs.y > (img_size/2-roi_width))
                                      & (df_locs.y < (img_size/2+roi_width))]
                df_locs = df_locs[df_locs.t <= 500]
                quick = '_quick'

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

            # save ('quick'-cropped) locs in nm and plot stats
            df_locs_nm = tools.df_convert2nm(df_locs, px2nm)
            path_nm = os.path.splitext(path)[0]+quick+args.suffix+'_nm.csv'
            df_locs_nm.to_csv(path_nm, index=False)
            path_plots_loc = tools.getOutputpath(
                path_nm, 'plots', keepFilename=True)

            tau_bleach = plot_diffusion.plot_loc_stats(
                df_locs_nm, path_plots_loc, dt=dt)

            # prepare rest of the paths
            path_params = tools.getOutputpath(
                path_nm, 'paramfiles', keepFilename=True) + f'_{args.tracker}'
            path_output = os.path.splitext(path_nm)[0] + f'_{args.tracker}'
            path_plots = tools.getOutputpath(
                path_nm, 'plots', keepFilename=True) + f'_{args.tracker}'

            # Choose tracking algorithmus
            if args.tracker == 'trackpy':
                print('Using trackpy.\n')
                # export parameters to yaml
                with open(path_params + '.yaml', 'w') as f:
                    yaml.dump(vars(args), f)

                df_tracksTP = link.link_locs_trackpy(
                    df_locs,
                    search=args.search,
                    memory=args.memory)

                # linked file is saved with pixel-corrected coordinates and
                # swiftGUI compatible columns, and unique track.ids
                df_tracks = tools.df_convert2nm(df_tracksTP, px2nm)
                df_tracks['seg.id'] = df_tracksTP['track.id']
                if 'roi' in path:
                    df_tracks = tools.get_unique_trackIDs(df_tracks)
                df_tracks.to_csv(path_output + '.csv', index=False)

            if args.tracker == 'swift':
                print('Using swift.')
                # getting arguments
                precision = df_locs_nm.loc_precision.median()
                if args.tau_bleach:
                    p_bleach = 1/(args.tau_bleach)
                else:
                    p_bleach = 1/(tau_bleach)

                if args.diff_limit:
                    diffraction_limit = args.diff_limit
                else:
                    diffraction_limit = df_locs_nm.nearest_neighbor.min()
                # else:
                    # D_guess = 0.4 #um2/s (for cells 0.1, for SLB 0.5?)
                    # mjd_median = np.sqrt(4*(1e6*D_guess*dt)+4*(precision**2))

                swift_kwargs = {
                    'diffraction_limit': diffraction_limit,  # calculated
                    'precision': precision,  # calculated
                    'exp_displacement': args.expected_mjd,
                    'exp_displacement_max': 0,  # 'auto'
                    'exp_displacement_max_pp': 0,  # 'auto'
                    'exp_noise_rate': args.exp_noise_rate,
                    'p_switch': args.p_switch,
                    'w_diffusion': args.w_diffusion,
                    'w_immobile': args.w_immobile,
                    'p_blink': args.p_blink,
                    'p_reappear': args.p_reappear,
                    'max_blinking_duration': args.memory,
                    'p_bleach': p_bleach,
                    'tau': 1000*dt,
                    'ignore_intensity': args.intensity
                }
                link.create_paramfile(path_params+'.json', **swift_kwargs)
                df_tracks = link.link_locs_swift(
                    path_nm, path_output, path_params, roi_boolean)

                # Iterations
                if args.iterate:
                    iterator_threshold = 5  # max 5nm difference
                    iterator_loop_threshold = 5  # max 5 iterations
                    if args.expected_mjd is None:  # use last saved value from default file
                        with open(path_params+'.json') as json_file:
                            initial_config = json.load(json_file)
                        mjdList = [initial_config['exp_displacement']]
                    else:
                        mjdList = [args.expected_mjd]

                    delta = np.inf
                    idx_it = 0
                    while (abs(delta) > iterator_threshold) & (idx_it < iterator_loop_threshold):
                        print(
                            f'Improving Swift tracking, iteration {len(mjdList)}')
                        # get updated expected displacement (=mjd) and update kwargs
                        mjd_median = link.get_expected_mjd(df_tracks)
                        swift_kwargs.update({'exp_displacement': mjd_median})

                        # create updated parameter file, save it and load the tracked
                        # file as if it were a locs file
                        path_params_updated = f'{path_params}_it{len(mjdList)}'
                        link.create_paramfile(
                            path_params_updated+'.json', **swift_kwargs)
                        # Link original locs with new params
                        df_tracks = link.link_locs_swift(
                            path_nm, path_output, path_params_updated, roi_boolean)

                        mjdList.append(mjd_median)
                        delta = mjdList[-1]-mjdList[-2]
                        idx_it += 1

                    plot_diffusion.plot_iteration(mjdList, path_params)
                    link.update_default_params(
                        exp_displacement=mjd_median)  # remove this?

                # move meta.json files to paramfiles folder
                meta_files = glob(os.path.dirname(path) + '/*meta.json')
                for meta_file in meta_files:
                    path_meta_file = tools.getOutputpath(
                        meta_file, 'paramfiles', keepFilename=True)
                    shutil.move(meta_file, path_meta_file+'.json')

            # Analysis and Plotting

            print('Calculating and plotting particle-wise diffusion analysis...\n')
            df_stats = link.get_particle_stats(df_tracks,
                                               dt=dt,
                                               particle='track.id',
                                               t='t')

            # adding ROI stats to track stat file
            if roi_boolean:
                df_stats = df_stats.merge(
                    df_roi[['path', 'contour', 'area', 'centroid', 'cell_id']], on='cell_id', how='left')

            # Save dataframe with track statistics (unfiltered)
            if os.path.isfile(path_output + '_stats.hdf'):
                os.remove(path_output + '_stats.hdf')  # force overwriting
            df_stats.to_hdf(path_output + '_stats.hdf',
                            key='df_stats', mode='w')

        # Filter short tracks and immobile particles
        # if not args.coloc:
            df_statsF = link.filter_df(
                df_stats, filter_length=10, filter_D=0.01)
            plot_diffusion.plot_track_stats(
                df_tracks, df_stats, df_statsF, path_plots, dt=dt)

        except Exception:
            skippedPaths.append(path)

            print('--------------------------------------------------------')
            print(f'Path {path} could not be analyzed. Skipping...\n')
            traceback.print_exc()

    print('--------------------------------------------------------')
    print('/////////////////////FINISHED//////////////////////////')
    print('--------------------------------------------------------')
    if skippedPaths:
        print('Analysis failed on paths:')
        for skippedPath in skippedPaths:
            print(f'\n{skippedPath}')
# %% colocalize


def _colocalize(args):
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

    # print params
    ws = '  '
    print('Colocalize - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        print(f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')

    # format paths according to a specified ending, e.g. "ch2_locs.csv"
    dirpath = args.files
    ch0 = args.channel0
    ch1 = args.channel1

    # getting all filenames of the first channel, later look for corresponding second channel files
    if os.path.isdir(dirpath):
        print('Analyzing directory...')
        pathsCh0 = glob(dirpath + f'//**//*{ch0}*_locs.csv', recursive=True)
        # remove filepaths that do match string
        if args.avoidstring:
            keptpaths = []
            for path in pathsCh0:
                if args.avoidstring not in path:
                    keptpaths.append(path)
            pathsCh0 = keptpaths

        # remove filepaths that do not match string
        if args.matchstring:
            keptpaths = []
            for path in pathsCh0:
                if args.matchstring in path:

                    keptpaths.append(path)
            pathsCh0 = keptpaths
        print(pathsCh0)
        print(f'Found {len(pathsCh0)} files for channel 0...')
        for path in pathsCh0:
            print(path)
    else:
        raise FileNotFoundError('Directory not found')
    print('--------------------------------------------------------')
    skippedPaths = []
    # main loop
    for idx, pathCh0 in tqdm(enumerate(pathsCh0), desc='Looking for colocalizations...'):
        try:
            dirname = os.path.dirname(pathCh0)
            if ch1 == None:
                raise FileNotFoundError('Second channel not declared.')
                print('--------------------------------------------------------')
            else:
                pathCh1 = glob(dirname + f'/**{ch1}*_locs.csv')[0]
                print(f'\nFound a second channel for file {idx}.')

                if not args.dt == None:
                    dt = args.dt
                else:
                    resultPath = os.path.join(os.path.dirname(
                        pathCh0), tools.getRunName(pathCh0))+'_result.txt'
                    resultTxt = open(resultPath, 'r')
                    resultLines = resultTxt.readlines()
                    dtStr = tools.find_string(
                        resultLines, 'Camera Exposure')[17:-1]
                    dt = 0.001 * \
                        int(float((''.join(c for c in dtStr if (c.isdigit() or c == '.')))))

                # read in the linked files
                df_locs_ch0 = pd.read_csv(pathCh0)
                df_locs_ch1 = pd.read_csv(pathCh1)
                # pixels to nm
                df_locs_ch0 = tools.df_convert2nm(df_locs_ch0, args.pixels)
                df_locs_ch1 = tools.df_convert2nm(df_locs_ch1, args.pixels)

                # get colocalizations
                df_colocs = coloc.colocalize_from_locs(df_locs_ch0, df_locs_ch1,
                                                       threshold_dist=args.threshold)

                # get right output paths
                pathOutput = os.path.splitext(pathCh0)[0][:-9] + args.suffix
                pathPlots = tools.getOutputpath(
                    pathCh0, 'plots', keepFilename=True)[:-9] + args.suffix

                print('Saving colocalizations...')
                df_colocs_px = tools.df_convert2px(df_colocs)
                df_colocs_px.to_csv(pathOutput + '_colocs.csv', index=False)
                print('Calculating and plotting colocalization analysis.')

                plot_coloc.plot_coloc_stats(df_locs_ch0, df_locs_ch1, df_colocs,
                                            threshold=args.threshold,
                                            path=pathPlots, dt=dt, roll_param=5)
                print('--------------------------------------------------------')

        except Exception:
            skippedPaths.append(pathCh0)
            print('--------------------------------------------------------')
            print(f'Path {path} could not be analyzed. Skipping...\n')
            traceback.print_exc()

        # export parameters to yaml
        with open(pathOutput + '_colocs.yaml', 'w') as f:
            yaml.dump(vars(args), f)

    print('                                                        ')

    print('--------------------------------------------------------')
    print('/////////////////////FINISHED//////////////////////////')
    print('--------------------------------------------------------')
    if skippedPaths:
        print('Skipped paths:')
        for skippedPath in skippedPaths:
            print(f'\n{skippedPath}\n')

# %% frap


def _frap(args):
    import os
    import traceback
    from glob import glob
    from tqdm import tqdm
    from spit import frap as frap

    # check if file directory is specified
    if args.files is None:
        raise FileNotFoundError('No directory specified')
    else:
        files = args.files

    # print all parameters
    ws = '  '
    print('FRAP analysis - Parameter Settings:')
    print(f'{"No":<6} | {ws+"Label":<18} | {ws+"Value":<10}')
    for index, element in enumerate(vars(args)):
        print(f'{index+1:<6} | {ws+element:<18} | {ws+str(getattr(args,element)):<10}')
    print('--------------------------------------------------------')

    # format filepaths
    if os.path.isdir(files):
        print('Analyzing directory...')
        if args.recursive == True:
            paths = glob(files + '/**/**PreFrap.raw', recursive=True)
        else:
            paths = glob(files + '/*PreFrap.raw')
    elif os.path.isfile(files):
        paths = glob(files)

    # remove filepaths that do match string
    if args.avoidstring:
        keptpaths = []
        for path in paths:
            if args.avoidstring not in path:
                keptpaths.append(path)
        paths = keptpaths

    # remove filepaths that do not match string
    if args.matchstring:
        keptpaths = []
        for path in paths:
            if args.matchstring in path:
                keptpaths.append(path)
        paths = keptpaths

    skippedPaths = []
    # print all kept paths
    for path in paths:
        print(path)
    print(f'A total of {len(paths)} files detected...')
    print('--------------------------------------------------------')

    channel = args.channel
    for path in tqdm(paths, desc='Analyzing files...'):
        try:
            textpath = os.path.join(os.path.split(path)[0], os.path.split(
                os.path.split(path)[0])[1]+'_result.txt')
            if os.path.isfile(textpath):
                if 'Aborted' in open(textpath).read():
                    print("\nRun was aborted, skipped this path.")
                else:
                    frap.bilayerAnalysis(path, channel)
            else:
                print("Run was not completed, skip this path.")

        except Exception:
            skippedPaths.append(path)
            print('--------------------------------------------------------')
            print(f'Path {path} could not be analyzed. Skipping...\n')
            traceback.print_exc()

    print('--------------------------------------------------------')
    print('/////////////////////FINISHED//////////////////////////')
    print('--------------------------------------------------------')
    if skippedPaths:
        print('Skipped paths:')
        for skippedPath in skippedPaths:
            print(f'\n{skippedPath}\n')


def main():
    import argparse

    # parser
    parser = argparse.ArgumentParser(prog='spit')
    subparsers = parser.add_subparsers(dest='module')

    # table
    table_parser = subparsers.add_parser(
        'table', help='create wikitable from Run directory'
    )
    table_parser.add_argument(
        'path', nargs='?',
        help='folder containing movie files specified by a unix style path pattern'
    )

    # localize
    localize_parser = subparsers.add_parser(
        'localize', help='identify and fit single molecule spots'
    )
    localize_parser.add_argument(
        'files', nargs='?',
        help='raw file or a folder containing movie files specified by a unix style path pattern'
    )
    localize_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    localize_parser.add_argument(
        '-as', '--avoidstring', type=str, default=None, help='string to avoid in filepath'
    )
    localize_parser.add_argument(
        '-ms', '--matchstring', type=str, default=None, help='string to match in filepath'
    )
    localize_parser.add_argument(
        '-su', '--suffix', type=str, default='',
        help='suffix to add to filename'
    )
    localize_parser.add_argument(
        '-px', '--pixels', type=int,
        default=108, help='nm per pixel'
    )
    localize_parser.add_argument(
        '-b', '--box-side-length', type=int, default=7, help='box side length'
    )
    localize_parser.add_argument(
        '-f',
        '--fit-method',
        choices=['com', 'lq'],
        default='lq',
        help='fit method: center-of-mass or least-square Gaussian'
    )
    localize_parser.add_argument(
        '-g', '--gradient', type=int, help='minimum net gradient for whole movie'
    )
    localize_parser.add_argument(
        '-g0', '--gradient0', type=int, help='minimum net gradient for channel 0 (left)'
    )
    localize_parser.add_argument(
        '-g1', '--gradient1', type=int, help='minimum net gradient for channel 1 (center)'
    )
    localize_parser.add_argument(
        '-g2', '--gradient2', type=int, help='minimum net gradient for channel 2 (right)'
    )
    localize_parser.add_argument(
        '-bl', '--baseline', type=int, default=100, help='camera baseline'
    )
    localize_parser.add_argument(
        '-s', '--sensitivity', type=float, default=0.6, help='camera sensitivity'
    )
    localize_parser.add_argument(
        '-ga', '--gain', type=int, default=1, help='camera gain'
    )
    localize_parser.add_argument(
        '-qe', '--qe', type=float, default=0.9, help='camera quantum efficiency'
    )
    localize_parser.add_argument(
        '-t', '--transform',
        action='store_true',  # store_true creates default False value
        help='NAClib: non-affine corrections of localizations'
    )
    localize_parser.add_argument(
        '-plot', '--plot',
        default=False, action='store_true',
        help='plot number of localizations, next neighbour distance and photon histogram'
    )

    # transform
    transform_parser = subparsers.add_parser(
        'transform', help='transform colocalizations'
    )
    transform_parser.add_argument(
        'files', nargs='?',
        help='directory where localized files are'
    )
    transform_parser.add_argument(
        '-as', '--avoidstring', type=str, default='transform', help='string to avoid in filepath'
    )
    transform_parser.add_argument(
        '-ms', '--matchstring', type=str,
        default=None, help='string to match in filepath'
    )
    transform_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    transform_parser.add_argument(
        '-su', '--suffix', type=str, default='',
        help='suffix to add to filename'
    )
    transform_parser.add_argument(
        '-ch0', '--channel0', type=str,
        default='ch0',
        help='x_locs.csv file for channel 0 (if batching, make sure to only include shared file ending)'
    )
    transform_parser.add_argument(
        '-ch1', '--channel1', type=str,
        default='ch1',
        help='x_locs.csv file for channel 1 (if batching, make sure to only include shared file ending)'
    )
    transform_parser.add_argument(
        '-ch2', '--channel2', type=str,
        default='ch2',
        help='x_locs.csv file for channel 2 (if batching, make sure to only include shared file ending)'
    )

    # roi
    roi_parser = subparsers.add_parser(
        'roi', help='restrict localizations to regions of interests'
    )
    roi_parser.add_argument(
        'files', nargs='?',
        help='directory where localized files are'
    )
    roi_parser.add_argument(
        '-as', '--avoidstring', type=str, default='roi', help='string to avoid in filepath'
    )
    roi_parser.add_argument(
        '-ms', '--matchstring', type=str,
        default=None, help='string to match in filepath'
    )
    roi_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    roi_parser.add_argument(
        '-su', '--suffix', type=str, default='',
        help='suffix to add to filename'
    )

    # link
    link_parser = subparsers.add_parser(
        'link', help='link localizations into trajectories'
    )
    link_parser.add_argument(
        'files', nargs='?',
        help='directory where localized files are'
    )
    link_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    link_parser.add_argument(
        '-as', '--avoidstring', type=str, default=None, help='string to avoid in filepath'
    )
    link_parser.add_argument(
        '-ms', '--matchstring',
        type=str, default=None, help='string to match in filepath'
    )
    link_parser.add_argument(
        '-su', '--suffix', type=str, default='',
        help='suffix to add to filename'
    )
    link_parser.add_argument(
        '-coloc', '--coloc',
        default=False, action='store_true',
        help='linking colocalizations'
    )
    link_parser.add_argument(
        '-roi', '--roi',
        default=False, action='store_true',
        help='use ROI'
    )
    link_parser.add_argument(
        '-quick', '--quick',
        default=False, action='store_true',
        help='Select center FOV and first 500 frames for linking.'
    )
    link_parser.add_argument(
        '-px', '--pixels', type=float,
        default=108, help='nm per pixel'
    )
    link_parser.add_argument(
        '-dt', '--dt',
        type=float, default=None, help='exposure time in s'
    )
    # tracking parameters
    link_parser.add_argument(
        '-tr', '--tracker', nargs='?', default='trackpy',
        help='which tracker to use: trackpy OR swift (or trackpy parameter scan: trackpy-scan)'
    )
    link_parser.add_argument(
        '-sr', '--search', type=float,
        nargs='?', default=None, help='max search range for trackpy linking in px'
    )
    link_parser.add_argument(
        '-mm', '--memory', type=int,
        nargs='?', default=None, help='max gap length '
    )
    link_parser.add_argument(
        '-it', '--iterate',
        default=False, action='store_true',
        help='Swift: Iterate estimation of mjd'
    )
    link_parser.add_argument(
        '-mjd', '--expected_mjd', type=float,
        nargs='?', default=None, help='Swift: expected displacement in nm'
    )
    link_parser.add_argument(
        '-diff', '--diff_limit', type=float,
        nargs='?', default=None, help='Swift: diffraction limit (nearest resolvable neighbor)'
    )
    link_parser.add_argument(
        '-noise', '--exp_noise_rate', type=float,
        nargs='?', default=None, help='Swift: percentage of localizations that are noise'
    )
    link_parser.add_argument(
        '-pswitch', '--p_switch', type=float,
        nargs='?', default=None, help='Swift: diffusion mode switching rate per particle and frame'
    )
    link_parser.add_argument(
        '-wdiff', '--w_diffusion', type=float,
        nargs='?', default=None, help='Swift: diffusion mode weight: free diffusion'
    )
    link_parser.add_argument(
        '-wimm', '--w_immobile', type=float,
        nargs='?', default=None, help='Swift: diffusion mode weight: immobile particle'
    )
    link_parser.add_argument(
        '-pblink', '--p_blink', type=float,
        nargs='?', default=None, help='Swift: blinking rate per particle and frame'
    )
    link_parser.add_argument(
        '-preappear', '--p_reappear', type=float,
        nargs='?', default=None, help='Swift: reappearing rate per particle and frame'
    )
    link_parser.add_argument(
        '-taubleach', '--tau_bleach', type=float,
        nargs='?', default=None, help='Swift: bleaching constant tau in frames'
    )
    link_parser.add_argument(
        '-int', '--intensity',
        default=False, action='store_true',
        help='Swift: Ignore intensity values'
    )
    # link_parser.add_argument(
    #     '-slist', '--searchlist', type=float, nargs='+',
    #     default=[2,3,4,5,6], help='search range parameters to scan'
    #     )

    # colocalize
    colocalize_parser = subparsers.add_parser(
        'colocalize', help='find colocalizations in linked tracks'
    )
    colocalize_parser.add_argument(
        'files', nargs='?',
        help='directory where localized files are'
    )
    colocalize_parser.add_argument(
        '-as', '--avoidstring', type=str, default=None, help='string to avoid in filepath'
    )
    colocalize_parser.add_argument(
        '-ms', '--matchstring', type=str,
        default=None, help='string to match in filepath'
    )
    colocalize_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    colocalize_parser.add_argument(
        '-su', '--suffix', type=str, default='',
        help='suffix to add to filename'
    )
    colocalize_parser.add_argument(
        '-px', '--pixels', type=float,
        default=108, help='nm per pixel'
    )
    colocalize_parser.add_argument(
        '-dt', '--dt', type=float,
        default=None, help='exposure time in s'
    )
    colocalize_parser.add_argument(
        '-ch0', '--channel0', type=str,
        default='ch0',
        help='x_locs.csv file for channel 0 (if batching, make sure to only include shared file ending)'
    )
    colocalize_parser.add_argument(
        '-ch1', '--channel1', type=str,
        default='ch1',
        help='x_locs.csv file for channel 1 (if batching, make sure to only include shared file ending)'
    )
    colocalize_parser.add_argument(
        '-th', '--threshold', type=int,
        default=250, help='Threshold for colocalization'
    )

    # frap
    frap_parser = subparsers.add_parser(
        'frap', help='FRAP, has to be 100 frames of 100ms exp. time'
    )
    frap_parser.add_argument(
        'files', nargs='?'
    )
    frap_parser.add_argument(
        '-r', '--recursive',
        default=False, action='store_true',
        help='recursive search subdirectories'
    )
    frap_parser.add_argument(
        '-as', '--avoidstring', type=str, default=None, help='string to avoid in filepath'
    )
    frap_parser.add_argument(
        '-ms', '--matchstring', type=str, default=None, help='string to match in filepath'
    )
    frap_parser.add_argument(
        '-ch', '--channel', type=str, default=2, help='channel 0, 1 or  2 (left, middle or right) '
    )


    # parse arguments
    args = parser.parse_args()
    if args.module:
        # library banner
        print(r'SPIT v.1.0')
        if args.module == 'table':
            _table(args)
        elif args.module == 'localize':
            _localize(args)
        elif args.module == 'transform':
            _transform(args)
        elif args.module == 'roi':
            _roi(args)
        elif args.module == 'link':
            _link(args)
        elif args.module == 'colocalize':
            _colocalize(args)
        elif args.module == 'frap':
            _frap(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
