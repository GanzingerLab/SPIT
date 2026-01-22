import os
import numpy as np
import pandas as pd
import numba
from tqdm import tqdm
import trackpy as tp
from scipy.optimize import curve_fit
import logging
from spit import tools
import json
import pdb
logger = logging.getLogger('trackpy')
logger.setLevel(level=logging.CRITICAL)


# %% Trackpy
# Linking with trackpy
def link_locs_trackpy(df_locs, search, memory):
    """
    Returns trackpy-linked locs in dataframes, filtered for short tracks.

    Parameters
    ----------
    locs : Pandas dataframe
        Localizations to link.
    search :Int
        Range is the max search range possible to look for next position.
    memory: Int
        Number of frames that can be empty before the next loc appears.

    Returns
    -------
    linked : Pandas dataframe
        Linked localizations.

    Notes
    -----
    A new column 'track.id' is created to label the different tracks. It can be discontinuous due to tracks that were removed because too short.

    """
    if 't' in df_locs.columns:
        df_locs = df_locs.rename(columns={'t': 'frame'})
    #imlement velocity prediction? 
    # pred = tp.predict.NearestVelocityPredict()
    #df_tracks  = pred.link(df_locs, search_range=search,    memory=memory,    adaptive_step=0.95,    adaptive_stop = 2,    link_strategy='hybrid')
    df_tracks = tp.link(df_locs,
                        search_range=search,
                        memory=memory,
                        adaptive_step=0.95,
                        adaptive_stop = 2,
                        link_strategy='hybrid'
                        )

    df_tracks = df_tracks.rename(columns={'particle': 'track.id',
                                          'frame': 't'})

    # add column with length of track
    loc_count = df_tracks['track.id'].value_counts()
    loc_count = loc_count.reset_index(
        name='loc_count').rename(columns={'index': 'track.id'})

    df_tracks = df_tracks.merge(loc_count, on='track.id')
    if 'cell_id' in df_tracks.columns:
        df_tracks = df_tracks.sort_values(['cell_id', 't', 'track.id'])
    else:
        df_tracks = df_tracks.sort_values(['t', 'track.id'])

    return df_tracks


# %% Analysis: individual particles
"""
Set of functions to analyze diffusion properties per particle.
Takes linked localizations (tracks) and returns
dataframe with diffusion properties per particle.
"""


def get_var(df_tracks):
    '''
    Get various properties for individual particles in linked localizations

    Args:
        df(pandas.DataFrame): Linked localization list, i.e. tracked.csv
    Returns:
        pandas.Series:            
                - ``loc_count``:      Number of localizations
                - ``x std``:          Standard deviation of group in ``x``
                - ``y std``:          Standard deviation of group in ``y``
                - ``length``:      max_frame-min_frame (see above)
                '''

    s_out = pd.Series(dtype='float64')
    s_out['track.id'] = df_tracks['track.id'].mean()

    if 'cell_id' in df_tracks.columns:
        s_out['cell_id'] = df_tracks.cell_id.mean()

    # Set sx and sy to standard deviation of x,y (locs!) instead of PSF width
    s_out['x_std'] = np.percentile(df_tracks['x'], 75)-np.percentile(df_tracks['x'], 25)
    s_out['y_std'] = np.percentile(df_tracks['y'], 75)-np.percentile(df_tracks['y'], 25)

    # Add min/max of frames
    s_out['length'] = df_tracks.t.max()-df_tracks.t.min()+1
    s_out['loc_count'] = len(df_tracks)
    return s_out


def fit_msd_free(lagtimes, msd, offset=False):
    '''
    Unweighted least square fit of invidual msd by linear model ``msd=a*lagtimes+b``, see analytic_expressions.msd_free(),
    i.e. assuming free Browninan motion. If there was less then two data-points or fit was not succesfull 
    NaNs are returned as optimum parameters.

    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
        offset(bool=False): If True offset is used in linear fit model, if False
    Returns:
        pandas.Series: Column ``a`` corresponds to slope, ``b`` corresponds to offset of linear function applied. 

    '''
    import warnings
    warnings.filterwarnings("ignore")  # supress OptimizeWarning
    x = lagtimes
    y = msd
    N = len(y)

    if N >= 2:  # Try ftting if more than one point in msd
        # Init fit parameters
        p0 = [(y[-1]-y[0])/N, y[0]]  # Start values

        try:
            if offset == True:
                popt, pcov = curve_fit(tools.msd_free, x, y, p0=p0)
            else:
                popt, pcov = curve_fit(tools.msd_free, x, y, p0=p0[0])
        except:
            popt = np.full(2, np.nan)

    else:
        popt = np.full(2, np.nan)

    # Assign to output
    if offset == True:
        s_out = pd.Series({'a': popt[0], 'b': popt[1]})
    else:
        s_out = pd.Series({'a': popt[0], 'b': 0})
    return s_out


def fit_msd_anomal(lagtimes, msd):
    '''
    Unweighted least square fit of invidual msd by anomalous model ``msd=a*lagtimes**b``, 
    see analytic_expressions.msd_anomal(). If there was less then two data-points or fit was not succesfull 
    NaNs are returned as optimum parameters.

    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
    Returns:
        pandas.Series: Column ``a`` corresponds to slope, ``b`` corresponds to diffusion mode. 

    '''
    x = lagtimes
    y = msd
    N = len(y)

    if N >= 2:  # Try ftting if more than one point in msd
        # Init fit parameters
        p0 = [(y[-1]-y[0])/N, 1.0]  # Start values
        try:
            popt, pcov = curve_fit(tools.msd_anomal, x, y, p0=p0)
        except:
            popt = np.full(2, np.nan)
    else:
        popt = np.full(2, np.nan)

    # Assign to output
    s_out = pd.Series({'a': popt[0], 'b': popt[1]})

    return s_out


def fit_msd_free_iterative(lagtimes, msd, max_it=5):
    '''
    Unweighted least square fit of invidual msd by linear model ``msd=a*lagtimes+b`` in **iterative manner**
    to find optimum fitting range of msd according to: Xavier Michalet, Physical Review E, 82, 2010 (michalet_).
    In first iteration msd is fitted up to a maximum lagtime of ``lag_max=0.5*Nmsd`` with ``Nmsd`` being the full msd length.
    Notice that motion_metrics.displacement_moments() calculates msd only up to ``Nmsd=0.25*N`` hence ``lag_max=0.125*N``
    with Nbeing the full lenght of the trajectory. Then fitting range is updated according to rule 
    ``lag_max=int(np.round(2+2.3*(b/a)**0.52))``. For a detailed illustration please see SI of spt_.

    Args:
        lagtimes(np.array): Array of msd lagtimes
        msd(np.array):      Mean square displacement (msd) at lagtimes
        max_it(int=5):      Maximum number of iterations
    Returns:
        pandas.Series:

            - ``a`` slope  of linear function applied.
            - ``b`` offset of linear function applied
            - ``p`` maximum lagtime up to which msd was fitted
            - ``max_it`` resulting number of iterations until convergence was achieved
    '''
    # Set inital track length that will be fitted to half the msd,
    # which is already set to only 0.25 of full track length, hence only 12.5% are used!
    p = [int(np.floor(0.5*len(msd)))]

    i = 0
    while i < max_it:
        # Truncate msd up to p for optimal fitting result
        t = lagtimes[:p[-1]]
        y = msd[:p[-1]]

        # Fit truncated msd
        s_out = fit_msd_free(t, y, offset=True)

        # Update x
        # x=np.abs((4*lp)/s_out['a'])
        x = np.abs(s_out['b']/(s_out['a']))

        # Assign iteration and fitted track length
        s_out['p'] = p[-1]
        s_out['max_it'] = i

        # Update optimal track length to be fitted
        try:
            p_update = int(np.round(2+2.3*x**0.52))
            if p_update <= 2:
                p_update = 2
            p = p+[p_update]
        except:
            break

        if np.abs(p[-1]-p[-2]) < 1:
            break
        i += 1

    return s_out


@numba.jit(nopython=True, nogil=True, cache=True)
def displacement_moments(t, x, y):
    '''
    Numba optimized calulation of trajectory ``(t,x,y)`` moments. Calulation accounts for short gaps, i.e. missed localizations 
    recovered by allowed ``memory`` values. Moments are only calulated up to maximum lag time of ``l_max = 0.25*N`` with ``N=len(t)``.
    Calulated moments are:

        - Mean square displacement (MSD)
        - Mean displacement moment of 4th order (MSD corresponds to 2nd order)
        - Mean maximal excursion of 2nd order (MME)
        - Mean maximal excursion of 4th order

    MME is calculated according to: Vincent Tejedor, Biophysical Journal, 98, 7, 2010 (tejedor)
    https://www.cell.com/biophysj/fulltext/S0006-3495(09)06097-4

    Args:
        t(np.array): time
        x(np.array): x-position
        y(np.array): y-position
    Returns:
        np.array of size ``(l_max,5)``:

            - ``[:,0]``: lag time
            - ``[:,1]``: MSD
            - ``[:,2]``: Mean displacement moment of 4th order
            - ``[:,3]``: MME
            - ``[:,4]``: Mean maximal excursion of 4th order
    '''
    N = t[-1]-t[0]+1  # Trajectory length

    max_lag = int(np.floor(0.25*N))  # Set maximum lagtime to 0.25*trajectory length

    # Create nan arrays of length N to distribute x,y and fill gaps with nans
    x_gap = np.ones(N)*np.nan
    y_gap = np.ones(N)*np.nan
    idx = t-t[0]  # Indices where we find finite coordinates

    ### Fill in values
    x_gap[idx] = x
    y_gap[idx] = y

    jumps = np.ones((N-1))  # Init jumps

    for i in range(N-1):
        # Distance traveled per frame
        dx = x_gap[i+1]-x_gap[i] #calculate jumps in x
        dy = y_gap[i+1]-y_gap[i] #calculates jumps in y
        jumpsSq = dx**2 + dy**2 #calculates sum of the squares of the jumps in x and y.
        jumps[i] = np.sqrt(jumpsSq) #save jump as square root of jumpSq

    moments = np.ones((max_lag, 5), dtype=np.float32) #initilize variable. 
    r2max_leql = np.zeros(N+1, dtype=np.float64)  # Init max distance traveled
    r4max_leql = np.zeros(N+1, dtype=np.float64)
    for l in range(max_lag):
        # One dimensional jumps for lag l
        dx = x_gap[l:]-x_gap[:N-l] #calculates jumps in x with a distance of l frames in distance
        dy = y_gap[l:]-y_gap[:N-l] #calculates the same for y. 

        # Two dimensional jumps to the power of two and four
        r2_l = dx**2+dy**2 #sums the distance to the power of 2
        r4_l = dx**4+dy**4 #sums the distance to the the power of 4. 

        # Assign mean moments
        moments[l, 0] = l
        moments[l, 1] = np.nanmean(r2_l)
        moments[l, 2] = np.nanmean(r4_l)

        # Update rXmax_leql to maximum of past steps and current
        # rXmax_leql will be always shortened
        # to the size of rX_l while going through the loop!
        r2max_leql = np.maximum(r2max_leql[:-1], r2_l)
        r4max_leql = np.maximum(r4max_leql[:-1], r4_l)

        # Assign max moments
        moments[l, 3] = np.nanmean(r2max_leql)
        moments[l, 4] = np.nanmean(r4max_leql)

    # Remove first entry
    moments = moments[1:, :]
    # Remove NaNs due to gaps in trace
    moments = moments[np.isfinite(moments[:, 1])]
    jumps = jumps[np.isfinite(jumps)]

    return moments, jumps


def getfit_moments(df_tracks):
    '''
    Calculate msd of single trajectory using metrics.displacement_moments() and apply both linear iterative fitting
    according to fit_msd_free_iterative() and anomalous diffusion model fitting using fit_msd_anomal() to msd.

    Args:
        df(pandas.DataFrame): Trajectories (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.Series: 
            Concatenated output of fit_msd_free_iterative() and fit_msd_anomal().

            - ``msd_fit_a`` slope  of iterative linear fit
            - ``msd_fit_b`` offset of iterative linear fit
            - ``p_iter`` maximum lagtime up to which msd was fitted for iterative linear fit
            - ``max_iter`` resulting number of iterations until convergence was achieved for iterative linear fit
    '''

    # Get displacement moments and jumps
    moments, jumps = displacement_moments(df_tracks.t.values,
                                          df_tracks.x.values,
                                          df_tracks.y.values)
    # MSD fitting
    x = moments[:, 0]  # Define lagtimes, x values for fit
    y = moments[:, 1]  # Define MSD, y values for fit

    # Iterative fit
    s_iter = fit_msd_free_iterative(x, y).rename(
        {'a': 'msd_fit_a', 'b': 'msd_fit_b', 'p': 'p_iter', 'max_it': 'max_iter'})

    # Asign output series
    s_lagtimes = pd.Series({'lagtimes': x})
    s_msd = pd.Series({'msd': y})

    s_jumps = pd.Series({'jumps': jumps})
    s_out = pd.concat([s_iter, s_jumps, s_lagtimes, s_msd,
                       ])

    return s_out


def get_props(df_tracks):
    """ 
    Wrapper function for df.apply
    Combination of immobile_props.get_var(df) and getfit_moments(df).

    Args:
        df(pandas.DataFrame): Trajectories (_pickedxxxx.hdf5) as obtained by linklocs.main()
    Returns:
        pandas.Series:       Concatenated output of immobile_props.get_var(df) and getfit_moments(df).
    """

    # Call individual functions
    s_var = get_var(df_tracks)
    
    s_msd = getfit_moments(df_tracks)

    # Combine output
    s_out = pd.concat([s_var, s_msd])

    return s_out


def apply_jd_analysis(df_stats, dt):
    """ 
    Fit two-population jump distance distribution to all jumps.

    Args:
        df_props: DataFrame
        Diffusion properties per particle, including jump distances.
    dt : Float
        Exposure time in seconds.

    Returns:
        Fit parameters for two-population jump distance distribution. 

    """
    nm2um = 0.001
    # pdf fit for 2 species: mobile, immobile

    def jd_fit(r, d_coef1, a1, d_coef2, a2):
        return (a1*r/(2*d_coef1*dt)) * np.exp(-(r*r) / (4*d_coef1*dt)) + (a2*r/(2*d_coef2*dt)) * np.exp(-(r*r) / (4*d_coef2*dt))

    jd_all = [nm2um*jump for track in df_stats.jumps for jump in track]
    freq, bin_edges = np.histogram(jd_all, bins='fd', density=False)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])

    pars, *cov = curve_fit(f=jd_fit, xdata=bin_centers, ydata=freq, maxfev=10000)

    if pars[0] > pars[2]:
        d0 = pars[2]
        a0 = pars[3]
        d1 = pars[0]
        a1 = pars[1]
    else:
        d0 = pars[0]
        a0 = pars[1]
        d1 = pars[2]
        a1 = pars[3]

    # Assign to output
    df_stats['D_jd0'] = d0
    df_stats['A_jd0'] = a0
    df_stats['D_jd1'] = d1
    df_stats['A_jd1'] = a1

    return df_stats


def get_particle_stats(df_tracks, dt, particle='track.id', t='t'):
    """ 
    Final wrapper function:
    Trajectories list (tracked.csv) with x and y in [nm].
    Apply get_props() to each track to get mobile properties.

    Args:
        df(pandas.DataFrame): Trajectories list (tracked.csv)
    dt : Float
        Exposure time in seconds.
    Returns:
        pandas.DataFrame:     Output of get_props() for each track in ``df`` (groupby-apply approach).     
    """
    nm2um = 0.001
    tqdm.pandas()  # For progressbar under apply
    
    df_stats = df_tracks.groupby('track.id').progress_apply(get_props)
    df_stats = apply_jd_analysis(df_stats, dt)

    # add D_msd (basically convert msd_fit_a to um2/s value)
    df_stats['D_msd'] = df_stats.msd_fit_a*(nm2um**2/(4*dt))

    # track.id as integers
    df_stats['track.id'] = df_stats['track.id'].astype(dtype='int64')
    
    # drop unused columns/indices
    df_stats.drop('p_iter', inplace=True, axis=1)
    df_stats.drop('max_iter', inplace=True, axis=1)
    df_stats = df_stats.reset_index(drop=True)
    return df_stats

def tracks_greaterT(df_tracks, dt):
    """
    Get number of trajectories (bright times) per particle (group or pick) greater or equal to T, i.e. TPP in `spt`_.

    Parameters
    ----------
    linked : Pandas dataframe
        Tracks of the different particles.

    Returns
    -------
    list
        - [0](numpy.array): T in frames.
        - [1](numpy.array): Number of tracks larger than T for all T.

    """
    tracks_per_frame = df_tracks.pivot_table(columns=['t'], aggfunc='size')
    track_lengths = df_tracks.pivot_table(columns=['track.id'], aggfunc='size')*dt
    track_lengths = track_lengths.to_numpy()

    # Define Ts
    Ts = np.concatenate((np.arange(1, 49.91, 0.1),
                        np.arange(50, 99.1, 1),
                        np.arange(100, 398.1, 2),
                        np.arange(400, 1998.1, 5),
                        np.arange(2000, 4981, 20),
                        np.arange(5000, 50001, 1000)), axis=0)

    # Init observables
    gT = np.zeros(len(Ts))

    for idx, T in enumerate(Ts):
        survivors = track_lengths >= T  # Which tracks are longer than T?
        gT[idx] = np.sum(survivors)

    # For normalization: try averaging track number over 10 frames after the first 10
    try:
        NgT = np.array([i/np.mean(tracks_per_frame[10:20]) for i in gT])
    except:  # exception: if mean is zero, try first non-zero value and average from there for 10 frames
        tracks_per_frame_firstnonzero = np.nonzero(tracks_per_frame[10:])[0][0]
        NgT = np.array(
            [i/np.mean(tracks_per_frame[10+tracks_per_frame_firstnonzero:tracks_per_frame_firstnonzero+20]) for i in gT])

    return [Ts, NgT]

# %% Swift


def create_paramfile(path,
                     **kwargs):
    default_config_path = os.path.dirname(
        os.path.realpath(__file__)) + '/paramfiles/swift_config.json'
    with open(default_config_path) as json_file:
        default_config = json.load(json_file)

    # remove keys without values
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    default_config.update(kwargs)

    with open(path, 'w') as json_file:
        json.dump(default_config, json_file, indent=4)


def update_default_params(**kwargs):
    default_config_path = os.path.dirname(
        os.path.realpath(__file__)) + '/paramfiles/swift_config.json'
    default_config_file = open(default_config_path, 'r')
    default_config = json.load(default_config_file)
    default_config_file.close()
    # remove keys without values
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    default_config.update(kwargs)

    default_config_file_updated = open(default_config_path, 'w')
    json.dump(default_config, default_config_file_updated, indent=4)
    default_config_file_updated.close()
    print(f'Updated default configuration file with {kwargs}.')


def link_locs_swift(path_df_locs, path_save, path_config, roi=False):
    path_config = path_config + '.json'
    path_save = path_save + '.csv'
    # config = tools.getOutputpath(filepath,'paramfiles',keepFilename=True)+'.json'
    print(
        f'Configuration file used: {path_config}. \nSaving tracked file to {path_save}.')
    if roi:
        os.system(f"swft \"{path_df_locs}\" --splitby \"cell_id\" -c \"{path_config}\" -f --out_values \"noise motion mjd mjd_n D_msd D_msd_std_err track.loc_count\" -o \"{path_save}\"")
    else:
        os.system(
            f"swft \"{path_df_locs}\" -c \"{path_config}\" -f --out_values \"noise motion mjd mjd_n D_msd D_msd_std_err track.loc_count\" -o \"{path_save}\"")
    linked = pd.read_csv(path_save)
    linked = linked.sort_values(['t', 'track.id'])
    linked = linked.rename(columns={'track.loc_count': 'loc_count'})
    return linked


def get_expected_mjd(df_tracks):
    # Remove outliers (0.1% of total) and non-diffusing particles
    df_tracks_mobile = df_tracks.loc[df_tracks['seg.D_msd'] > 0.001]
    mjd_median = np.average(df_tracks_mobile['seg.mjd'],
                            weights=df_tracks_mobile['seg.mjd_n'])
    return mjd_median
# %% Misc


def filter_df(df_stats, filter_length, filter_D=None):
    """
    Filter particles according to their individual diffusion coefficient
    and track length. 
    Parameters
    ----------
    df_props : Dataframe
        Dataframe containing diffusion properties of the different particles.
    filterD : Float
        Minimum diffusion constant to be considered mobile in um2/s
    filterLength: Int
        Minimum track length in frames
    Returns
    -------
    df_props_filter : Dataframe
        Dataframe containing only tracks with a minimum duration
        and minimum diffusion coefficient. 
    """
    keep_lengths = df_stats.loc_count > filter_length
    if filter_D is not None:
        keep_D = df_stats.D_msd > filter_D
        keep = keep_lengths & keep_D
    else:
        keep = keep_lengths
    df_statsF = df_stats[keep]

    return df_statsF


# %%


def filter_dfNEW(df, **kwargs):
    '''
    Keep only rows with column values in 
        - a given interval, if providing a list
            - except for cell_id: all list values ROI are kept
        - greater than the argument, if providing a single value
            - for df_tracks: track lengths (loc_counts) are calculated and filtered on the spot.

    '''
    for key, value in kwargs.items():
        # if list, keep values in between (except for cell_id, keep list values)
        if type(value) == list:
            if key == 'cell_id':
                df = df.loc[df['cell_id'].isin(value)]
            else:
                df = df.loc[(df[key] > value[0]) & (df[key] < value[1])]
        else:
            # if single value, keep values bigger than that value
            # loc_counts only exist for df_stats, calculate on the spot for df_tracks
            if ((key == 'loc_count') & ('loc_count' not in df.columns)):
                loc_counts = df['track.id'].value_counts()
                df = df.loc[df['track.id'].isin(loc_counts[loc_counts > value].index)]
            else:
                df = df.loc[df[key] > value]

    if 'cell_id' in df.columns:
        df = df.sort_values(['cell_id', 't', 'track.id'])
    else:
        df = df.sort_values(['t', 'track.id'])

    return df
