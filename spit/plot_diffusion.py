# plotting diffusion stuff
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties
from spit import tools
from spit import linking as link
from spit import localize


# %% Localization precision
def plot_localization_precision(df_locs, path=None, ax=None):
    # Remove outliers (0.1% of total)
    loc_precision_filtered = df_locs.loc_precision.loc[df_locs.loc_precision < df_locs.loc_precision.quantile(
        0.999)]
    loc_precision_median = loc_precision_filtered.median()

    data_precision = loc_precision_filtered.values

    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Localization precision \n'+os.path.split(path)[1], pad=40)
        save_plot = True

    # Histogram of localization precisions
    _, bins, _ = ax.hist(data_precision,
                         bins='fd',
                         histtype='step',
                         density=1,
                         color='black'
                         )
    # Fit log-norm distribution
    shape, loc, scale = scipy.stats.lognorm.fit(loc_precision_filtered.values)
    log_norm_fit = scipy.stats.lognorm.pdf(bins, shape, loc, scale)
    ax.plot(bins, log_norm_fit, color='orange')
    ax.text(0.95, 0.05, s=rf'$\sigma = $ {loc_precision_median:.0f} nm',
            ha='right', va='bottom', color='k', transform=ax.transAxes,
            bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))
    ax.axvline(loc_precision_median,
               lw=1.5,
               ls='--',
               color='black',
               alpha=0.3,
               label='Median')
    ax.set_xlabel('Localization precision [nm]')
    ax.set_ylabel('PDF [a.u.]')

    # Histogram of log localization precisions with fit
    ins = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
    _, bins_log, _ = ins.hist(np.log(data_precision),
                              bins='fd',
                              histtype='step',
                              density=1,
                              color='black')

    # Fit normal distribution
    mu, sigma = scipy.stats.norm.fit(np.log(data_precision))
    norm_fit = scipy.stats.norm.pdf(bins_log, mu, sigma)
    ins.plot(bins_log, norm_fit, color='darkorange')
    ins.tick_params(axis='x', bottom=False, labelbottom=False)
    ins.yaxis.set_visible(False)
    ins.set_xlim(mu-1.8, mu+1.8)
    ins.set_xlabel('log-transformed')

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_precision.png', dpi=200)

# df_locs.hist('loc_precision', by='cell_id', grid=False, bins = 'fd')

# plot_localization_precision(df_locs, path)
# %% Nearest neighbors


def plot_nearest_neighbor(df_locs, path=None, color='black', split=False, ax=None):
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Nearest neighbor distance \n'+os.path.split(path)[1], pad=40)
        save_plot = True

    nearest_neighbor = localize.get_nearest_neighbor(df_locs)

    # remove NaN or inf values
    nearest_neighbor_fin = nearest_neighbor[np.isfinite(nearest_neighbor)]

    _, bins, _ = ax.hist(nearest_neighbor_fin,
                         bins='fd',
                         histtype='step',
                         density=1,
                         color=color,
                         )
    if not split:
        ax.axvline(nearest_neighbor_fin.quantile(q=0.5),
                   lw=1, ls='--', label='_nolegend_',
                   color='darkslategrey')
        ax.axvline(nearest_neighbor_fin.quantile(q=0.1),
                   lw=1, ls='--', label='_nolegend_',
                   color='darkslategrey')
        ax.text(0.95, 0.05,
                f'Of all NN distances: \n 50% > {nearest_neighbor.quantile(q=0.5):.1f} nm \n 90%  > {nearest_neighbor.quantile(q=0.1):.1f} nm',
                ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))
    # ax.text(0.5, 0.20, s=f'Median loc. precision xy \n= {loc_precision_median:.0f} nm',
        # ha='left', color='k', transform = ax.transAxes)
    ax.set_xlabel('Nearest neighbor [um]')
    ax.set_ylabel('PDF [a.u.]')

    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks/1000).astype(int))
    ax.set_xlim(0, np.quantile(nearest_neighbor_fin, q=0.999))

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_nearest_neighbor.png', dpi=200)
    return ax
# plot_nearest_neighbor(df_locs_masked, path)
# %% Expected displacement  * unused currently


def plot_expected_displacement(df_tracks, path=None, ax=None):
    # Remove outliers (0.1% of total) and non-diffusing particles
    df_tracks_mobile = df_tracks.loc[df_tracks['seg.motion'] == 'diffusion']
    df_tracks_mobile = df_tracks_mobile.loc[df_tracks_mobile['seg.mjd']
                                            < df_tracks_mobile['seg.mjd'].quantile(q=0.999)]

    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Expected displacement \n'+os.path.split(path)[1], pad=40)
        save_plot = True

    # np.average(data_mjd['jumps'], weights=data_mjd['loc_count'])
    _, bins, _ = ax.hist(df_tracks_mobile['seg.mjd'],
                         bins='fd',
                         histtype='step',
                         density=1,
                         color='black')
    # weights = data_mjd['loc_count']) #weigh histogram with trajectory length

    # fit normal distribution
    mu, sigma = scipy.stats.norm.fit(df_tracks_mobile['seg.mjd'])
    exp_displacement_max = df_tracks_mobile['seg.mjd'].quantile(q=0.999)
    exp_displacement_max_pp = df_tracks_mobile['seg.mjd'].max()
    norm_fit = scipy.stats.norm.pdf(bins, mu, sigma)
    ax.plot(bins, norm_fit, color='orange')

    ax.axvline(mu,
               lw=1.5,
               ls='--',
               color='black',
               alpha=0.8,
               label='Gauss center')

    ax.axvline(exp_displacement_max,
               lw=1.5,
               ls='--',
               color='darkslategray',
               alpha=0.2,
               label='99% percentile')
    ax.axvline(exp_displacement_max_pp,
               lw=1.5,
               ls='--',
               color='darkslategray',
               alpha=0.2,
               label='Max. displacement')

    weighted_median = np.median(df_tracks_mobile['seg.mjd'])
    ax.text(0.5, 0.85, s=f'Expected displacement \n = {weighted_median:.0f} nm',
            ha='left', color='k', transform=ax.transAxes)
    ax.set_xlabel('Mean (mobile) jump distance [nm]')
    ax.set_ylabel('PDF [a.u.]')
    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_exp_displacement.png', dpi=200)

# plot_expected_displacement(df_tracks,path)

# %% Expected displacement iteration


def plot_iteration(data, path):
    f = plt.figure(figsize=[5, 5])
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
    f.clear()
    ax = f.add_subplot(111)
    ax.set_title('Expected displacement \n'+os.path.split(path)[1], pad=40)
    ax.plot(data, 'o--')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Expected displacement parameter [nm]')
    plt.tight_layout()
    plt.savefig(path + '_exp_displacement_iteration.png', dpi=200)

# %% MSDs


def plot_msd(df_stats, dt, path=None, plot_loglog=True, ax=None):
    """
    Plot individual MSDs and the fit of the median MSD value. 
    Displays weighted median of the diffusion constant.

    Parameters
    ----------
    df_tracks : Dataframe
        Dataframe containing tracks, with their respective diffusion properties
    path : String
        Path to the location where the plot should be saved.
    px : Float
        Conversion of unit in dataframe to micrometer.
    dt : Float
        Exposure time in seconds.
    plot_loglog: Boolean
        Plotting with log-log axes.

    """
    # Prepare data
    nm2um = 0.001

    # prepare 25-50-75 quantile fits
    msd_fit_x = np.arange(1, df_stats.length.max()*0.25, 0.1)

    y_fit_q05 = tools.msd_free(msd_fit_x,
                               df_stats.msd_fit_a.quantile(q=0.5),
                               df_stats.msd_fit_b.quantile(q=0.5))

    y_fit_q075 = tools.msd_free(msd_fit_x,
                                df_stats.msd_fit_a.quantile(q=0.75),
                                df_stats.msd_fit_b.quantile(q=0.75))

    y_fit_q025 = tools.msd_free(msd_fit_x,
                                df_stats.msd_fit_a.quantile(q=0.25),
                                df_stats.msd_fit_b.quantile(q=0.25))

    save_plot = False
    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Iterative MSD fits \n'+os.path.split(path)[1])
        save_plot = True

    for row in df_stats.itertuples():
        data, = ax.plot(row.lagtimes*dt,
                        row.msd*nm2um**2,
                        '-',
                        c='gray',
                        lw=3,
                        alpha=0.3,
                        label='Data'
                        )

    # plot quantile fits
    median, = ax.plot(msd_fit_x*dt,
                      y_fit_q05*nm2um**2,
                      '-',
                      c='orange',
                      lw=3,
                      label='Median',
                      )

    interquartile = ax.fill_between(msd_fit_x * dt,
                                    y_fit_q025*nm2um**2,
                                    y_fit_q075*nm2um**2,
                                    color='orange',
                                    alpha=0.4,
                                    label='Interquartile range',
                                    zorder=10)

    # D_msd = df_tracks.msd_fit_a*(nm2um**2/(4*dt))
    df_stats = df_stats.loc[~df_stats.D_msd.isnull()]
    D_msd_median_w = np.average(
        df_stats.D_msd, weights=df_stats.loc_count)  # weighted median
    # D_msd_median = df_tracks.D_msd.quantile(q=0.5)

    ax.text(0.05, 0.95, s=rf'D_MSD = {D_msd_median_w:.2f} $\mu m^2/s$',
            ha='left', va='top', color='k', transform=ax.transAxes,
            bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

    ax.set_xlabel('Lag-time [s]')
    ax.set_ylabel(r'MSD $[\mu m^2]$')
    ax.set_xlim(left=dt)
    if plot_loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        loglog = '_log_'
    else:
        loglog = '_lin_'

    if save_plot:
        ax.legend(handles=[data, median, interquartile], loc='upper left')
        plt.tight_layout()
        plt.savefig(path + loglog + 'msd.png', dpi=200)

# plot_msd(df_tracks, path, dt, plot_loglog = True)
# %% D_msd Histogram


def plot_Dmsd(df_stats, dt, path=None, ax=None, color='black', split=False):
    """
    Plot histogram of diffusion coefficient distribution based on MSD of individual trajectories.

    Parameters
    ----------
    df_tracks : Dataframe
        Dataframe containing diffusion properties of the different particles
    path : String
        Path to the location where the plot should be saved.
    px : Float
        Conversion of unit in dataframe to micrometer.
    dt : Float
        Exposure time in seconds.
    """

    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Diffusion constants \n'+os.path.split(path)[1])
        save_plot = True

    df_stats = df_stats.loc[~df_stats.D_msd.isnull()]
    df_stats = df_stats.loc[df_stats.D_msd < df_stats.D_msd.quantile(q=0.99)]
    # df_tracks = df_tracks.loc[df_tracks.D_msd >0]

    # Bins: arrange such that 99% of the values are included and ignore negative D
    ax.hist(df_stats.D_msd,
            bins=np.linspace(0, df_stats.D_msd.quantile(q=0.999), 50),
            edgecolor=color,
            label='Data',
            alpha=1,
            weights=df_stats.loc_count,
            histtype='step',
            density=True
            )

    D_msd_median_w = np.average(df_stats.D_msd, weights=df_stats.loc_count)

    if not split:
        ax.text(0.95, 0.95, s=rf'D_MSD = {D_msd_median_w:.2f} $\mu m^2/s$',
                ha='right', va='top', color='k', transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

        ax.axvline(D_msd_median_w,
                   lw=3,
                   ls='--',
                   color='darkorange',
                   alpha=0.8,
                   label='Weighted median')

    ax.set_xlabel(r' $D_{\mathrm{msd}} \ [\mu m^2/s]$')
    ax.set_ylabel('# Tracks [a.u.]')

    if save_plot:
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path + '_D_msd.png', dpi=200)

    return D_msd_median_w

# plot_Dmsd(df_tracks, path, dt)


def plot_Dmsd_seg(df_tracks, dt, path=None, ax=None):
    """
    Plot weighted histogram of diffusion coefficients based on swift-calculated MSD of individual segments.
    df_tracks : Dataframe
        Dataframe containing diffusion properties of the different particles
    path : String
        Path to the location where the plot should be saved.
    px : Float
        Conversion of unit in dataframe to micrometer.
    dt : Float
        Exposure time in seconds.
    """

    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Diffusion constants \n'+os.path.split(path)[1])
        save_plot = True

    df_tracks = df_tracks.loc[~df_tracks['seg.D_msd'].isnull()]
    df_tracks = df_tracks.loc[df_tracks['seg.D_msd']
                              < df_tracks['seg.D_msd'].quantile(q=0.99)]
    # df_tracks = df_tracks.loc[df_tracks.D_msd >0]

    ax.hist(df_tracks['seg.D_msd'],
            bins='fd',
            color='gray',
            edgecolor='k',
            label='Data',
            alpha=1,
            histtype='step'
            )

    # Calculate and plot median D_msd
    D_msd_median_w = df_tracks['seg.D_msd'].quantile(q=0.5)
    ax.axvline(D_msd_median_w,
               lw=3,
               color='darkorange',
               alpha=0.8,
               label='Median')

    ax.text(0.95, 0.95, s=rf'D_MSD = {D_msd_median_w:.2f} $\mu m^2/s$',
            ha='right', va='top', color='k', transform=ax.transAxes,
            bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

    ax.set_xlabel(r' $D_{\mathrm{msd}} \ [\mu m^2/s]$')
    ax.set_ylabel('# Tracks')

    if save_plot:
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path + '_seg_D_msd.png', dpi=200)

# path=r'D:\test'
# dt=0.04
# df_tracksF=link.filter_df(df_tracks,10,0.001)
# plot_Dmsd_seg(df_tracks, dt, path)


# %% Jump distance
def plot_jd(df_stats, dt, path=None, ax=None):
    """
    Plot histogram of jump distribution for all particles. 

    Parameters
    ----------
    df_stats : Dataframe
        Dataframe containing diffusion properties of the different particles.
    path : String
        Path to the location where the plot should be saved.
    px : Float
        Conversion of unit in dataframe to micrometer.
    dt : Float
        Exposure time in seconds.
    """
    save_plot = False
    # Plot
    if ax is None:
        # plot data
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Jump Distance \n'+os.path.split(path)[1])
        save_plot = True

    nm2um = 0.001

    pars = df_stats.iloc[0][['D_jd0', 'A_jd0', 'D_jd1', 'A_jd1']].values
    jd_all = [nm2um*jump for track in df_stats.jumps for jump in track]

    # compute expected frequencies using obtained parameters
    # double the x-range to have surfaces filled
    exp_x = np.arange(0, 2*max(jd_all), step=0.001)
    exp_freq = []
    exp_freq_sub1 = []
    exp_freq_sub2 = []

    for x in exp_x:
        f = tools.pdf_jd(x, dt, *pars)
        exp_freq.append(f)

    for x in exp_x:
        f = tools.pdf_jd_sub(x, dt, pars[0], pars[1])
        exp_freq_sub1.append(f)

    for x in exp_x:
        f = tools.pdf_jd_sub(x, dt, pars[2], pars[3])
        exp_freq_sub2.append(f)

    ax.hist(jd_all,
            bins='fd',
            color='gray',
            density=False,
            ec='k',
            label='Data',
            alpha=0.4,
            histtype='stepfilled')

    ax.plot(exp_x,
            exp_freq,
            '-',
            color='darkorange',
            lw=2,
            alpha=1,
            zorder=50,
            )
    ax.fill(exp_x,
            exp_freq_sub2,
            '-',
            color='saddlebrown',
            alpha=0.35,
            label='Mobile')

    ax.fill(exp_x,
            exp_freq_sub1,
            '-',
            color='darkolivegreen',
            alpha=0.35,
            label='Immobile')

    ax.text(0.95, 0.95, s=rf'D1 = {pars[0]:.2} $\mu m^2/s$ \nD2 = {pars[2]:.2} $\mu m^2/s$',
            ha='right', va='top', color='k', transform=ax.transAxes, zorder=100,
            bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

    ax.set_ylabel('# Jumps')
    ax.set_xlabel('Jump distance $[nm]$')
    xtick = ax.get_xticks()
    ax.set_xticks(xtick)
    ax.set_xticklabels((xtick*1e3).astype(int))
    ax.set_xlim(0, np.quantile(jd_all, q=0.999))
    if save_plot:
        ax.legend()
        plt.tight_layout()
        plt.savefig(path + '_jd_hist.png', dpi=200)


# plot_jd(df_stats, dt=0.04, path=r'D:\test\Run00001\plots', ax=None)
# %% Plot Trajectories


def plot_tracks_filtered(df_tracks, df_tracksF, df_stats, path=None, px2nm=108, ax=None):
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Trajectories \n'+os.path.split(path)[1])
        save_plot = True

    df_tracks_trashed = df_tracks.loc[~df_tracks['track.id'].isin(
        df_tracksF['track.id'].unique())]

    df_tracksF_unstacked = df_tracksF.set_index(['track.id', 't'])[['x', 'y']].unstack()

    colormap = plt.cm.tab20b
    unique_colors = [colormap(i) for i in range(0, 10)]

    # re-number particles for proper coloring
    df_tracksF_unstacked.reset_index(drop=True, inplace=True)
    df_tracksF_unstacked.x = df_tracksF_unstacked.x.fillna(
        method='ffill', axis=1)  # bridge gaps by filling NaNs forward
    df_tracksF_unstacked.y = df_tracksF_unstacked.y.fillna(
        method='ffill', axis=1)  # bridge gaps by filling NaNs forward

    ax.scatter(df_tracks_trashed.x, df_tracks_trashed.y,
               marker='.', color='gray', alpha=0.1)
    for i, track in df_tracksF_unstacked.iterrows():
        ax.plot(track.x, track.y, color=unique_colors[track.name % 10])

    if 'contour' in df_stats.columns:
        nROI = df_tracks['cell_id'].unique()
        colors = iter([plt.cm.Accent(i) for i in range(len(nROI))])
        contours = df_stats.groupby('cell_id').contour.mean()
        for idx, roi in enumerate(nROI):
            contour = contours.loc[contours.index == roi].values[0]*px2nm
            # bridge gap between first and last point of the ROI
            contour_filled = np.append(contour, [contour[0]], axis=0)
            ax.plot(contour_filled[:, 0], contour_filled[:, 1],
                    color=next(colors), lw=2, ls='--')

    extent = df_tracks['y'].max()-df_tracks['y'].min()
    units = 'um'
    fp = FontProperties()
    fp.set_size(10)
    bar = AnchoredSizeBar(ax.transData, 10000, f'10 {units}', loc='lower right', pad=0.5, borderpad=0.5, sep=5,
                          frameon=False, size_vertical=extent*0.05, prop=fp)
    ax.add_artist(bar)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_aspect('equal')
    # ax.axis('equal','box')

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_tracks.png', dpi=200)

# plot_tracks_filtered(df_tracks, df_tracks, df_stats, path)

# %% Loc/Track per frame


def plot_x_per_frame(df, data_type, dt, path=None, roll_param=10, ignore_start=0, color='black', split=False, ax=None):
    '''
    '''
    # Prepare data (locs use 'frame' column, linked use 't' column)
    if 't' in df.columns:
        data_per_frame = df.pivot_table(columns=['t'], aggfunc='size')
    else:
        data_per_frame = df.pivot_table(columns=['frame'], aggfunc='size')

    # normalization
    data_per_frame = data_per_frame/data_per_frame.max()

    # Prepare fit (consider frames starting at ignore_start/dt)
    popt = tools.fit_decay(data_per_frame[int(ignore_start/dt):])
    tau_bleach = popt[1]
    x_fit = np.arange(0, data_per_frame.size*10, 1)
    y_fit = tools.exp_single(x_fit, *popt)

    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title(f'{data_type} per frame \n'+os.path.split(path)[1], pad=40)
        save_plot = True

    if not split:  # only plot fits if using whole dataset
        data_plot, = ax.plot(data_per_frame.rolling(roll_param).mean(),
                             '-', c=color, alpha=1)
        data_fit_plot, = ax.plot(x_fit,
                                 y_fit,
                                 '--', c='darkorange')
        ax.text(0.95, 0.05, s=f'tau_bleach = {tau_bleach:.2e} frames',
                ha='right', va='bottom', color='k', transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', alpha=1, edgecolor='lightgrey'))

    data_plot, = ax.plot(data_per_frame.rolling(roll_param).mean(),
                         '-', c=color, alpha=0.7)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, data_per_frame.size])
    ax.set_ylabel(f'# {data_type}')
    ax.set_xlabel('Time [s]')
    xtick = np.linspace(0, data_per_frame.size, 5)
    ax.set_xticks(xtick)
    ax.set_xticklabels((xtick*dt).astype(int))

    if save_plot:
        ax.legend(handles=[data_plot, data_fit_plot],
                  labels=[f'{data_type}', 'Exponential fit'])
        plt.tight_layout()
        plt.savefig(path+f'_{data_type}_per_frame.png', dpi=200)

    return tau_bleach  # bleaching rate
# plot_x_per_frame(df, 'Locs', path, dt, roll_param=10)
# %% NgT


def plot_NgT(df_tracks, dt, path=None, color='black', split=False, ax=None):
    '''
    '''
    Ts, NgT = link.tracks_greaterT(df_tracks, dt)
    Tcrit = Ts[np.argmax(NgT < 0.5)]
    # max_time = dt*linked.frame.max()

    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('TPP Plot \n'+os.path.split(path)[1])
        save_plot = True

    ax.plot(Ts, NgT, color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.loglog(Ts, NgT, color = color)

    if not split:
        ax.hlines(0.5, 0, Tcrit, color='darkorange', ls='--',
                  zorder=100, linewidth=2, alpha=0.2)
        ax.vlines(Tcrit, 0, 0.5, color='darkorange', ls='--',
                  zorder=100, linewidth=2, alpha=0.2)
        ax.text(0.95, 0.95, s=f'T_crit = {Tcrit:.1f} s',
                ha='right', va='top', color='k', transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))
    # ax.set_xlim(1,max_time+0.1*max_time)
    ax.set_ylim(bottom=0.01)
    ax.set_xlim(right=200)
    ax.set_ylabel('# TPP > t ')
    ax.set_xlabel('Time [s]')

    if save_plot:
        plt.tight_layout()
        plt.savefig(path+'_NgT.png', dpi=200)

    return Tcrit

# plot_NgT(linked, dt, path, color=color)

# %% Track length weighted histogram


def plot_track_lengths(df_stats, dt, path=None, color='black', split=False, ax=None, weighted=True):
    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Track lengths \n'+os.path.split(path)[1])
        save_plot = True
    if weighted:
        average_length = np.average(df_stats.length, weights=df_stats.length)
        _, bins = np.histogram(df_stats.length*dt, bins='doane')
        ax.hist(df_stats.length*dt,
                color=color,
                bins=bins,
                density=1,
                label='Data',
                weights=df_stats.length,
                histtype='step',
                zorder=100)

    else:
        average_length = np.average(df_stats.length)

        _, bins = np.histogram(df_stats.length*dt, bins='doane')
        ax.hist(df_stats.length*dt,
                color=color,
                bins=bins,
                density=1,
                label='Data',
                histtype='step',
                zorder=100)
        # ax.set_yscale('log')

    if not split:
        ax.text(0.95, 0.95, s=f'Duration = {average_length*dt:.1f} s',
                ha='right', va='top', color='k', transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

        ax.axvline(average_length*dt,
                   color='darkorange',
                   ls='--',
                   zorder=100,
                   linewidth=2,
                   )

    ax.set_ylabel('# Tracks [a.u.]')
    ax.set_xlabel('Track duration [s]')

    if save_plot:
        plt.tight_layout()
        plt.savefig(path+'track_lengths.png', dpi=200)

    return average_length*dt

# df_stats = df_stats.loc[df_stats.length>5]
# plot_track_lengths(df_stats_roiF, dt, path=path, color='black', alpha=1, lines = True, ax=None)
# %% WRAPPER: Localization stats


def plot_loc_stats(df_locs, path, dt=None):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
    # f.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.75, wspace=0.5, hspace=0.5)

    if 'cell_id' in df_locs.columns:
        nROI = np.sort(df_locs['cell_id'].unique())
        if len(nROI) == 1:
            colors = iter(['black'])
        else:
            colors = iter([plt.cm.Accent(i % 7) for i in range(len(nROI))])
        for idx, roi in enumerate(nROI):
            df_locs_subset = df_locs.loc[df_locs.cell_id == roi]
            plot_nearest_neighbor(df_locs_subset, ax=ax2,
                                  color=[next(colors)], split=True)
        ax2.legend(['roi '+str(x) for x in nROI])

    plot_localization_precision(df_locs, ax=ax1)
    plot_nearest_neighbor(df_locs, ax=ax2)
    tau_bleach = plot_x_per_frame(df_locs, 'Localizations',
                                  dt, roll_param=10, ignore_start=0, split=False, ax=ax3)

    f.suptitle(os.path.split(path)[1])
    plt.tight_layout()
    plt.savefig(path+'_stats.png', dpi=200)

    return tau_bleach
# plot_loc_stats(locs, path, dt=0.04)

# %% WRAPPER: Tracking stats


def plot_track_stats(df_tracks, df_stats, df_statsF, path, px2nm, dt=None):
    keep_particles = df_statsF['track.id'].values
    df_tracksF = df_tracks.loc[df_tracks['track.id'].isin(keep_particles)]

    f, axs = plt.subplots(2, 3, figsize=(12, 6))
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                      top=0.75, wspace=0.5, hspace=0.5)

    if 'cell_id' in df_tracks.columns:
        nROI = np.sort(df_tracks['cell_id'].unique())
        if len(nROI) == 1:
            colors = iter(['black'])
        else:
            colors = iter([plt.cm.Accent(i % 7) for i in range(len(nROI))])
        for idx, roi in enumerate(nROI):
            colorC = next(colors)
            df_tracksF_roi = df_tracksF.loc[df_tracksF.cell_id == roi]
            df_statsF_roi = df_statsF.loc[df_statsF.cell_id == roi]
            if df_statsF_roi.shape[0]:
                plot_x_per_frame(df_tracksF_roi,
                                 'Tracks',
                                 dt,
                                 roll_param=10,
                                 ignore_start=0,
                                 color=colorC,
                                 split=True,
                                 ax=axs[0, 0])
                
                plot_track_lengths(df_statsF_roi,
                                   dt,
                                   color=colorC,
                                   split=True,
                                   ax=axs[0, 1])
    
                plot_NgT(df_tracksF_roi,
                         dt,
                         color=colorC,
                         split=True,
                         ax=axs[0, 2],)

        axs[0, 2].legend([f'roi {x}' for x in nROI], loc='lower right')

    plot_x_per_frame(df_tracksF, 'Tracks', dt, roll_param=10, ignore_start=0,
                     ax=axs[0, 0])  # should be the filtered ones
    plot_track_lengths(df_statsF, dt, ax=axs[0, 1])
    plot_NgT(df_tracksF, dt, ax=axs[0, 2])
    plot_jd(df_stats, dt, ax=axs[1, 0])
    plot_msd(df_statsF, dt, plot_loglog=True, ax=axs[1, 1])
    plot_Dmsd(df_statsF, dt, ax=axs[1, 2])
    f.suptitle(os.path.split(path)[1])
    # f.suptitle(f'{os.path.split(path)[1]}\n filtered: loc_count>{filter_length}, D_msd>{filter_D}')
    plt.tight_layout()
    plt.savefig(path+'_stats.png', dpi=200)

    # speed up plotting by taking tracks only from first and last 100 frames
    df_tracks_1k = df_tracks.loc[(df_tracks.t < 100) | (
        df_tracks.t > df_tracks.t.max()-100)]
    df_tracksF_1k = df_tracksF.loc[(df_tracks.t < 100) | (
        df_tracks.t > df_tracks.t.max()-100)]

    plot_tracks_filtered(df_tracks_1k, df_tracksF_1k, df_stats, path, px2nm=px2nm)


# plot_track_stats(linked, track_stats, track_stats_filtered, path, dt=0.04)
