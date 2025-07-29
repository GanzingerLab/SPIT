import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.font_manager import FontProperties


def plot_colocDistances(df_colocs, threshold, path=None, color='black', ax=None):
    """
    Plot histogram of colocalization distances.

    colocEvents : Dataframe
        Dataframe containing colocalization events with distances in [nm]
    path : String
        Path to the location where the plot should be saved.
    """
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Colocalization distances\n'+os.path.split(path)[1])
        save_plot = True

    ax.hist(df_colocs.dist,
            bins=np.linspace(0, threshold, 40),
            color=color,
            alpha=1,
            histtype='step'
            )
    ax.axvline(threshold,
               lw=2,
               ls='--',
               color='orange',
               alpha=1,)

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_coloc_distances.png', dpi=200)
    ax.set_xlabel(r' Distance [nm]')
    ax.set_ylabel('# Events')

# %% Locs and colocs per frame


def plot_colocs_locs_per_frame(df_locs_ch0, df_locs_ch1, df_colocs, dt, roll_param=5, color_ch0='darkslategrey', color_ch1='darkmagenta', path=None, ax=None):
    '''
    '''
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('(Co-)localizations per frame \n'+os.path.split(path)[1])
        save_plot = True

    # Prepare data
    # locsCh0_per_frame = df_locs_ch0.pivot_table(columns=['t'], aggfunc='size')
    # locsCh1_per_frame = df_locs_ch1.pivot_table(columns=['t'], aggfunc='size')
    frames_idx = range(max(df_locs_ch0.t.max(), df_locs_ch1.t.max()) + 1)

    # Count per frame and reindex to fill missing frames with 0
    locsCh0_per_frame = df_locs_ch0.groupby('t').size().reindex(frames_idx, fill_value=0)
    locsCh1_per_frame = df_locs_ch1.groupby('t').size().reindex(frames_idx, fill_value=0)
    colocs_per_frame = df_colocs.groupby('t').size().reindex(frames_idx, fill_value=0)
    # Ensure it's a proper Series indexed by frame (t)
    if isinstance(locsCh1_per_frame, pd.Series):
        locsCh1_per_frame.index.name = 't'
    else:
        locsCh1_per_frame = locsCh1_per_frame.T.squeeze()
        locsCh1_per_frame.index.name = 't'
    # colocs_per_frame = df_colocs.pivot_table(columns=['t'], aggfunc='size')

    # Fill frame rows with 0 where there is no coloc instead of just leaving the row out
    frames = pd.Series(np.zeros(max(df_locs_ch0.t.max(), df_locs_ch1.t.max()) + 1))

    frames.index.name = 't'
    # colocs_per_frame = colocs_per_frame.combine_first(frames)
    # if not frames.empty and not frames.dropna(how='all').empty:
    #     if colocs_per_frame.empty or colocs_per_frame.dropna(how='all').empty:
    #         colocs_per_frame = frames.copy()
    #     else:
    #         colocs_per_frame = colocs_per_frame.combine_first(frames)

    ax.plot(colocs_per_frame.rolling(roll_param).mean(),
            '-', c='darkorange', alpha=1)
    ax.plot(locsCh0_per_frame.rolling(roll_param).mean(),
            '-', c=color_ch0)
    ax.plot(locsCh1_per_frame.rolling(roll_param).mean(),
            '-', c=color_ch1)

    ax.set_ylabel('# (Co-)localizations')
    ax.set_xlabel('Time [s]')
    ax.set_ylim(bottom=0)
    ax.set_xlim([0, colocs_per_frame.size])
    xtick = np.linspace(0, colocs_per_frame.size, 5)
    ax.set_xticks(xtick)
    ax.set_xticklabels((xtick*dt).astype(int))

    # Save plot
    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_colocs_per_frame.png', dpi=200)


# %% Ratio of colocalization spot to total
def plot_colocs_ratio(df_locs_ch0, df_locs_ch1, df_colocs, dt, roll_param=5, path=None, color='black', ax=None):
    '''
    '''
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[7, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Colocalization share per frame \n'+os.path.split(path)[1])
        save_plot = True

    # Prepare data
    locsCh0_per_frame = df_locs_ch0.pivot_table(columns=['t'], aggfunc='size')
    locsCh1_per_frame = df_locs_ch1.pivot_table(columns=['t'], aggfunc='size')
    colocs_per_frame = df_colocs.pivot_table(columns=['t'], aggfunc='size')

    # Find minimum locs per frame because detectable dimers are restricted by the number of
    # visible particles in the less dense channel
    locs_min_per_frame = pd.concat(
        [locsCh0_per_frame, locsCh1_per_frame], axis=1).min(axis=1)
    colocs_ratio = colocs_per_frame/locs_min_per_frame
    colocs_ratio = colocs_ratio.fillna(0)

    ax.plot(colocs_ratio.rolling(roll_param).mean(),
            '-', c=color, alpha=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Colocalization ratio [%]')

    ax.set_ylim(bottom=0)
    ytick = np.linspace(0, 0.5, 5)
    ax.set_yticks(ytick)
    ax.set_yticklabels((ytick*100).astype(int))

    ax.set_xlim([0, colocs_ratio.size])
    xtick = np.linspace(0, colocs_ratio.size, 5)
    ax.set_xticks(xtick)
    ax.set_xticklabels((xtick*dt).astype(int))

    # Save plot
    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_colocs_ratio.png', dpi=200)


# %% Aggregated colocalization events


def plot_raw_colocs(df_colocs, color_dist=True, path=None, ax=None):
    # Plot
    save_plot = False
    if ax is None:
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Colocalizations \n'+os.path.split(path)[1])
        save_plot = True

    # color code either distance of coloc events, or frame number they appear in
    if color_dist:
        s = ax.scatter(df_colocs.x,
                       df_colocs.y,
                       c=df_colocs.dist,
                       cmap=plt.cm.get_cmap('plasma'),
                       s=0.3)
        colorlabel = 'Distance [nm]'

    else:
        s = ax.scatter(df_colocs.x,
                       df_colocs.y,
                       c=df_colocs.t,
                       cmap=plt.cm.get_cmap('viridis'),
                       s=0.3)
        colorlabel = 'Frame'

    # arrange colorbar
    cb = plt.colorbar(s, shrink=0.85, ax=ax)
    cb.set_label(colorlabel)
    units = 'um'

    fp = FontProperties()
    fp.set_size(20)
    bar = AnchoredSizeBar(ax.transData, 10000, f'10 {units}', loc='lower right', pad=0.2, borderpad=0.5, sep=3,
                          frameon=False, size_vertical=1000, prop=fp)
    ax.add_artist(bar)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_aspect('equal')

    # Save plot
    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_colocs_raw.png', dpi=200)


# %% WRAPPER: overview of colocs and locs
def plot_coloc_stats(df_locs_ch0, df_locs_ch1, df_colocs, threshold, path=None, dt=None, roll_param=5):
    f, axs = plt.subplots(2, 2, figsize=(8, 5))
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                      top=0.75, wspace=0.5, hspace=0.5)

    plot_colocDistances(df_colocs, threshold, ax=axs[0, 0])
    plot_raw_colocs(df_colocs,  ax=axs[0, 1])
    plot_colocs_locs_per_frame(df_locs_ch0, df_locs_ch1, df_colocs, dt, roll_param,
                               color_ch0='darkslategrey', color_ch1='darkmagenta', ax=axs[1, 0])
    plot_colocs_ratio(df_locs_ch0, df_locs_ch1, df_colocs, dt, roll_param, ax=axs[1, 1])

    f.suptitle(os.path.split(path)[1])
    # f.suptitle(f'{os.path.split(path)[1]}\n filtered: loc_count>{filter_length}, D_msd>{filter_D}')
    plt.tight_layout()
    plt.savefig(path+'_coloc_stats.png', dpi=200)

# %% Aggregated tracks and coloc-tracks


def plot_tracks_coloctracks(df_tracks0, df_tracks1, df_tracks_coloc, path=None, ax=None):
    save_plot = False
    if ax is None:
        f, axs = plt.subplots(1, 3, figsize=[10, 6])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.85)
        save_plot = True

    # Colors
    number_of_colors = 4
    colormap = plt.cm.tab20c
    unique_colors0 = [colormap(i) for i in range(0, number_of_colors)]
    unique_colors1 = [colormap(i) for i in range(12, 12+number_of_colors)]
    unique_colors2 = [colormap(i) for i in range(4, 4+number_of_colors)]
    unique_colors = [unique_colors0, unique_colors1, unique_colors2]

    # needed only for track length dependent alpha
    # colors = ['royalblue', 'indigo', 'darkorange']
    # data_props = [tracks_props0, tracks_props1, colocProps]

    data = [df_tracks0, df_tracks1, df_tracks_coloc]

    for idx, dataset in enumerate(data):
        # re-number particles for proper coloring
        unstacked = dataset.set_index(['track.id', 't'])[['x', 'y']].unstack()
        unstacked = unstacked.reset_index(drop=True)
        # bridge gaps by filling NaNs forward
        unstacked.x = unstacked.x.fillna(method='ffill', axis=1)
        unstacked.y = unstacked.y.fillna(method='ffill', axis=1)
        for i, trajectory in tqdm(unstacked.iterrows(), total=unstacked.shape[0]):
            axs[idx].plot(trajectory.x,
                          trajectory.y,
                          color=unique_colors[idx][trajectory.name % number_of_colors])
            # max_traj = data_props[idx].n_locs.max()
            # ax[idx].plot(trajectory.x,
            #               trajectory.y,
            #               color=colors[idx],
            #               alpha = trajectory.dropna().x.shape[0]/5*max_traj) #alpha by traj. length

    for i, subplot in np.ndenumerate(axs):
        subplot.axes.xaxis.set_visible(False)
        subplot.axes.yaxis.set_visible(False)
        subplot.set_xlim(1000, 70000)
        subplot.set_ylim(1000, 70000)
        subplot.set_aspect('equal')
    axs[0].set_title('Channel 0')
    axs[1].set_title('Channel 1')
    axs[2].set_title('Colocalization')
    plt.suptitle(os.path.dirname(path), y=0.8)

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_colocs_trio.png', dpi=200)


# plot_tracks_coloctracks(df_tracks0, df_tracks1, df_tracks_coloc, path=None, ax=None)
