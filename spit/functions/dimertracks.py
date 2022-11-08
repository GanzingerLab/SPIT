from matplotlib.collections import LineCollection
import os
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm
from spit import tools
from spit import linking as link

# %% Track and colocs
# Load and filter tracked data for track/coloc-track plots


def prepare_tracks(path_df_tracks, filter_length, start=0, end=50000, filter_D=None):
    df_tracks = pd.read_csv(path_df_tracks)
    df_tracks = df_tracks.loc[(df_tracks.t > start) & (df_tracks.t < end)]
    path_df_stats = os.path.splitext(path_df_tracks)[0]+'_stats.hdf'
    df_stats = pd.read_hdf(path_df_stats)
    df_statsF = link.filter_df(df_stats, filter_length, filter_D=filter_D)
    keep_particles = df_statsF['track.id'].values
    df_tracksF = df_tracks.loc[df_tracks['track.id'].isin(keep_particles)]
    if df_tracksF.empty:
        print('Dataframe is empty after filtering. Returning heads of original dataframes.')
        df_tracksF = df_tracks.head()
        df_statsF = df_stats.head()
    return df_tracksF, df_statsF

# Piehler Plot


def plot_tracks(df_tracks, color=None, path=None, ax=None):
    # color needs to be provided as integer 0/1/2
    # Plot
    save_plot = False
    number_of_colors = 4
    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Tracks \n'+os.path.split(path)[1])
        colormap = plt.cm.tab10
        color = [colormap(i) for i in range(0, 10)]
        save_plot = True
        number_of_colors = 10

    # colormap = plt.cm.tab20c
    # color_list = [[colormap(i) for i in range(j, j+4)] for j in [0, 12, 4]]
    # color = color_list[color]

    color = [color, tools.adjust_lightness(color, amount=0.4), tools.adjust_lightness(
        color, amount=0.6), tools.adjust_lightness(color, amount=0.8)]

    # re-number particles for proper coloring
    unstacked = df_tracks.set_index(['track.id', 't'])[['x', 'y']].unstack()
    unstacked = unstacked.reset_index(drop=True)
    # bridge gaps by filling NaNs forward
    unstacked.x = unstacked.x.fillna(method='ffill', axis=1)
    unstacked.y = unstacked.y.fillna(method='ffill', axis=1)
    for i, trajectory in tqdm(unstacked.iterrows(), total=unstacked.shape[0]):
        ax.plot(trajectory.x,
                trajectory.y,
                color=color[trajectory.name % number_of_colors])

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(1000, 70000)
    ax.set_ylim(1000, 70000)
    ax.set_aspect('equal')

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_tracks.png', dpi=400)


def plot_tracks_timecoded(df_tracks, path=None, ax=None):
    save_plot = False

    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_xlim(df_tracks.x.min(), df_tracks.x.max())
        ax.set_ylim(df_tracks.y.min(), df_tracks.y.max())
        ax.set_title('Time-coded tracks')
        save_plot = True

     # unstack (t and track.id exchanged)
    df_tracks = df_tracks.sort_values(by=['t'])
    df_tracks_unstacked = df_tracks.set_index(['t', 'track.id'])[['x', 'y']].unstack()

    df_tracks_unstacked.x = df_tracks_unstacked.x.fillna(
        method='ffill', axis=0)  # bridge gaps by filling NaNs forward
    df_tracks_unstacked.y = df_tracks_unstacked.y.fillna(
        method='ffill', axis=0)  # bridge gaps by filling NaNs forward

    color_numbers = np.arange(df_tracks.t.min(), df_tracks.t.max())
    for particle in df_tracks_unstacked.x:
        points = np.array([df_tracks_unstacked.x[particle].values,
                           df_tracks_unstacked.y[particle].values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=plt.cm.viridis, path_effects=[
            path_effects.Stroke(capstyle="round")])

        lc.set_array(color_numbers)
        # im = ax.add_collection(lc)
        ax.add_collection(lc)

    ax.set_aspect('equal')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_xlim(0, 73656)
    ax.set_ylim(0, 73656)
    if save_plot:
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(path + f'_timecoded.png', dpi=400)


def plot_all_dimer_tracks(df_dimers, path):
    # Wrapper for dimertracks.plot_tracks to have all dimer tracks in one figure
    for concentration in df_dimers['ligand_conc'].unique():

        df_dimersC = df_dimers.loc[df_dimers['ligand_conc'] == concentration]

        subplot_rows = int(np.ceil(np.sqrt(len(df_dimersC))))

        f, axs = plt.subplots(subplot_rows, subplot_rows,
                              figsize=(subplot_rows * 4, subplot_rows * 4))
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                          top=0.75, wspace=0.5, hspace=0.5)

        for idx, path_coloc in enumerate(df_dimersC.path):
            current_axes = (idx % subplot_rows, idx//subplot_rows)
            df_tracks_coloc, df_stats_coloc = prepare_tracks(
                path_coloc, filter_length=20, filter_D=0.01)
            plot_tracks(df_tracks_coloc, color='skyblue',
                        ax=axs[current_axes])
            axs[current_axes].set_rasterized(True)
            axs[current_axes].xaxis.set_visible(False)
            axs[current_axes].yaxis.set_visible(False)
            axs[current_axes].set_xlim(1000, 70000)
            axs[current_axes].set_ylim(1000, 70000)
            axs[current_axes].set_title(os.path.dirname(
                path_coloc[28:]), fontsize='xx-small')
        plt.tight_layout()
        plt.savefig(path+f'_dimer_tracks_{concentration}M.png', dpi=200)


# %% Dual color stuff


def prepare_tracks_cotracks(df_cotracks, df_tracks0, df_tracks1, path=None, save_csv=False):
    '''
    Load df_cotracks and df_tracks, merge df_tracks to df_tracks_all,
    add channel and coloc identifiers and unique track IDs
    '''
    df_tracks0['channel'] = 0
    df_tracks1['channel'] = 1

    df_tracks0 = df_tracks0.rename(columns={'locID': 'locID0'})
    df_tracks1 = df_tracks1.rename(columns={'locID': 'locID1'})

    df_tracks_all = pd.concat([df_tracks0, df_tracks1], ignore_index=True)
    df_tracks_all = tools.get_unique_trackIDs(df_tracks_all, group='channel')
    df_tracks_all['seg.id'] = df_tracks_all['track.id']

    # add coloc identifier
    df_tracks_all['coloc'] = 0

    df_tracks_all.loc[df_tracks_all.locID0.isin(df_cotracks.locID0.values) |
                      df_tracks_all.locID1.isin(df_cotracks.locID1.values),
                      ['coloc']] = 1

    if save_csv:
        df_tracks_all.to_csv(os.path.splitext(path)[0]+'_tracks.csv')

    return df_tracks_all


def get_trackIDs(df_tracks, df_cotracksC):
    '''
    get track IDs of beginning and end of current colocalization event
    '''
    # locID0 and locID1 at timepoint
    locIDs_start = df_cotracksC.loc[df_cotracksC.t ==
                                    df_cotracksC.t.min(), ['locID0', 'locID1']]
    locIDs_end = df_cotracksC.loc[df_cotracksC.t ==
                                  df_cotracksC.t.max(), ['locID0', 'locID1']]
    trackID0_start = df_tracks.loc[df_tracks.locID0 ==
                                   locIDs_start.iloc[0, 0], 'track.id'].iloc[0]
    trackID0_end = df_tracks.loc[df_tracks.locID0 ==
                                 locIDs_end.iloc[0, 0], 'track.id'].iloc[0]
    trackID1_start = df_tracks.loc[df_tracks.locID1 ==
                                   locIDs_start.iloc[0, 1], 'track.id'].iloc[0]
    trackID1_end = df_tracks.loc[df_tracks.locID1 ==
                                 locIDs_end.iloc[0, 1], 'track.id'].iloc[0]
    return [trackID0_start, trackID0_end], [trackID1_start, trackID1_end]


def get_stubs(trackIDsC, df_tracks, df_cotracksC, stub_length=10):
    '''
    Grab 10 closest localizations around the colocalizaton event
    '''
    # check if tracks continue and keep up to 10 closest locs in time
    track_pre = df_tracks.loc[(df_tracks['track.id'] ==
                               trackIDsC[0]) & (df_tracks.t <= df_cotracksC.t.min())].tail(n=stub_length)
    track_post = df_tracks.loc[(df_tracks['track.id'] ==
                                trackIDsC[1]) & (df_tracks.t >= df_cotracksC.t.max())].head(n=stub_length)
    return track_pre, track_post


def match_trackIDs(df_tracks_all, df_cotracksC):
    ''' # this is still buggy!
    Match track IDs of beginning, end and during colocalization event
        in an effort to also catch splitting and merging tracks
    '''
    [trackID0_start, trackID0_end], [trackID1_start,
                                     trackID1_end] = get_trackIDs(df_tracks_all, df_cotracksC)

    # forward trackID through coloc section (assign trackID to all locs that are also in cotracksC)
    df_tracks_all.loc[df_tracks_all.locID0.isin(
        df_cotracksC.locID0.values), 'track.id'] = trackID0_start
    df_tracks_all.loc[df_tracks_all.locID1.isin(
        df_cotracksC.locID1.values), 'track.id'] = trackID1_start

    # forward trackID also through split section (assign trackID after coloc event)
    df_tracks_all.loc[(df_tracks_all['track.id'] ==
                      trackID0_end) & (df_tracks_all.t > df_cotracksC.t.max()), 'track.id'] = trackID0_start
    df_tracks_all.loc[(df_tracks_all['track.id'] ==
                      trackID1_end) & (df_tracks_all.t > df_cotracksC.t.max()), 'track.id'] = trackID1_start

    df_tracks_all['seg.id'] = df_tracks_all['track.id']
    return df_tracks_all


def plot_individual_dimer_tracks(df_cotracksC, df_tracks, color0='mediumorchid', color1='seagreen', color_coloc='lightblue', shift=0, stub_length=10, coloc_only=False, path=None, ax=None, box_length=None):
    '''
    Plot individual tracks with overlayed colocalization track
    '''
    trackIDs = get_trackIDs(df_tracks, df_cotracksC)

    track_pre0, track_post0 = get_stubs(
        trackIDs[0], df_tracks, df_cotracksC, stub_length=stub_length)
    track_pre1, track_post1 = get_stubs(
        trackIDs[1], df_tracks, df_cotracksC, stub_length=stub_length)
    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[8, 8])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        cotrack = df_cotracksC['track.id'].iloc[0]
        ax.set_title(
            f'co-track {cotrack:05} \ntracks0: {trackIDs[0][0]:05}, {trackIDs[0][1]:05} \ntracks1: {trackIDs[1][0]:05}, {trackIDs[1][1]:05}',
            fontsize=18)
        save_plot = True

    # Visual parameters
    coloc_width = 5
    mono_width = 0.7
    triangle_size = 5
    alpha = 0.8

    # just to ensure time-sortedness since it is important for line-plots
    df_cotracksC = df_cotracksC.sort_values(by=['t'])

    ax.plot(df_cotracksC.loc[df_cotracksC.colocID >= 0].x,
            df_cotracksC.loc[df_cotracksC.colocID >= 0].y,
            color=color_coloc, alpha=0.7, lw=coloc_width, solid_capstyle='round')

    ax.plot(df_cotracksC.loc[df_cotracksC.colocID >= 0].loc0x+shift,
            df_cotracksC.loc[df_cotracksC.colocID >= 0].loc0y+shift,
            color=color0, alpha=1, lw=mono_width, solid_capstyle='round')

    ax.plot(df_cotracksC.loc[df_cotracksC.colocID >= 0].loc1x-shift,
            df_cotracksC.loc[df_cotracksC.colocID >= 0].loc1y-shift,
            color=color1, alpha=0.6, lw=mono_width, solid_capstyle='round')
    dt = 0.04

    if not coloc_only:

        # triangles marking start- and endpoints
        markerstyle = '>'
        markercolor = 'grey'
        ax.scatter(track_pre0.head(n=1).x, track_pre0.head(n=1).y, alpha=0.9,
                   marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
        ax.scatter(track_pre1.head(n=1).x, track_pre1.head(n=1).y, alpha=0.9,
                   marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
        markerstyle = 's'
        ax.scatter(track_post0.tail(n=1).x, track_post0.tail(n=1).y, alpha=0.9,
                   marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
        ax.scatter(track_post1.tail(n=1).x, track_post1.tail(n=1).y, alpha=0.9,
                   marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)

        ax.plot(track_pre0.x[:-1], track_pre0.y[:-1], color='white',
                alpha=alpha, lw=mono_width, solid_capstyle='round')
        ax.plot(track_pre1.x[:-1], track_pre1.y[:-1], color='white',
                alpha=alpha, lw=mono_width, solid_capstyle='round')
        ax.plot(track_post0.x[1:], track_post0.y[1:], color='white',
                alpha=alpha, lw=mono_width, solid_capstyle='round')
        ax.plot(track_post1.x[1:], track_post1.y[1:], color='white',
                alpha=alpha, lw=mono_width, solid_capstyle='round')

        ax.plot(track_pre0.x, track_pre0.y, color=color0,
                alpha=alpha, solid_capstyle='round')
        ax.plot(track_pre1.x, track_pre1.y, color=color1,
                alpha=alpha, solid_capstyle='round')
        ax.plot(track_post0.x, track_post0.y, color=color0,
                alpha=alpha, solid_capstyle='round')
        ax.plot(track_post1.x, track_post1.y, color=color1,
                alpha=alpha, solid_capstyle='round')

    # text annotation for start and end
    # textoffset = (-15, 50)
    # box_alpha = 0.7
    # box_color = '0.95'

    # ax.annotate(text=f't = {df_cotracksC.shape[0]*dt:.2f} s', xy=(df_cotracksC.tail(n=1).x, df_cotracksC.tail(n=1).y),
    #             zorder=100, color='k', xytext=textoffset, textcoords='offset points', size=10,
    #             bbox=dict(fc=box_color, ec='none', alpha=box_alpha))

    ax.text(0.93, 0.93, s=f't = {df_cotracksC.shape[0]*dt:.2f} s',
            ha='right', va='top', color='k', transform=ax.transAxes,
            bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='white'))

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if not box_length:
        box_length = max((xmax-xmin), (ymax-ymin))

    ax.set_xlim(xmin-((box_length-(xmax-xmin))/2),
                xmax+((box_length-(xmax-xmin))/2))
    ax.set_ylim(ymin-((box_length-(ymax-ymin))/2),
                ymax+((box_length-(ymax-ymin))/2))
    # print(f'Box length: {box_length} nm')
    if save_plot:
        plt.savefig(path + f'_cotrack.png', dpi=200)


# %% Animation dual color
def get_df_interaction(df_cotracksC, df_tracks_all):
    trackIDs = get_trackIDs(df_tracks_all, df_cotracksC)

    track_pre0, track_post0 = get_stubs(
        trackIDs[0], df_tracks_all, df_cotracksC, stub_length=40)
    track_pre1, track_post1 = get_stubs(
        trackIDs[1], df_tracks_all, df_cotracksC, stub_length=40)

    track_pre0 = track_pre0.rename(columns={'x': 'loc0x', 'y': 'loc0y'}).iloc[:-1]
    track_post0 = track_post0.rename(columns={'x': 'loc0x', 'y': 'loc0y'}).iloc[1:]
    track_pre1 = track_pre1.rename(columns={'x': 'loc1x', 'y': 'loc1y'}).iloc[:-1]
    track_post1 = track_post1.rename(columns={'x': 'loc1x', 'y': 'loc1y'}).iloc[1:]

    track0 = pd.concat([track_pre0, track_post0])
    track1 = pd.concat([track_pre1, track_post1])

    df_interaction = pd.concat([track0.merge(track1, on='t'), df_cotracksC]).sort_values(by='t')[
        ['t', 'loc0x', 'loc0y', 'loc1x', 'loc1y', 'x', 'y']]
    return df_interaction


def animate_individual_dimer_tracks(df_interaction, time_resolution, codec, box_length, path):
    # prepare figure
    f, ax = plt.subplots(figsize=[8, 8])
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
    # ax.axis('off')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.invert_yaxis()

    x_max = max(df_interaction.x.max(), df_interaction.loc0x.max(),
                df_interaction.loc1x.max())
    x_min = min(df_interaction.x.min(), df_interaction.loc0x.min(),
                df_interaction.loc1x.min())
    y_max = max(df_interaction.y.max(), df_interaction.loc0y.max(),
                df_interaction.loc1y.max())
    y_min = min(df_interaction.y.min(), df_interaction.loc0y.min(),
                df_interaction.loc1y.min())

    # box_length = 1.05*np.ceil(max((x_max-x_min), (y_max-y_min)))

    ax.set_xlim(x_min-((box_length-(x_max-x_min))/2),
                x_max+((box_length-(x_max-x_min))/2))
    ax.set_ylim(y_min-((box_length-(y_max-y_min))/2),
                y_max+((box_length-(y_max-y_min))/2))

    # Visual parameters
    color_0 = 'mediumorchid'
    color_1 = 'seagreen'
    color_coloc = 'lightblue'
    coloc_width = 15
    mono_width = 2
    triangle_size = 25

    alpha = 0.8
    artists = []
    dt = 0.04
    start = df_interaction.t.min()
    end = df_interaction.t.max()

    for t in tqdm(np.arange(start, end, time_resolution)):
        df_interactionT = df_interaction.loc[df_interaction.t <= t]

        markerstyle = '>'
        markercolor = 'grey'
        triangle_start0 = ax.scatter(df_interactionT.head(n=1).loc0x, df_interactionT.head(n=1).loc0y,
                                     marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
        triangle_start1 = ax.scatter(df_interactionT.head(n=1).loc1x, df_interactionT.head(n=1).loc1y,
                                     marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)

        # raw_data = plot_raw_data(movie, t, ax=ax, vmin=vmin, vmax=vmax)
        ax0, = ax.plot(df_interactionT.x, df_interactionT.y, color=color_coloc,
                       alpha=0.95, lw=coloc_width, solid_capstyle='round')
        ax1, = ax.plot(df_interactionT.loc0x, df_interactionT.loc0y, color=color_0,
                       alpha=1, lw=mono_width, solid_capstyle='round')
        axC, = ax.plot(df_interactionT.loc1x, df_interactionT.loc1y, color=color_1,
                       alpha=0.6, lw=mono_width, solid_capstyle='round')

        xmin = np.min(df_interaction.x)
        ymin = np.min(df_interaction.y)
        tC = t-start
        time = ax.text(0.95, 0.95, s=f't = {tC*dt:3.2f} s', fontsize=15, ha='right', va='top', color='k', transform=ax.transAxes,
                       bbox=dict(facecolor='ghostwhite', alpha=0.8, edgecolor='lightgrey'))

        ax.set_aspect('equal')
        artists.append([triangle_start0, triangle_start1, ax0, ax1, axC, time])

    markerstyle = 's'
    triangle_end0 = ax.scatter(df_interactionT.tail(n=1).loc0x, df_interactionT.tail(n=1).loc0y,
                               marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
    triangle_end1 = ax.scatter(df_interactionT.tail(n=1).loc1x, df_interactionT.tail(n=1).loc1y,
                               marker=markerstyle, color=markercolor, s=triangle_size, zorder=100)
    artists.append([triangle_start0, triangle_start1, triangle_end0,
                   triangle_end1, ax0, ax1, axC, time])

    plt.close()
    ani = animation.ArtistAnimation(f, artists, interval=10)
    if codec == 'gif':
        ani.save(f'{path}_ani_indivi_dimer.gif', writer='pillow', dpi=200)
    else:
        ani.save(f'{path}_ani_indivi_dimer.mp4',
                 fps=40, extra_args=['-vcodec', 'libx264'])
