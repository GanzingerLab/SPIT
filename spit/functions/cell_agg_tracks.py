import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from spit import tools
from spit import linking as link
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def plot_aggregated_tracks(df_tracks, df_stats, start, duration, color=None, cell_id='0', scalebar=True, dt=0.08, px2nm=108, path=None, ax=None):
    save_plot = False

    if ax is None:
        # plot data
        f = plt.figure(figsize=[5, 5])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title(f' Start: {start*dt}s')
        save_plot = True

    # set up figure
    if not color:
        cmap = plt.get_cmap('tab20c')
        colors = cmap.colors
    else:
        color_base = color
        colors = [tools.adjust_lightness(color_base, amount=0.4), tools.adjust_lightness(
            color_base, amount=0.6), tools.adjust_lightness(color_base, amount=0.8), tools.adjust_lightness(color_base, amount=1)]

    ax.set_prop_cycle(color=colors)

    # set up aggregation
    end = start + duration
    df_tracks_agg = df_tracks.loc[(start < df_tracks.t) & (df_tracks.t < end)]

    # Tracks
    for particle in df_tracks_agg['track.id'].unique():
        df_particle = df_tracks_agg.loc[df_tracks_agg['track.id'] == particle]
        ax.plot(df_particle.x, df_particle.y)

    # ROI and scalebar
    contour = df_stats.contour.values[0]*px2nm
    # contour_filled = np.append(contour, [contour[0]], axis=0) #not needed anymore
    ax.plot(contour[:, 0], contour[:, 1], '--', color='dimgray')
    ax.set_title(f' Start: {start*dt}s')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.set_title('Frames ' + str(start) + ' to ' + str(end))
    if scalebar:
        scalebar_height = (ax.get_ylim()[1]-ax.get_ylim()[0])*0.03

        bar = AnchoredSizeBar(ax.transData, 10000, '', loc='lower right', pad=0.2, borderpad=1, sep=9,
                              frameon=False, size_vertical=scalebar_height, color='black')
        ax.add_artist(bar)

    if save_plot:
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(
            path + f'_cell_{df_tracks.cell_id[0]}_agg_tracks_{start}.png', dpi=300)
