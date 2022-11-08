from spit import plot_diffusion
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from spit import linking as link
from spit import tools


def concat_data(paths):
    df_tracks_list = []
    df_stats_list = []
    df_keys = []

    for path in tqdm(paths):
        df_tracks_list.append(pd.read_csv(path))
        df_stats_list.append(pd.read_hdf(os.path.splitext(path)[0]+'_stats.hdf'))
        df_keys.append(tools.find_between(path, 'locs_', '_nm_swift'))

    df_tracks = pd.concat(df_tracks_list, keys=df_keys).reset_index(
        level=0).rename(columns={'level_0': 'param'})
    df_stats = pd.concat(df_stats_list, keys=df_keys).reset_index(
        level=0).rename(columns={'level_0': 'param'})

    df_tracks = tools.get_unique_trackIDs(df_tracks, group='param')
    df_stats = tools.get_unique_trackIDs(df_stats, group='param')
    return df_tracks, df_stats


def plot_param_sweep_stats(df_tracks, df_stats, df_statsF, title, path, dt):
    keep_particles = df_statsF['track.id'].values
    df_tracksF = df_tracks.loc[df_tracks['track.id'].isin(keep_particles)]

    f, axs = plt.subplots(2, 4, figsize=(16, 6))
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                      top=0.75, wspace=0.5, hspace=0.5)

    nParam = np.sort(df_tracks['param'].unique())
    if len(nParam) == 1:
        colors = iter(['black'])
    else:
        colors = iter(plt.cm.viridis(np.linspace(0, 1, len(nParam))))
    for idx, param in enumerate(nParam):
        colorC = next(colors)
        df_tracksF_param = df_tracksF.loc[df_tracksF.param == param]
        df_statsF_param = df_statsF.loc[df_statsF.param == param]

        tau_bleach = plot_diffusion.plot_x_per_frame(df_tracksF_param,
                                                     'Tracks',
                                                     dt,
                                                     roll_param=10,
                                                     ignore_start=0,
                                                     color=colorC,
                                                     split=True,
                                                     ax=axs[0, 0])

        duration = plot_diffusion.plot_track_lengths(df_statsF_param,
                                                     dt,
                                                     color=colorC,
                                                     split=True,
                                                     ax=axs[0, 1])

        Tcrit = plot_diffusion.plot_NgT(df_tracksF_param,
                                        dt,
                                        color=colorC,
                                        split=True,
                                        ax=axs[0, 2],)

        D_msd = plot_diffusion.plot_Dmsd(df_statsF_param,
                                         dt, ax=axs[0, 3],
                                         color=colorC,
                                         split=True)

        axs[1, 0].scatter(x=idx, y=tau_bleach*dt, color=colorC)
        axs[1, 0].set_ylabel('Bleaching time [s]')
        axs[1, 1].scatter(x=idx, y=duration, color=colorC)
        axs[1, 1].set_ylabel('Track duration [s]')
        axs[1, 2].scatter(x=idx, y=Tcrit, color=colorC)
        axs[1, 2].set_ylabel('T 1/2 [s]')
        axs[1, 3].scatter(x=idx, y=D_msd, color=colorC)
        axs[1, 3].set_ylabel(r' $D_{\mathrm{msd}} \ [\mu m^2/s]$')
        # print(df_tracksF_param)
    for ax in axs[1, :]:
        ax.set_xlabel('ID')
        ax.set_xticks(np.arange(0, len(nParam), 1))
        ax.set_xticklabels(np.arange(0, len(nParam), 1))

    f.legend([f'{x}' for x in nParam], borderaxespad=0.1, loc='upper left',
             bbox_to_anchor=(1, 0.8), title='Files')

    f.suptitle(title, size='xx-large')
    # # f.suptitle(f'{os.path.split(path)[1]}\n filtered: loc_count>{filter_length}, D_msd>{filter_D}')
    plt.tight_layout()
    plt.savefig(path+'_stats.png', dpi=200, bbox_inches='tight')
    plt.savefig(path+'_stats.pdf', dpi=200, bbox_inches='tight')
