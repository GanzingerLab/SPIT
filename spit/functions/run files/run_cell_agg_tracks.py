import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from spit import tools
from spit import linking as link
from spit.analysis.functions import cell_agg_tracks

path = r'Y:\04 DNA paint tracking\Data\20220505 JF dyes'
path = r'Y:\04 DNA paint tracking\Data\20211212 dspe-rgd repeat\PEG-RGD'
path = r'Y:\04 DNA paint tracking\Data\20220726 bg dna sd\long'
path = r'Y:\04 DNA paint tracking\Data\20220810 jurkat long videos 2'
paths = tools.scrape_data(path, matchstring='trackpy.csv')


duration = 250


for path in tqdm(paths):
    df_tracks = pd.read_csv(path)
    df_stats = pd.read_hdf(os.path.splitext(path)[0]+'_stats.hdf')
    # df_tracks['cell_id'] = 0 # for old datasets
    pathPlots = tools.getOutputpath(path, r'plots', keepFilename=(True))

    df_statsF = link.filter_df(df_stats, filter_length=10, filter_D=0.01)
    keep_particles = df_statsF['track.id'].values
    df_tracksF = df_tracks.loc[df_tracks['track.id'].isin(keep_particles)]

    for cell in df_statsF.cell_id.unique():
        df_tracksFC = df_tracksF.loc[df_tracksF.cell_id == cell]
        df_statsFC = df_statsF.loc[df_statsF.cell_id == cell]

        # Plot
        f, axs = plt.subplots(1, 3, figsize=(15, 5))
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                          top=0.75, wspace=0.5, hspace=0.5)
        for idx, start in enumerate([0, int(df_tracks.t.max()/2-duration), df_tracks.t.max()-duration]):
            cell_agg_tracks.plot_aggregated_tracks(df_tracksFC, df_statsFC, start,
                                                   duration, cell_id=cell, color='chocolate', ax=axs[idx])
        plt.tight_layout()
        plt.savefig(pathPlots + f'_cell_{int(cell)}.png', dpi=300)
