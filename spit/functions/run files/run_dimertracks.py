import os
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm
from spit import tools
from spit import linking as link
from spit.analysis.functions import dimertracks as dimertracks
from spit.analysis.functions import dimerKD as dimerKD


# % Aggregated tracks and coloc tracks
# path = r'Y:\11 FKBP interaction\Data\20220614 dimer antibody kon koff\Ligand 1E-7 M Ab\Long video\vis\Run00010'
path = r'Y:\11 FKBP interaction\Data\20220404 Dimer kinetic\2 During ligand\4.2 pM protein\Run00002'
path = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel\Ligand 1E-8 M\4%6%laserpower\Run00005'

path_ch0 = glob(path + '//**//*ch0*locs_nm_*.csv', recursive=True)[0]
path_ch1 = glob(path + '//**//*ch1*locs_nm_*.csv', recursive=True)[0]
path_coloc = glob(path + '//**//*colocs_nm_*.csv', recursive=True)[0]

df_tracks_ch0, df_stats_ch0 = dimertracks.prepare_tracks(
    path_ch0, filter_length=10, filter_D=0.01)
df_tracks_ch1, df_stats_ch1 = dimertracks.prepare_tracks(
    path_ch1, filter_length=10, filter_D=0.01)
df_tracks_coloc, df_stats_coloc = dimertracks.prepare_tracks(
    path_coloc, filter_length=10, filter_D=0.01)

# Colors
# colormap = plt.cm.tab20c
# unique_colors0 = [colormap(i) for i in range(0, 4)]
# unique_colors1 = [colormap(i) for i in range(12, 16)]
# unique_colors2 = [colormap(i) for i in range(4, 8)]

unique_colors0 = 'orchid'  # individual tracks: mediumorchid
unique_colors1 = 'limegreen'  # individual tracks: seagreen
unique_colors2 = 'skyblue'  # individual tracks: lightblue
# Plot
f, axs = plt.subplots(1, 3, figsize=(6, 3))
f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                  top=0.75, wspace=0.5, hspace=0.5)

dimertracks.plot_tracks(df_tracks_ch0, unique_colors0, path, ax=axs[0])
dimertracks.plot_tracks(df_tracks_ch1, unique_colors1, path, ax=axs[1])
dimertracks.plot_tracks(df_tracks_coloc, unique_colors2, path, ax=axs[2])

outputPath = tools.getOutputpath(path, 'plots/'+os.path.basename(path))
f.suptitle('Tracks and colocalized tracks\n'+os.path.split(path)[1])
plt.tight_layout()
plt.savefig(outputPath+'_colocs_trio.png', dpi=200)

# %% Aggregated coloc tracks with time-coding
# path_coloc = r'Y:\11 FKBP interaction\Data\20220614 dimer antibody kon koff\Ligand 1E-7 M Ab\Long video\vis\Run00010\Run00010_record_colocs_nm_trackpy.csv'
path_coloc = r'Y:\11 FKBP interaction\Data\20220404 Dimer kinetic\2 During ligand\4.2 pM protein\Run00002\Run00002_record_colocs_nm_trackpy.csv'

pathPlots = tools.getOutputpath(path_coloc, 'plots', keepFilename=True)
df_cotracks, df_tracks_all = dimertracks.prepare_tracks_cotracks(
    path_coloc, save_csv=False)
df_cotracksF = df_cotracks.loc[df_cotracks['track.loc_count'] > 100]
dimertracks.plot_tracks_timecoded(df_cotracksF, path=pathPlots, ax=None)

# # %%  Dual-color aggregated dimer tracks with individual colors
# path_coloc = r'C:\Users\niederauer\Dropbox\fkbp\Run00003\Run00003_record_colocs_nm_swift.csv'
# pathPlots = tools.getOutputpath(path_coloc, 'plots', keepFilename=True)
# df_cotracks, df_tracks_all = dimertracks.prepare_tracks_cotracks(
#     path_coloc, save_csv=False)

# df_cotracksF = df_cotracks.loc[df_cotracks['track.loc_count'] > 100]


# f = plt.figure(figsize=[8, 8])
# f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
# ax = f.add_subplot(111)

# for cotrack in tqdm(df_cotracksF['track.id'].unique()):
#     # select current co-track
#     df_cotracksC = df_cotracksF.loc[df_cotracksF['track.id'] == cotrack]
#     dimertracks.plot_individual_dimer_tracks(
#         df_cotracksC, df_tracks_all, ax=ax, aggregated=True)

# ax.set_aspect('equal')
# ax.set_xlim(0, 73656)
# ax.set_ylim(0, 73656)

# plt.tight_layout()
# plt.savefig(pathPlots + f'_cotrack.pdf', dpi=200)
# %% Dual-color individual dimer tracks (with and without matched trackIDs)
# path_coloc = r'Y:\11 FKBP interaction\Data\20220614 dimer antibody kon koff\Ligand 1E-7 M Ab\Long video\vis\Run00010\Run00010_record_colocs_nm_trackpy.csv'
path_coloc = r'Y:\11 FKBP interaction\Data\20220404 Dimer kinetic\2 During ligand\4.2 pM protein\Run00002\Run00002_record_colocs_nm_swift.csv'
path_coloc = r'Y:\11 FKBP interaction\Data\20220404 Dimer kinetic\2 During ligand\4.2 pM protein\Run00002\Run00002_record_colocs_nm_trackpy.csv'
path_coloc = r'Y:\11 FKBP interaction\Data\20220817 best ligands\PAINT\Ligand\Run00042\Run00042_record_colocs_nm_trackpy.csv'
path_coloc = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel\Ligand 1E-8 M\4%6%laserpower\Run00005\Run00005_record_colocs_nm_swift.csv'

df_cotracks = pd.read_csv(path_coloc)
path_coloc = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel\Ligand 1E-8 M\4%6%laserpower\Run00005\Run00005_record_colocs_nm_trackpy.csv'


# merge some cotracks
# df_cotracks = tools.join_segments(df_cotracks, [1139, 2255])

df_tracks0 = pd.read_csv(path_coloc.replace('colocs', 'ch0_locs'))
df_tracks1 = pd.read_csv(path_coloc.replace('colocs', 'ch1_locs'))


pathPlots = tools.getOutputpath(path_coloc, 'plots', keepFilename=True)
df_tracks_all = dimertracks.prepare_tracks_cotracks(
    df_cotracks, df_tracks0, df_tracks1, path_coloc, save_csv=True)


matched = False
if matched:  # this is still buggy!
    # Prepare matched dataframe
    df_tracks_matched = df_tracks_all.copy(deep=True)
    for cotrack in tqdm(df_cotracks['track.id'].unique()):
        df_cotracksC = df_cotracks.loc[df_cotracks['track.id'] == cotrack]
        df_tracks_matched = dimertracks.match_trackIDs(df_tracks_matched, df_cotracksC)
    df_tracks_matched.to_csv(os.path.splitext(path_coloc)[0]+'_matched.csv')

# Filter colocalization events by length
# df_cotracksF = df_cotracks.loc[df_cotracks['track.loc_count'] > 800]
df_cotracksF = df_cotracks.loc[df_cotracks['track.loc_count'] > 800]

box_length = 1.1*np.ceil(df_cotracksF.groupby('track.id').apply(
    lambda l: (max(l.x.max()-l.x.min(), l.y.max()-l.y.min()))).max())

for cotrack in tqdm(df_cotracksF['track.id'].unique()):
    # select current co-track
    df_cotracksC = df_cotracksF.loc[df_cotracksF['track.id'] == cotrack]
    if matched:
        dimertracks.plot_individual_dimer_tracks(
            df_cotracksC, df_tracks_matched, path=pathPlots+str(cotrack)+'_matched', box_length=box_length)
    else:
        dimertracks.plot_individual_dimer_tracks(df_cotracksC, df_tracks_all,
                                                 path=pathPlots+str(cotrack), box_length=box_length)


# %% Dual-color individual dimer tracks ANIMATED
path_coloc = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel\Ligand 1E-8 M\4%6%laserpower\Run00005\Run00005_record_colocs_nm_swift.csv'
pathPlots = tools.getOutputpath(path_coloc, 'plots', keepFilename=True)

df_cotracks = pd.read_csv(path_coloc)
# df_cotracks = tools.join_segments(df_cotracks, [1139, 2255])
path_coloc = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel\Ligand 1E-8 M\4%6%laserpower\Run00005\Run00005_record_colocs_nm_trackpy.csv'
df_tracks0 = pd.read_csv(path_coloc.replace('colocs', 'ch0_locs'))
df_tracks1 = pd.read_csv(path_coloc.replace('colocs', 'ch1_locs'))

df_tracks_all = dimertracks.prepare_tracks_cotracks(
    df_cotracks, df_tracks0, df_tracks1, path_coloc, save_csv=False)

df_cotracksF = df_cotracks.loc[df_cotracks['track.loc_count'] > 100]

# common box size for all dimer tracks
box_length = 1.5*np.ceil(df_cotracksF.groupby('track.id').apply(
    lambda l: (max(l.x.max()-l.x.min(), l.y.max()-l.y.min()))).max())

time_resolution = 1
codec = 'mp4'
# df_cotracks['coloc'] = 1

for cotrack in df_cotracksF['track.id'].unique():
    df_cotracksC = df_cotracks.loc[df_cotracks['track.id'] == cotrack]
    df_interaction = dimertracks.get_df_interaction(df_cotracksC, df_tracks_all)
    dimertracks.animate_individual_dimer_tracks(
        df_interaction, time_resolution, codec, box_length, path=pathPlots+str(cotrack))
