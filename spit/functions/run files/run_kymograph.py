import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm
from spit import tools
from spit import linking as link
from spit.analysis.functions import kymograph as kymo

# Prepare data
# path_movie = r'Y:\04 DNA paint tracking\Data\20220301 Jurkat 27\Run00012\Run00012_record.raw'
# path_df_tracks = r'Y:\04 DNA paint tracking\Data\20220301 Jurkat 27\Run00012\Run00012_record_ch1_locs_roi1.trackedTP.csv'  # trackpy
# path_df_stats = os.path.splitext(path_df_tracks)[0]+'_D.hdf'


path_movie = r'Y:\04 DNA paint tracking\Data\20220221 jurkat 26 differential labeling\Run00015\Run00015_record.raw'
path_df_tracks = r'Y:\04 DNA paint tracking\Data\20220221 jurkat 26 differential labeling\Run00015\Run00015_record_ch1_locs_trsr5mm4_nm_trackpy.csv'  # trackpy
path_df_stats = os.path.splitext(path_df_tracks)[0]+'_stats.hdf'


# Load data
movieRaw, info = tools.load_raw(path_movie)
movieSplit = np.array_split(movieRaw[:, :, 1:2047], 3, axis=2)
movie = movieSplit[1]
df_tracks = pd.read_csv(path_df_tracks)
df_tracks[['x', 'y']] = df_tracks[['x', 'y']]/108  # convert to pixel coordinates
df_stats = pd.read_hdf(path_df_stats)

id_list = [23, 3897, 6431]
df_tracks = tools.join_segments(df_tracks, id_list)

path_plots = tools.getOutputpath(path_df_tracks, 'polka', keepFilename=True)

# because of reworking analysis pipeline 05/2022
# df_stats = df_stats.rename(columns={'length': 'loc_count'})

# filter for track lengths and bridge agp
df_tracksF = link.filter_df(df_stats, filter_length=1, filter_D=0.00000000000001)
keepParticles = df_tracksF['track.id'].values
df_tracksF = df_tracks.loc[df_tracks['track.id'].isin(keepParticles)]
df_tracksF = tools.fill_track_gaps(df_tracksF)

# %% put tif as background in swift to quickly check for tracks to connect
kymo.crop_tif(movie, df_tracksF, path_plots)
# use
# id_list = [main track number, joining segments A, etc.]
# df_tracks = tools.join_segments(df_tracks, id_list)

# %% kymograph plots (run this first to decide which particle to choose)
for particle in tqdm(keepParticles):
    kymo.plot_wrapper_kymo(movie, df_tracksF, particle, path_plots, dt=0.08,
                           boxsize=7, smoothing=5,
                           polka=True, kymo=True, animate=False)

# %% Localizations on cell
particleH = 23  # 4898
frame = df_tracks.loc[df_tracks['track.id'] == particleH].t.min()
kymo.plot_wrapper_cell(movie, df_tracksF, frame, path_movie,
                       path_plots, particleH=particleH, animate=True, vmin=130, vmax=400)


# %% animated intensity plot
# select current particle
# particle = 4898
dt = 0.08
boxArray, df_particleH = kymo.get_boxes(movie, df_tracks, particleH)
boxMean, photons, netgradient = kymo.get_intensities(
    boxArray, df_particleH, smoothing=2)

plt.style.use(
    r'C:\Users\niederauer\Dropbox\Work\!DNA tracking 2021\Figures\paper.mplstyle')
# Plot intensities/photon values
fig = plt.figure(figsize=[9, 6])
fig.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
fig.clear()
ax = fig.add_subplot(111)

artists = []
data, = ax.plot(boxMean, '.-', color='tomato', label='Box Mean', alpha=0.2)
artists.append([data])
for i in tqdm(np.arange(0, len(boxMean))):
    timeline = ax.axvline(i,
                          lw=5,
                          ls='--',
                          color='black',
                          alpha=0.3,
                          label='Median')
    data_ani, = ax.plot(boxMean[0:i], '.-', color='tomato')
    artists.append([data, data_ani, timeline])

ani = animation.ArtistAnimation(fig, artists, interval=10, blit=True)

xtick = np.arange(0, boxMean.shape[0], 40/dt)
ax.set_xticks(xtick)
ax.set_xticklabels((xtick*dt).astype(int))
# ax.set_ylim([130, 180])
ax.set_ylabel('Intensity [a.u.]')
ax.set_yticks([])
ax.set_xlabel('Time [s]')


# ani.save(f'{path_plots}_intensity_particle{particleH}.gif', writer='pillow', dpi=200)
ani.save(f'{path_plots}_intensity_particle{particleH}.mp4',
         fps=25, extra_args=['-vcodec', 'libx264'])


plt.style.use('default')
