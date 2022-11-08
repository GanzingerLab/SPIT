'''
Extracts spots from movies based on tracked localizations and creates
 1) animated overlay of localizations marked as squares/cross on real movie data
 2) overview of all spots extracted from the movie based on the localization coordinates
 3) Kymograph of above spots
 4) intensity/photon values of these spots
Gaps in the track are handled by keeping the previous value.

Input: raw movie file (.raw) and tracks (.csv)
'''
import trackpy as tp
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd
from tqdm import tqdm
import matplotlib.animation as animation
from PIL import Image
from spit import tools
from spit import linking as link
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# %% Cell: Spot tracking


def crop_tif(movie, df_tracks, path):
    ''' 
    Generate .tif file to use in swift with offset.

    df_tracks: in pixel format
    movie: already split
    '''
    square_length = np.max((df_tracks.x.max()-df_tracks.x.min(),
                            df_tracks.y.max()-df_tracks.y.min()))
    padding = -1

    movie_crop = movie[:, int(df_tracks.y.min()-padding): int(padding+df_tracks.y.min()+square_length),
                       int(df_tracks.x.min()-padding): int(padding+df_tracks.x.min()+square_length)]

    y_offset = df_tracks.y.min()-padding
    x_offset = df_tracks.x.min()-padding

    imlist = []
    for m in tqdm(movie_crop):
        imlist.append(Image.fromarray(m))
    imlist[0].save(f'{path}_crop_yoff_{y_offset}_xoff_{x_offset}.tif',
                   save_all=True, append_images=imlist[1:])


def plot_cell_locs(df_tracks, frame, ax, edgecolors='yellow'):
    aspectsetting = 1
    # take only the row with current frame
    locsCurrent = df_tracks.loc[df_tracks.t == frame]
    # plot localization in frame as a square
    squares = ax.scatter(locsCurrent.x/aspectsetting,
                         locsCurrent.y/aspectsetting,
                         marker='s',
                         facecolors='none',
                         edgecolors=edgecolors,
                         s=300,
                         linewidths=2,
                         alpha=1)

    # plot localization in frame as a cross
    cross = ax.scatter(locsCurrent.x/aspectsetting,
                       locsCurrent.y/aspectsetting,
                       marker='x',
                       color='red',
                       s=30,
                       linewidths=1,
                       alpha=0.00)

    return squares, cross


def plot_cell_data(movie, frame, ax, vmin=None, vmax=None):
    # plot overlayed in background, adjust contrast manually
    movie_data = ax.imshow(movie[frame],
                           aspect='auto',
                           vmin=vmin,
                           vmax=vmax,
                           cmap=plt.get_cmap('Greys_r'))

    return movie_data


def plot_wrapper_cell(movie, df_tracks, choosen_frame, path_movie, path_plots, particleH=None, animate=False, codec='mp4', vmin=None, vmax=None):
    # Prepare figure
    fig, ax = plt.subplots(figsize=[6, 6])
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')

    squares, cross = plot_cell_locs(df_tracks, choosen_frame, ax=ax)
    background = plot_cell_data(movie, choosen_frame, ax=ax, vmin=vmin, vmax=vmax)
    if particleH:
        # Highlight one particle
        df_tracksH = df_tracks.loc[df_tracks['track.id'] == particleH]
        squaresH, crossH = plot_cell_locs(
            df_tracksH, choosen_frame, ax=ax, edgecolors='red')
        particleH = f'_particle{particleH}'
    else:
        particleH = ''

    square_length = np.max((df_tracks.x.max()-df_tracks.x.min(),
                           df_tracks.y.max()-df_tracks.y.min()))
    padding = -1
    ax.set_xlim([df_tracks.x.min()-padding, padding+df_tracks.x.min()+square_length])
    ax.set_ylim([df_tracks.y.min()-padding, padding+df_tracks.y.min()+square_length])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{path_plots}_cell_frame_{choosen_frame}{particleH}.pdf', dpi=300)

    if animate:
        # animation start-end values
        start = df_tracksH.t.min()
        end = df_tracksH.t.max()
        fig, ax = plt.subplots(figsize=[6, 6])
        # ax.set_title(os.path.splitext(path_movie)[0])
        ax.axis('off')
        ax.invert_yaxis()
        ax.set_xlim([df_tracks.x.min(), df_tracks.x.max()])
        ax.set_ylim([df_tracks.y.min(), df_tracks.y.max()])
        # ax.set_aspect('equal', adjustable='datalim')

        artists = []
        for frame in tqdm(range(start, end)):
            # background, squares, cross = plot_cell_locs(movie, df_tracks, frame, ax)
            squares, cross = plot_cell_locs(df_tracks, frame, ax)
            squaresH, crossH = plot_cell_locs(df_tracksH, frame, ax, edgecolors='red')
            background = plot_cell_data(movie, frame, ax=ax, vmin=vmin, vmax=vmax)

            bar = AnchoredSizeBar(ax.transData, 46, '', loc='lower right', pad=-0.5, borderpad=3, sep=-3,
                                  frameon=False, size_vertical=2, color='white')
            ax.add_artist(bar)
            time = ax.text(331, 495, f'{frame*0.08:.2f} s', color='white', fontsize=25)

            track = df_tracksH.loc[(frame-50 <= df_tracksH.t)
                                   & (df_tracksH.t <= frame)]
            trackWorm, = ax.plot(track.x, track.y, color='red', lw=2, alpha=0.7)

            artists.append([background, squares, cross,
                           squaresH, crossH, time, trackWorm])
            ax.set_aspect('equal')

            # plt.tight_layout()
        ani = animation.ArtistAnimation(fig, artists, interval=40)
        if codec == 'gif':
            ani.save(f'{path_plots}_cell_{particleH}.gif', writer='pillow', dpi=200)
        else:
            ani.save(f'{path_plots}_cell_{particleH}.mp4',
                     fps=25, extra_args=['-vcodec', 'libx264'])


# %% Kymographs
def get_boxes(movie, df_tracks, particle, boxsize=7):
    # Extract boxes around current centroid
    # initialize variables
    boxList = []
    box = np.zeros((boxsize, boxsize))
    length = int((boxsize-1)/2)
    # select current particle data
    df_particle = df_tracks.loc[df_tracks['track.id'] == int(particle)]

    framesList = df_particle.t.to_list()

    for frame in framesList:
        centroid = df_particle.loc[df_particle.t == frame][['y', 'x']].values.astype(int)[
            0]
        box = movie[frame, (centroid[0]-length):(centroid[0]+length+1),
                    (centroid[1]-length):(centroid[1]+length+1)]
        boxList.append(box)

    boxArray = np.array(boxList)
    boxArray = boxArray  # - np.min(boxArray)

    return boxArray, df_particle


def get_polka(boxArray):
    polkaLong = boxArray.shape[0]*boxArray.shape[1]
    polkaShort = boxArray.shape[2]
    polkaSquareLength = np.ceil(np.sqrt(polkaLong*polkaShort)/polkaShort).astype(int)
    polkaSquarePadding = np.zeros(
        (polkaShort, polkaShort*polkaSquareLength*polkaSquareLength-polkaLong))

    boxArrayPadded = np.concatenate((boxArray.ravel(), polkaSquarePadding.ravel()))
    polkadots = boxArrayPadded.reshape(polkaSquareLength, polkaSquareLength, polkaShort, polkaShort).swapaxes(
        1, 2).reshape(polkaSquareLength*polkaShort, polkaSquareLength*polkaShort)
    return polkadots


def get_kymo(boxArray):
    kymo = np.concatenate((np.mean(boxArray, axis=1),
                           np.mean(boxArray, axis=2)),
                          axis=1)
    kymo = np.transpose(kymo)
    return kymo


def get_intensities(boxArray, df_particle, smoothing=3):
    boxMean = np.convolve(np.mean(boxArray, axis=(1, 2)),
                          np.ones(smoothing)/smoothing, mode='valid')
    photons = np.convolve(df_particle.intensity.values,
                          np.ones(smoothing)/smoothing, mode='valid')
    netgradient = np.convolve(df_particle.net_gradient.values,
                              np.ones(smoothing)/smoothing, mode='valid')
    photon_scaling_factor = (photons.max()-photons.min())/(boxMean.max()-boxMean.min())
    boxMean_photon_rescaled = photon_scaling_factor * \
        (boxMean-boxMean.min()) + photons.min()
    # background = np.convolve(df_particle.bg.values, np.ones(smoothing)/smoothing, mode='valid')
    return boxMean_photon_rescaled, photons, netgradient


def plot_wrapper_kymo(movie, df_tracks, particle, path_plots, dt,
                      boxsize=7, smoothing=5,
                      polka=False, kymo=False, animate=False, codec='mp4'):

    # Prepare data
    boxArray, df_particle = get_boxes(movie, df_tracks, particle, boxsize)

    # Plot intensities/photon values
    boxMean, photons, netgradient = get_intensities(boxArray, df_particle)

    f, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    f.subplots_adjust(left=0.15, right=0.85, bottom=0.2,
                      top=0.75, wspace=0.5, hspace=0.5)

    axs[0].plot(boxMean, '.-', color='black', label='Box Mean')
    axs[1].plot(photons, '.-', color='orange', label='Photons')
    axs[2].plot(netgradient, '.-', color='gray', label='Net Gradient')

    xtick = np.arange(0, boxMean.shape[0], 10/dt)
    axs[2].set_xticks(xtick)
    axs[2].set_xticklabels((xtick*dt).astype(int))
    axs[2].set_xlabel('Time [s]')
    axs[0].set_title(f'Particle {particle:.0f}')
    for ax in axs:
        ax.legend(loc='upper right')
        ax.set_ylabel('Intensity [a.u.]')

    plt.tight_layout()
    plt.savefig(f'{path_plots}_photons_{particle}.png', dpi=200)

    if polka:
        # Polka dot plot
        polkadots = get_polka(boxArray)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_title('Particle '+str(int(particle)))
        ax.imshow(polkadots, vmin=100, cmap=plt.get_cmap('Greys_r'))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{path_plots}+polka_{particle}.png')

    if kymo:
        # Plot kymograph
        kymo = get_kymo(boxArray)
        fig, ax = plt.subplots(figsize=[8, 2])
        ax.imshow(kymo, cmap='gray')
        xtick = np.linspace(0, kymo.shape[1], 5)
        ax.set_xticks(xtick)
        ax.set_xticklabels((xtick*0.08).astype(int))
        ax.set_xlabel('Time [s]')
        ax.set_title(f'Particle {particle}')
        ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{path_plots}+kymo_{particle}.png')

    if animate:
        arr = []
        framesList = df_particle.t.to_list()
        fig, ax = plt.subplots(figsize=[10, 10])
        j = 0  # this is for going through the boxArray
        for frame in range(framesList[0], framesList[-1]):
            if frame in framesList:
                spot = boxArray[j]
                j += 1
            else:  # blinking
                spot = np.zeros(boxArray.shape[1:])

            ax.imshow(spot, cmap='gray')
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            arr.append(img)
            ax.cla()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        if codec == 'gif':
            imageio.mimsave(f'{path_plots}_polka_{particle}.gif', arr, fps=25)
        else:
            imageio.mimwrite(f'{path_plots}_polka_{particle}.mp4',
                             arr, fps=25, quality=8)
        plt.close()
