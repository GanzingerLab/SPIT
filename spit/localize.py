import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import distance as dist
from spit import tools
import naclib.stpol
import naclib.util


# %% naclib transform for color correction
def transform_locs(df_locs, df_coefs, channel, fig_size):
    """
    Load the calculated Zernike coefficients and calculate the polynomials from that

    Parameters
    ----------
    df_locs : Dataframe
        Dataframe containing localizations.
    df_coefs : Dataframe
        Dataframe containing coefficients for S/T polynomial decomposition.
    channel : Int
        Channel where the correction is applied.
    fig_size : Tuple
        Size of image.

    Returns
    -------
    df_locsCorrected : Dataframe
        Original localizations with correction applied.
    dataset : String
        Path indicating experiment used to extract coefficients.
    """

    # Read the path of the dataset that was used for creating the Zernike coefficients
    dataset = df_coefs.iloc[-1].value
    # Read parameters and get polynomials of channel 0
    df_S = df_coefs.loc[(df_coefs['type'] == 'S') & (df_coefs['channel'] == channel)]
    df_T = df_coefs.loc[(df_coefs['type'] == 'T') & (df_coefs['channel'] == channel)]
    a_S = {row['term']: row['value'] for _, row in df_S.iterrows()}
    a_T = {row['term']: row['value'] for _, row in df_T.iterrows()}
    j_max_S = int(df_S['term'].max())
    j_max_T = int(df_T['term'].max())
    stpol = naclib.stpol.STPolynomials(j_max_S=j_max_S, j_max_T=j_max_T)

    # scale to unit circle
    locsScaled, scale = naclib.util.loc_to_unitcircle(
        df_locs[['x', 'y']].values, fig_size)
    # get correction field and correct for weird float/string bug
    a_S = {int(key): float(value) for key, value in a_S.items()}
    a_T = {int(key): float(value) for key, value in a_T.items()}
    P = stpol.get_field(locsScaled, a_S, a_T)

    # corect spot locations, scale back
    locsScaledCorrected = locsScaled + P
    locsCorrected = naclib.util.unitcircle_to_loc(locsScaledCorrected, fig_size)
    df_locsCorrected = df_locs.copy()
    df_locsCorrected['x'] = locsCorrected[:, 0]
    df_locsCorrected['y'] = locsCorrected[:, 1]
    return df_locsCorrected, dataset

# %% Get nearest neighbors distribution


def get_nearest_neighbor(df_locs):
    """
    Compute nearest neighbor distance between particles, frame by frame.

    Args:
        locs : Dataframe containing localizations in (picasso output, in px).
    """
    def get_proximity(df_locs):
        ''' 
        Get distances in a single frame.
        '''
        distances = dist.cdist(df_locs[['x', 'y']],
                               df_locs[['x', 'y']], metric='euclidean')
        min_distances = np.where(distances > 0, distances, np.inf).min(axis=1)
        s_out = pd.Series(min_distances)
        return s_out

    nearest_neighbor = df_locs.groupby('t').apply(get_proximity).reset_index(drop=True)
    return nearest_neighbor

# %% Plot stats from localiztions


def plot_loc_stats(df_locs, path, combFit=False, centersInit=(1000, 2000)):
    '''
    Plot number of localizations and density, nearest neighbor distribution and photon histogram.

    Parameters
    ----------
    path : String
        Path to the location where the plot should be saved.
    locs : Dataframe
        Dataframe containing all localizations.
    nLocs : List
        Number of localization per frame.
    nearest_neighbor: Series
        Distance for each particle's nearest neighbor.
    combFit : Bool, optional
        To indicate if a Gaussian comb should be fitted or not. The default is False.
    centersInit : Tuple, optional
        Coordinates of center for Gaussian comb. The default is (1000,2000).

    Returns
    -------
    None.

    '''
    fig, ax = plt.subplots(3, figsize=(6, 7))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=-0.4, top=0.75)
    fig.suptitle(
        f'Data: {os.path.split(os.path.split(path)[0])[1]}/{os.path.split(path)[1]}')
    area = 682*2048*0.108*0.108
    nLocs = df_locs.pivot_table(columns=['t'], aggfunc='size')
    nLocs_um = nLocs/area

    # Localizations per frame
    # First y axis
    ax[0].plot(nLocs, '-', c='tomato', alpha=0)
    ax[0].set_ylabel('# Localizations', color='k')
    ax[0].yaxis.set_tick_params(color='k')
    ax[0].set_ylim(0, 800)
    ax[0].set_xlabel('Frame')
    # Second y axis
    ax2 = ax[0].twinx()
    ax2.plot(nLocs_um, 'o', c='tomato')
    ax2.set_ylabel('# Localizations/um2', color='k')
    ax2.set_ylim(0, 800/area)

    # Nearest neighbor distance
    nearest_neighbor = get_nearest_neighbor(df_locs)
    bins = np.arange(0, 45, 1)
    ax[1].hist(nearest_neighbor,
               bins=bins,
               fc='tomato',
               label='hist',)

    ax[1].axvline(nearest_neighbor.quantile(q=0.5), lw=2, ls='--',
                  label='_nolegend_', color='darkslategrey')
    ax[1].axvline(nearest_neighbor.quantile(q=0.1), lw=2, ls='--', label='_nolegend_')
    # ax[1].set_ylim(0,4000)
    ax[1].set_xlim(0, 50)
    # ax[1].set_yticks([0,1000,2000,3000,4000])
    ax[1].set_yticks(ax[1].get_yticks())
    ax[1].set_yticklabels((ax[1].get_yticks()/1000).astype(int))
    ax[1].text(25,
               ax[1].get_ylim()[1]*0.6,
               f'Of all NN distances: \n 50% > {nearest_neighbor.quantile(q=0.5):.1f} px \n 90%  > {nearest_neighbor.quantile(q=0.1):.1f} px',
               fontsize=10)
    ax[1].set_xlabel('Nearest neighbor distance [px]')
    ax[1].set_ylabel('# Localizations x 1000')

    # Photon histogram
    y, x = np.histogram(df_locs.intensity,
                        bins='auto',
                        )
    x = x[:-1]+(x[1]-x[0])/2

    # Scale x and y
    x_scaler = 1000
    y_scaler = 1000

    ax[2].bar(x/x_scaler,
              y/y_scaler,
              width=((x[1]-x[0])*1.1)/x_scaler,
              color='tomato',
              align='center',
              label='bar',
              alpha=0.9,
              )
    # ax.axvline(fits[2,0]/x_scaler,ls='--',label='vline')
    if combFit == True:
        N, x, y, yopt, fits = gaussianComb(df_locs, centersInit)
        # Choose colors
        colors = plt.get_cmap('magma')
        colors = [colors(i) for i in np.linspace(0, 1, 6)]
        # Plot of 1 level fit
        ax[2].plot(x/x_scaler, yopt/y_scaler, '-', c='k', lw=2)

        # Plot of N level fit
        for n in range(N):
            x0 = (n+1)*fits[2, N-1]
            sigma = np.sqrt(n+1)*fits[3, N-1]
            A = fits[4+n, N-1]
            yn = gauss_1D(x, x0, sigma, A)
            ax[2].plot(x/x_scaler, yn/y_scaler, c=colors[n])

    ax[2].set_xlabel('Photons x 1000')
    ax[2].set_xlim(0, np.percentile(df_locs.intensity, 99.9)/y_scaler)
    # ax[2].set_xticks(np.arange(0,np.percentile(locs.intensity,99.9),1))

    ax[2].set_ylabel('#Localizations x 1000')
    # ax[2].set_yticks(np.arange(0,8,2))

    fig.tight_layout()
    fig.savefig(path+'_localization.png', dpi=200)


# %% Gaussian comb fit on photon levels
def gaussianComb(df_locs, centersInit):
    """
    Fitting a combination of Gaussian.

    Parameters
    ----------
    locs : Dataframe
        Dataframe containing all localizations.
    centersInit : Tuple
        Coordinates of center.

    Returns
    -------
    N : Int
        Number of Gaussian functions that are summed up.
    x : Array
        Positions where the Gaussians were evaluated
    y : Array
        Observed values.
    yopt : Array
        Computed values using the Gaussians at x.
    fits : Array
        Obtained values for fit.

    """
    # Try fitting gaussian comb
    # Net gradient instead of photons yields levels with lower spread
    data = df_locs.intensity.values
    data = data[(data < np.percentile(data, 99.9))]  # Cut off highest 0.1% values
    data = data.reshape(-1, 1)
    centersInit = np.array(centersInit)

    N, x, y, yopt, fits = fit_levels(data, centersInit)
    # print('Fitting gaussian comb...')
    return N, x, y, yopt, fits


# %
def fit_Ncomb(x, y, centers_init, N):

    # Prepare start parameters for fit
    p0 = []
    p0.extend([centers_init[0]])
    p0.extend([p0[0]/4])
    p0.extend([max(y) for i in range(N)])

    # Define comb of N gaussians
    def gauss_comb(x, *p): return gauss_Ncomb(x, p, N)

    def redchi(x, y, p):
        yopt = gauss_comb(x, *p)  # Get reduced chi
        chi = np.divide((yopt-y)**2, y)
        chi = chi[np.isfinite(chi)]
        chi = np.sum(chi)/(len(chi)-len(p))
        return chi

    # Fit
    levels = np.zeros(4+6)
    try:
        # Try first cluster peak as start parameter
        popt0, pcov = curve_fit(gauss_comb, x, y, p0=p0)
        popt0 = np.absolute(popt0)

        # Try second cluster peak as start parameter
        p0[0] = centers_init[1]
        popt1, pcov = curve_fit(gauss_comb, x, y, p0=p0)
        popt1 = np.absolute(popt1)

        # Select better chi fit
        if redchi(x, y, popt0) < redchi(x, y, popt1):
            popt = popt0
        else:
            popt = popt1

        # Try half of first cluster peak
        p0[0] = centers_init[0]/2
        popt3, pcov = curve_fit(gauss_comb, x, y, p0=p0)
        popt3 = np.absolute(popt3)

        # Select better chi fit
        if redchi(x, y, popt3) < redchi(x, y, popt):
            popt = popt3

    except:
        levels[0] = N
        levels[1] = np.nan  # set chi to nan
        return levels

    # Remove peaks outside of data range
    centers = np.array([(i+1)*popt[0] for i in range(N)])  # Location of ith peak
    popt[2:][centers >= max(x)] = 0  # Assign zeros to out of range

    # Set peaks with an amplitude lower than 1% than the peak amplitude to zero
    Acrit = 0.01*max(popt[2:])
    popt[2:][popt[2:] < Acrit] = 0

    # Now calculate fit reduced chi
    chi = redchi(x, y, popt)

    # Assign to end result
    levels[0] = N
    levels[1] = chi
    levels[2:2+len(popt)] = popt

    return levels


def fit_levels(data, centers_init):

    # Prepare data for fit
    y, x = np.histogram(data,
                        bins='auto',
                        )
    x = x[:-1]+(x[1]-x[0])/2

    # Fit comb of Gaussians for different number of levels
    fits = np.zeros((10, 6))
    for N in range(1, 7):
        fits[:, N-1] = fit_Ncomb(x, y, centers_init, N)

    # Decide on the appropriate number of levels based on following criteria
    levels = fits.copy()

    # 1. Set chi value of fits having two extrema within peak amplitudes to NaN
    for i in range(np.shape(levels)[1]):
        diff_sign = np.sign(np.diff(levels[4:, i]))
        diff_sign = diff_sign[diff_sign != 0]
        n_maxima = np.sum(np.absolute(np.diff(diff_sign)) == 2)
        if n_maxima > 2:
            levels[1, i] = np.nan

    # 2. Remove NaNs in chi, corresponding to not succesful fits
    levels = levels[:, np.isfinite(levels[1, :])]

    # 3. Search for k with minimum reduced chisquare, go back in ks and return value where first jump bigger than 10% occurs
    k = np.argmin(levels[1, :])
    chis = levels[1, :]
    if k > 0:
        for i in range(k, 0, -1):
            if np.absolute((chis[i]-chis[i-1])/chis[i-1]) > 0.1:
                k = i
                break

    # After decision was made prepare return
    N = int(levels[0, k])

    pN = levels[2:2+2+N, k]
    yopt = gauss_Ncomb(x, pN, N)

    N = np.sum(levels[4:, k] > 0)

    return N, x, y, yopt, fits


def gauss_1D(x, x0, sigma, A):
    '''        
    Simple 1D non-normalized Gaussian function: ``y=np.absolute(A)*np.exp(-(x-x0)**2/sigma**2)``
    '''
    y = np.absolute(A)*np.exp(-(x-x0)**2/sigma**2)

    return y


def gauss_Ncomb(x, p, N):
    '''
    Sum of N 1D Gaussian functions, see gauss_1D(x,x0,sigma,A).
    The nth Gaussian function with ``n in [0,N[`` is:

        - centered at multiples of first Gaussian center ``(n+1)*x0``
        - has a width of ``sqrt(n+1)*sigma`` assuming Poissonian broadening
        - but decoupled Amplitudes ``An``

    Args:
        x(np.array):  Values at which function is evaluated
        p(list):      ``[x0,sigma,A0,A1,...,AN]`` input parameters for sum of Gaussians (len=N+2)
        N(integer):   Number of Gaussian functions that are summed up
    Returns:
        np.array: Evaluation of function at ``x``, ``p``, ``N``
    '''
    y = 0
    for i in range(N):
        # Version with least degrees of freedom
        y += gauss_1D(x, (i+1)*p[0], np.sqrt(i+1)*p[1], p[2+i])

    return y
