import os
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm
from spit import tools
from spit import linking as link
from spit.analysis.functions import dimertracks as dimertracks

# %% Dimer data
# Scrape coloc data


def create_df_dimer(path, coloc_track_length=10, **kwargs):
    dict_dimers = {'path': [], 'tracks_ch0': [], 'tracks_ch0 [0:100 frames]': [],
                   'tracks_ch1': [], 'tracks_ch1 [0:100 frames]': [],
                   'ligand_conc': [],
                   'dimers': [], 'dimers [0:100 frames]': []}

    dict_dimers['path'] = glob(path + '/**/**_colocs_nm_trackpy.csv', recursive=True)

    for path in tqdm(dict_dimers['path']):
        df_colocs = pd.read_csv(path)
        df_locs0 = pd.read_csv(path[:-21]+'ch0_locs.csv')
        df_locs1 = pd.read_csv(path[:-21]+'ch1_locs.csv')
        concentration = float(tools.find_between(path, 'Ligand ', ' M'))

        df_colocsC = df_colocs.loc[df_colocs.loc_count > coloc_track_length]

        dict_dimers['tracks_ch0'].append(df_locs0.pivot_table(
            columns=['t'], aggfunc='size').mean())
        dict_dimers['tracks_ch0 [0:100 frames]'].append(df_locs0.pivot_table(
            columns=['t'], aggfunc='size')[0:100].mean())
        dict_dimers['tracks_ch1'].append(df_locs1.pivot_table(
            columns=['t'], aggfunc='size').mean())
        dict_dimers['tracks_ch1 [0:100 frames]'].append(df_locs1.pivot_table(
            columns=['t'], aggfunc='size')[0:100].mean())
        dict_dimers['ligand_conc'].append(concentration)
        dict_dimers['dimers'].append(df_colocsC.pivot_table(
            columns=['t'], aggfunc='size').mean())
        dict_dimers['dimers [0:100 frames]'].append(df_colocsC.pivot_table(
            columns=['t'], aggfunc='size')[0:100].mean())
    df_dimers = pd.DataFrame(dict_dimers)

    for key, value in kwargs.items():
        df_dimers[key] = value

    # downstream calculations
    M_to_um3 = 6.022*1E2
    df_dimers['ligand_conc_um3'] = df_dimers['ligand_conc']*M_to_um3

    channel_ratio_a = df_dimers['tracks_ch0'].div(df_dimers['tracks_ch1'])
    channel_ratio_b = df_dimers['tracks_ch1'].div(df_dimers['tracks_ch0'])
    df_dimers['channel_ratio'] = channel_ratio_a.combine(
        channel_ratio_b, max, fill_value=0)

    df_dimers['dimers_ratiocorrected'] = df_dimers['dimers'] * \
        df_dimers['channel_ratio']

    # 2x because of single- and dual-color dimers
    df_dimers['tracks_combined_dimercorrected'] = df_dimers['tracks_ch0'] + \
        df_dimers['tracks_ch1']+2*df_dimers['dimers_ratiocorrected']

    # 2x 2x because single- and dual-color dimers, and because a dimer consists of two molecules
    df_dimers['fraction'] = 2*(2*df_dimers['dimers_ratiocorrected']) / \
        df_dimers['tracks_combined_dimercorrected']

    return df_dimers


def func_dimer_fraction(var_indep, K_X, K_B, correction_factor=1):
    '''
    var_indep: independent variables [L, R_tot]
    K_X: dissociation constant of lateral receptor crosslinking (RL+R <=> RLR = C)
    K_B: dissociation constant of ligand binding from bulk (R+L <=> RL)
    L: bulk ligand concentration
    C = RLR: receptor complex
    R: free receptor
    RL: ligand-bound receptor
    R_tot: x-total number-x surface density of receptors R_tot =  2R0 = 2C + R + RL
    f: fraction of bound receptors f = C/R0
    '''
    L = var_indep[0]
    R0 = 0.5*var_indep[1]*correction_factor
    aux = (K_X*(2*L+K_B)**2)/(2*R0*4*L*K_B)
    func = correction_factor*(1-(((2*aux+aux**2)**0.5)-aux))
    return func


def get_dimer_fit_params(df_dimers, correction=False):
    """
    """
    M_to_um3 = 6.022*1E2
    FOV_to_um2 = 1/5425

    density = df_dimers.groupby(by='ligand_conc').mean()[
        'tracks_combined_dimercorrected'].values * FOV_to_um2
    density_std = df_dimers.groupby(by='ligand_conc').std()[
        'tracks_combined_dimercorrected'].values * FOV_to_um2

    dimers_fraction = df_dimers.groupby(by='ligand_conc').mean()[
        'fraction'].values

    dimers_fraction_corr = df_dimers.groupby(by='ligand_conc').mean()[
        'fraction'].values - df_dimers.loc[df_dimers['ligand_conc'] == 0, 'fraction'].mean()

    dimers_fraction_std = df_dimers.groupby(by='ligand_conc').std()[
        'fraction'].values

    # convert: M/L to #/um^3
    ligand_conc = df_dimers.groupby(by='ligand_conc_um3').mean().index

    # Fit (not considering the 0nM ligand value, seems to mess with the fitting)
    var_indep = [ligand_conc[1:], density[1:]]

    if correction:
        p0 = [1e-9, 1e-9, 1]
        popt, pcov = curve_fit(func_dimer_fraction,
                               var_indep, dimers_fraction_corr[1:],
                               p0=p0,
                               bounds=((0, 0, 0),
                                       (np.inf, np.inf, 1)),
                               maxfev=10000)
    else:
        p0 = [1e-9, 1e-9]
        popt, pcov = curve_fit(func_dimer_fraction,
                               var_indep, dimers_fraction_corr[1:],
                               p0=p0,
                               bounds=((0, 0),
                                       (np.inf, np.inf)),
                               maxfev=10000)

    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def add_Kd_df_dimer(df_dimers, popt, perr):
    df_dimers['K_X'] = popt[0]
    df_dimers['K_B'] = popt[1]
    df_dimers['K_X_error'] = perr[0]
    df_dimers['K_B_error'] = perr[1]
    if popt.size > 2:
        df_dimers['correction_factor'] = popt[2]
        df_dimers['correction_factor_error'] = perr[2]
    return df_dimers

# %% Plotting


def plot_dimer_raw(df_dimers, path=None, ax=None):
    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[4, 4])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('Dimer numbers (raw) \n'+os.path.split(path)[1])
        save_plot = True

    ax.errorbar(df_dimers.groupby(by='ligand_concentration').mean().index*1E6, df_dimers.groupby(by='ligand_concentration').mean()[
        'dimers'], yerr=df_dimers.groupby(by='ligand_concentration').std()[
        'dimers'], ls='',
        marker='d', c='lightsalmon', ecolor='lightsalmon', capsize=4, alpha=1)
    ax.scatter(df_dimers['ligand_concentration']*1E6,
               df_dimers.dimers, marker='.', color='black', zorder=100)
    ax.set_ylabel('Raw number of dimers per FOV')
    ax.set_xlabel('Ligand concentration [uM]')
    ax.set_ylim(bottom=0)
    ax.set_xscale('log')
    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_dimers_raw.pdf', dpi=200)


def plot_dimer_fit(df_dimers, color='black', annotation=True, path=None, ax=None):
    # Plot
    save_plot = False
    if ax is None:
        # plot data
        f = plt.figure(figsize=[4, 4])
        f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
        f.clear()
        ax = f.add_subplot(111)
        ax.set_title('K_D fit \n'+os.path.split(path)[1])
        save_plot = True

    M_to_um3 = 6.022*1E2
    FOV_to_um2 = 1/5425

    popt = [df_dimers['K_X'].mean(), df_dimers['K_B'].mean()]
    perr = [df_dimers['K_X_error'].mean(), df_dimers['K_B_error'].mean()]

    if 'correction_factor' in df_dimers.columns:
        popt.append(df_dimers.correction_factor.mean())
        perr.append(df_dimers.correction_factor_error.mean())

    # Evaluate fit for plotting
    xlim_log_min = np.log10(min(
        df_dimers.loc[df_dimers['ligand_conc_um3'] != 0, 'ligand_conc_um3']))-1
    xlim_log_max = np.log10(df_dimers['ligand_conc_um3'].max())+1
    xfit = np.logspace(xlim_log_min, xlim_log_max, 100)
    yfit = func_dimer_fraction(
        [xfit, df_dimers.tracks_combined_dimercorrected.mean() * FOV_to_um2], *popt)
    fit_peak = xfit[np.argmax(yfit)]

    # Actual plot
    ax.plot(xfit, yfit, c=color)

    ax.errorbar(df_dimers.groupby(by='ligand_conc_um3').mean().index, df_dimers.groupby(by='ligand_conc').mean()[
        'fraction'], yerr=df_dimers.groupby(by='ligand_conc').std()[
        'fraction'], ls='',
        marker='o', c=color, ecolor=color, capsize=4, alpha=1)

    ax.set_ylabel(r'% of molecules dimerised')
    ax.set_xlabel('Ligand concentration [M]')
    ax.set_xscale('log')
    range_x = 1E7
    ax.set_xlim(fit_peak/range_x, fit_peak*range_x)
    # ax.set_ylim(bottom=0)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    xticks = np.logspace(xlim_log_min, xlim_log_max, 4)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: '{0:.0e}'.format(x/M_to_um3)))

    ax.set_yticks(np.arange(0, 1.1*np.max(yfit), step=np.max(yfit)/4))
    # yticklabels: round maximum value to nearest tens place
    ax.set_yticklabels(ticker.FormatStrFormatter('%.0f').format_ticks(
        np.linspace(0, 10*np.ceil(10*np.max(yfit)), 5)))

    s = f'K_X = {popt[0]:.1e} +/- {perr[0]:.1e} [1/um2] \nK_B = {popt[1]:.1e} +/- {perr[1]:.1E} [1/um3] =  {popt[1]/M_to_um3:.1e} M'
    if len(popt) > 2:
        s = s+f'\ncorr. factor = {popt[2]: .1e} +/- {perr[2]: .1e}'
    print(s)
    print(f'Fit peak: {fit_peak/M_to_um3:.1e} M = K_B/2')
    if annotation:
        ax.text(0.05, 0.95, s, ha='left', va='top', color='k', size='x-small', transform=ax.transAxes,
                bbox=dict(facecolor='ghostwhite', alpha=0.5, edgecolor='lightgrey'))

    if save_plot:
        plt.tight_layout()
        plt.savefig(path + '_fit.pdf', dpi=200)
