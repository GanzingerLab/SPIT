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


# %% Create dimer dataframe only from cotracks with individual channel locs
path = r'Y:\11 FKBP interaction\Data\20220614 dimer antibody parallel'
# path = r'Y:\11 FKBP interaction\Data\20220601 Dimer ligand parallel'
# path = r'Y:\11 FKBP interaction\Data\20220602 Dimer ligand parallel'
# path = r'Y:\11 FKBP interaction\Data\20220608 Dimer ligand parallel'

df_dimers = dimerKD.create_df_dimer(path, ligand='Antibody')
# df_dimers = dimerKD.create_df_dimer(path, ligand='AP20187')

# Additional information
# df_dimers.loc[df_dimers['path'].str.contains('120'), 'incubation_time [min]'] = 120
# df_dimers.loc[df_dimers['path'].str.contains('5'), 'incubation_time [min]'] = 5

# Save dataframe
outputPath = os.path.join(path, os.path.basename(path))
df_dimers.to_csv(outputPath+'_df_dimers_all.csv', index=False)

# %% Plot dimer fit

df_dimers = pd.read_csv(outputPath+'_df_dimers_all.csv')
# because of mistake in antibody weight (2.54x less concentrated than thought)
df_dimers.ligand_conc = df_dimers.ligand_conc/2.54
df_dimers.ligand_conc_um3 = df_dimers.ligand_conc_um3/2.54
# # # Filter
df_dimers = df_dimers.loc[~df_dimers.path.str.contains('replicate')]
# df_dimers = df_dimers.loc[df_dimers.path.str.contains('replicate')]

popt, perr = dimerKD.get_dimer_fit_params(df_dimers, correction=True)

df_dimers_Kd = dimerKD.add_Kd_df_dimer(df_dimers, popt, perr)

# df_dimers_Kd.to_csv(outputPath+'_df_dimers_KD_replicate_corr.csv', index=False)
df_dimers_Kd.to_csv(outputPath+'_df_dimers_KD_sample_corr.csv', index=False)

# dimerKD.plot_dimer_fit(df_dimers_Kd, path=outputPath+'_replicate_corr')
dimerKD.plot_dimer_fit(df_dimers_Kd, path=outputPath+'_sample_corr')


# %%
# Plot all dimer tracks and save in batch analysis folder
pathOutput = tools.getOutputpath(path, 'batch analysis', keepFilename=False)
dimertracks.plot_all_dimer_tracks(df_dimers, path=pathOutput)
