from spit import plot_diffusion
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from spit import linking as link
from spit import tools
from spit.analysis.functions import swift_optimization

# plt.style.use(r'C:\Users\niederauer\Dropbox\Work\!DNA tracking 2021\Figures\paper.mplstyle')
plt.style.use('default')

path_dir = r'D:\test\Run00001\tau_bleach'
# path_dir = r'Y:\04 DNA paint tracking\Data\20220221 jurkat 26 differential labeling\Run00015'
paths = tools.scrape_data(path_dir, matchstring='_swift.csv')

df_tracks, df_stats = swift_optimization.concat_data(paths)
df_statsF = link.filter_df(df_stats, filter_length=10, filter_D=0.01)

path_plot = os.path.join(path_dir, 'tau bleach')
title = 'bleaching parameter screen'

swift_optimization.plot_param_sweep_stats(
    df_tracks, df_stats, df_statsF, title, path_plot, dt=0.04)
