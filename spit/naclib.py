import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spit.tools as tools
import spit.linking as link
import naclib.stpol
import naclib.util

# python -m spit localize "Y:\05 K2 alignment\20220520 calibration slides\Run00005" -g0 1500 -g1 1500 -g2 1500 -f lq

path = r'Y:\05 K2 alignment\20220520 calibration slides\Run00005'

# %%
'''
Track across channels for each frame, instead of tracking across frames for each channel.
'''

runName = tools.getExperimentName(path)
paths = glob.glob(path+'/*locs.csv')

# Load localizations
df_locs0, info = tools.load_locs(paths[0])
df_locs1, info = tools.load_locs(paths[1])
df_locs2, info = tools.load_locs(paths[2])


df_locs_all = pd.concat([df_locs0, df_locs1, df_locs2], keys=['0', '1', '2']).reset_index(
    level=0).rename(columns={'t': 'grid', 'level_0': 'frame'})


def apply_tracking(df_locs_all):
    df_linked = link.link_locs_trackpy(df_locs_all, search=20, memory=0)
    return df_linked


df_tracks = df_locs_all.groupby('grid').apply(
    apply_tracking).reset_index(level=0, drop=True)
df_tracks = df_tracks.loc[df_tracks.loc_count > 2]
df_tracks = tools.get_unique_trackIDs(df_tracks, group='grid')

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax = df_tracks.plot.scatter(x='x', y='y', c='t', colormap='brg',
                            marker='.', colorbar=False, ax=ax, alpha=0.3)
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
ax.set_aspect('equal')
# plt.savefig(os.path.join(path,'plots',runName+'_uncorrected.png'),dpi=200,bbox_inches='tight',transparent=False)
# plt.close()

points = [df_tracks.loc[df_tracks.t == i, ['x', 'y']].values for i in [0, 1, 2]]

'''
Plot uncorrected beads, bead per bead
'''


# %%
'''
Rescale to unit circle.
'''
fig_size = [
    682, 682]  # Original image (channel) size, needed for rescaling to unit circle.
locs0_scaled, scale = naclib.util.loc_to_unitcircle(points[0], fig_size)
locs1_scaled, scale = naclib.util.loc_to_unitcircle(points[1], fig_size)
locs2_scaled, scale = naclib.util.loc_to_unitcircle(points[2], fig_size)

'''
Get distortion grid.
'''
D0 = np.zeros(locs0_scaled.shape)
D0[:, 0] = locs1_scaled[:, 0] - locs0_scaled[:, 0]
D0[:, 1] = locs1_scaled[:, 1] - locs0_scaled[:, 1]

D2 = np.zeros(locs2_scaled.shape)
D2[:, 0] = locs1_scaled[:, 0] - locs2_scaled[:, 0]
D2[:, 1] = locs1_scaled[:, 1] - locs2_scaled[:, 1]

# channel0error = np.zeros([15,15])
# channel2error = np.zeros([15,15])


j_max_S = 20  # Max term for S polynomials.
j_max_T = 20  # Max term for T polynomials.

# Get ST polynomial decomposition coefficients.
stpol = naclib.stpol.STPolynomials(j_max_S=j_max_S, j_max_T=j_max_T)
a_S0, a_T0 = stpol.get_decomposition(locs0_scaled, D0)
a_S2, a_T2 = stpol.get_decomposition(locs2_scaled, D2)


# Save resulting coefficients.
df_coefs = pd.DataFrame(columns=['channel', 'type', 'term', 'value'])
for term, value in a_S0.items():
    df_coefs = df_coefs.append(
        {'channel': '0', 'type': 'S', 'term': term, 'value': value}, ignore_index=True)
for term, value in a_T0.items():
    df_coefs = df_coefs.append(
        {'channel': '0', 'type': 'T', 'term': term, 'value': value}, ignore_index=True)
for term, value in a_S2.items():
    df_coefs = df_coefs.append(
        {'channel': '2', 'type': 'S', 'term': term, 'value': value}, ignore_index=True)
for term, value in a_T2.items():
    df_coefs = df_coefs.append(
        {'channel': '2', 'type': 'T', 'term': term, 'value': value}, ignore_index=True)

df_coefs = df_coefs.append({'value': path}, ignore_index=True)
df_coefs.to_csv(os.path.join(path, 'plots', runName+'_naclib_coefficients.csv'))
# Save in GitHub folder
df_coefs.to_csv(
    r'C:\Users\niederauer\Documents\GitHub\SPI\spit\paramfiles\naclib_coefficients.csv')


'''
Prepare plots.
'''
theta = np.linspace(0, 2*np.pi, 100)

# Distortion field
magnitude0 = np.hypot(D0[:, 0], D0[:, 1]) * scale
magnitude2 = np.hypot(D2[:, 0], D2[:, 1]) * scale
magnitude_max = np.max([np.max(magnitude0), np.max(magnitude2)])

# Correction field
P0 = stpol.get_field(locs0_scaled, a_S0, a_T0)
magnitude_P0 = np.hypot(P0[:, 0], P0[:, 1]) * scale
P2 = stpol.get_field(locs2_scaled, a_S2, a_T2)
magnitude_P2 = np.hypot(P2[:, 0], P2[:, 1]) * scale

# Residual distortion field
locs0_scaled_corrected = locs0_scaled + P0
D_res0 = np.zeros(locs0_scaled_corrected.shape)
D_res0[:, 0] = locs1_scaled[:, 0] - locs0_scaled_corrected[:, 0]
D_res0[:, 1] = locs1_scaled[:, 1] - locs0_scaled_corrected[:, 1]
magnitude_res0 = np.hypot(D_res0[:, 0], D_res0[:, 1]) * scale

locs2_scaled_corrected = locs2_scaled + P2
D_res2 = np.zeros(locs2_scaled_corrected.shape)
D_res2[:, 0] = locs1_scaled[:, 0] - locs2_scaled_corrected[:, 0]
D_res2[:, 1] = locs1_scaled[:, 1] - locs2_scaled_corrected[:, 1]
magnitude_res2 = np.hypot(D_res2[:, 0], D_res2[:, 1]) * scale


'''
Plot
'''
# Plot distortion fields.
fig = plt.figure(figsize=(16, 16))
ax0 = fig.add_subplot(321)
q0 = plt.quiver(locs0_scaled[:, 0], locs0_scaled[:, 1], D0[:, 0], D0[:, 1],
                magnitude0, pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)

plt.plot(np.cos(theta), np.sin(theta), c='k')
clb = plt.colorbar(q0)
plt.title('Distortion field: Channel 0')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))


ax1 = fig.add_subplot(322)
q2 = plt.quiver(locs0_scaled[:, 0], locs0_scaled[:, 1], D2[:, 0], D2[:, 1],
                magnitude2, pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)
plt.plot(np.cos(theta), np.sin(theta), c='k')
clb = plt.colorbar(q2)
plt.title('Distortion field: Channel 2')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))

# Plot correction fields.
ax0 = fig.add_subplot(323)
q0 = plt.quiver(locs0_scaled[:, 0], locs0_scaled[:, 1], P0[:, 0], P0[:, 1], magnitude_P0,
                pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)
clb = plt.colorbar(q0)
plt.plot(np.cos(theta), np.sin(theta), c='k')
plt.title('Correction field: Channel 0')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))

ax1 = fig.add_subplot(324)
q2 = plt.quiver(locs0_scaled[:, 0], locs0_scaled[:, 1], P2[:, 0], P2[:, 1], magnitude_P2,
                pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)
clb = plt.colorbar(q2)
plt.plot(np.cos(theta), np.sin(theta), c='k')
plt.title('Correction field: Channel 2')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))


# Plot residual distortion fields.
ax0 = fig.add_subplot(325)
q0 = plt.quiver(locs0_scaled[:, 0], locs0_scaled[:, 1], D_res0[:, 0], D_res0[:, 1], magnitude_res0,
                pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)
clb = plt.colorbar(q0)
plt.plot(np.cos(theta), np.sin(theta), c='k')
plt.title('Residual distortion field: Channel 0')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))

ax1 = fig.add_subplot(326)
q2 = plt.quiver(locs2_scaled[:, 0], locs2_scaled[:, 1], D_res2[:, 0], D_res2[:, 1], magnitude_res2,
                pivot='mid', clim=(0, 1.2 * magnitude_max), scale=0.07)
clb = plt.colorbar(q2)
plt.plot(np.cos(theta), np.sin(theta), c='k')
plt.title('Residual distortion field: Channel 2')
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('distortion (px)')
plt.axis('equal')
plt.xlim((-1, 1))
plt.ylim((-1, 1))

plt.savefig(os.path.join(path, 'plots', runName+'_fields'+str(j_max_S) +
            str(j_max_T)+'.png'), dpi=200, bbox_inches='tight', transparent=False)

print(str(np.mean(magnitude_res0)), str(np.mean(magnitude_res2)))

# %%

# Correct spot locations, scale back from unit circle to original scale, and save.
locs0_corrected = locs0_scaled + P0
locs2_corrected = locs2_scaled + P2

locs0_corrected = naclib.util.unitcircle_to_loc(locs0_scaled_corrected, fig_size)
locs2_corrected = naclib.util.unitcircle_to_loc(locs2_scaled_corrected, fig_size)


# Show localization improvement
# Original grid.
fig = plt.figure(figsize=(16, 7))
fig.add_subplot(121)
plt.title('Original spots - center of image')
color = ['b', 'g', 'r']
for k in [0, 1, 2]:
    plt.scatter(points[k][:, 0], points[k][:, 1], c=color[k], marker='.', s=2)
    plt.xlabel('x (px)')
    plt.ylabel('y (px)')
    plt.axis('equal')
    plt.title('Uncorrected beads')
    # plt.legend(['Channel 0','Channel 1','Channel 2'])

# Corrected grid.
fig.add_subplot(122)
plt.scatter(locs0_corrected[:, 0], locs0_corrected[:, 1], c='b', marker='.', s=2)
plt.scatter(points[1][:, 0], points[1][:, 1], c='g', marker='.', s=2)
plt.scatter(locs2_corrected[:, 0], locs2_corrected[:, 1], c='r', marker='.', s=2)
plt.xlabel('x (px)')
plt.ylabel('y (px)')
plt.axis('equal')
plt.title('Corrected beads')
# plt.legend(['Channel 0','Channel 1','Channel 2'])
plt.savefig(os.path.join(path, 'plots', runName+'_corrected.png'),
            dpi=200, bbox_inches='tight', transparent=False)
plt.show()
plt.close()
