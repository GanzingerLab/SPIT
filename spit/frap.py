import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import os
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.optimize import curve_fit
from scipy.special import iv
from scipy.stats import chisquare
import spit.tools as tools

def splitData(movie,channel):
    """
    Keep only relevant channel for FRAP analysis.

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    channel : Int
        Coordinate of the channel to keep.

    Returns
    -------
    Numpy array
        Images for relevant channel, at all time frames.

    """
    movieFRAP = np.array_split(movie[:,:,1:2047], 3, axis=2)
    return movieFRAP[int(channel)]


def edgeDetection(movie, sigma=30, low_threshold=2.5, high_threshold=3):
    """
    Automated registration of bleached spot
    Uses canny edge detection algorithm to detect perimeter of bleached spot and
    applies hough circle transform to find best circle (radius, xy-position) 
    on the perimeter.
    https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    sigma : Int, optional
        The width of the Gaussian. The default is 30.
    low_threshold : Float, optional
        Lower bound for hysteresis. The default is 2.5.
    high_threshold : Float, optional
        Upper bound for hysteresis. The default is 3.

    Returns
    -------
    List
        Coordinates of the center of the bleached spot.
    Float
        Radius of the bleached spot.

    """
    imgEdges = np.zeros(movie[1][0].shape)
    edges = canny(movie[1][0], sigma, low_threshold, high_threshold)
    imgEdges[:, :] = edges
    
    # hough circle transform, radius range estimate: 140 - 160px
    hough_radius = np.arange(140, 160)
    hough_res = hough_circle(imgEdges, hough_radius)
    
    # select optimal circle
    accums, cx, cy, radius = hough_circle_peaks(hough_res, hough_radius, total_num_peaks=1)
    return [int(cx), int(cy)], int(radius)


def cmask(movie,c_xy, radius):
    """
    Turns circle coordinates and radius into boolean array, creating a mask
    to divide the movie data into inside bleached spot and outside of bleached spot.

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    c_xy : List
        Coordinates of the center of the bleached spot.
    radius : Float
        Radius of the bleached spot. 

    Returns
    -------
    imgmask : Numpy array
        Boolean array indicating position of the bleached spot. 

    """
    cx, cy = c_xy
    nx, ny = movie[1][0].shape
    x, y = np.ogrid[-cy:nx-cy,-cx:ny-cx]
    imgmask = x*x + y*y <= radius*radius
    return imgmask


def soumpasis(t, td, a0, a1):
    """
    Fit Soumpasis recovery function to fluorescence intensity ratio of 
    outside and inside (accounting for overall photobleaching)
    Soumpasis, 1983 https://www.cell.com/biophysj/pdf/S0006-3495(83)84410-5.pdf
    
    """
    return a0 + a1*np.exp(-2*td/t) * (iv(0, (2*td/t)) + iv(1, (2*td / t)))


def getIntensity(movie,c_xy,radius):
    """
    Create list of area-averaged intensity values inside 
    and outside of bleached area'''

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    c_xy : List of integer array
        Detected pixel position of the bleached spot
    radius : Integer array
        Detected radius of the bleached spot in pixels.

    Returns
    -------
    Normalized area-averaged intensities outside, and inside
    of the bleached spot over time, and their ratio.
    """        
    avgPreInt = []
    avgPreExt = []
    avgFrapInt = []
    avgFrapExt = []
    imgmask = cmask(movie,c_xy,radius)
    for i in range(movie[0].shape[0]):
        avgPreInt.append(np.mean(movie[0][i][imgmask]))
        avgPreExt.append(np.mean(movie[0][i][~imgmask]))        
  
    for i in range(movie[1].shape[0]):
        avgFrapInt.append(np.mean(movie[1][i][imgmask]))
        avgFrapExt.append(np.mean(movie[1][i][~imgmask]))
                
    avgInt = np.array(avgPreInt+avgFrapInt)        
    avgExt = np.array(avgPreExt+avgFrapExt)
    
    # normalize values 
    avgIntNormalized = avgInt/avgInt[0]     
    avgExtNormalized = avgExt/avgExt[0]
    avgRatio = avgInt/avgExt
    avgRatioNormalized = avgRatio/avgRatio[0]

    return avgIntNormalized,avgExtNormalized,avgRatioNormalized
    

def recoveryFit(movie,avgRatioNormalized,radius):
    """
    Recovery function fit and chisquare test

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    avgRatioNormalized : Array
        Ratio inside/outside of normalized area-averaged intensities.
    radius : Float
        Detected radius of the bleached spot.

    Returns
    -------
    fitData : List
        Estimated intensities for all frame, to estimate recovery.
    diffCoef : Float
        Estimated diffusion coefficient.

    """
    recovery_time = np.arange(1,movie[1].shape[0]+1)
    recovery_intensity = avgRatioNormalized[movie[0].shape[0]:]
    fitData = []
    pars, cov = curve_fit(f=soumpasis, xdata=recovery_time, ydata=recovery_intensity,p0=[5,0.1,0.1])
    for frame in recovery_time:
        frame_intensity = soumpasis(frame,*pars)
        fitData.append(frame_intensity)
    chisq, p_val = chisquare(recovery_intensity, f_exp=fitData)

    # estimated diffusion coefficient, w(cm) and td(sec): w^2 / 4*td
    pars[0] = (pars[0] * 1)
    # w = radius[0]*108 / 1e+3
    w = radius*108 / 1e+3
    diffCoef = '{:.2f}'.format(w**2 / (4*pars[0]))
    
    # format estimated parameters and chisquare test to 2 decimals
    # td_f = '{:.2f}'.format(pars[0])
    # a0_f = '{:.2f}'.format(pars[1])
    # a1_f = '{:.2f}'.format(pars[2])
    # chisq_f = '{:.4f}'.format(chisq)
  
    return fitData, diffCoef

def plotIntensities(movie,avgIntNormalized,avgExtNormalized,avgRatioNormalized, filename):
    """
    Plot raw data.

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    avgIntNormalized : Array
        Normalized area-averaged intensities inside the bleached spot over time.
    avgExtNormalized : Array
        Normalized area-averaged intensities outside the bleached spot over time.
    avgRatioNormalized : Array
        Ratio inside/outside of normalized area-averaged intensities.
    filename : String
        Name of the file to save.

    Returns
    -------
    None.

    """
    time = list(range(-movie[0].shape[0]+1, movie[1].shape[0]+1))  
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Fluorescence intensity [a.u.]')
    ax.plot(time, avgExtNormalized, 'oy', alpha=0.1, label='Exterior',ms=6)
    ax.plot(time, avgIntNormalized,'or', alpha=0.1, label='Interior',ms=6)
    ax.plot(time, avgRatioNormalized,'ok', label='Interior divided by exterior',ms=6,mfc='None')
    ax.set_ylim(top=1.1)
    ax.legend()
    ax.set_title('Data:...'+filename[-19:])
    plt.savefig(os.path.splitext(filename)[0].replace('_PreFrap','FRAP_values_raw.png'))
    plt.savefig(os.path.splitext(filename)[0].replace('_PreFrap','FRAP_values_raw.pdf'))
    plt.close()

def plotRecovery(movie, c_xy, radius, avgRatioNormalized, fitData, diffCoef, filename):
    """
    Plots first frame, after 10s, after 100s for visual output. Also shows detected bleaching spot.
    Creates a second plot of the fluorescence intensity recovery.
    Displays the Soumpasis fit and diffusion coefficient and recovery percentage (mobile fraction).

    Parameters
    ----------
    movie : list of arrays (PreFRAP.raw and Pat01.raw)
        .raw files of a FRAP measurement with 1s intervals and 100s duration.
    c_xy : List of integer array
        Detected pixel position of the bleached spot
    radius : Integer array
        Detected radius of the bleached spot in pixels.
    avgRatioNormalized : Float array
        Normalized ratio of intensities outside, and inside of the bleached spot over time.
    recovery_fit : List of float values
        Values of Soumpasis fit to avgRatioNormalized
    diff_coef : String
        Calculated value of diffusion coefficient.

    Returns
    -------
    Creates plot and saves it in the data folder (.pdf and .png)

    """
    scalebar_length = 30
    plt.figure(figsize=(6,6))
    
    plt.subplot(2,4,1)
    plt.imshow(movie[0][0], cmap='gray',clim=(100,np.mean(movie[1][0,...])+3*np.std(movie[0][0,...])))
    plt.title('Pre-bleach', fontsize = 12)
    plt.axis('off')
    
    plt.subplot(2,4,2)
    plt.imshow(movie[1][0,...], cmap='gray',clim=(100,np.mean(movie[1][0,...])+3*np.std(movie[1][0,...])))
    plt.title('0 s', fontsize = 12)
    imgCircle = plt.Circle(c_xy, radius, color='r', fill=False)
    ax = plt.gca()
    ax.add_artist(imgCircle)  
    plt.axis('off')
    
    plt.subplot(2,4,3)
    plt.imshow(movie[1][10,...], cmap='gray',clim=(100,np.mean(movie[1][10,...])+3*np.std(movie[1][10,...])))
    plt.title('10 s', fontsize = 12)
    plt.axis('off')
    
    plt.subplot(2,4,4)
    plt.imshow(movie[1][99,...], cmap='gray',clim=(100,np.mean(movie[1][99,...])+3*np.std(movie[1][99,...])))
    plt.title('100 s', fontsize = 12)
    plt.axis('off')
    plt.gca()
    plt.gcf()
    scalebar = AnchoredSizeBar(plt.gca().transData,
                               scalebar_length/0.108,
                               label=str(scalebar_length)+' μm',
                               color='white',
                               frameon=False,
                               size_vertical=10,
                               bbox_to_anchor=(670,670), loc="lower right",
                               bbox_transform=plt.gca().transData,
                               sep=5,
                               )
    plt.gca().add_artist(scalebar)
    
    time = list(range(-movie[0].shape[0]+1, movie[1].shape[0]+1))  
    plt.subplot(2,4,(5,8))
    plt.plot(time, avgRatioNormalized,'ok', label='Data',ms=6,mfc='None')
    plt.plot(fitData, 'r', ls='--', label='Fit',linewidth=4)
    plt.xlabel('Time [s]', fontsize = 12)
    plt.ylabel('Fluorescence intensity [a.u.]', fontsize = 12)
    plt.ylim(0,1.05)
    plt.legend()
    recovery = '{:.1f}'.format(100*avgRatioNormalized[-1])
    
    plt.text(0.1, -1.7, f'Recovery of {recovery}% after {len(movie[1])} seconds',
            size=12, horizontalalignment = 'left', transform = ax.transAxes)
    plt.text(0.1, -2, f'Diffusion Coefficient: {diffCoef}μm$^2$/s',
            size=12, horizontalalignment = 'left', transform = ax.transAxes)
    # ax.text(0.3, 0.05, f'Chi-square: {chisq_f}, p-value: {p_val}',
    #                     size=12, horizontalalignment = 'left', transform = ax.transAxes)
    plt.suptitle('Data: '+os.path.split(filename)[1],fontsize=16)
    # radius_um = radius[0]*0.108
    # plt.suptitle('FRAP of DOPE-ATTO390 SLB, \n Bleached circle: {:.1f} um'.format(radius_um), fontsize=18);
    plt.tight_layout() #to prevent cutting off of the plot labels
    plt.savefig(os.path.splitext(filename)[0].replace('_PreFrap','FRAP.png'),dpi=200)
    plt.savefig(os.path.splitext(filename)[0].replace('_PreFrap','FRAP.pdf'),dpi=200)
    plt.close()
    
def bilayerAnalysis(filename,channel):
    """
    General function to run the whole analysis.

    Parameters
    ----------
    filename : String
        Name of the movie to load.
    channel : Int
        Coordinate of the channel to analyze. 

    Returns
    -------
    None.

    """
    pathAnalysis = os.path.join(os.path.dirname(filename),'plots')
    if not os.path.exists(pathAnalysis):
        os.mkdir(os.path.join(os.path.dirname(filename),'plots'))


    movie = []
    movie.append(tools.load_raw(filename)[0])
    movie.append(tools.load_raw(filename.replace('PreFrap','Pat01'))[0])
    movie[0] = splitData(movie[0],channel)
    movie[1] = splitData(movie[1],channel)
    c_xy,radius = edgeDetection(movie)
 
    
    (avgIntNormalized,avgExtNormalized,avgRatioNormalized) = getIntensity(movie,c_xy,radius)
    (fitData,diffCoef) = recoveryFit(movie,avgRatioNormalized,radius)

    dirName = os.path.split(os.path.split(os.path.split(filename)[0])[0])[1]
    filenameAnalysis = os.path.join(os.path.split(filename)[0],'plots',f'{dirName}_{os.path.split(filename)[1]}')

    
    plotIntensities(movie,avgIntNormalized,avgExtNormalized,avgRatioNormalized, filenameAnalysis)
    plotRecovery(movie,c_xy,radius,avgRatioNormalized,fitData,diffCoef, filenameAnalysis)
