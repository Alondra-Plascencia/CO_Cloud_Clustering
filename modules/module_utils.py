# Libraries

# Standard
import numpy as np
import pandas as pd
import scipy
import math
import os

import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from matplotlib.ticker import LogFormatter
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle


# Astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

# Data Cube
from spectral_cube import SpectralCube

def Gauss_area(H,FWHM):
        resultado = H * FWHM / (0.3989423 * 2.3548200)
        return resultado

def rms(image):
    """""
    Returns root mean square error (rms) of an image (2d array).

    Parameters:
        image(np.darray):The 2d array used to calculate the rms.

    Returns:
        rms(float):The rms of the image.   
    """
    
    rms = np.sqrt((np.mean(image**2.0)))
    return rms

# Image smoothing using a Gaussian Kernel
def smooth(image, kern_px=1):
    """
    Returns a smoothed image in the first HDU of the input file.

    Parameters:
        image(2d np.darray): image to be smoothed.
        kern_px: FWHM of kernel in pixels.

    Returns:
        f1(2d np.darray): Smoothed 2d array.
    """

    f1=scipy.ndimage.gaussian_filter(image, kern_px/(2*math.sqrt(2*math.log(2))))
    return f1

def cube_mom8(cube_path,velmin,velmax,output_path,write_fits=False):
    """
    Returns the moment 8 (max intensity) image of a data cube.

    Parameters:
        cube(SpectralCube): data cube from which the moment is computed.
        velmin(float): min value of the spectral range (in km/s)
        velmax(float): max value of the spectral range (in km/s)

    Returns:
        moment(SpectralCube 2d): Moment 8 image (2d) of the data cube.
    """

    cube = SpectralCube.read(cube_path)
    cube.allow_huge_operations = True
    cube_slab = cube.spectral_slab(velmin *u.km / u.s, velmax *u.km / u.s)
    moment = cube_slab.max(axis = 0)

    if write_fits:
        moment.write(output_path, overwrite=True)

    return moment

def cube_mom0(cube_path,velmin,velmax,output_path,write_fits=False):
    """
    Returns the moment 0 (mean intensity) image of a data cube.

    Parameters:
        cube(SpectralCube): data cube from which the moment is computed.
        velmin(float): min value of the spectral range (in km/s)
        velmax(float): max value of the spectral range (in km/s)

    Returns:
        moment(SpectralCube 2d): Moment 0 image (2d) of the data cube.
    """

    cube = SpectralCube.read(cube_path)
    cube_slab = cube.spectral_slab(velmin*u.km/u.s, velmax*u.km/u.s)
    moment = cube_slab.with_spectral_unit(u.km/u.s).moment(order=0)

    if write_fits:
        moment.write(output_path, overwrite=True)
        
    return moment

def cube_smoothing(data_path, mask_path, output_path, prefix_source, prefix_emission, efficiency=1.0, kernel_px=1, apply_mask=False, write_fits=False):
    hdu = fits.open(data_path)[0]
    if apply_mask:
        mask = np.load(os.path.join(mask_path,'mask_edges.npy'))

    for v in range(0,hdu.data.shape[0]):
        if apply_mask:
            hdu.data[v,:,:] = np.where(mask, hdu.data[v,:,:], np.nan)
        hdu.data[v,:,:] = smooth(hdu.data[v,:,:]/efficiency,kern_px=kernel_px)

    print('Smoothing done for:', data_path)
    
    if write_fits:
        print('Writing new cube in following path:', output_path)
        hdu.writeto(os.path.join(output_path,prefix_source+'_'+prefix_emission+'_smoothed.fits'),
                overwrite = True)

def plot_mom8(path, output_path, prefix_source, prefix_emission, gamma=1.0, vmin=0.0, vmax=25.0):
    hdu = fits.open(path)[0]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection = WCS(hdu.header))

    im = ax.imshow(hdu.data, cmap = 'RdBu_r',
                   norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))

    ### Axis parameters ###
    lat = ax.coords['glat']
    lat.set_axislabel('Galactic Latitude', size = 12, alpha = 1.0)
    lat.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lat.set_ticklabel(size = 12, exclude_overlapping=True)
    lat.display_minor_ticks(True)

    lon = ax.coords['glon']
    lon.set_axislabel('Galactic Longitude', size = 12, alpha = 1.0)
    lon.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lon.set_ticklabel(size = 12, exclude_overlapping=True)
    lon.display_minor_ticks(True)

    ### Annotations ###
    ax.annotate(prefix_source + ', ' + prefix_emission + ' Peak Temperature', xy = (5,5), xytext = (5, 5), color='black',
            fontsize = 8, bbox = dict(boxstyle = "round", fc = "w", alpha = 0.0))

    ### Colorbar ###
    cbar = plt.colorbar(im, pad=.01)
    cbar.set_label(r'$T_{\rm MB}^{\rm \ peak}$ [K]', labelpad = 4, y = 0.5, rotation=90, size = 12)
    #cbar.ax.tick_params(labelsize=14)
    #cbar.ax.locator_params(nbins=6)

    plt.savefig(os.path.join(output_path, prefix_source + '_' + prefix_emission + '_mom8.pdf'),
                bbox_inches = 'tight')
    plt.close()

def plot_mom8_not_smoothed(path, mask_path, output_path, prefix_source, prefix_emission, gamma=1.0, vmin=0.0, vmax=25.0, use_mask=False):
    hdu = fits.open(path)[0]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection = WCS(hdu.header))

    im = ax.imshow(hdu.data, cmap = 'RdBu_r',
                   norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))

    ### Axis parameters ###
    lat = ax.coords['glat']
    lat.set_axislabel('Galactic Latitude', size = 12, alpha = 1.0)
    lat.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lat.set_ticklabel(size = 12, exclude_overlapping=True)
    lat.display_minor_ticks(True)

    lon = ax.coords['glon']
    lon.set_axislabel('Galactic Longitude', size = 12, alpha = 1.0)
    lon.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lon.set_ticklabel(size = 12, exclude_overlapping=True)
    lon.display_minor_ticks(True)

    ### Annotations ###
    ax.annotate(prefix_source + ', ' + prefix_emission + ' Peak Temperature', xy = (5,5), xytext = (5, 5), color='black',
            fontsize = 8, bbox = dict(boxstyle = "round", fc = "w", alpha = 0.0))

    ### Colorbar ###
    cbar = plt.colorbar(im, pad=.01)
    cbar.set_label(r'$T_{\rm MB}^{\rm \ peak}$ [K]', labelpad = 4, y = 0.5, rotation=90, size = 12)
    #cbar.ax.tick_params(labelsize=14)
    #cbar.ax.locator_params(nbins=6)

    if use_mask:
        mask = np.load(os.path.join(mask_path,'mask_edges.npy'))
        ax.imshow(mask, alpha=0.3)

    plt.savefig(os.path.join(output_path, prefix_source + '_' + prefix_emission + '_mom8_not_smoothed.pdf'),
                bbox_inches = 'tight')
    plt.close()    

def plot_mom8_comparison(mom_path, plots_path, catalog_path, prefix_source, prefix_emission, dropped = True, gamma=1.0, vmin=0.0, vmax=25.0):
    hdu = fits.open(mom_path)[0]

    if dropped:
        catalog = pd.read_csv(os.path.join(catalog_path, f"{prefix_source}_catalog_{prefix_emission}_dropped.csv"))
        prefix_out = 'dropped'
        print('Saving figure after dropped indexes')
    else:
        catalog = pd.read_csv(os.path.join(catalog_path, f"{prefix_source}_catalog_{prefix_emission}.csv"))
        prefix_out = 'not_dropped'
        print('Saving figure before dropped indexes')

    fig = plt.figure()

    ax = fig.add_subplot(111, projection = WCS(hdu.header))

    im = ax.imshow(hdu.data, cmap = 'RdBu_r',
                   norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))

    ### Axis parameters ###
    lat = ax.coords['glat']
    lat.set_axislabel('Galactic Latitude', size = 12, alpha = 1.0)
    lat.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lat.set_ticklabel(size = 12, exclude_overlapping=True)
    lat.display_minor_ticks(True)

    lon = ax.coords['glon']
    lon.set_axislabel('Galactic Longitude', size = 12, alpha = 1.0)
    lon.set_ticks(width = 1, spacing = 0.1 * u.deg)
    lon.set_ticklabel(size = 12, exclude_overlapping=True)
    lon.display_minor_ticks(True)

    ### Annotations ###
    ax.annotate(prefix_source + ', ' + prefix_emission + ' Peak Temperature', xy = (5,5), xytext = (5, 5), color='black',
            fontsize = 8, bbox = dict(boxstyle = "round", fc = "w", alpha = 0.0))

    ### Colorbar ###
    cbar = plt.colorbar(im, pad=.01)
    cbar.set_label(r'$T_{\rm MB}^{\rm \ peak}$ [K]', labelpad = 4, y = 0.5, rotation=90, size = 12)
    #cbar.ax.tick_params(labelsize=14)
    #cbar.ax.locator_params(nbins=6)

    ## Ellipses

    for i in range(0,len(catalog)):
        el_x = catalog['x_cen'][i]
        el_y = catalog['y_cen'][i]
        el_w = catalog['major_sigma'][i]
        el_h = catalog['minor_sigma'][i]
        el_a = catalog['position_angle'][i]
        el = Ellipse(xy = (el_x, el_y), width = el_w, height = el_h, angle = el_a, color = 'red',
                 linewidth  = 0.1, linestyle = '-', fill = False)
        ax.scatter(catalog['x_cen'][i],
               catalog['y_cen'][i],
               marker = '+',
               c = 'orange',
               s = 0.1)
        ax.add_patch(el)
    
    

    texts = [ax.text(catalog['x_cen'][i]+2, catalog['y_cen'][i]+0,
                    str(catalog.index[i]), ha='center', va='center', size=5) for i in range(len(catalog))]

    plt.savefig(os.path.join(plots_path, f"{prefix_source}_{prefix_emission}_catalog_{prefix_out}.pdf"),
                bbox_inches = 'tight')
    plt.close()

def mask_edges(data_path, mask_path, width=10, height=10, angle=0, x0=0, y0=0):
    hdu = fits.open(data_path)[0]
    print(f'data cube has the following dimensions: {hdu.data.shape}')

    # Parameters
    angle = np.radians(angle)

    # Generate coordinate grid
    x, y = np.meshgrid(np.arange(hdu.data.shape[2]), np.arange(hdu.data.shape[1]))

    # Rotate coordinates around (x0, y0)
    x_rot = (x - x0) * np.cos(angle) - (y - y0) * np.sin(angle)
    y_rot = (x - x0) * np.sin(angle) + (y - y0) * np.cos(angle)

    # Define the rotated rectangle mask
    mask = (np.abs(x_rot) < width) & (np.abs(y_rot) < height)

    np.save(os.path.join(mask_path,'mask_edges.npy'), mask)
    print('Saved mask for data edges')

def distance_parallax(data_frame):
    """
    Computes stellar distances from a CSV file containing parallax data using
    a Bayesian method with an exponentially decreasing space density prior.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing at least two columns:
        'parallax' and 'parallax_error'.

    Behavior
    --------
    For each star in the dataset:
    - If the parallax or its error is NaN, stores -1.
    - Otherwise, applies the main_exp method to estimate:
        - Mode
        - Median
        - 5th and 95th percentiles
        - Posterior normalization factor

    Output
    ------
    Saves a new CSV file '../data/distancias.csv' with the original parallax 
    values and the computed distances in parsecs.
    """

    from Distance.distance import main_exp

    df = pd.read_csv(data_frame, usecols=['SOURCE_ID', 'l', 'b','parallax', 'parallax_error'])

    # List of parallaxes and errors
    designation = df['SOURCE_ID']
    l = df['l']
    b = df['b']
    parallax = df['parallax']
    errors = df['parallax_error']

    distancias = []

    for d, l, b, w, s in zip(designation, l, b, parallax, errors):
        if np.isnan(w) or np.isnan(s):
            continue
        else:
            try:
                r_5, r_mode, r_median, r_95, n = main_exp(np.float64(w),np.float64(s))
                distancias.append({
                    'source_id': designation,
                    'l': l,
                    'b': b,
                    'parallax': w,
                    'error': s,
                    'r_mode_pc': r_mode,
                    'r_median_pc': r_median,
                    'r_5%': r_5,
                    'r_95%': r_95,
                    'n_points': n
                })
            except Exception as e:
                print(f"Error con w={w}, s={s}: {e}")
                continue
    # Convert to DataFrame and save to CSV
    distancias_df = pd.DataFrame(distancias)
    distancias_df.to_csv('../catalog/distancias.csv', index=False)
    
    
def vot_to_csv(votable_path,prefix):
    """
    Converts a VOTable (.vot or .xml) file into a CSV file.

    This function uses `astropy.io.votable` to parse a VOTable file and converts its first data table 
    into a pandas DataFrame. It then saves the DataFrame as a CSV file in the `../data/` directory, 
    using the specified prefix as the base name.

    Parameters
    ----------
    votable_path : str
        Path to the VOTable (.vot or .xml) file containing the data table.
    
    prefix : str
        Prefix to use for naming the output CSV file.

    Returns
    -------
    None
        The function writes the CSV file directly to disk under the `../data/` directory.

    Example
    -------
    >>> vot_to_csv("stars.vot", "stars_data")
    # This will generate the file ../data/stars_data.csv

    Notes
    -----
    - Make sure the VOTable contains at least one valid table.
    - The CSV file will be overwritten if a file with the same name already exists.
    - Requires the `astropy` and `pandas` packages to be installed.
    """
    from astropy.io.votable import parse
    votable = parse(votable_path)
    data_frame = pd.DataFrame(votable.get_first_table().array.data)
    data_frame.to_csv('../data/' + prefix + '.csv', index=False)