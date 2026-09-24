# Libraries

# Standard
from tqdm import tqdm
from typing import Union
import numpy as np
import pandas as pd
import scipy
from scipy.special import erf
from sklearn.isotonic import IsotonicRegression
import math
import os
import emcee
import corner

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

# External
from module_data_path import cube_data_path, plot_data_path, fits_data_path, mask_data_path, catalog_data_path

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

def distance_parallax(data_input: Union[str, pd.DataFrame]) -> None:
    """Computes stellar distance estimators using a Bayesian exponentially decreasing space density prior.

    This function accepts either a file path to a CSV catalog or an in-memory
    pandas DataFrame. It validates and extracts necessary astrometric (parallax,
    parallax_error) and photometric extinction columns, computes the extinction
    uncertainty (a_g_error) if missing, and iteratively estimates stellar
    distances using the `Distance.distance.main_exp` routine. The resulting
    catalog includes mode, median, 5th and 95th percentiles, normalization
    factors, and spatial cloud membership flags, saving the final table to disk.

    Parameters
    ----------
    data_input : str or pd.DataFrame
        File path to a CSV file or an existing DataFrame containing Gaia
        astrometry and photometry. Required columns include:
        'source_id', 'l', 'b', 'parallax', 'parallax_error', 'a_g_val',
        'a_g_percentile_lower', 'a_g_percentile_upper', and 'on_cloud'.

    Returns
    -------
    None
        The function does not return an object; it writes the computed
        catalog directly to disk as 'distances.csv' inside the catalog
        directory returned by `catalog_data_path()`.

    Raises
    ------
    TypeError
        If `data_input` is neither a string representing a valid path
        nor a `pandas.DataFrame`.
    FileNotFoundError
        If `data_input` is a string path that does not exist on disk.
    KeyError
        If any of the required columns are missing from the input data.

    Example
    -------
    >>> from module_utils import distance_parallax
    >>> distance_parallax("../catalog/dr21_gaia_classified.csv")
    Calculating distances: 100%|██████████| 511/511 [00:35<00:00, 14.20it/s]
    Catalog of distances saved to: ../catalog/distances.csv (511 stars processed)

    Notes
    -----
    - Uses an exponentially decreasing space density prior as formulated in
      Bailer-Jones (2015) and Astraatmadja & Bailer-Jones (2016).
    - Rows with missing values (`NaN`) in 'parallax', 'parallax_error',
      or 'a_g_val' are automatically dropped prior to computation.
    - Numerical integration failures for individual stars are caught, logged,
      and skipped without interrupting the entire processing loop.
    - Requires `pandas`, `numpy`, `tqdm`, and the local `Distance` package.
    """
    
    from Distance.distance import main_exp

    # Check input type
    if isinstance(data_input, str):
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError("data_input must be a file path (str) or a pandas DataFrame")

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    cols = ['source_id', 'l', 'b', 'parallax', 'parallax_error', 
            'a_g_val', 'a_g_percentile_lower', 'a_g_percentile_upper', 'on_cloud']

    df = df[cols].dropna(subset=['parallax', 'parallax_error', 'a_g_val'])

    df['a_g_error'] = (df['a_g_percentile_upper'] - df['a_g_percentile_lower']) / 2.0

    distances = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Calculating distances"):
        w = np.float64(row.parallax)
        s = np.float64(row.parallax_error)
        
        try:
            r_5, r_mode, r_median, r_95, n = main_exp(w, s)
            distances.append({
                'source_id': row.source_id,
                'l': row.l,
                'b': row.b,
                'parallax': w,
                'error': s,
                'a_g_val': row.a_g_val,
                'a_g_error': row.a_g_error,
                'on_cloud': row.on_cloud,
                'r_mode_pc': r_mode,
                'r_median_pc': r_median,
                'r_5%': r_5,
                'r_95%': r_95,
                'n_points': n
            })
        except Exception as e:
            print(f"Error with star {row.source_id} (w={w}, s={s}): {e}")
            continue

    distances_df = pd.DataFrame(distances)
    output_path = os.path.join(catalog_data_path(), 'distances.csv')
    distances_df.to_csv(output_path, index=False)
    print(f"Catalog of distances saved to: {output_path} ({len(distances_df)} stars processed)")

    
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
    
    data_frame.columns = [c.lower() for c in data_frame.columns]
    
    output_file = os.path.join(cube_data_path(), f"{prefix}.csv")
    data_frame.to_csv(output_file, index=False)
    print(f"Archivo guardado exitosamente en: {output_file}")

def classify_and_filter_stars(
    catalog_df: pd.DataFrame,
    fits_path: str,
    mask_path: str,
    prefix_source: str = "dr21",
    prefix_emission: str = "12co",
    min_parallax: float = 0.25,
    min_ag_error: float = 0.05,
) -> pd.DataFrame:
    """Classifies Gaia stars as on-cloud or off-cloud and applies quality and distance cutoffs.

    This function projects Galactic coordinates (l, b) into image pixel space
    (x, y) using the WCS header of a reference FITS map. It evaluates spatial
    overlap with a 2D collapsed dendrogram mask to determine cloud membership
    ('on_cloud'). In accordance with the methodology of Yan et al. (2019), it
    computes the extinction uncertainty (a_g_error) and applies lower-bound
    thresholds to parallax and extinction uncertainty to discard background
    noise and pathological weights.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        Input catalog containing Gaia astrometric and photometric columns
        ('l', 'b', 'parallax', 'a_g_percentile_lower', 'a_g_percentile_upper').
    fits_path : str
        Directory path containing the reference FITS moment map
        (e.g., '{prefix_source}_{prefix_emission}_mom8.fits').
    mask_path : str
        Directory path containing the 3D Boolean mask array
        (e.g., '{prefix_source}_{prefix_emission}_masks_dropped.npy').
    prefix_source : str, optional
        Prefix identifying the target molecular cloud source, by default 'dr21'.
    prefix_emission : str, optional
        Prefix identifying the molecular line emission tracer, by default '12co'.
    min_parallax : float, optional
        Minimum parallax cutoff in milliarcseconds (mas) to discard distant
        background stars, by default 0.25 (corresponding to ~4000 pc).
    min_ag_error : float, optional
        Minimum uncertainty threshold in G-band extinction (mag) to prevent
        overweighting in subsequent isotonic regression, by default 0.05.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing stars located within the spatial bounds
        of the FITS map that satisfy the quality cutoffs, including added
        'a_g_error' and 'on_cloud' columns.

    Raises
    ------
    FileNotFoundError
        If the specified FITS file or mask `.npy` file does not exist.
    KeyError
        If the input DataFrame lacks required astrometric or photometric columns.

    Example
    -------
    >>> import pandas as pd
    >>> raw_df = pd.DataFrame(
    ...     {
    ...         "l": [81.57, 80.00],
    ...         "b": [0.76, 5.00],
    ...         "parallax": [2.90, 0.10],
    ...         "a_g_percentile_lower": [0.17, 0.05],
    ...         "a_g_percentile_upper": [0.74, 0.10],
    ...     }
    ... )
    >>> clean_stars = classify_and_filter_stars(
    ...     catalog_df=raw_df,
    ...     fits_path="../fitsfiles",
    ...     mask_path="../mask",
    ...     prefix_source="dr21",
    ...     prefix_emission="12co",
    ... )
    # Returns a DataFrame with inside-boundary stars and 'on_cloud' classification.

    Notes
    -----
    - The 3D cluster mask is collapsed into a 2D sky footprint via `np.any(masks_3d, axis=0)`.
    - Coordinates outside the range [0, num_x) and [0, num_y) are strictly discarded.
    - Requires `astropy`, `numpy`, and `pandas` installed.
    """
    # 1. Load WCS header and 2D footprint
    fits_file = os.path.join(fits_path, f'{prefix_source}_{prefix_emission}_mom8.fits')
    header = fits.getheader(fits_file)
    wcs_proj = WCS(header)
    
    mask_file = os.path.join(mask_path, f'{prefix_source}_{prefix_emission}_masks_dropped.npy')
    masks_3d = np.load(mask_file)
    mask_2d = np.any(masks_3d, axis=0)
    num_y, num_x = mask_2d.shape

    # 2. Standardize column names
    df = catalog_df.copy()
    df.columns = [col.lower() for col in df.columns]

    # 3. Compute A_G error (Yan et al. 2019, Eq. 1)
    df['a_g_error'] = (df['a_g_percentile_upper'] - df['a_g_percentile_lower']) / 2.0

    # 4. Project Galactic coordinates (l, b) to pixel coordinates (x, y)
    pix_x, pix_y = wcs_proj.all_world2pix(df['l'], df['b'], 0)

    # 5. Filter stars inside the spatial bounds of the map
    inside_map = (pix_x >= 0) & (pix_x < num_x) & (pix_y >= 0) & (pix_y < num_y)
    df_filtered = df[inside_map].copy()
    valid_x = pix_x[inside_map].astype(int)
    valid_y = pix_y[inside_map].astype(int)

    # 6. Assign on-cloud boolean flag
    df_filtered['on_cloud'] = mask_2d[valid_y, valid_x]

    # 7. Apply parallax and extinction uncertainty cutoffs
    valid_parallax = df_filtered['parallax'] >= min_parallax
    valid_extinction = df_filtered['a_g_error'] >= min_ag_error
    clean_sample = df_filtered[valid_parallax & valid_extinction].copy()

    return clean_sample

def baseline_subtraction(df: pd.DataFrame, plots_path: str, prefix_source: str = 'dr'):
    """
    Performs baseline fitting and subtraction on the extinction data using isotonic regression.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the extinction data with columns 'r_mode_pc', 'a_g_val', 'a_g_error' and 'on_cloud'.
    plots_path : str
        Path to save the baseline fitting plot.
    prefix_source : str, optional
        Prefix for the source name used in the plot title, by default 'dr21'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the net extinction values after baseline subtraction.
    object
        The fitted isotonic regression model.
    """
    
    # Separate on-cloud and off-cloud stars
    df_off = df[df['on_cloud'] == False].copy()
    
    # Calculate the weights
    weights = 1 / (df_off['a_g_error'] ** 2+1e-6)  # Avoid division by zero
    
    # Adjust isotonic regression to handle weights
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip') # out_of_bounds='clip' to handle extrapolation
    ir.fit(df_off['r_mode_pc'], df_off['a_g_val'], sample_weight=weights)
    
    # Subtract the fitted baseline from the original data
    df['a_g_expected'] = ir.predict(df['r_mode_pc'])
    df['a_g_net'] = df['a_g_val'] - df['a_g_expected']
    
    # Preliminar plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_off['r_mode_pc'], df_off['a_g_val'], s=10, alpha=0.3, label='Off-cloud stars', color='blue') # Background stars
    df_on = df[df['on_cloud'] == True]
    plt.scatter(df_on['r_mode_pc'], df_on['a_g_val'], s=15, alpha=0.6, label='On-cloud stars', color='red') # On-cloud stars
    
    # Plot the fitted isotonic regression line
    x_plot = np.linspace(df['r_mode_pc'].min(), df['r_mode_pc'].max(), 500)
    y_plot = ir.predict(x_plot)
    plt.plot(x_plot, y_plot, color='green', linewidth=2, label='Fitted Baseline') # Isotonic Regression
    
    plt.xlabel('Distance (pc)')
    plt.ylabel('A_G (mag)')
    plt.title(f'Baseline Fitting for {prefix_source}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(os.path.join(plots_path, f'{prefix_source}_baseline_fit.pdf'), bbox_inches='tight')
    plt.close()
    
    return df, ir

def log_prior(theta):
    """
    Defines a uniform prior for the parameters of the model, the physical boundaries for the free parameters:
    Parameters
    ----------
    theta = [D, mu1, sigma1, mu2, sigma2 = theta]
    
    D: Distance to the cloud (pc) (500 - 3000 pc)
    mu1: Foreground mean extinction (close to 0)
    sigma1, sigma 2: Data spread 
    mu2: Background mean extinction (garter than mu1)
    """

    D, mu1, sigma1, mu2, sigma2 = theta
    
    if (500 < D < 3000) and (-0.5 < mu1 < 0.5) and (0 < sigma1 < 1) and (mu1 < mu2 < 5) and (0 < sigma2 < 2):
        return 0.0
    return -np.inf

def log_likelihood(theta, r, r_error, ag, a_g_error):
    """
    Defines the log probability function for the model, given the data and the parameters.
    
    Parameters
    ----------
    theta: [D, mu1, sigma1, mu2, sigma2 = theta]
    r: Distance to the star (pc)
    r_error: Distance error (pc)
    ag: Extinction value (mag)
    a_g_error: Extinction error (mag)
    
    Returns
    -------
    log_likelihood: float
        The log probability value for the given parameters and data.
    """
    
    D, mu1, sigma1, mu2, sigma2 = theta
    
    # Probabilityy that the star is in the foreground
    f_i = 0.5 * (1 + erf((r - D) / (np.sqrt(2) * r_error)))
    
    # Total variances 
    var1 = sigma1**2 + a_g_error**2
    var2 = sigma2**2 + a_g_error**2
    
    # Gaussian probabilities for foreground and background
    fore = (1 - f_i) * (1 / np.sqrt(2 * np.pi * var1)) * np.exp(-0.5 * ((ag - mu1)**2 / var1))
    back = f_i * (1 / np.sqrt(2 * np.pi * var2)) * np.exp(-0.5 * ((ag - mu2)**2 / var2))
    
    total_prob = fore + back
    
    return np.sum(np.log(np.clip(total_prob, 1e-10, None)))

def log_posterior(theta, r, r_error, ag, a_g_error):
    """
    Defines the log posterior probability function for the model, given the data and the parameters.
    Bayes Theorem: log_posterior = log_prior + log_likelihood
    """
    
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, r, r_error, ag, a_g_error)
    
    return lp + ll

def run_mcmc(df: pd.DataFrame, plots_path: str, prefix_source: str = 'dr21', nwalkers: int = 50, nsteps: int = 1000, nburn: int = 200) -> np.ndarray:
    """
    Runs emcee on the on-cloud stars to find the distance to the molecular cloud.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the extinction data
    plots_path : str
        Path to save the MCMC corner plot.
    prefix_source : str, optional
        Prefix for the source name used in the plot title, by default 'dr21'.
    
    Returns
    -------
    D_median : float
        Most probable distance to the cloud
    error_plus : float
        Upper error bound
    error_minus : float
        Lower error bound
    """
    
    # Extract data from DataFrame
    r = df['r_mode_pc'].values
    r_error = (df['r_95%'].values - df['r_5%'].values) / 3.29 # esto no estoy segura de los valores porque decia  r_error pero que según los catalogos de Bailer-Jones (Gaia DR2) es r_95% - r_5% / 3.29
    ag = df['a_g_net'].values
    ag_error = df['a_g_error'].values
    
    df_on = df[df['on_cloud'] == True].copy()
    
    # Extract variables and approximate distance errors using Bailer-Jones (2015) method
    r = df_on['r_mode_pc'].values
    r_error = (df_on['r_95%'].values - df_on['r_5%'].values) / 3.29
    ag = df_on['a_g_net'].values
    ag_error = df_on['a_g_error'].values
    
    # Initial guess: [D, mu1, sigma1, mu2, sigma2]
    initial_guess = np.array([1400, 0.0, 0.1, 1.0, 0.2])
    
    nwalkers = 32
    ndim = len(initial_guess)
    np.random.seed(42)  # For reproducibility
    pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)
    
    print("Running MCMC...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[r, r_error, ag, ag_error])
    
    # Run chain
    sampler.run_mcmc(pos, 1000, progress=True)
    
    # Discard burn-in
    samples = sampler.get_chain(discard=200, flat=True)
    
    # Corner plot
    fig = corner.corner(samples, labels=[r"$D$ (pc)", r"$\mu_1$", r"$\sigma_1$", r"$\mu_2$", r"$\sigma_2$"], quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.savefig(os.path.join(plots_path, f'{prefix_source}_mcmc_corner.pdf'))
    plt.close()
    
    # Extract median percentiles
    D_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    D_median = D_mcmc[1]
    error_minus = D_median - D_mcmc[0]
    error_plus = D_mcmc[2] - D_median
    
    print(f"\nFinal distance to the cloud: {D_median:.2f} pc (+{error_plus:.2f}, -{error_minus:.2f})")
    
    return D_median, error_plus, error_minus