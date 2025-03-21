# Libraries

# Standard
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from matplotlib.ticker import LogFormatter
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle

## Astropy
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.wcs import WCS

# Astrodendro
from astrodendro import Dendrogram
from astrodendro import pp_catalog, ppv_catalog
from astrodendro.analysis import PPStatistic
from astrodendro.analysis import PPVStatistic

def make_catalog(cube_path, dendrogram, prefix_source, prefix_emission, catalog_path, write_catalog=False):
    
    print(f"Making catalog for {prefix_source}")

    hdu = fits.open(cube_path)[0]

    # Metadata, necessary for analysis
    metadata = {}
    metadata['data_unit'] = u.Kelvin
    metadata['spatial_scale'] =  abs(hdu.header['CDELT1']) * u.degree # pixel grid
    metadata['beam_major'] =  46.0 * u.arcsec # FWHM
    metadata['beam_minor'] =  46.0 * u.arcsec # FWHM
    metadata['wavelength'] = 0.002730793549 * u.m # c18o wavelength

    cat = ppv_catalog(dendrogram.leaves, metadata, verbose=False) # only leaf structures
    cat_pd = cat.to_pandas()
    cat_pd['major_sigma'] = cat_pd['major_sigma'] / abs(hdu.header['CDELT1'])
    cat_pd['minor_sigma'] = cat_pd['minor_sigma'] / abs(hdu.header['CDELT1'])
    cat_pd['major_sigma'] = cat_pd['major_sigma'] * 2.0
    cat_pd['minor_sigma'] = cat_pd['minor_sigma'] * 2.0
    cat_pd['area_exact'] = cat_pd['area_exact'] / (abs(hdu.header['CDELT1'])**2.0)

    if write_catalog:
        cat_pd.to_csv(os.path.join(catalog_path,f"{prefix_source}_catalog_{prefix_emission}.csv"),
                  header = True,
                  index = False)
        print(f"Catalog saved for {prefix_source}")

    return cat_pd

def make_mask(dendrogram, prefix_source, prefix_emission, mask_path, write_mask=False):

    print(f"Making Mask for {prefix_source}")
    mask_list = []
    
    for i in range(0,len(dendrogram.leaves)):
        mask = dendrogram.leaves[i].get_mask()
    
        count_max = 0
        for j in range(0,mask.shape[0]):
            count = np.count_nonzero(mask[j,:,:])
    
            if count > count_max: # we use the maximum area in the sky
                count_max = count
                index_max = j
            
        #print(index_max,count_max)
    
        mask_t = mask[index_max,:,:] #2d array of True/False values
        mask_list.append(mask_t)

    if write_mask:
        mask_array = np.array(mask_list)
        np.save(os.path.join(mask_path,f'{prefix_source}_{prefix_emission}_masks.npy'), mask_array)
        print(f"Mask saved for {prefix_source}")

def make_clustering(cube_path, catalog_path, mask_path, T_min, T_delta, n_vox, prefix_source, prefix_emission, catalog=False, mask=False):

    print(f"Beggining dendrogram for {prefix_source}")

    hdu = fits.open(cube_path)[0]

    d = Dendrogram.compute(hdu.data,
                           min_value = T_min,
                           min_delta = T_delta,
                           min_npix = n_vox,
                           verbose = True)
    
    # leaves count
    count = 0
    for leaf in d.leaves:
        count += 1
    print('# of leaves (clusters) identified = ', count)

    if catalog:
        make_catalog(cube_path=cube_path, dendrogram=d, prefix_source=prefix_source, prefix_emission=prefix_emission, catalog_path=catalog_path, write_catalog=True)

    if mask:
        make_mask(dendrogram=d, prefix_source=prefix_source, prefix_emission=prefix_emission, mask_path=mask_path, write_mask=True)
    #return d

def make_plot_clusters(mom_path, catalog_path, mask_path, plots_path, prefix_source, prefix_emission, gamma=1.0, vmin=0.0, vmax=45.0):
    
    hdu = fits.open(mom_path)[0]
    catalog = pd.read_csv(os.path.join(catalog_path, f"{prefix_source}_catalog_{prefix_emission}.csv"))
    mask = np.load(os.path.join(mask_path, f'{prefix_source}_{prefix_emission}_masks.npy'))

    fig = plt.figure()

    ax = fig.add_subplot(111, projection = WCS(hdu.header))

    color = 'RdBu_r'
    #color = 'viridis'
    im = ax.imshow(hdu.data,
                   cmap=color,
                   norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax))

    ax.scatter(catalog['x_cen'],
                catalog['y_cen'],
                marker = '+',
                c = 'orange',
                s = 0.1)

    #ax.invert_yaxis()

    ### Axis parameters ###
    lat = ax.coords['glat']
    lon = ax.coords['glon']

    lat.set_axislabel('Galactic Latitude', size=12, alpha=1.0)
    lat.set_ticks(width=1, spacing=0.25*u.deg)
    lat.set_ticklabel(size=12, exclude_overlapping=True)
    lat.display_minor_ticks(True)

    lon.set_axislabel('Galactic Longitude', size=12, alpha=1.0)
    lon.set_ticks(width=1, spacing=0.25*u.deg)
    lon.set_ticklabel(size=12, exclude_overlapping=True)
    lon.display_minor_ticks(True)

    

    ### Colorbar ###
    position=fig.add_axes([0.91,0.13,0.02,0.75])
    ## the parameters are the specified position you set
    cbar = plt.colorbar(im, pad=.01, cax = position)
    cbar.set_label(r'$T_{\rm MB}^{\rm \ Peak}$', labelpad = 0.0, y = 0.5, rotation = 90, size = 12)

    ### Annotations ###
    ax.annotate(f'{prefix_source}, {prefix_emission}, MaxIntensity',
                xy = (3,3),
                xytext = (3, 3),
                color='black',
                fontsize = 12,
                bbox = dict(boxstyle = "round", fc = "w", alpha = 0.0))


    texts = [ax.text(catalog['x_cen'][i]+2, catalog['y_cen'][i]+1,
             str(catalog.index[i]),
             ha='center',
             va='center',
             size=8,
             alpha=1.0) for i in range(len(catalog))]
    
    for m in mask:
        mask_p = m.astype(int) #2d array to plot
        ax.contour(mask_p, levels=[0, 0.5, 1], fill=False, linewidth=0.05, alpha=0.5, color='red')
    
    plt.savefig(os.path.join(plots_path, f"{prefix_source}_{prefix_emission}_max_astrodendro_contours.pdf"),
                bbox_inches = 'tight')
    print(f'Max intensity with cluster contours for {prefix_source}')
    plt.close()

def catalog_mask_drop(catalog_path, mask_path, drop_list, prefix_source, prefix_emission):

    catalog = pd.read_csv(os.path.join(catalog_path, f"{prefix_source}_catalog_{prefix_emission}.csv"))
    mask = np.load(os.path.join(mask_path, f'{prefix_source}_{prefix_emission}_masks.npy'))
    print('len=',len(mask))

    catalog = catalog.drop(drop_list)
    catalog = catalog.reset_index()
    catalog.to_csv(os.path.join(catalog_path,f"{prefix_source}_catalog_{prefix_emission}_dropped.csv"),
                  header = True,
                  index = False)
    print(f'Dropped catalog saved for {prefix_source}')
    
    drop_index_sorted = sorted(drop_list, reverse=True)
    mask_dropped = np.delete(mask,drop_index_sorted,axis=0)
    np.save(os.path.join(mask_path,f'{prefix_source}_{prefix_emission}_masks_dropped.npy'), mask_dropped)
    print(f'Dropped mask saved for {prefix_source}')