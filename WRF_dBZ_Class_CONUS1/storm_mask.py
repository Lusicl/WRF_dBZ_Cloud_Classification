import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wrf
from netCDF4 import Dataset, date2num
from scipy.ndimage import label, generate_binary_structure
import shapefile
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from conv_strat_latlon import assign_radius, haversine, conv_strat_latlon
from matplotlib.lines import Line2D
import time
import datetime

start_time = time.time()

# '/home/krasmussen/scratch/DATA/PRECIP/June2017_meiyu/TerrainMod_1km/'
in_folder = '/bell-scratch/acole/june09_WRF/enkf_3km/'
# /home/krasmussen/scratch/RESEARCH/STORM_MODE_TRACKING/Rung_version/fig/TerrainMod/'
out_fig_folder = '/bell-scratch/rpanasawatwong/storm-mode/202006/fig/'
# '/home/krasmussen/scratch/RESEARCH/STORM_MODE_TRACKING/Rung_version/outfile/TerrainMod/'
out_fil_folder = '/bell-scratch/rpanasawatwong/storm-mode/202006/outfile/'

# Number of pixels needed to be a WCC - update WRF resolution as needed depending on input
resolution_WRF = 3  # km
WCC_pixels_required = 800/resolution_WRF**2
BSR_pixels_required = 30000/resolution_WRF**2
dbz_threshold = 30
height_threshold = 8

# Create filter for seeking neighboring data
s = generate_binary_structure(2, 2)

# Shapefile for SA Topography
# sf2 = shapefile.Reader('Research/andes_contour/andes_contour.shp')


# Loop through all output files and run the separation and then identify storm types by using the label tool
# This loop then plots the storm types, but does not save them off as text files or in any other way for further
# analysis. This may be a step that's needed eventually.

# Loop through all output files and run the separation and then identify storm types by using the label tool
# This loop then plots the storm types, but does not save them off as text files or in any other way for further
# analysis. This may be a step that's needed eventually.


def storm_mask(filename, resolution_WRF=1, dbz_threshold=30, height_threshold=8):
    '''
    Input: folder name to read WRF output 
    output: storm mode masks and time 
    '''

    WCC_pixels_required = 800/resolution_WRF**2
    BSR_pixels_required = 30000/resolution_WRF**2

    # Read in an hourly WRF output file
    nc = Dataset(filename, 'r')
    dbz = wrf.getvar(nc, 'REFL_10CM')
    lon = wrf.getvar(nc, 'lon').data
    lat = wrf.getvar(nc, 'lat').data
    z = wrf.getvar(nc, "z", units="m")
    vtime = wrf.extract_times(nc, None).astype('M8[ms]').astype('O')
    nc.close()

    # Interpolate reflectivity data to height - for DCC analysis
    dbz_10km = wrf.interplevel(dbz.data, z, height_threshold*1000)

    # This interpolation below is required for use in C/S diagnosis
    dbz_2km = wrf.interplevel(dbz.data, z, 2000)

    # C/S Classification
    cs, __, __ = conv_strat_latlon(dbz_2km, lat, lon)
    cs_original = cs
    cs[np.where(dbz_2km.data < 0)] = -1
    cs[np.where(np.isnan(dbz_2km))] = -1

    # Find DCCs ---------------------------------------------------------------------------------------------
    dbz_10km_boolean = np.zeros([lon.shape[0], lon.shape[1]])
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            if dbz_10km[i, j].data >= dbz_threshold and cs[i, j] in [1, 2, 3, 4, 5]:
                dbz_10km_boolean[i, j] = 1
            else:
                dbz_10km_boolean[i, j] = 0

    # Find all neighboring objects with ref >= dbz_threshold dBZ at the 10 km level
    labeled_array_DCC, num_features_DCC = label(
        dbz_10km_boolean, structure=s)
    DCC_array = np.zeros([lon.shape[0], lon.shape[1]])
    # Determine if identified objects meet area criteria for DCCs and create array of DCCs for plotting
    for i in range(num_features_DCC):
        j = i+1
        # print('DCC detected')
        # Do something to map DCCs
        for x in range(lon.shape[0]):
            for y in range(lon.shape[1]):
                if labeled_array_DCC[x, y] == j:
                    DCC_array[x, y] = 1

    label_id = []
    # Determine if dBZ is >= dbz_threshold dBZ and create Boolean array for those values -----------------------
    dbz_comp = np.zeros([lon.shape[0], lon.shape[1]])
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            dbz_comp[i, j] = np.max(dbz[:, i, j].data)

    dbz_WCC_boolean = np.zeros([lon.shape[0], lon.shape[1]])
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            if dbz_comp[i, j] >= dbz_threshold and cs[i, j] in [1, 2, 3, 4, 5]:
                dbz_WCC_boolean[i, j] = 1
            else:
                dbz_WCC_boolean[i, j] = 0

    # Find all contiguous elements
    labeled_array_WCC, num_features_WCC = label(
        dbz_WCC_boolean[:, :], structure=s)

    WCC_array = np.zeros([lon.shape[0], lon.shape[1]])
    # Determine if identified objects meet area criteria for WCCs and create array of WCCs for plotting
    for i in range(num_features_WCC):
        j = i+1
        num_pixels = len(np.where(labeled_array_WCC == j)[0])
        if num_pixels > WCC_pixels_required:
            label_id.append(j)
            # print('WCC detected')
            # Do something to map WCCs
            for x in range(lon.shape[0]):
                for y in range(lon.shape[1]):
                    if labeled_array_WCC[x, y] == j:
                        WCC_array[x, y] = 1

    # DWCC Detection - need to check for overlaps in DCC and WCC arrays -------------------------------------
    DWCC_array = np.zeros([lon.shape[0], lon.shape[1]])
    labeled_array_DCC_removal, num_features_DCC_removal = label(
        DCC_array, structure=s)
    labeled_array_WCC_removal, num_features_WCC_removal = label(
        WCC_array, structure=s)

    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            if DCC_array[i, j] == 1 and WCC_array[i, j] == 1:
                DWCC_array[i, j] = 1
                # print('DWCC Found')

                # remove those cells from DCC and WCC arrays, since these are exclusive categories
                # DCC_array[i, j] = 0
                # WCC_array[i, j] = 0

                # New Idea: relabel the WCC_array and DCC_array and reassign values based on this i,j location

                DCC_match_id = labeled_array_DCC_removal[i, j]
                for a in range(lon.shape[0]):
                    for b in range(lon.shape[1]):
                        if labeled_array_DCC_removal[a, b] == DCC_match_id:
                            DWCC_array[a, b] = 1
                            DCC_array[a, b] = 0

                WCC_match_id = labeled_array_WCC_removal[i, j]
                for a in range(lon.shape[0]):
                    for b in range(lon.shape[1]):
                        if labeled_array_WCC_removal[a, b] == WCC_match_id:
                            DWCC_array[a, b] = 1
                            WCC_array[a, b] = 0

                # for x in label_id:
                    # WCC_array[np.where(labeled_array_WCC == x)] = 0

    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            if DCC_array[i, j] == 1 and WCC_array[i, j] == 1:
                DWCC_array[i, j] = 1
                # print('DWCC Found')

                # remove those cells from DCC and WCC arrays, since these are exclusive categories
                DCC_array[i, j] = 0
                WCC_array[i, j] = 0

    # BSR Identification -------------------------------------------------------------------------------------
    BSR_boolean = np.zeros([lon.shape[0], lon.shape[1]])
    for i in range(lon.shape[0]):
        for j in range(lon.shape[1]):
            if cs[i, j] == 0:
                BSR_boolean[i, j] = 1
            else:
                BSR_boolean[i, j] = 0

    # Find all contiguous elements
    labeled_array_BSR, num_features_BSR = label(
        BSR_boolean[:, :], structure=s)

    BSR_array = np.zeros([lon.shape[0], lon.shape[1]])
    # Determine if identified objects meet area criteria for WCCs and create array of WCCs for plotting
    for i in range(num_features_BSR):
        j = i+1
        num_pixels = len(np.where(labeled_array_BSR == j)[0])
        if num_pixels > BSR_pixels_required:
            # print('BSR detected')
            # Do something to map WCCs
            for x in range(lon.shape[0]):
                for y in range(lon.shape[1]):
                    if labeled_array_BSR[x, y] == j:
                        BSR_array[x, y] = 1

    return DCC_array, DWCC_array, WCC_array, BSR_array, cs_original, cs, lat, lon, z, vtime


def mask_plot(DCC_array, DWCC_array, WCC_array, BSR_array, lat, lon, z, vtime, folder):
    '''
    Plot storm mode masks on the cartopy map
    '''

    # --------------------------------------------------------------------------------------------------
    # Storm Type Plotting

    dataproj = z.projection.cartopy()
    mapproj = dataproj

    # Get data to plot state and province boundaries
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='50m',
        facecolor='none')

    fig = plt.figure(1, figsize=(10, 10))
    ax = plt.subplot(111, projection=mapproj)
    # range of x and y. This one is for y
    ax.set_ylim(wrf.cartopy_ylim(z[:, :]))
    # range of x and y. This one is for x. absv before the range calls in that variable.
    ax.set_xlim(wrf.cartopy_xlim(z[:, :]))
    # plt.xlim(-3*10**5,1*10**6)
    # ax.set_xlim(wrf.cartopy_xlim(hght))
    # ax.set_ylim(wrf.cartopy_ylim(hght))
    ax.add_feature(states_provinces, edgecolor='black', linewidth=1)
    ax.gridlines(linestyle=':')
    ax.coastlines('50m', linewidth=1)

    # for shape in sf2.shapeRecords():
    #     x = [i[0] for i in shape.shape.points[:]]
    #     y = [i[1] for i in shape.shape.points[:]]
    #     #x,y = np.array(wrf.ll_to_xy(nc,y,x))
    # plt.plot(x,y,linestyle='solid',color='dimgrey',linewidth=1,transform=ccrs.PlateCarree())

    # str(level) allows for the level which is being plotted to appear on the graph from the variable declared above.
    plt.title("Storm Type Classification", loc='left')
    plt.title('Valid Time: {}'.format(vtime[0], vtime[0]), loc='right')
    # fig.text(0.025,.795,''.format(vtime[0]))
    if np.max(DCC_array) > 0:
        ax.contourf(lon, lat, DCC_array, np.arange(0.5, 2, 1),
                    transform=ccrs.PlateCarree(), colors=('red'))
    if np.max(DWCC_array) > 0:
        ax.contourf(lon, lat, DWCC_array, np.arange(0.5, 2, 1),
                    transform=ccrs.PlateCarree(), colors=('green'))
    if np.max(WCC_array) > 0:
        ax.contourf(lon, lat, WCC_array, np.arange(0.5, 2, 1),
                    transform=ccrs.PlateCarree(), colors=('blue'))
    if np.max(BSR_array) > 0:
        ax.contourf(lon, lat, BSR_array, np.arange(0.5, 2, 1),
                    transform=ccrs.PlateCarree(), colors=('orange'))

    # Create a legend
    # rect = patches.Rectangle((-56.5, -34.4),0.2,0.2,linewidth=1,edgecolor='r',facecolor='red',angle=5,transform=ccrs.PlateCarree())
    # ax.add_patch(rect)
    # plt.text(-56.215, -34.35,'DCC',fontsize=14, transform=ccrs.PlateCarree())
    # rect = patches.Rectangle((-56.46,-34.733),0.2,0.2,linewidth=1,edgecolor='g',facecolor='green',angle=5,transform=ccrs.PlateCarree())
    # ax.add_patch(rect)
    # plt.text(-56.173, -34.683,'DWCC',fontsize=14, transform=ccrs.PlateCarree())
    # rect = patches.Rectangle((-56.42,-35.06),0.2,0.2,linewidth=1,edgecolor='b',facecolor='blue',angle=5,transform=ccrs.PlateCarree())
    # ax.add_patch(rect)
    # plt.text(-56.13, -35.01,'WCC',fontsize=14, transform=ccrs.PlateCarree())
    # rect = patches.Rectangle((-56.38,-35.4),0.2,0.2,linewidth=1,edgecolor='orange',facecolor='orange',angle=5,transform=ccrs.PlateCarree())
    # ax.add_patch(rect)
    # plt.text(-56.088, -35.35,'BSR',fontsize=14, transform=ccrs.PlateCarree())

    custom_lines = [Line2D([0], [0], color='r', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='orange', lw=4)]
    ax.legend(custom_lines, ['DCC', 'DWCC', 'WCC', 'BSR'],
              loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    # Rename the folder as relevant
    plt.savefig(folder + vtime[0].strftime("%Y%m%d_%H%M")+'.png', dpi=300)
    plt.close()


def mask_netcdf(DCC_array, DWCC_array, WCC_array, BSR_array, cs, lat, lon, vtime, folder):
    '''
    Save the mask variables to the netcdf file for each time step
    '''
    vtime_str = folder + vtime[0].strftime("%Y%m%d_%H%M")
    fn = vtime_str + '.nc'
    ds = Dataset(fn, 'w', format='NETCDF4')

    # Dimensions

    TIME = ds.createDimension('time', None)
    LAT = ds.createDimension('lat', lat.shape[0])
    LON = ds.createDimension('lon', lon.shape[1])

    # Variables
    TIMES = ds.createVariable('time', 'f4', ('time',))
    TIMES.longname = 'time'
    TIMES.units = 'days since 1970-01-01 00:00'

    LATS = ds.createVariable('lat', 'f4', ('lat', 'lon',))
    LATS.longname = 'latitude'
    LATS.units = 'degrees_north'

    LONS = ds.createVariable('lon', 'f4', ('lat', 'lon',))
    LONS.longname = 'longitude'
    LONS.units = 'degrees_east'

    DCC_MASK = ds.createVariable('DCC_mask', 'f4', ('time', 'lat', 'lon',))
    DWCC_MASK = ds.createVariable('DWCC_mask', 'f4', ('time', 'lat', 'lon',))
    WCC_MASK = ds.createVariable('WCC_mask', 'f4', ('time', 'lat', 'lon',))
    BSR_MASK = ds.createVariable('BSR_mask', 'f4', ('time', 'lat', 'lon',))
    CS_MASK = ds.createVariable('CS_mask', 'f4', ('time', 'lat', 'lon',))

    TIMES[:] = date2num(
        vtime, units='days since 1970-01-01 00:00', calendar='standard')
    LATS[:] = lat
    LONS[:] = lon

    DCC_MASK[0, :] = DCC_array
    DWCC_MASK[0, :] = DWCC_array
    WCC_MASK[0, :] = WCC_array
    BSR_MASK[0, :] = BSR_array
    CS_MASK[0, :] = cs
    # Global attributes
    ds.TITLE = 'STORM MODE MASKS FROM WRF V4.1.3 MODEL REAL-DATA CASE OUTPUT'
    ds.SIMULATION_START_DATE = '2017-06-01_12:00:00'

    ds.close()


for filename in os.listdir(in_folder):
    # print(filename)
    # and filename[19:27] in ['01_22:00']:
    if filename.startswith('wrf') and filename[25:27] in ['00']:
        print(filename)
        dcc_mask, dwcc_mask, wcc_mask, bsr_mask, cs_original, cs, lat, lon, z, vtime = storm_mask(
            in_folder+filename)
        mask_plot(dcc_mask, dwcc_mask, wcc_mask, bsr_mask,
                  lat, lon, z, vtime, out_fig_folder)
        mask_netcdf(dcc_mask, dwcc_mask, wcc_mask, bsr_mask,
                    cs, lat, lon, vtime, out_fil_folder)

print("--- %s seconds ---" % (time.time() - start_time))
