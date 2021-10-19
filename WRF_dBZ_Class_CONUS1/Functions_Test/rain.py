'''
Storm Mode's Rain Characteristics and Ingredient

Rung Panasawatwong, September 2021
rung@colostate.edu

'''
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
from metpy.calc import divergence
import xarray as xr
from pint import UnitRegistry

# !!!   Don't forget to change the pixel size in volume_rain()  !!!


def load_wrf_rainfall(path_wrf):
    # path_wrf = "/home/krasmussen/scratch/DATA/PRECIP/June2017_meiyu/CTRL_fromJen_1km/"
    wrflist = []
    n = 0
    for filename in sorted(os.listdir(path_wrf)):
        if filename.startswith('wrf') and filename[25:27] in ['00']:
            wrflist.append(Dataset(path_wrf + filename))
            n += 1

    rain_accum = []
    for i in range(n):
        rain_accum.append(wrflist[i].variables['RAINNC'][:])
    rain_accum = np.array(np.squeeze(rain_accum))
    rain_hourly = np.concatenate(
        ([rain_accum[0]], (rain_accum[1:] - rain_accum[:-1])), axis=0)
    for i in range(n):
        wrflist[i].close()

    return rain_hourly


def loc(label_mask, lat, lon):
    '''Compute lat-lon coords of the storm's center'''
    lon_center = np.mean(lon[label_mask])
    lat_center = np.mean(lat[label_mask])

    return lat_center, lon_center


def mean_rain_rate(label_mask, rainrate):
    '''Compute mean rain rate (mean_rain) in each storm'''
    mean_rain = np.average(rainrate[label_mask])
    return mean_rain


def volume_rain(label_mask, rainrate, dx=1):
    '''Compute volumetric rain (vol_rain) in each storm'''
    vol_rain = np.sum(rainrate[label_mask])*(dx**2)

    return vol_rain


def max_rain_rate(label_mask, rainrate):
    '''Compute mean rain rate (mean_rain) in each storm'''
    max_rain = np.nanmax(rainrate[label_mask])
    return max_rain


# ----------------------------------------------------------------------------------------------


def storm_characters(label_mask, rainrate, lat, lon):
    lat_center, lon_center = loc(label_mask, lat, lon)
    mean_rain = mean_rain_rate(label_mask, rainrate)
    vol_rain = volume_rain(label_mask, rainrate)
    max_rain = max_rain_rate(label_mask, rainrate)
    return lat_center, lon_center, mean_rain, max_rain, vol_rain


def comp_and_write(filename, num_features, labeled_array, rainrate, lat, lon):
    for i in range(1, num_features+1):
        label_mask_DCC = np.array(labeled_array == i)
        DCC_characters = storm_characters(label_mask_DCC, rainrate, lat, lon)
        with open(filename, 'a') as file:
            for c in DCC_characters:
                file.write(str(c))
                file.write('\t')
            file.write('\n')


def compute_characters(out_folder, filename, dcc_mask, dwcc_mask, wcc_mask, bsr_mask, rainrate, lat, lon):
    '''
    Input:  Storm masks: DCC, DWCC, WCC, BSR
            Raw ingredients:
                Rain rate (RAINNC = ACCUMULATED TOTAL GRID SCALE PRECIPITATION)

    Output: Rain characteristics: 
                Mean rain rate
                Volumetric rain
            Computed ingredients: 
                Vertically integrated moisutre flux
                Vertically integrated moisture divergence
                850 hPa moisture flux
                700-900 hPa (Low-level) wind shear

    '''

# Labeling each storm
    s = generate_binary_structure(2, 2)
    labeled_array_DCC, num_features_DCC = label(dcc_mask, structure=s)
    labeled_array_DWCC, num_features_DWCC = label(dwcc_mask, structure=s)
    labeled_array_WCC, num_features_WCC = label(wcc_mask, structure=s)
    labeled_array_BSR, num_features_BSR = label(bsr_mask, structure=s)


# Compute storm character by each labeled
    comp_and_write(out_folder + 'DCC'+filename, num_features_DCC,
                   labeled_array_DCC, rainrate, lat, lon)
    comp_and_write(out_folder + 'DWCC'+filename, num_features_DWCC,
                   labeled_array_DWCC, rainrate, lat, lon)
    comp_and_write(out_folder + 'WCC'+filename, num_features_WCC,
                   labeled_array_WCC, rainrate, lat, lon)
    comp_and_write(out_folder + 'BSR'+filename, num_features_BSR,
                   labeled_array_BSR, rainrate, lat, lon)


def main():
    path = "/home/krasmussen/scratch/DATA/PRECIP/June2017_meiyu/CTRL_fromJen_1km/"
    mask_path = "/home/krasmussen/scratch/RESEARCH/STORM_MODE_TRACKING/Rung_version/outfile/CTRL/"
    out_folder = "./rain/"
    rainrate = load_wrf_rainfall(path)
    n = 0

    for filename in sorted(os.listdir(path)):
        if filename.startswith('wrf') and filename[25:27] in ['00']:
            filename = "wrfout_d01_2017-06-01_22:00:00"
            n = 11
            nc = Dataset(path+filename, 'r')
            lon = wrf.getvar(nc, 'lon').data
            lat = wrf.getvar(nc, 'lat').data
            nc.close()

            mask_filename = filename[11:15]+filename[16:18] + \
                filename[19:21]+'_'+filename[22:24]+filename[25:27]+'.nc'
            maskfile = mask_path + mask_filename
            nc = Dataset(maskfile, 'r')
            dcc_mask = nc.variables['DCC_mask'][:]
            dcc_mask = dcc_mask.squeeze()
            dcc_mask = dcc_mask.astype(int)

            dwcc_mask = nc.variables['DWCC_mask'][:]
            dwcc_mask = dwcc_mask.squeeze()
            dwcc_mask = dwcc_mask.astype(int)

            wcc_mask = nc.variables['WCC_mask'][:]
            wcc_mask = wcc_mask.squeeze()
            wcc_mask = wcc_mask.astype(int)

            bsr_mask = nc.variables['BSR_mask'][:]
            bsr_mask = bsr_mask.squeeze()
            bsr_mask = bsr_mask.astype(int)
            nc.close()

            date = mask_filename[:13]
            compute_characters(out_folder, '_rain_'+date, dcc_mask, dwcc_mask, wcc_mask, bsr_mask,
                               rainrate[n, :, :], lat, lon)
            n += 1
            break
    return


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
