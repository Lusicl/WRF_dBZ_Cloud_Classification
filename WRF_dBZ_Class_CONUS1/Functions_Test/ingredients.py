'''
Storm Mode's Rain Characteristics and Ingredient

Rung Panasawatwong, July 2021
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

# !!! Change dx in area() !!!


def dp_plev_sfc(plev, psfc):
    '''Return depth of pressure levels (dp) from given pressure levels (plev) and 
    surface pressure (psfc)'''
    # Setup dp array to be the same size as plev
    dp = np.zeros(plev.shape)
    for i in range(len(plev)):
        if psfc < plev[i]:
            dp[i] = 0
        else:  # psfc >= plev[i]
            if i == 0:
                dp[i] = psfc - plev[i]
            else:
                dp[i] = plev[i-1] - plev[i]
    return dp


def loc(label_mask, lat, lon):
    '''Compute lat-lon coords of the storm's center'''
    lon_center = np.mean(lon[label_mask])
    lat_center = np.mean(lat[label_mask])

    return lat_center, lon_center


def vim_flux(label_mask, u, v, Q, p, psfc):
    '''Compute vertically integrated moisture flux (vimf) and divergence (vimd) '''
    # Interpolate u*Q, v*Q from p to plev
    plev = np.array([100000., 97500., 95000., 92500., 90000., 87500., 85000., 82500., 80000.,
                     77500., 75000., 72500., 70000., 67500., 65000., 62500., 60000.,
                     55700., 55000., 52500., 50000., 45000., 40000., 35000., 30000., 25000., 20000.,
                     17500., 15000., 12500., 10000., 9000., 8000., 7000., 6000., 5000.])  # in Pa not hPa

    uq = wrf.interplevel(u*Q, p, plev)
    vq = wrf.interplevel(v*Q, p, plev)

    # Get VIMF: Integrate over dp, divided by g = 9.81
    uq_shape = uq.shape
    dp = np.zeros(uq_shape)
    for i in range(uq_shape[1]):
        for j in range(uq_shape[2]):
            dp[:, i, j] = dp_plev_sfc(plev, psfc[i, j])

    # vimf = (uq + vq) *dp / 9.81
    # vimf_integrated = np.sum(vimf, axis = 0)
    # mean_vimf = np.nanmean(vimf_integrated.data[label_mask])

    # GET VIMD: compute divergence of uq and vq
    ureg = UnitRegistry()
    onek = 1000.0 * ureg.meter
    vimd = divergence(uq, vq, dx=onek, dy=onek)
    # Integrate over dp, divided by g = 9.81
    vimd = vimd * dp / 9.81
    vimd_integrated = np.nansum(vimd, axis=0)
    mean_vimd = np.nanmean(vimd_integrated[label_mask])

    return mean_vimd.magnitude

# def qflux850(label_mask, w, Q, p):
#     '''Compute 850 hPa moisture flux (qflux) '''
#     # Interpolate w and Q (specific hum.) to 850 hPa
#     qflux = wrf.interplevel(w*Q, p, 85000)
#     mean_qflux = np.nanmean(qflux.data[label_mask])

#     return mean_qflux


def qflux850(label_mask, w, Q, p):
    '''Compute 850 hPa moisture flux (qflux) '''
    # Interpolate w and Q (specific hum.) to 850 hPa
    w_850 = wrf.interplevel(w, p, 85000)
    q_850 = wrf.interplevel(Q, p, 85000)

    w_mean = np.nanmean(w_850.data[label_mask])
    q_mean = np.nanmean(q_850.data[label_mask])

    mean_qflux = w_mean * q_mean

    return mean_qflux


def lls_700_900(label_mask, u, v, p):
    '''Compute 700-900 hPa (Low-level) wind shear (lls)'''
    # Interpolate u, v wind to 700 and 900 hPa level
    u_700 = wrf.interplevel(u, p, 70000.)
    u_900 = wrf.interplevel(u, p, 90000.)
    v_700 = wrf.interplevel(v, p, 70000.)
    v_900 = wrf.interplevel(v, p, 90000.)

    u_700_2 = np.nanmean(u_700.data[label_mask])
    u_900_2 = np.nanmean(u_900.data[label_mask])
    v_700_2 = np.nanmean(v_700.data[label_mask])
    v_900_2 = np.nanmean(v_900.data[label_mask])

    # Subtract wind field
    lls = np.sqrt((u_900_2-u_700_2)**2 + (v_900_2-v_700_2)**2)

    return lls


def vert_wind(label_mask, omega, p):
    '''Compute mean vertical wind (w_mean) and max vertical wind (w_up_max) in each storm'''
    # Interpolate to proper level
    omega_850 = wrf.interplevel(omega, p, 85000)

    # label mask and take mean and max
    omega850_mask = omega_850.data[label_mask]
    omega850_mean = np.nanmean(omega850_mask)
    omega850_up = np.nanmean(omega850_mask[omega850_mask > 0.])
    omega850_max = np.nanmax(omega850_mask)

    return omega850_mean, omega850_up, omega850_max

# Temperature


def temperature(label_mask, temp, theta_e, p):
    '''Compute average temperature (temp, in K) and equivalent potential temperature (theta_e, in K)'''
    # Interpolate to proper level
    temp_850 = wrf.interplevel(temp, p, 85000)
    theta_e_850 = wrf.interplevel(theta_e, p, 85000)

    # label mask and take mean
    temp_mean = np.nanmean(temp_850.data[label_mask])
    theta_e_mean = np.nanmean(theta_e_850.data[label_mask])

    return temp_mean, theta_e_mean


# LANDMASK = land/sea mask
# HGT = terrain height
def land_flag(label_mask, landmask, hgt):
    '''Find land-sea mask (1 for land, 0 for water) and terrain height (in metre) at the center of the storm'''

    # center point
    count = np.count_nonzero(label_mask)
    x_center, y_center = np.argwhere(label_mask == 1).sum(0)/count

    # get land-sea mask and terrain height of the center point
    landmask_i = landmask[round(x_center), round(y_center)]
    hgt_i = hgt[round(x_center), round(y_center)]

    return landmask_i, hgt_i

# Echo top


def echo_top(label_mask, refl, z):
    '''Find echo top height of the specified dbz (30 and 0 dbZ, in m) of the storm'''
    # interpolate reflectivity to zlev
    zlev = np.arange(500, 20001, 500)  # in metre

    refl_zlev = wrf.interplevel(refl, z, zlev)

    # filter with label mask
    label_mask_3d = np.broadcast_to(label_mask, refl_zlev.shape)
    refl_mask = np.ma.array(refl_zlev, mask=~label_mask_3d)

    # find max at each level
    refl_max = np.asarray([np.nanmax(refl_mask[i, :, :])
                          for i in range(refl_mask.shape[0])])

    # find echo top height of 30dbz and 0dbz
    try:
        top_30 = zlev[max(np.argwhere(refl_max >= 30.))][0]
    except ValueError:
        top_30 = np.NaN

    try:
        top_0 = zlev[max(np.argwhere(refl_max > 0.))][0]
    except ValueError:
        top_0 = np.NaN

    return top_30, top_0


# Area
def area(label_mask, dx=1):
    '''Compute the area of each storm'''
    # ara = no. pixel * dx**2
    area_rain = np.sum(label_mask) * (dx**2)
    return area_rain


# ----------------------------------------------------------------------------------------------


def storm_characters(label_mask, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z):
    lat_center, lon_center = loc(label_mask, lat, lon)
    vimd = vim_flux(label_mask, u, v, Q, p, psfc)
    qflux = qflux850(label_mask, w, Q, p)
    lls = lls_700_900(label_mask, u, v, p)
    omega850_mean, omega850_up, omega850_max = vert_wind(label_mask, omega, p)
    temp_mean, theta_e_mean = temperature(label_mask, temp, theta_e, p)
    landmask_i, hgt_i = land_flag(label_mask, landmask, hgt)
    top_30, top_0 = echo_top(label_mask, refl, z)
    area_rain = area(label_mask)

    return lat_center, lon_center, vimd, qflux, lls, omega850_mean, omega850_up, omega850_max, \
        temp_mean, theta_e_mean, landmask_i, hgt_i, top_30, top_0, area_rain


def comp_and_write(filename, num_features, labeled_array, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z):
    for i in range(1, num_features+1):
        label_mask_each = np.array(labeled_array == i)
        all_characters = storm_characters(
            label_mask_each, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)
        with open(filename, 'a') as file:
            for c in all_characters:
                file.write(str(c))
                file.write('\t')
            file.write('\n')


def compute_characters(out_folder, filename, dcc_mask, dwcc_mask, wcc_mask, bsr_mask, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z):
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
    comp_and_write(out_folder+'DCC'+'_'+filename+'.txt', num_features_DCC,
                   labeled_array_DCC, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)
    comp_and_write(out_folder+'DWCC'+'_'+filename+'.txt', num_features_DWCC,
                   labeled_array_DWCC, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)
    comp_and_write(out_folder+'WCC'+'_'+filename+'.txt', num_features_WCC,
                   labeled_array_WCC, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)
    comp_and_write(out_folder+'BSR'+'_'+filename+'.txt', num_features_BSR,
                   labeled_array_BSR, lat, lon, u, v, w, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)


def main():
    path = "/home/krasmussen/scratch/DATA/PRECIP/June2017_meiyu/CTRL_fromJen_1km/"
    mask_path = "/home/krasmussen/scratch/RESEARCH/STORM_MODE_TRACKING/Rung_version/outfile/CTRL/"
    out_folder = "./ingredients/"
    for filename in sorted(os.listdir(path)):
        if filename.startswith('wrf') and filename[25:27] in ['00']:
            filename = "wrfout_d01_2017-06-01_22:00:00"
            nc = Dataset(path+filename, 'r')
            lon = wrf.getvar(nc, 'lon').data
            lat = wrf.getvar(nc, 'lat').data
            p = wrf.getvar(nc, 'pres').data
            psfc = wrf.getvar(nc, 'PSFC').data
            z = wrf.getvar(nc, "z", units="m")
            u = wrf.getvar(nc, 'U')
            v = wrf.getvar(nc, 'V')
            w = wrf.getvar(nc, 'W').data
            w_up_max = wrf.getvar(nc, 'W_UP_MAX').data
            # rainnc = wrf.getvar(nc, 'RAINNC').data
            mr = wrf.getvar(nc, 'QVAPOR')  # mixing ratio
            # vtime = wrf.extract_times(nc, None).astype('M8[ms]').astype('O')
            refl = wrf.getvar(nc, 'REFL_10CM').data
            landmask = wrf.getvar(nc, 'LANDMASK').data
            hgt = wrf.getvar(nc, 'HGT').data
            temp = wrf.getvar(nc, 'temp').data
            theta_e = wrf.getvar(nc, 'eth').data
            omega = wrf.getvar(nc, 'omg').data
            nc.close()

            # destag wind
            w_destag = wrf.destagger(w, stagger_dim=0)
            v_destag = wrf.destagger(v, stagger_dim=1)
            u_destag = wrf.destagger(u, stagger_dim=2)

            Q = mr/(1+mr)  # specific humidity

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
            compute_characters(out_folder, date, dcc_mask, dwcc_mask, wcc_mask, bsr_mask,
                               lat, lon, u_destag, v_destag, w_destag, Q, p, psfc, omega, temp, theta_e, landmask, hgt, refl, z)
            break

    return


start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
