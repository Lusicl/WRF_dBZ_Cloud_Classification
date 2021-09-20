
# ## Retrieve 3D Reflectivity from WRF Simulations using WRF-Python. 
# 
# For [High Resolution WRF Simulations of the Current and Future Climate of North America](https://rda.ucar.edu/datasets/ds612.0/).
# Save one variable (dBZ) to daily files only for storage.
#
# How to run:
# python wrf_py_dbz_1var_daily.py [CTRL3D/PGW3D] [start_date_yyyymmdd] [end_date_yyyymmdd]
#
# Hungjui Yu 20210910


import sys
import time
import datetime as dt
import pytz
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import wrf
from wrf import (getvar, dbz, extract_times, destagger)

# %% 
# **Set input files paths and names:**

def set_input_names(file_date):

    file_path_1 = '/gpfs/fs1/collections/rda/data/ds612.0'
    file_path_2 = '/' + wrf_sim_type # '/CTRL3D'
    file_path_3 = '/{}'.format(file_date.strftime('%Y'))

    file_names = dict( P = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_P_{}.nc'.format(file_date.strftime('%Y%m%d')), 
					  TK = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_TK_{}.nc'.format(file_date.strftime('%Y%m%d')), 
					  QVAPOR = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_QVAPOR_{}.nc'.format(file_date.strftime('%Y%m%d')), 
					  QRAIN = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_QRAIN_{}.nc'.format(file_date.strftime('%Y%m')), 
					  QSNOW = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_QSNOW_{}.nc'.format(file_date.strftime('%Y%m%d')), 
					  QGRAUP = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_QGRAUP_{}.nc'.format(file_date.strftime('%Y%m')), 
					  Z = file_path_1 + file_path_2 + file_path_3 + 
					  '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_Z_{}.nc'.format(file_date.strftime('%Y%m%d'))
					 )
    
    return file_names

# file_name_list = set_file_paths_names(file_date_time)
# print(file_name_list)


# %% 
# **Get wrf output variables:**

def get_wrf_vars(file_name, var_name, time_index):

    wrf_file = Dataset(file_name)
    # wrf_var = getvar(wrf_file, wrf_var_to_read, timeidx=time_index_1) # This doesn't work for CONUS run files.
    wrf_var = getvar(wrf_file, var_name, timeidx=time_index, meta=False)
    # wrf_var_time = wrf.extract_times(wrf_file, timeidx=time_index)
    # print(wrf_var_time)
    
    return wrf_var


# %% 
# **Calculation for dBZ:**

def calculate_wrf_dbz(wrf_pres, wrf_temp, wrf_qv, wrf_qr, wrf_qs, wrf_qg):
    
    wrf_dbz = dbz(wrf_pres, 
				  wrf_temp, 
				  wrf_qv, 
				  wrf_qr, 
				  wrf_qs,
				  wrf_qg,
				# use_varint=True
				  use_liqskin=False,
				  meta=True
				 )
    
    return wrf_dbz

# %%
# **Set output file path and name:**

def set_output_name(output_file_datetime):

    # output_path = '/glade/u/home/hungjui/2scratch/DATA_WRF_CONUS_1_dBZ_v1.0'
    
    output_time = pd.to_datetime(output_file_datetime).strftime('%Y%m%d') # If input time type is numpy.datetime64:
    
    # output_name = output_path + '/wrf3d_d01_dbz_{}.nc'.format(file_date_time.strftime('%Y%m%d%H'))
    # output_name = output_path + '/wrf3d_d01_dbz_{}.nc'.format(output_time)
    # output_name = '/wrf3d_d01_dbz_{}.nc'.format(output_time)
    output_name = '/wrf3d_d01_' + wrf_sim_type[0:-2] + '_dbz_{}.nc'.format(output_time)

    return output_name


# %%
# ### Main Function:
# %%

def main_function(file_date_time):
    
    ## Set file datetime:
    # file_date_time = dt.datetime(2013, 9, 13, 0, 0, 0, tzinfo=pytz.utc)
    
    # print('Processing: {}'.format(file_date_time.strftime('%Y%m%d')), end=' ')
    
    ## Set input files paths and names:
    file_name_dict = set_input_names(file_date_time)

    ## Get the 3-hourly time list from P and QRAIN files:
    wrf_3hour_list_1 = wrf.extract_times(Dataset(file_name_dict['P']), timeidx=wrf.ALL_TIMES, meta=False, do_xtime=False)
    wrf_3hour_list_2 = wrf.extract_times(Dataset(file_name_dict['QRAIN']), timeidx=wrf.ALL_TIMES, meta=False, 
										 do_xtime=False)

    ## Set wrf variable list for reflectivity retrieval:
    wrf_vars_list = ['P', 'TK', 'QVAPOR', 'QRAIN', 'QSNOW', 'QGRAUP']

    ## Set dBZ data array and append calculated data:
    # wrf_dbz = xr.zeros_like(wrf_dataset_out['P'])
    
    for hi in range(len(wrf_3hour_list_1)):
        
        # print(str(hi) + ' | ', end='')

        ## Get the index for common time in different files (every 3-hour):
        common_index_2 = np.intersect1d(wrf_3hour_list_1[hi], wrf_3hour_list_2, return_indices=True)[2][0]

        ## Get wrf output variables:
        wrf_vars = {}
        for vname in wrf_vars_list:

            file_name = file_name_dict[vname]

            if ( vname in ['QRAIN', 'QGRAUP'] ):
                wrf_vars['{}'.format(vname)] = get_wrf_vars(file_name, vname, common_index_2)
                # wrf_vars['{}'.format(vname)] = xr.open_dataset(file_name)
            else:
                wrf_vars['{}'.format(vname)] = get_wrf_vars(file_name, vname, hi)
                # wrf_vars['{}'.format(vname)] = xr.open_dataset(file_name)


        ## Calculation for dBZ:
        wrf_dbz_3hr = calculate_wrf_dbz(wrf_vars['P'],
                                        wrf_vars['TK'], 
                                        wrf_vars['QVAPOR'],
                                        wrf_vars['QRAIN'],
                                        wrf_vars['QSNOW'],
                                        wrf_vars['QGRAUP']
                                        ) # .to_dataset()
        
        # wrf_dbz_3hr = wrf_dbz_3hr.expand_dims({'TimeDim': 8})
        # print(wrf_dbz_3hr)
        
        if ( hi == 0 ):
            wrf_dbz = wrf_dbz_3hr
        else:
            wrf_dbz = xr.concat([wrf_dbz, wrf_dbz_3hr], dim='TimeDim')

    # print(wrf_dbz)
            
    ## Set output dataset:
    wrf_dataset_out = xr.open_dataset(file_name_dict['P'])    
      
    ## Add dBZ to output dataset:
    wrf_dataset_out['dBZ'] = (['Time', 'bottom_top', 'south_north', 'west_east'], wrf_dbz)
    #print(wrf_dataset_out)
    
    ## Drop P from the output dataset:
    wrf_dataset_out = wrf_dataset_out.drop_vars('P')
            
    ## Unstagger Z vertical grids:
    # wrf_var_Z_unstag = wrf.destagger(getvar(Dataset(file_name_dict['Z']), 'Z', timeidx=hi, meta=False), 0)

    ## Get AGL:
    # wrf_var_Z_AGL = wrf.g_geoht.get_height_agl(Dataset(file_name_dict['Z']), 'Z', meta=False)

    ## Add Z and dBZ into dataset of P:
    # wrf_dataset_P_Z_dBZ = xr.open_dataset(file_name_dict['P']).isel(Time = hi)
    # wrf_dataset_out = xr.open_dataset(file_name_dict['P'])
    # wrf_dataset_out['dBZ'] = (['bottom_top', 'south_north', 'west_east'], wrf_dbz)
    
    # wrf_dataset_P_Z_dBZ['TK'] = (['bottom_top', 'south_north', 'west_east'], wrf_vars['TK'])
    # wrf_dataset_P_Z_dBZ['Z'] = (['bottom_top', 'south_north', 'west_east'], wrf_var_Z_unstag)
        
    ## Calculate Z using the U.S. standard atmosphere & Hypsometric eqn.:
    # Z_standard = mpcalc.pressure_to_height_std((wrf_dataset_P_Z_dBZ['P'].values) * units.units.Pa)
    # wrf_dataset_P_Z_dBZ['Z_standard'] = (['bottom_top', 'south_north', 'west_east'], Z_standard)

    ## Set output file path and name:
    output_path_1 = '/glade/u/home/hungjui/2scratch/DATA_WRF_CONUS_1_dBZ_v1.0/' + wrf_sim_type
    # output_path_2 = '/20130913'
    output_path_2 = '/{}'.format(file_date_time.strftime('%Y'))
    output_file_name = set_output_name(file_date_time)
    wrf_dataset_out.to_netcdf(output_path_1 + output_path_2 + output_file_name)
        
    # print('Finish this date.')
        
# %% 
# ### Main Program:
# %%

start = time.time()

## WRF Model Simulation Category:
# wrf_sim_type = 'CTRL3D'
# wrf_sim_type = 'PGW3D'
wrf_sim_type = sys.argv[1]

## Loop through a period:
# target_date_range = pd.date_range(start='2013-9-13', end='2013-9-13', tz=pytz.utc)
target_date_range = pd.date_range(start=sys.argv[2], end=sys.argv[3], tz=pytz.utc)

for dayi in target_date_range:
    
    main_function(dayi)
    
    #main_calc = delayed(main_function)(dayi)
    
# main_calc.compute()
# main_calc.visualize()

end = time.time()

# print("RUNTIME：%f SEC" % (end - start))
# print("RUNTIME：%f MIN" % ((end - start)/60))
# print("RUNTIME：%f HOUR" % ((end - start)/3600))

run_time_txt = open('./run_time.log','a')
run_time_txt.write(sys.argv[1] + '\n')
run_time_txt.write(sys.argv[2] + ' - ' + sys.argv[3] + '\n')
run_time_txt.write("RUNTIME：%f SEC \n" % (end - start))
run_time_txt.write("RUNTIME：%f MIN \n" % ((end - start)/60))
run_time_txt.write("RUNTIME：%f HOUR \n" % ((end - start)/3600))
