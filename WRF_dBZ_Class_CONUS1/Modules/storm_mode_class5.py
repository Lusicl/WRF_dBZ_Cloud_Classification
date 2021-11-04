
## ======================================================================
## 
## Storm-Mode/Cloud-Type Classification for 3D gridded radar reflectivity from observations or models. 
##
## The Storm Mode Classification methodology is based on the algorithm from 
## [ Houze et al (2015): The variable nature of convection in the tropics and subtropics: 
##   A legacy of 16 years of the Tropical Rainfall Measuring Mission satellite ]
## , and is updated to identify [ 5 ] different classes of storm/cloud types from the composite reflectivity
## according to [ moderate ] and [ strong ] thresholds.
## 
## The strom modes are,
##     1: Deep Convective Cores (DCC)
##     2: Ordinary Convective Cores (OCC)
##     3: Wide Convective Cores (WCC)
##     4: Deep and Wide COnvective Cores (DWCC)
##     5: Broad Stratiform Region (BSR)
##
## Please refer to [ https://github.com/yuhungjui/WRF_dBZ_Cloud_Classification ] for more informaiton on the algorithm.
## 
##
## Run:             DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask = stm.storm_mode_c5( refl
##                                                                                       , reflc
##                                                                                       , CS_mask
##                                                                                       , geo_H
##                                                                                       , grid_res
##                                                                                       , thresholds_type
##                                                                                       )
##
##                  Storm_Mode = stm.merge_to_Storm_Mode(DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask)
##
##
## Input:
##
##     refl:        Derived 3D reflectivity
##     reflc:       Derived 2D max. reflectivity composite
##     CS_mask:     2D Convective/Stratifom mask at 12th sigma level (~1.5–2km agl.)
##     geoH:        3D Geopotential-height (msl. in meter)
##
##
## Output (masked as 1/0):
##
##     DCC_mask:    1 if identified as DCC, otherwise 0
##     OCC_mask:    1 if identified as OCC, otherwise 0
##     WCC_mask:    1 if identified as WCC, otherwise 0
##     DWCC_mask:   1 if identified as DWCC, otherwise 0
##     BSR_mask:    1 if identified as BSR, otherwise 0
##
##     Storm_Mode:  0: others
##                  1: DCC
##                  2: OCC
##                  3: WCC
##                  4: DWCC
##                  5: BSR
## 
##
## Hungjui Yu 20211103
##
## ======================================================================

import numpy as np
from scipy.ndimage import label, generate_binary_structure
from wrf import interplevel

## ======================================================================

def set_classification_thresholds(threshold_type):
    
    ## Make sure the thresholds are either Mocderate or Strong:
    assert threshold_type in ['moderate', 'strong']
    
    if ( threshold_type == 'moderate' ):
        dbz_threshold = 30 # dBZ
        height_threshold = 8 # km
        WCC_threshold = 800 # km^2
        BSR_threshold = 40000 # km^2
    else:
        dbz_threshold = 40 # dBZ
        height_threshold = 10 # km
        WCC_threshold = 1000 # km^2
        BSR_threshold = 50000 # km^2
        
    return dbz_threshold, height_threshold, WCC_threshold, BSR_threshold

## ======================================================================

def dbz_geoh_interp(refl, geoH, interp_lev_km):
    
    ## Use linear Z for interpolation:
    refl_linear = 10**(refl/10.)
    
    ## Interpolation:
    ## !!! convert interpolation level to the same as geo-H (meter) !!!
    refl_linear_lev = interplevel(refl_linear, geoH, interp_lev_km*1000)
    
    ## Convert back to dBz after interpolation:
    refl_lev = 10.0 * np.log10(refl_linear_lev)
    
    return refl_lev

## ======================================================================

def test_modes_distiction(DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask):
    
    modes_distinct_ind = np.max( DCC_mask + OCC_mask + WCC_mask + DWCC_mask + BSR_mask )
    
    return modes_distinct_ind

## ======================================================================

def storm_mode_c5( refl, reflc, CS_mask
                 , geoH
                 , data_grid_res # km grid spacing 
                 , threshold_type # moderate/strong
                 # , dbz_threshold # dBZ
                 # , height_threshold # km
                 # , WCC_threshold # km^2
                 # , BSR_threshold # km^2
                 ):
    
    ## Set thresholds:
    dbz_threshold, height_threshold, WCC_threshold, BSR_threshold = set_classification_thresholds(threshold_type)
    
    ## Set pixel numbers required for WCC & BSR:
    WCC_pixels_required = WCC_threshold/(data_grid_res**2)
    BSR_pixels_required = BSR_threshold/(data_grid_res**2)
    
    ## Interpolate reflectivity to height threshold:
    refl_lev = dbz_geoh_interp(refl, geoH, height_threshold)
    
    ## Generate a structuring element that will consider features connected even if they touch diagonally:
    se = generate_binary_structure(2, 2)
    
    
    ##
    ## 1:
    ## Threshold - Max. Composite dBZ & Convective Mask:
    reflc_boo_tmp = np.where( (reflc >= dbz_threshold) & (CS_mask > 0), 1, 0 )
    
    ## DCC & OCC masking:
    DCC_mask = np.where( ((reflc_boo_tmp == 1) & (refl_lev >= dbz_threshold)), 1, 0 )
    OCC_mask = np.where( ((reflc_boo_tmp == 1) & (refl_lev < dbz_threshold)), 1, 0 )
    
    
    ##
    ## 2:
    ## WCC & DWCC masking:
    labeled_array_reflc_boo, num_features_reflc_boo = label( reflc_boo_tmp, structure=se )
    
    WCC_mask = np.zeros_like(labeled_array_reflc_boo)
    DWCC_mask = np.zeros_like(labeled_array_reflc_boo)

    for feati in np.arange(num_features_reflc_boo):
        feat_id = feati+1
        if ( (labeled_array_reflc_boo == feat_id).sum() > WCC_pixels_required ):
            if ( (DCC_mask[np.where(labeled_array_reflc_boo == feat_id)]).sum() == 0 ):
                WCC_mask = np.where( (labeled_array_reflc_boo == feat_id), 1, WCC_mask )
            else:
                DWCC_mask = np.where( (labeled_array_reflc_boo == feat_id), 1, DWCC_mask )
            
    
    ##
    ## 3:
    ## DCC mask adjustment for DWCC:
    DCC_mask[np.where(DWCC_mask == 1)] = 0
    
    ## OCC mask adjustment for WCC and DWCC:
    OCC_mask[np.where( (WCC_mask) | (DWCC_mask == 1) )] = 0
        
    
    ##
    ## 4:
    ## Threshold - Stratiform Mask:
    stratiform_boo_tmp = np.where( (CS_mask == 0), 1, 0 )
    
    ## BSR masking:
    labeled_array_BSR, num_features_BSR = label( stratiform_boo_tmp, structure=se )
    
    BSR_mask = np.zeros_like(labeled_array_BSR)

    for feati in np.arange(num_features_BSR):
        feat_id = feati+1
        if ( (labeled_array_BSR==feat_id).sum() > BSR_pixels_required ):
            BSR_mask = np.where( (labeled_array_BSR==feat_id), 1, BSR_mask )
      

    ##
    ## 5:
    ## Test the distinction among storm modes:
    
    modes_distinct = test_modes_distiction(DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask)
    if ( modes_distinct > 1 ):
        print('Overlaps among the Storm Modes!')
    
    
    return DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask

## ======================================================================
    
def merge_to_Storm_Mode( DCC_mask
                       , OCC_mask
                       , WCC_mask
                       , DWCC_mask
                       , BSR_mask
                       ):
    
    Storm_Mode = np.zeros_like(DCC_mask)
    Storm_Mode = np.where( (DCC_mask==1), 1, Storm_Mode )
    Storm_Mode = np.where( (OCC_mask==1), 2, Storm_Mode )
    Storm_Mode = np.where( (WCC_mask==1), 3, Storm_Mode )
    Storm_Mode = np.where( (DWCC_mask==1), 4, Storm_Mode )
    Storm_Mode = np.where( (BSR_mask==1), 5, Storm_Mode )

    # Storm_Mode[np.where(Storm_Mode == 0)] = np.nan
    
    return Storm_Mode

