# WRF Cloud Type (Storm Mode) Classification Modules

The modules include the **Convective/Stratiform masking** module and the **Storm Mode classification** module.

## 1. Convective/Stratiform Masking:

`conv_stra_mask.py`

* The module indentifies (masks) the 2D Convective vs. Stratiform gird points from 3D radar relectivity (dBZ) gridded observations or (WRF) model outputs.
* The methodology is based on [Steiner, Houze, and Yuter 1995](https://journals.ametsoc.org/view/journals/apme/34/9/1520-0450_1995_034_1978_ccotds_2_0_co_2.xml).
* Codes originally generated by Nick Guy (Apr 2011) and Brody Fuchs (May 2018), modifed by Zach Bruick (October 2018) for use with WRF data for RELAMPAGO.
* How to run?
```
cs, cc, bkgnd = conv_stra_mask.conv_stra_sep(2d_refl, lat, lon)
```
where **2d_refl** is the specified level of interpolated reflectivity, for example at 2-km height or the 12th &sigma;-level. **lat** and **lon** are the corresponding latitude and longitude coordinates.The output  **cs** is the derived convective/stratiform mask with the same size as the inputs.
* Masks:
  * -99999. missing data
  * 0       stratiform
  * 1       convective (z <= 25 dBZ)
  * 2       convective (25 < z <= 30 dBZ)
  * 3       convective (30 < z <= 35 dBZ)
  * 4       convective (35 < z <= 40 dBZ)
  * 5       convective (z > 40 dBZ)

## 2. Storm Mode Classification:

`storm_mode_class5.py`

* The module classifies the storm modes (or cloud types) from 3D radar relectivity (dBZ) gridded observations or (WRF) model outputs. 
* The classification is based on the methodology in [Houze et al. 2015](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015RG000488) which identifies four types of clouds from the TRMM satellite reflectivity, including **Deep convective cores (DCCs)**, **Wide convective cores (WCCs)**, **Deep and Wide convective cores (DWCCs)**, and **Broad stratiform regions (BSRs)**. Each category are identified according to the **moderate** and **strong** thresholds.
* This module specifically identifies the 5th category: the **Ordinay Convective Cores (OCCs)** to represents those convective cores that are neither deep or wide enough according to the thresholds (likely convection during intensifying or decaying stages).
* Codes originally generated by Zach Bruick (October 2018) for use with WRF data for RELAMPAGO, updated by Hungjui Yu in Oct 2021 to add OCC type.
* * How to run?
```
DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask = storm_mode_class5.storm_mode_c5( 3d_refl, reflc, CS_mask, geo_H, grid_res, thresholds_type)
```
```
Storm_Mode = storm_mode_class5.merge_to_Storm_Mode(DCC_mask, OCC_mask, WCC_mask, DWCC_mask, BSR_mask)
```
where **3d_refl** is the 3D gridded reflectivity, of which **reflc** is the 2D max. composite, and **CS_mask** is the convective/stratiform masks identified by the previous module. **geo_H** is the 3D gridded geopotential heights which serves as the height coordinate. **grid_res** is the horizontal resolution of the input reflectivity in km. And **thresholds_type** is either `moderate` or `strong`.
* The output ***_masks** are indicated in 1 and 0.
* The output **Storm_Mode** is masked as 
  * 0: others
  * 1: DCC
  * 2: OCC
  * 3: WCC
  * 4: DWCC
  * 5: BSR
* Flow Chart:
![](https://github.com/yuhungjui/WRF_dBZ_Cloud_Classification/blob/main/WRF_dBZ_Class_CONUS1/Storm_Mode_Flow.png)

## Specific Dependencies:

* [wrf-python](https://wrf-python.readthedocs.io/en/latest/index.html)
* Numpy
* Scipy

Last update - 20211108 - Hungjui Yu