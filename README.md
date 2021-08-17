# WRF dBZ Retrieval & Cloud Type Classification

This project calculates the simulated radar relectivity (dBZ) from WRF model outputs (3D), and classifies the cloud type based on the 3D reflectivity.

So far, the project focuses on the outputs from [High Resolution WRF Simulations of the Current and Future Climate of North America](https://rda.ucar.edu/datasets/ds612.0/). \
The radar reflectivity retrival is based on the [wrf-python algorithm](https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.dbz.html). \
And the cloud type classification is based on the methodology in [Houze et al. 2015](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015RG000488) which identifies four types of clouds from the TRMM satellite reflectivity, including Deep convective cores (DCCs), Wide convective cores (WCCs), Deep and Wide convective cores (DWCCs), and Broad stratiform regions (BSRs).

Last update - 20210817 - Hungjui Yu

## Specific Dependencies:

* [wrf-python](https://wrf-python.readthedocs.io/en/latest/index.html)
* netcdf4
* Numpy
* xarray
* pandas
