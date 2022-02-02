# Combine NSIDC data from 2008-06-28 to 2017-12-31. 3474 days.
# water year from 2007 to 2018

import numpy as np
import xarray as xa

# Load CO_PRISM
file = '/tempest/duan0000/SWE/CO_SWE/ppt_data'
ppt = xa.open_dataset(file)
lat_co = ppt.lat
lon_co = ppt.lon

swes = []

years = np.arange(2007, 2019)
for year in years:
    file = '/tempest/duan0000/SWE/NSIDC/4km_SWE_Depth_WY' + str(year) + '_v01.nc'
    data = xa.open_dataset(file)
    swe = data.SWE.sel(lat=slice(lat_co.min().data, lat_co.max().data))
    swe = swe.sel(lon=slice(lon_co.min().data, lon_co.max().data))
    swes.append(swe)
swes = xa.concat(swes, dim='time')
swes = swes.sel(time=slice('2008-06-28', '2017-12-31'))
swes.to_netcdf('SWES_NSIDC.nc')

years = np.arange(1982, 2000)
for year in years:
    file = '/tempest/duan0000/SWE/NSIDC/4km_SWE_Depth_WY' + str(year) + '_v01.nc'
    data = xa.open_dataset(file)
    swe = data.SWE.sel(lat=slice(lat_co.min().data, lat_co.max().data))
    swe = swe.sel(lon=slice(lon_co.min().data, lon_co.max().data))
    swes.append(swe)
swes = xa.concat(swes, dim='time')
swes.to_netcdf('SWES_NSIDC_hist.nc')
