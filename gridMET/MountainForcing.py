# Extract forcings for mountain ranges. 

import numpy as np
import xarray as xa



forcings = ['pr', 'rmax', 'rmin', 'sph', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
# Sierra
for forcing in forcings:
    file = forcing + '_1980_2018.nc'
    data = xa.open_dataarray(file)
    data = data.sel(lat=slice(39.7, 35.2), lon=slice(-121, -117))
    data.to_netcdf('SIERRA/sierra_' + forcing + '_1980_2018.nc')
print(data.shape)
# North
for forcing in forcings:
    file = forcing + '_1980_2018.nc'
    data = xa.open_dataarray(file)
    data = data.sel(lat=slice(47.1, 41.5), lon=slice(-116.8, -108))
    data.to_netcdf('NORTH/north_' + forcing + '_1980_2018.nc')
print(data.shape)
# Cascade
for forcing in forcings:
    file = forcing + '_1980_2018.nc'
    data = xa.open_dataarray(file)
    data = data.sel(lat=slice(49.1, 40.8), lon=slice(-123.1, -119.9))
    data.to_netcdf('CASCADE/cascade_' + forcing + '_1980_2018.nc')
print(data.shape)
# Utah
for forcing in forcings:
    file = forcing + '_1980_2018.nc'
    data = xa.open_dataarray(file)
    data = data.sel(lat=slice(41.7, 36.8), lon=slice(-114.2, -108.8))
    data.to_netcdf('UTAH/utah_' + forcing + '_1980_2018.nc')
print(data.shape)