import numpy as np
import xarray as xa
from tqdm import tqdm

snotel = xa.open_dataset('../SNOTEL/raw_wus_snotel_topo.nc')
lat = snotel.latitude.data
lon = snotel.longitude.data
min_lat = np.min(lat) - 0.5
max_lat = np.max(lat) + 0.5
min_lon = np.min(lon) - 0.5
max_lon = np.max(lon) + 0.5
del snotel
forcings = ['pr', 'rmax', 'rmin', 'sph', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
years = np.arange(1980, 2019)
for forcing in tqdm(forcings, position=0, desc='forcings'):
    f_data = []
    time_index = []
    print(forcing)
    for year in tqdm(years, position=1, desc='years', leave=False):
        print(year)
        data = xa.open_dataarray(forcing + '_' + str(year) + '.nc')
        data = data.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
        f_data.append(data.data)
        time_index.append(data.day.data)
    print('concat: ')
    f_data = np.concatenate(f_data, axis=0)
    time_index = np.concatenate(time_index)
    f_data = xa.DataArray(f_data, dims=['time', 'lat', 'lon'],
                          coords={'time': time_index, 'lat': data.lat, 'lon': data.lon})
    print(f_data.shape)
    f_data.to_netcdf(forcing + '_1980_2018.nc')
