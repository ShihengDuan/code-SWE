import numpy as np
import xarray as xa
from tqdm import tqdm


def locate_forcing(forcing_data, lat, lon):
    forcing = forcing_data.interp(lat=lat, lon=lon)
    return forcing


def find_forcing(forcing_data, lat, lon):
    f_lons = forcing_data.lon.data.reshape(-1)
    f_lats = forcing_data.lat.data.reshape(-1)
    diff_lat = np.abs(f_lats - lat)
    diff_lon = np.abs(f_lons - lon)
    lat_arg = np.argmin(diff_lat)
    lon_arg = np.argmin(diff_lon)
    forcing = forcing_data.isel(lat=lat_arg, lon=lon_arg)
    return forcing


snotel = xa.open_dataset('../SNOTEL/raw_wus_snotel_topo.nc')
print(snotel)
lats = snotel.latitude
lons = snotel.longitude
forcings = ['pr', 'rmax', 'rmin', 'sph', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
f_data = []
for forcing in forcings:
    data = xa.open_dataarray(forcing + '_1980_2018.nc')
    data = data.rename(forcing)
    f_data.append(data)

f_data = xa.merge(f_data)
print(f_data)
n_stations = snotel.n_stations
station_forcings = {}
station_forcings['pr'] = []
station_forcings['rmax'] = []
station_forcings['rmin'] = []
station_forcings['sph'] = []
station_forcings['srad'] = []
station_forcings['tmmn'] = []
station_forcings['tmmx'] = []
station_forcings['vpd'] = []
station_forcings['vs'] = []

for n in tqdm(range(765)):  # 765 stations:
    lat = lats.isel(n_stations=n).data
    lon = lons.isel(n_stations=n).data
    forcing = find_forcing(f_data, lat, lon)

    for forc in forcings:
        station_forcings[forc].append(forcing[forc])
time_index = forcing.time
for forcing in forcings:
    station_forcing = np.empty((14245, 765))
    for i in tqdm(range(765), desc='stations'):
        station_forcing[:, i] = station_forcings[forcing][i]
    print(station_forcing.shape)
    station_forcing = xa.DataArray(station_forcing, dims=['time', 'n_stations'],
                                   coords={'time': time_index, 'n_stations': n_stations})
    station_forcing.to_netcdf(forcing + '_wus.nc')
