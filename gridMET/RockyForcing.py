import numpy as np
import xarray as xa

swe_ROCKY = xa.open_dataarray('/tempest/duan0000/SWE/NSIDC/SWES_NSIDC.nc')
lats = swe_ROCKY.lat.data
lons = swe_ROCKY.lon.data
min_lat, max_lat = np.min(lats), np.max(lats)
min_lon, max_lon = np.min(lons), np.max(lons)

pr_forcing = xa.open_dataarray('pr_1980_2018.nc')
forcing_lats = pr_forcing.lat.data
forcing_lons = pr_forcing.lon.data
diff_lat = forcing_lats - min_lat
lat_ind_min = np.where(diff_lat < 0, diff_lat, -np.inf)
lat_ind_min = np.argmax(lat_ind_min)
diff_lat = forcing_lats - max_lat
lat_ind_max = np.where(diff_lat > 0, diff_lat, np.inf)
lat_ind_max = np.argmin(lat_ind_max)

diff_lon = forcing_lons - min_lon
lon_ind_min = np.where(diff_lon < 0, diff_lon, -np.inf)
lon_ind_min = np.argmax(lon_ind_min)
diff_lon = forcing_lons - max_lon
lon_ind_max = np.where(diff_lon > 0, diff_lon, np.inf)
lon_ind_max = np.argmin(lon_ind_max)

forcings = ['pr', 'rmax', 'rmin', 'sph', 'srad', 'tmmn', 'tmmx', 'vpd', 'vs']
for forcing in forcings:
    file = forcing + '_1980_2018.nc'
    data = xa.open_dataarray(file)
    data = data.isel(lat=slice(lat_ind_max, lat_ind_min + 1), lon=slice(lon_ind_min, lon_ind_max + 1))
    data.to_netcdf('Rocky/' + forcing + '_1980_2018.nc')
