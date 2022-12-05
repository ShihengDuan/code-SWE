# PRISM grid spacing: 0.008
# LOCA grid spacing: 0.06
import numpy as np
import xarray as xa
from tqdm import tqdm

prism_topo = xa.open_dataset('../wus_prism_topo_800m.nc')

mountains = ['rocky', 'cascade', 'north', 'sierra', 'utah']
for mountain in mountains:
    origin_forcing = xa.open_dataarray('ori_data/ACCESS1-3/pr_'+mountain+'_1981-2100.nc')
    lat = origin_forcing.lat
    lon = origin_forcing.lon
    num_lat = len(lat)
    num_lon = len(lon)
    elevation_ave = np.zeros((num_lat, num_lon)) # lat, lon, average 
    elevation_nn = np.zeros((num_lat, num_lon)) # nearest neighbor
    dah_ave = np.zeros((num_lat, num_lon))
    dah_nn = np.zeros((num_lat, num_lon))
    trasp_ave = np.zeros((num_lat, num_lon))
    trasp_nn = np.zeros((num_lat, num_lon))
    longitude = np.zeros((num_lat, num_lon))
    latitude = np.zeros((num_lat, num_lon))
    lons = origin_forcing.lon
    lats = origin_forcing.lat
    min_lat = np.min(lats)
    min_lon = np.min(lons)
    max_lat = np.max(lats)
    max_lon = np.max(lons)
    print(min_lat.data, ' ', max_lat.data)
    print(min_lon.data, ' ', max_lon.data)
    for i, lat in tqdm(enumerate(lats)):
        for j, lon in enumerate(lons):
            longitude[i, j] = lon
            latitude[i, j] = lat
            # ele_station = ele.sel(y=lat, x=lon, method='nearest').data[0]
            ele_station = prism_topo.elevation_prism.sel(lat=slice(lat-0.03, lat+0.03), lon=slice(lon-0.03, lon+0.03))
            elevation_ave[i, j] = ele_station.mean(skipna=True).data
            elevation_nn[i, j] = prism_topo.elevation_prism.sel(lat=lat, lon=lon, method='nearest').data

            dah_station = prism_topo.dah.sel(lat=slice(lat-0.03, lat+0.03), lon=slice(lon-0.03, lon+0.03)).mean(skipna=True).data
            dah_ave[i, j] = dah_station
            dah_nn[i, j] = prism_topo.dah.sel(lat=lat, lon=lon, method='nearest').data

            trasp_station = prism_topo.trasp.sel(lat=slice(lat-0.03, lat+0.03), lon=slice(lon-0.03, lon+0.03)).mean(skipna=True).data
            trasp_ave[i, j] = trasp_station
            trasp_nn[i, j] = prism_topo.trasp.sel(lat=lat, lon=lon, method='nearest').data
            
    elevation_ave = xa.DataArray(elevation_ave, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    dah_ave = xa.DataArray(dah_ave, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    trasp_ave = xa.DataArray(trasp_ave, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    longitude = xa.DataArray(longitude, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    latitude = xa.DataArray(latitude, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    static_data = xa.Dataset({
        'dah':dah_ave,
        'trasp':trasp_ave,
        'elevation_prism':elevation_ave,
        'longitude':longitude, 
        'latitude': latitude,
        })
    static_data.to_netcdf(mountain+'_topo_file_prism_ave.nc')

    elevation_nn = xa.DataArray(elevation_nn, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    dah_nn = xa.DataArray(dah_nn, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    trasp_nn = xa.DataArray(trasp_nn, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    static_data = xa.Dataset({
        'dah':dah_nn,
        'trasp':trasp_nn,
        'elevation_prism':elevation_nn,
        'longitude':longitude, 
        'latitude': latitude,
        })
    static_data.to_netcdf(mountain+'_topo_file_prism_nn.nc')