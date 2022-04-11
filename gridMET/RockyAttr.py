import numpy as np
import xarray as xa
from tqdm import tqdm


def find_nearest(lon, lat, attr_lon, attr_lat):
    diff_lon = np.abs(attr_lon - lon)
    diff_lat = np.abs(attr_lat - lat)
    lon_arg = np.argmin(diff_lon)
    lat_arg = np.argmin(diff_lat)
    return lon_arg, lat_arg


rocky_pr = xa.open_dataarray('Rocky/pr_1980_2018.nc')
lats = rocky_pr.lat.data
lons = rocky_pr.lon.data

attr_topo = xa.open_dataset('/tempest/duan0000/SWE/wus_prism_topo_800m.nc')
print(attr_topo)
attr_lon = attr_topo.lon.data
attr_lat = attr_topo.lat.data
attr_elevation = attr_topo.elevation_prism.data  # lat, lon
attr_dah = attr_topo.dah.data
attr_trasp = attr_topo.trasp.data

longitude = np.empty((lats.shape[0], lons.shape[0]))
latitude = np.empty((lats.shape[0], lons.shape[0]))
elevation_prism = np.empty((lats.shape[0], lons.shape[0]))
dah = np.empty((lats.shape[0], lons.shape[0]))
trasp = np.empty((lats.shape[0], lons.shape[0]))
for i, lon in tqdm(enumerate(lons)):
    for j, lat in enumerate(lats):
        # attr_rocky = attr_topo.interp(lat=lat, lon=lon, method='nearest')
        lon_arg, lat_arg = find_nearest(lon, lat, attr_lon, attr_lat)
        longitude[j, i] = lon
        latitude[j, i] = lat
        elevation_prism[j, i] = attr_elevation[lat_arg, lon_arg]
        dah[j, i] = attr_dah[lat_arg, lon_arg]
        trasp[j, i] = attr_trasp[lat_arg, lon_arg]

longitude = xa.DataArray(longitude, dims=['lat', 'lon'], coords={'lon': lons, 'lat': lats})
latitude = xa.DataArray(latitude, dims=['lat', 'lon'], coords={'lon': lons, 'lat': lats})
elevation_prism = xa.DataArray(elevation_prism, dims=['lat', 'lon'], coords={'lon': lons, 'lat': lats})
dah = xa.DataArray(dah, dims=['lat', 'lon'], coords={'lon': lons, 'lat': lats})
trasp = xa.DataArray(trasp, dims=['lat', 'lon'], coords={'lon': lons, 'lat': lats})

RockyAttr = xa.Dataset(data_vars={'longitude': longitude, 'latitude': latitude, 'elevation_prism': elevation_prism,
                                  'dah': dah, 'trasp': trasp})
print('DataSet: *******')
print(RockyAttr)
RockyAttr.to_netcdf('Rocky/topo_file.nc')
