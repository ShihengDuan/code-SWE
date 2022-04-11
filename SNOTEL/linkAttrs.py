import numpy as np
import xarray as xa

wus = '../wus_swe_snotel_prism_topo.nc'
wus = xa.open_dataset(wus)
print(wus)
wus_lat = wus.latitude
wus_lon = wus.longitude
wus_dah = wus.dah
wus_trasp = wus.trasp
wus_elevation_prism = wus.elevation_prism

snotel = xa.open_dataset('raw_WUS_snotel.nc')
print(snotel)

snotel_dah = []
snotel_trasp = []
snotel_index = []
snotel_elevation = []
ind = []
for station in snotel.n_stations[:]:
    record = snotel.sel(n_stations=station)
    lat = record.latitude.data
    lon = record.longitude.data
    print(lat, ' ', lon, ' ', lat in wus_lat.data, ' ', lon in wus_lon.data)
    wus_ind_lat = wus_lat.where(wus_lat == lat, drop=True).n_stations
    wus_ind_lon = wus_lon.where(wus_lon == lon, drop=True).n_stations
    wus_ind = np.intersect1d(wus_ind_lon, wus_ind_lat)
    if len(wus_ind) > 0:
        snotel_dah.append(wus_dah.sel(n_stations=wus_ind[0]))
        snotel_trasp.append(wus_trasp.sel(n_stations=wus_ind[0]))
        snotel_elevation.append(wus_elevation_prism.sel(n_stations=wus_ind[0]))
        snotel_index.append(wus_ind)
        ind.append(station)  # snotel index.

snotel_dah = np.array(snotel_dah)
snotel_trasp = np.array(snotel_trasp)
snotel_elevation = np.array(snotel_elevation)
snotel_dah = xa.DataArray(snotel_dah, coords={'n_stations': ind}, dims=['n_stations'])
snotel_trasp = xa.DataArray(snotel_trasp, coords={'n_stations': ind}, dims=['n_stations'])
snotel_elevation = xa.DataArray(snotel_elevation, coords={'n_stations': ind}, dims=['n_stations'])

subset_snotel = snotel.sel(n_stations=ind)
subset_snotel = subset_snotel.assign(variables={'dah': snotel_dah})
subset_snotel = subset_snotel.assign(variables={'trasp': snotel_trasp})
subset_snotel = subset_snotel.assign(variables={'elevation_prism': snotel_elevation})

print('SUBSET: ')
print(subset_snotel)
subset_snotel.to_netcdf('raw_wus_snotel_topo.nc')
