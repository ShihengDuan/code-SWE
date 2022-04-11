import numpy as np
import xarray as xa

topo = xa.open_dataset('../SNOTEL/raw_wus_snotel_topo.nc')
print(topo)
swe = topo.SWE.sel(time=slice('1980-10-01', '1999-09-30'))
max_swe = swe.max(dim='time')
print(max_swe)
del_stations = []
for station in topo.n_stations:
    station_max_swe = max_swe.sel(n_stations=station)
    if station_max_swe == 0:
        # print('ZERO')
        del_stations.append(station.data)
    elif np.isnan(station_max_swe):
        # print('NAN')
        del_stations.append(station.data)
print(del_stations)
print(len(del_stations))
topo_clean = topo.drop_sel(n_stations=del_stations)
topo_clean.to_netcdf('../SNOTEL/raw_wus_snotel_topo_clean.nc')

# Load forcings.
pr_wus = xa.open_dataarray('pr_wus.nc')
rmax_wus = xa.open_dataarray('rmax_wus.nc')
rmin_wus = xa.open_dataarray('rmin_wus.nc')
sph_wus = xa.open_dataarray('sph_wus.nc')
srad_wus = xa.open_dataarray('srad_wus.nc')
tmmn_wus = xa.open_dataarray('tmmn_wus.nc')
tmmx_wus = xa.open_dataarray('tmmx_wus.nc')
vpd_wus = xa.open_dataarray('vpd_wus.nc')
vs_wus = xa.open_dataarray('vs_wus.nc')
# drop stations
pr_clean = pr_wus.drop_sel(n_stations=del_stations)
rmax_clean = rmax_wus.drop_sel(n_stations=del_stations)
rmin_clean = rmin_wus.drop_sel(n_stations=del_stations)
sph_clean = sph_wus.drop_sel(n_stations=del_stations)
srad_clean = srad_wus.drop_sel(n_stations=del_stations)
tmmn_clean = tmmn_wus.drop_sel(n_stations=del_stations)
tmmx_clean = tmmx_wus.drop_sel(n_stations=del_stations)
vpd_clean = vpd_wus.drop_sel(n_stations=del_stations)
vs_clean = vs_wus.drop_sel(n_stations=del_stations)
# save to netcdf
pr_clean.to_netcdf('pr_wus_clean.nc')
rmax_clean.to_netcdf('rmax_wus_clean.nc')
rmin_clean.to_netcdf('rmin_wus_clean.nc')
sph_clean.to_netcdf('sph_wus_clean.nc')
srad_clean.to_netcdf('srad_wus_clean.nc')
tmmn_clean.to_netcdf('tmmn_wus_clean.nc')
tmmx_clean.to_netcdf('tmmx_wus_clean.nc')
vpd_clean.to_netcdf('vpd_wus_clean.nc')
vs_clean.to_netcdf('vs_wus_clean.nc')
