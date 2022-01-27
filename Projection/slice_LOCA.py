import numpy as np
import xarray as xa


def hist_process(model):
    vars = ['pr', 'tasmin', 'tasmax', 'qair', 'shortwave_in', 'ws', 'rh', 'SWE']
    names = ['pr', 'tmmn', 'tmmx', 'sph', 'srad', 'vs', 'rave', 'SWE']
    for i, var in enumerate(vars):
        print(var)
        data = xa.open_dataarray('../LOCA/' + model + '-hist-' + var + '.nc')
        if i > 2:
            data = data.sel(Time=slice(np.datetime64('1981-10-01'), np.datetime64('2001-09-30')))
            data = data.sel(Lon=slice(-108.97499996666667, -104.51666663333334),
                            Lat=slice(35.025000000000006, 41.983333333333334))
            data_lon = data.Lon.data
            data_lat = data.Lat.data
            time = data.Time.data
        else:
            data['time'] = data.indexes['time'].normalize()
            data = data.sel(time=slice(np.datetime64('1981-10-01'), np.datetime64('2001-09-30')))
            data['lon'] = data['lon'] - 360
            data = data.sel(lon=slice(-108.97499996666667, -104.51666663333334),
                            lat=slice(35.025000000000006, 41.983333333333334))
            data_lon = data.lon.data
            data_lat = data.lat.data
            time = data.time.data
        print(data.shape)
        slice_data = xa.DataArray(data=data.data, dims=['time', 'lat', 'lon'],
                                  coords={'lat': data_lat, 'lon': data_lon, 'time': time})
        slice_data.to_netcdf(model + '-hist-' + names[i] + '.nc')


def rcp_process(model):
    vars = ['pr', 'tasmin', 'tasmax', 'qair', 'shortwave_in', 'ws', 'rh', 'SWE']
    names = ['pr', 'tmmn', 'tmmx', 'sph', 'srad', 'vs', 'rave', 'SWE']
    for i, var in enumerate(vars):
        print(var)
        data = xa.open_dataarray('../LOCA/' + model + '-85-' + var + '.nc')
        if i > 2:
            data = data.sel(Time=slice(np.datetime64('2071-10-01'), np.datetime64('2091-09-30')))
            data = data.sel(Lon=slice(-108.97499996666667, -104.51666663333334),
                            Lat=slice(35.025000000000006, 41.983333333333334))
            data_lon = data.Lon.data
            data_lat = data.Lat.data
            time = data.Time.data
        else:
            data['time'] = data.indexes['time'].normalize()
            data = data.sel(time=slice(np.datetime64('2071-10-01'), np.datetime64('2091-09-30')))
            data['lon'] = data['lon'] - 360
            data = data.sel(lon=slice(-108.97499996666667, -104.51666663333334),
                            lat=slice(35.025000000000006, 41.983333333333334))
            data_lon = data.lon.data
            data_lat = data.lat.data
            time = data.time.data
        print(data.shape)
        slice_data = xa.DataArray(data=data.data, dims=['time', 'lat', 'lon'],
                                  coords={'lat': data_lat, 'lon': data_lon, 'time': time})
        slice_data.to_netcdf(model + '-85-' + names[i] + '.nc')


models = ['CanESM2', 'CESM-CAM5', 'CNRM-CM5', 'GFDL-ESM2M', 'HadGEM2-ES', 'MIROC5']
models = ['EC-EARTH']
for model in models:
    print(model)
    # hist_process(model)
    rcp_process(model)
