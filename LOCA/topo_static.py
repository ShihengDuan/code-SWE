import numpy as np
import xarray as xa
import glob
from tqdm import tqdm
import rasterio
from osgeo import gdal
import os

mountains = ['rocky', 'cascade', 'north', 'sierra', 'utah']
mountains = ['utah']
for mountain in mountains:
    origin_forcing = xa.open_dataarray('ori_data/ACCESS1-3/pr_'+mountain+'_1981-2100.nc')
    lat = origin_forcing.lat
    lon = origin_forcing.lon
    num_lat = len(lat)
    num_lon = len(lon)
    elevation_30m = np.zeros((num_lat, num_lon)) # lat, lon
    dah_30m = np.zeros((num_lat, num_lon))
    trasp_30m = np.zeros((num_lat, num_lon))
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

    min_int_lon = int(np.floor(-min_lon))
    min_int_lat = int(np.floor(min_lat))
    print(min_int_lon, min_int_lat)
    max_int_lon = int(np.floor(-max_lon))
    max_int_lat = int(np.floor(max_lat))
    print(max_int_lon, max_int_lat)

    file_pre = '/tempest/duan0000/BAIR/metaearth/data/3dep-seamless/'
    files = []
    for i in range(min_int_lat, max_int_lat+2):
        for j in range(max_int_lon, min_int_lon+2):
            # print(i, j)
            ele = glob.glob(file_pre+'n'+str(i)+'w'+str(j)+'-1/*.tif')[0]
            # print(ele)
            files.append(ele)
    print(len(files))

    files_to_mosaic = files
    g = gdal.Warp("tmp_aspect_slope/all_ele.tif", files_to_mosaic, format="GTiff",
            options=["COMPRESS=LZW", "TILED=YES"]) # if you want
    g = None # Close file and flush to disk
    ele = xa.open_dataarray("tmp_aspect_slope/all_ele.tif")
    gdal.DEMProcessing('tmp_aspect_slope/all_slope.tif', "tmp_aspect_slope/all_ele.tif", 'slope')
    gdal.DEMProcessing('tmp_aspect_slope/all_aspect.tif', "tmp_aspect_slope/all_ele.tif", 'aspect', zeroForFlat=True)
    slope = xa.open_dataarray('tmp_aspect_slope/all_slope.tif')
    aspect = xa.open_dataarray('tmp_aspect_slope/all_aspect.tif')

    # DAH
    asp_max = 202.5*np.pi/180
    dah = np.cos(asp_max-aspect*np.pi/180)*np.arctan(slope*np.pi/180)
    trasp = 0.5 - 0.5*np.cos((np.pi/180)*(aspect-30))
    print('SLOPE: ', slope.shape, ' ASPECT: ', aspect.shape)
    print('DAH: ', dah.shape, ' TRASP: ', trasp.shape)
    print(np.min(dah).data)

    for i, lat in tqdm(enumerate(lats)):
        for j, lon in enumerate(lons):
            longitude[i, j] = lon
            latitude[i, j] = lat
            ele_station = ele.sel(y=lat, x=lon, method='nearest').data[0]
            elevation_30m[i, j] = ele_station
            
            dah_station = dah.sel(y=lat, x=lon, method='nearest').data[0]
            # print(dah_station)
            while np.isnan(dah_station):
                dah = dah.interpolate_na('x', method='linear')
                dah = dah.interpolate_na('y', method='linear')
                dah_station = dah.sel(y=lat, x=lon, method='nearest').data
                print('interpolate DAH')
            dah_30m[i, j] = dah_station
            trasp_station = trasp.sel(y=lat, x=lon, method='nearest').data[0]
            while np.isnan(trasp_station):
                trasp = trasp.interpolate_na('x', method='linear')
                trasp = trasp.interpolate_na('y', method='linear')
                trasp_station = trasp.sel(y=lat, x=lon, method='nearest').data
                print('interpolate TRASP')
            trasp_30m[i, j] = trasp_station

    elevation_30m = xa.DataArray(elevation_30m, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    dah_30m = xa.DataArray(dah_30m, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    trasp_30m = xa.DataArray(trasp_30m, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    longitude = xa.DataArray(longitude, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    latitude = xa.DataArray(latitude, dims=['lat', 'lon'], coords={'lat':origin_forcing.lat, 'lon':origin_forcing.lon})
    static_data = xa.Dataset({
        'dah_30m':dah_30m,
        'trasp_30m':trasp_30m,
        'elevation_30m':elevation_30m,
        'longitude':longitude, 
        'latitude': latitude,
        })
    
    static_data.to_netcdf(mountain+'_topo_file_30m.nc')
