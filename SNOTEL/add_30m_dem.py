import numpy as np
import xarray as xa
import glob
from tqdm import tqdm
import rasterio
from osgeo import gdal
import os

raw_snotel = xa.open_dataset('raw_wus_snotel_topo_clean.nc')
print(raw_snotel)
elevation_30m = []
dah_30m = []
trasp_30m = []

for station in range(581):
# for station in [2]:
    station_lon = raw_snotel.longitude.isel(n_stations=station).data
    station_lat = raw_snotel.latitude.isel(n_stations=station).data
    int_lon = int(np.floor(-station_lon))
    int_lat = int(np.floor(station_lat))
    print(station, station_lon, station_lat, int_lon, int_lat)
    file_pre = '/tempest/duan0000/BAIR/metaearth/data/3dep-seamless/'
    ele1 = glob.glob(file_pre+'n'+str(int_lat)+'w'+str(int_lon)+'-1/*.tif')[0]
    ele2 = glob.glob(file_pre+'n'+str(int_lat+1)+'w'+str(int_lon+1)+'-1/*.tif')[0]
    ele3 = glob.glob(file_pre+'n'+str(int_lat)+'w'+str(int_lon+1)+'-1/*.tif')[0]
    ele4 = glob.glob(file_pre+'n'+str(int_lat+1)+'w'+str(int_lon)+'-1/*.tif')[0]
    print('ELE1: ', ele1)
    print('ELE2: ', ele2)
    print('ELE3: ', ele3)
    print('ELE4: ', ele4)
    # Merge DEM with GDAL
    files_to_mosaic = [ele1, ele2, ele3, ele4]
    g = gdal.Warp("tmp_aspect_slope/"+str(station)+"_ele.tif", files_to_mosaic, format="GTiff",
              options=["COMPRESS=LZW", "TILED=YES"]) # if you want
    g = None # Close file and flush to disk
    ele = xa.open_dataarray("tmp_aspect_slope/"+str(station)+"_ele.tif")
    print('ELE x: ', np.min(ele.x.data), ' ', np.max(ele.x.data))
    print('ELE y: ', np.min(ele.y.data), ' ', np.max(ele.y.data))
    ele_station = ele.sel(y=station_lat, x=station_lon, method='nearest').data[0]
    elevation_30m.append(ele_station)
    # Slope and Aspect
    gdal.DEMProcessing('tmp_aspect_slope/'+str(station)+'_slope.tif', "tmp_aspect_slope/"+str(station)+"_ele.tif", 'slope')
    gdal.DEMProcessing('tmp_aspect_slope/'+str(station)+'temp_aspect.tif', "tmp_aspect_slope/"+str(station)+"_ele.tif", 'aspect', zeroForFlat=True)
    slope = xa.open_dataarray('tmp_aspect_slope/'+str(station)+'_slope.tif')
    aspect = xa.open_dataarray('tmp_aspect_slope/'+str(station)+'temp_aspect.tif')
    
    # DAH
    asp_max = 202.5*np.pi/180
    dah = np.cos(asp_max-aspect*np.pi/180)*np.arctan(slope*np.pi/180)
    trasp = 0.5 - 0.5*np.cos((np.pi/180)*(aspect-30))
    print('SLOPE: ', slope.shape, ' ASPECT: ', aspect.shape)
    print('DAH: ', dah.shape, ' TRASP: ', trasp.shape)
    print('Y target and range: ', station_lat, ' ', np.max(dah.y).data, ' ', np.min(dah.y).data)
    print('X target and range: ', station_lon, ' ', np.max(dah.x).data, ' ', np.min(dah.x).data)
    print(np.min(dah).data)
    
    dah_station = dah.sel(y=station_lat, x=station_lon, method='nearest').data[0]
    print(dah_station)
    while np.isnan(dah_station):
        dah = dah.interpolate_na('x', method='linear')
        dah = dah.interpolate_na('y', method='linear')
        dah_station = dah.sel(y=station_lat, x=station_lon, method='nearest').data
        print('interpolate DAH')
    dah_30m.append(dah_station)
    trasp_station = trasp.sel(y=station_lat, x=station_lon, method='nearest').data[0]
    while np.isnan(trasp_station):
        trasp = trasp.interpolate_na('x', method='linear')
        trasp = trasp.interpolate_na('y', method='linear')
        trasp_station = trasp.sel(y=station_lat, x=station_lon, method='nearest').data
        print('interpolate TRASP')
    trasp_30m.append(trasp_station)
    del dah, trasp, ele

elevation_30m = xa.DataArray(elevation_30m, dims=['n_stations'], coords={'n_stations':raw_snotel.n_stations})
dah_30m = xa.DataArray(dah_30m, dims=['n_stations'], coords={'n_stations':raw_snotel.n_stations})
trasp_30m = xa.DataArray(trasp_30m, dims=['n_stations'], coords={'n_stations':raw_snotel.n_stations})
raw_snotel = raw_snotel.drop('elevation_prism')
raw_snotel = raw_snotel.drop('dah')
raw_snotel = raw_snotel.drop('trasp')
raw_snotel = raw_snotel.assign({
    'dah_30m':dah_30m,
    'trasp_30m':trasp_30m,
    'elevation_30m':elevation_30m,
    })
raw_snotel.to_netcdf('raw_snotel_topo_30m.nc')
