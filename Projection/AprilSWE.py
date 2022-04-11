import numpy as np
import xarray as xa
from tqdm import tqdm

pr_hist = xa.open_dataarray("CanESM2-hist-pr.nc")
print(pr_hist.shape)
time_hist = pr_hist.time
pr_rcp85 = xa.open_dataarray("CanESM2-85-pr.nc")
time_85 = pr_rcp85.time
lat = pr_hist.lat
lon = pr_hist.lon
elevation = xa.open_dataset("topo_file_LOCA.nc")
dah = elevation.dah
trasp = elevation.trasp
latitude = elevation.latitude
longitude = elevation.longitude
elevation = elevation.elevation_prism
print(elevation)

# nsidc = xa.open_dataarray("../co_prediction/SWES_NSIDC_all.nc")
nsidc = xa.open_dataarray('../NSIDC/SWES_NSIDC_hist.nc')
nsidc = nsidc.reindex(lat=nsidc.lat[::-1])
nsidc = nsidc.transpose("lat", "lon", "time")
print(nsidc.shape)
print(nsidc.time)
nsidc_lon = nsidc.lon.data
nsidc_lat = nsidc.lat.data
max_nsidc = nsidc.max(dim='time')
print(max_nsidc.shape)

def load_data(model):
    hist_data = np.load("LSTM_proj/" + model + "-hist_0.npy")
    hist_ens = np.zeros(hist_data.shape)
    for i in range(10):
        ens = np.load("LSTM_proj/" + model + "-hist_" + str(i) + ".npy")
        hist_ens += ens
    hist_ens = hist_ens / 10
    hist_ens = xa.DataArray(hist_ens, dims=["lat", "lon", "time"], coords={"lat": lat, "lon": lon, "time": time_hist[179:]})
    rcp_data = np.load("LSTM_proj/" + model + "-85_0.npy")
    rcp_ens = np.zeros(rcp_data.shape)
    for i in range(10):
        ens = np.load("LSTM_proj/" + model + "-85_" + str(i) + ".npy")
        rcp_ens += ens
    rcp_ens = rcp_ens / 10
    rcp_ens = xa.DataArray(rcp_ens, dims=["lat", "lon", "time"], coords={"lat": lat, "lon": lon, "time": time_85[179:]})
    return hist_ens, rcp_ens

def build_xarray(array, lat, lon):
    data = xa.DataArray(array, dims=['latitude', 'longitude'], coords={'latitude': lat, 'longitude': lon})
    return data
def upsample(nsidc, snow):
    snow_xa = build_xarray(snow, dah.lat.data.reshape(-1), dah.lon.data.reshape(-1))
    nsidc_xa = build_xarray(nsidc, nsidc_lat, nsidc_lon)
    # up_nsidc_xa = nsidc_xa.interp_like(snow_xa, 'nearest')
    up_nsidc_xa = np.zeros_like(snow)
    for i, lat in enumerate(dah.lat.data):
        for j, lon in enumerate(dah.lon.data):
            up_nsidc_xa[i, j] = nsidc_xa.sel(longitude=lon, method='nearest').sel(latitude=lat, method='nearest')
    print(up_nsidc_xa.shape)
    up_nsidc_xa = build_xarray(up_nsidc_xa, dah.lat.data.reshape(-1), dah.lon.data.reshape(-1))
    return up_nsidc_xa

MIROC_hist_peak = np.load('Metrics/MIROC_hist_peak.npy')
up_nsidc_max = upsample(max_nsidc, MIROC_hist_peak)

def April_SWE(model):
    hist_ens, rcp_ens = load_data(model)
    hist_april_swe = []
    rcp_april_swe = []
    for year in tqdm(range(1982, 2000)):
        slice_data = hist_ens.sel(time=(str(year) + "-04-01"))
        swe_slice = slice_data.data*up_nsidc_max.data
        hist_april_swe.append(swe_slice.reshape(1, swe_slice.shape[0], swe_slice.shape[1]))
    hist_april_swe = np.concatenate(hist_april_swe, axis=0)
    # hist_april_swe = np.mean(hist_april_swe, axis=0)
    for year in tqdm(range(2072, 2090)):
        slice_data = rcp_ens.sel(time=(str(year) + "-04-01"))
        swe_slice = slice_data.data*up_nsidc_max.data
        rcp_april_swe.append(swe_slice.reshape(1, swe_slice.shape[0], swe_slice.shape[1]))
    rcp_april_swe = np.concatenate(rcp_april_swe, axis=0)
    # rcp_april_swe = np.mean(rcp_april_swe, axis=0)
    return hist_april_swe, rcp_april_swe

models = ['HadGEM2-ES', 'MIROC5', 'EC-EARTH', 'CNRM-CM5', 'CESM-CAM5', 'GFDL-ESM2M']

for model in models:
    hist_april_swe, rcp_april_swe = April_SWE(model)
    np.save('Metrics/'+model+'_HistAprilSWE', hist_april_swe)
    np.save('Metrics/'+model+'_RcpAprilSWE', rcp_april_swe)
    