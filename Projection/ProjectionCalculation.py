import cartopy
import matplotlib.ticker as mticker
import numpy as np
import xarray as xa
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm

def load_loca(model):
    hist_data = xa.open_dataarray(model+'-hist-SWE.nc')
    rcp_data = xa.open_dataarray(model+'-85-SWE.nc')
    hist_data = hist_data.transpose("lat", "lon", "time")
    rcp_data = rcp_data.transpose("lat", "lon", "time")
    return hist_data, rcp_data

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
'''
EC_hist, EC_rcp = load_data("EC-EARTH")
CESM_hist, CESM_rcp = load_data("CESM-CAM5")
CNRM_hist, CNRM_rcp = load_data("CNRM-CM5")
GFDL_hist, GFDL_rcp = load_data("GFDL-ESM2M")
HadGEM_hist, HadGEM_rcp = load_data("HadGEM2-ES")

MIROC_hist, MIROC_rcp = load_data("MIROC5")


EC_hist_loca, EC_rcp_loca = load_loca("EC-EARTH")
CESM_hist_loca, CESM_rcp_loca = load_loca("CESM-CAM5")
CNRM_hist_loca, CNRM_rcp_loca = load_loca("CNRM-CM5")
GFDL_hist_loca, GFDL_rcp_loca = load_loca("GFDL-ESM2M")
HadGEM_hist_loca, HadGEM_rcp_loca = load_loca("HadGEM2-ES")
MIROC_hist_loca, MIROC_rcp_loca = load_loca("MIROC5")
print(CESM_hist_loca.shape, " ", CESM_rcp_loca.shape)
'''
def acc_melt_day(slice_data):
    acc_days = np.zeros((112, 72))
    melt_days = np.zeros((112, 72))
    peaks = np.zeros((112, 72))
    peak_days = np.zeros((112, 72))
    for i in range(112):
        for j in range(72):
            series = slice_data[i, j].data
            peaks[i, j] = np.max(series[:300])
            # print(peaks[i, j])
            diff_series = series - peaks[i, j] * 0.1  # 10% peak of SWE
            # acc_days[i, j] = np.argmin(abs(diff_series[:180]))
            if (diff_series[: np.argmax(series[:300])] < 0).any():
                acc_days[i, j] = np.where(diff_series[: np.argmax(series[:300])] < 0)[0][-1]
            else:
                acc_days[i, j] = 0

            peak_days[i, j] = np.where(series == np.max(series[:300]))[0][0]
            # melt_days[i, j] = len(series)-np.argmin(abs(diff_series[180::-1]))-1
            # melt_days[i, j] = peak_days[i, j] + np.argmin(abs(diff_series[np.argmax(series) :]))
            if (diff_series[np.argmax(series[:300]) :] <= 0).any():
                melt_days[i, j] = np.where(diff_series[np.argmax(series[:300]) :] <= 0)[0][0] + peak_days[i, j]
            else:
                melt_days[i, j] = 365
            # print(i, j)
    return acc_days, melt_days, peaks, peak_days
def calculate_model(swe_hist, swe_rcp):
    hist_acc = []
    hist_melt = []
    hist_days = []
    hist_peaks = []
    for year in tqdm(range(1982, 2000)):
        slice_data = swe_hist.sel(time=slice(str(year) + "-10-01", str(year + 1) + "-09-30"))
        acc_days, melt_days, peaks, peak_days = acc_melt_day(slice_data)
        hist_acc.append(acc_days)
        hist_melt.append(melt_days)
        hist_days.append(peak_days)
        hist_peaks.append(peaks)
    hist_acc = np.array(hist_acc)
    hist_melt = np.array(hist_melt)
    hist_days = np.array(hist_days)
    hist_peaks = np.array(hist_peaks)

    rcp_acc = []
    rcp_melt = []
    rcp_days = []
    rcp_peaks = []
    for year in tqdm(range(2072, 2090)):
        slice_data = swe_rcp.sel(time=slice(str(year) + "-10-01", str(year + 1) + "-09-30"))
        acc_days, melt_days, peaks, peak_days = acc_melt_day(slice_data)
        rcp_acc.append(acc_days)
        rcp_melt.append(melt_days)
        rcp_days.append(peak_days)
        rcp_peaks.append(peaks)
    rcp_acc = np.array(rcp_acc)
    rcp_melt = np.array(rcp_melt)
    rcp_days = np.array(rcp_days)
    rcp_peaks = np.array(rcp_peaks)

    snow_range_hist = np.mean(hist_melt - hist_acc, axis=0)
    snow_range_rcp = np.mean(rcp_melt - rcp_acc, axis=0)
    snow_acc_hist = np.mean(hist_acc, axis=0)
    snow_acc_rcp = np.mean(rcp_acc, axis=0)
    snow_melt_hist = np.mean(hist_melt, axis=0)
    snow_melt_rcp = np.mean(rcp_melt, axis=0)
    snow_peak_hist = np.mean(hist_peaks, axis=0)
    snow_peak_rcp = np.mean(rcp_peaks, axis=0)
    snow_date_hist = np.mean(hist_days, axis=0)
    snow_date_rcp = np.mean(rcp_days, axis=0)
    snow_accdate_hist = np.mean(hist_days - hist_acc, axis=0)
    snow_accdate_rcp = np.mean(rcp_days - rcp_acc, axis=0)
    snow_meltdate_hist = np.mean(hist_melt - hist_days, axis=0)
    snow_meltdate_rcp = np.mean(rcp_melt - rcp_days, axis=0)

    return (
        snow_range_hist,
        snow_range_rcp,
        snow_acc_hist,
        snow_acc_rcp,
        snow_melt_hist,
        snow_melt_rcp,
        snow_peak_hist,
        snow_peak_rcp,
        snow_date_hist,
        snow_date_rcp,
    )
'''
(
    EC_hist_range,
    EC_rcp_range,
    EC_hist_acc,
    EC_rcp_acc,
    EC_hist_melt,
    EC_rcp_melt,
    EC_hist_peak,
    EC_rcp_peak,
    EC_date_hist,
    EC_date_rcp,
) = calculate_model(EC_hist, EC_rcp)
(
    CESM_hist_range,
    CESM_rcp_range,
    CESM_hist_acc,
    CESM_rcp_acc,
    CESM_hist_melt,
    CESM_rcp_melt,
    CESM_hist_peak,
    CESM_rcp_peak,
    CESM_date_hist,
    CESM_date_rcp,
) = calculate_model(CESM_hist, CESM_rcp)
(
    CNRM_hist_range,
    CNRM_rcp_range,
    CNRM_hist_acc,
    CNRM_rcp_acc,
    CNRM_hist_melt,
    CNRM_rcp_melt,
    CNRM_hist_peak,
    CNRM_rcp_peak,
    CNRM_date_hist,
    CNRM_date_rcp,
) = calculate_model(CNRM_hist, CNRM_rcp)
(
    GFDL_hist_range,
    GFDL_rcp_range,
    GFDL_hist_acc,
    GFDL_rcp_acc,
    GFDL_hist_melt,
    GFDL_rcp_melt,
    GFDL_hist_peak,
    GFDL_rcp_peak,
    GFDL_date_hist,
    GFDL_date_rcp,
) = calculate_model(GFDL_hist, GFDL_rcp)
(
    HadGEM_hist_range,
    HadGEM_rcp_range,
    HadGEM_hist_acc,
    HadGEM_rcp_acc,
    HadGEM_hist_melt,
    HadGEM_rcp_melt,
    HadGEM_hist_peak,
    HadGEM_rcp_peak,
    HadGEM_date_hist,
    HadGEM_date_rcp,
) = calculate_model(HadGEM_hist, HadGEM_rcp)

(
    MIROC_hist_range,
    MIROC_rcp_range,
    MIROC_hist_acc,
    MIROC_rcp_acc,
    MIROC_hist_melt,
    MIROC_rcp_melt,
    MIROC_hist_peak,
    MIROC_rcp_peak,
    MIROC_date_hist,
    MIROC_date_rcp,
) = calculate_model(MIROC_hist, MIROC_rcp)

np.save('Metrics/MIROC_hist_range', MIROC_hist_range)
np.save('Metrics/MIROC_rcp_range', MIROC_rcp_range)
np.save('Metrics/MIROC_hist_acc', MIROC_hist_acc)
np.save('Metrics/MIROC_rcp_acc', MIROC_rcp_acc)
np.save('Metrics/MIROC_hist_melt', MIROC_hist_melt)
np.save('Metrics/MIROC_rcp_melt', MIROC_rcp_melt)
np.save('Metrics/MIROC_hist_peak', MIROC_hist_peak)
np.save('Metrics/MIROC_rcp_peak', MIROC_rcp_peak)
np.save('Metrics/MIROC_date_hist', MIROC_date_hist)
np.save('Metrics/MIROC_date_rcp', MIROC_date_rcp)

np.save('Metrics/HadGEM_hist_range', HadGEM_hist_range)
np.save('Metrics/HadGEM_rcp_range', HadGEM_rcp_range)
np.save('Metrics/HadGEM_hist_acc', HadGEM_hist_acc)
np.save('Metrics/HadGEM_rcp_acc', HadGEM_rcp_acc)
np.save('Metrics/HadGEM_hist_melt', HadGEM_hist_melt)
np.save('Metrics/HadGEM_rcp_melt', HadGEM_rcp_melt)
np.save('Metrics/HadGEM_hist_peak', HadGEM_hist_peak)
np.save('Metrics/HadGEM_rcp_peak', HadGEM_rcp_peak)
np.save('Metrics/HadGEM_date_hist', HadGEM_date_hist)
np.save('Metrics/HadGEM_date_rcp', HadGEM_date_rcp)

np.save('Metrics/GFDL_hist_range', GFDL_hist_range)
np.save('Metrics/GFDL_rcp_range', GFDL_rcp_range)
np.save('Metrics/GFDL_hist_acc', GFDL_hist_acc)
np.save('Metrics/GFDL_rcp_acc', GFDL_rcp_acc)
np.save('Metrics/GFDL_hist_melt', GFDL_hist_melt)
np.save('Metrics/GFDL_rcp_melt', GFDL_rcp_melt)
np.save('Metrics/GFDL_hist_peak', GFDL_hist_peak)
np.save('Metrics/GFDL_rcp_peak', GFDL_rcp_peak)
np.save('Metrics/GFDL_date_hist', GFDL_date_hist)
np.save('Metrics/GFDL_date_rcp', GFDL_date_rcp)

np.save('Metrics/CNRM_hist_range', CNRM_hist_range)
np.save('Metrics/CNRM_rcp_range', CNRM_rcp_range)
np.save('Metrics/CNRM_hist_acc', CNRM_hist_acc)
np.save('Metrics/CNRM_rcp_acc', CNRM_rcp_acc)
np.save('Metrics/CNRM_hist_melt', CNRM_hist_melt)
np.save('Metrics/CNRM_rcp_melt', CNRM_rcp_melt)
np.save('Metrics/CNRM_hist_peak', CNRM_hist_peak)
np.save('Metrics/CNRM_rcp_peak', CNRM_rcp_peak)
np.save('Metrics/CNRM_date_hist', CNRM_date_hist)
np.save('Metrics/CNRM_date_rcp', CNRM_date_rcp)

np.save('Metrics/CESM_hist_range', CESM_hist_range)
np.save('Metrics/CESM_rcp_range', CESM_rcp_range)
np.save('Metrics/CESM_hist_acc', CESM_hist_acc)
np.save('Metrics/CESM_rcp_acc', CESM_rcp_acc)
np.save('Metrics/CESM_hist_melt', CESM_hist_melt)
np.save('Metrics/CESM_rcp_melt', CESM_rcp_melt)
np.save('Metrics/CESM_hist_peak', CESM_hist_peak)
np.save('Metrics/CESM_rcp_peak', CESM_rcp_peak)
np.save('Metrics/CESM_date_hist', CESM_date_hist)
np.save('Metrics/CESM_date_rcp', CESM_date_rcp)

np.save('Metrics/EC_hist_range', EC_hist_range)
np.save('Metrics/EC_rcp_range', EC_rcp_range)
np.save('Metrics/EC_hist_acc', EC_hist_acc)
np.save('Metrics/EC_rcp_acc', EC_rcp_acc)
np.save('Metrics/EC_hist_melt', EC_hist_melt)
np.save('Metrics/EC_rcp_melt', EC_rcp_melt)
np.save('Metrics/EC_hist_peak', EC_hist_peak)
np.save('Metrics/EC_rcp_peak', EC_rcp_peak)
np.save('Metrics/EC_date_hist', EC_date_hist)
np.save('Metrics/EC_date_rcp', EC_date_rcp)


(
    EC_hist_range,
    EC_rcp_range,
    EC_hist_acc,
    EC_rcp_acc,
    EC_hist_melt,
    EC_rcp_melt,
    EC_hist_peak,
    EC_rcp_peak,
    EC_date_hist,
    EC_date_rcp,
) = calculate_model(EC_hist_loca, EC_rcp_loca)
(
    CESM_hist_range,
    CESM_rcp_range,
    CESM_hist_acc,
    CESM_rcp_acc,
    CESM_hist_melt,
    CESM_rcp_melt,
    CESM_hist_peak,
    CESM_rcp_peak,
    CESM_date_hist,
    CESM_date_rcp,
) = calculate_model(CESM_hist_loca, CESM_rcp_loca)
(
    CNRM_hist_range,
    CNRM_rcp_range,
    CNRM_hist_acc,
    CNRM_rcp_acc,
    CNRM_hist_melt,
    CNRM_rcp_melt,
    CNRM_hist_peak,
    CNRM_rcp_peak,
    CNRM_date_hist,
    CNRM_date_rcp,
) = calculate_model(CNRM_hist_loca, CNRM_rcp_loca)
(
    GFDL_hist_range,
    GFDL_rcp_range,
    GFDL_hist_acc,
    GFDL_rcp_acc,
    GFDL_hist_melt,
    GFDL_rcp_melt,
    GFDL_hist_peak,
    GFDL_rcp_peak,
    GFDL_date_hist,
    GFDL_date_rcp,
) = calculate_model(GFDL_hist_loca, GFDL_rcp_loca)
(
    HadGEM_hist_range,
    HadGEM_rcp_range,
    HadGEM_hist_acc,
    HadGEM_rcp_acc,
    HadGEM_hist_melt,
    HadGEM_rcp_melt,
    HadGEM_hist_peak,
    HadGEM_rcp_peak,
    HadGEM_date_hist,
    HadGEM_date_rcp,
) = calculate_model(HadGEM_hist_loca, HadGEM_rcp_loca)

(
    MIROC_hist_range,
    MIROC_rcp_range,
    MIROC_hist_acc,
    MIROC_rcp_acc,
    MIROC_hist_melt,
    MIROC_rcp_melt,
    MIROC_hist_peak,
    MIROC_rcp_peak,
    MIROC_date_hist,
    MIROC_date_rcp,
) = calculate_model(MIROC_hist_loca, MIROC_rcp_loca)

np.save('Metrics/MIROC_hist_range_loca', MIROC_hist_range)
np.save('Metrics/MIROC_rcp_range_loca', MIROC_rcp_range)
np.save('Metrics/MIROC_hist_acc_loca', MIROC_hist_acc)
np.save('Metrics/MIROC_rcp_acc_loca', MIROC_rcp_acc)
np.save('Metrics/MIROC_hist_melt_loca', MIROC_hist_melt)
np.save('Metrics/MIROC_rcp_melt_loca', MIROC_rcp_melt)
np.save('Metrics/MIROC_hist_peak_loca', MIROC_hist_peak)
np.save('Metrics/MIROC_rcp_peak_loca', MIROC_rcp_peak)
np.save('Metrics/MIROC_date_hist_loca', MIROC_date_hist)
np.save('Metrics/MIROC_date_rcp_loca', MIROC_date_rcp)

np.save('Metrics/HadGEM_hist_range_loca', HadGEM_hist_range)
np.save('Metrics/HadGEM_rcp_range_loca', HadGEM_rcp_range)
np.save('Metrics/HadGEM_hist_acc_loca', HadGEM_hist_acc)
np.save('Metrics/HadGEM_rcp_acc_loca', HadGEM_rcp_acc)
np.save('Metrics/HadGEM_hist_melt_loca', HadGEM_hist_melt)
np.save('Metrics/HadGEM_rcp_melt_loca', HadGEM_rcp_melt)
np.save('Metrics/HadGEM_hist_peak_loca', HadGEM_hist_peak)
np.save('Metrics/HadGEM_rcp_peak_loca', HadGEM_rcp_peak)
np.save('Metrics/HadGEM_date_hist_loca', HadGEM_date_hist)
np.save('Metrics/HadGEM_date_rcp_loca', HadGEM_date_rcp)

np.save('Metrics/GFDL_hist_range_loca', GFDL_hist_range)
np.save('Metrics/GFDL_rcp_range_loca', GFDL_rcp_range)
np.save('Metrics/GFDL_hist_acc_loca', GFDL_hist_acc)
np.save('Metrics/GFDL_rcp_acc_loca', GFDL_rcp_acc)
np.save('Metrics/GFDL_hist_melt_loca', GFDL_hist_melt)
np.save('Metrics/GFDL_rcp_melt_loca', GFDL_rcp_melt)
np.save('Metrics/GFDL_hist_peak_loca', GFDL_hist_peak)
np.save('Metrics/GFDL_rcp_peak_loca', GFDL_rcp_peak)
np.save('Metrics/GFDL_date_hist_loca', GFDL_date_hist)
np.save('Metrics/GFDL_date_rcp_loca', GFDL_date_rcp)

np.save('Metrics/CNRM_hist_range_loca', CNRM_hist_range)
np.save('Metrics/CNRM_rcp_range_loca', CNRM_rcp_range)
np.save('Metrics/CNRM_hist_acc_loca', CNRM_hist_acc)
np.save('Metrics/CNRM_rcp_acc_loca', CNRM_rcp_acc)
np.save('Metrics/CNRM_hist_melt_loca', CNRM_hist_melt)
np.save('Metrics/CNRM_rcp_melt_loca', CNRM_rcp_melt)
np.save('Metrics/CNRM_hist_peak_loca', CNRM_hist_peak)
np.save('Metrics/CNRM_rcp_peak_loca', CNRM_rcp_peak)
np.save('Metrics/CNRM_date_hist_loca', CNRM_date_hist)
np.save('Metrics/CNRM_date_rcp_loca', CNRM_date_rcp)

np.save('Metrics/CESM_hist_range_loca', CESM_hist_range)
np.save('Metrics/CESM_rcp_range_loca', CESM_rcp_range)
np.save('Metrics/CESM_hist_acc_loca', CESM_hist_acc)
np.save('Metrics/CESM_rcp_acc_loca', CESM_rcp_acc)
np.save('Metrics/CESM_hist_melt_loca', CESM_hist_melt)
np.save('Metrics/CESM_rcp_melt_loca', CESM_rcp_melt)
np.save('Metrics/CESM_hist_peak_loca', CESM_hist_peak)
np.save('Metrics/CESM_rcp_peak_loca', CESM_rcp_peak)
np.save('Metrics/CESM_date_hist_loca', CESM_date_hist)
np.save('Metrics/CESM_date_rcp_loca', CESM_date_rcp)

np.save('Metrics/EC_hist_range_loca', EC_hist_range)
np.save('Metrics/EC_rcp_range_loca', EC_rcp_range)
np.save('Metrics/EC_hist_acc_loca', EC_hist_acc)
np.save('Metrics/EC_rcp_acc_loca', EC_rcp_acc)
np.save('Metrics/EC_hist_melt_loca', EC_hist_melt)
np.save('Metrics/EC_rcp_melt_loca', EC_rcp_melt)
np.save('Metrics/EC_hist_peak_loca', EC_hist_peak)
np.save('Metrics/EC_rcp_peak_loca', EC_rcp_peak)
np.save('Metrics/EC_date_hist_loca', EC_date_hist)
np.save('Metrics/EC_date_rcp_loca', EC_date_rcp)

'''
## NSIDC

nsidc = xa.open_dataarray('../NSIDC/SWES_NSIDC_hist.nc')
nsidc = nsidc.reindex(lat=nsidc.lat[::-1])
nsidc = nsidc.transpose("lat", "lon", "time")

def acc_melt_day_nsidc(slice_data):
    acc_days = np.zeros((167, 107))
    melt_days = np.zeros((167, 107))
    peaks = np.zeros((167, 107))
    peak_days = np.zeros((167, 107))
    for i in range(167):
        for j in range(107):
            series = slice_data[i, j].data
            peaks[i, j] = np.max(series)
            if np.isnan(peaks[i, j]):
                acc_days[i, j] = np.nan
                peak_days[i, j] = np.nan
                melt_days[i, j] = np.nan
            else:
                diff_series = series - peaks[i, j] * 0.1  # 10% peak of SWE
                acc_days[i, j] = np.argmin(abs(diff_series[:180]))
                peak_days[i, j] = np.argmax(series)
                melt_days[i, j] = peak_days[i, j] + np.argmin(abs(diff_series[np.argmax(series) :]))

    return acc_days, melt_days, peaks, peak_days
hist_acc_nsidc = []
hist_melt_nsidc = []
hist_days_nsidc = []
hist_peaks_nsidc = []
for year in tqdm(range(1982, 1999)):
    slice_data = nsidc.sel(time=slice(str(year) + "-10-01", str(year + 1) + "-09-30"))
    acc_days, melt_days, peaks, peak_days = acc_melt_day_nsidc(slice_data)
    hist_acc_nsidc.append(acc_days)
    hist_melt_nsidc.append(melt_days)
    hist_days_nsidc.append(peak_days)
    hist_peaks_nsidc.append(peaks)
hist_acc_nsidc = np.array(hist_acc_nsidc)
hist_melt_nsidc = np.array(hist_melt_nsidc)
hist_days_nsidc = np.array(hist_days_nsidc)
hist_peaks_nsidc = np.array(hist_peaks_nsidc)
np.save('Metrics/NSIDC_date_hist', hist_days_nsidc)
np.save('Metrics/NSIDC_hist_acc', hist_acc_nsidc)
np.save('Metrics/NSIDC_hist_melt', hist_melt_nsidc)
np.save('Metrics/NSIDC_hist_peak', hist_peaks_nsidc)
