import numpy as np
from tqdm import tqdm
import xarray as xa
import os
import pickle

# Get cold and hot WY
tmin = xa.open_dataarray('gridMET/tmmn_wus_clean.nc')
tmax = xa.open_dataarray('gridMET/tmmx_wus_clean.nc')
tmean = (tmin+tmax)/2
wy_temp = []
wys = np.arange(1982, 2018)
for wy in range(1982, 2018):
    wy_temp.append(tmean.sel(time=slice(str(wy-1)+'-10-01', str(wy)+'-09-30')).mean().data)
sorted_wys = [wy for _, wy in sorted(zip(wy_temp, wys))]
print(wys)
print(sorted_wys) # should be from coolest to hottest. 
print(wy_temp) 
## train with hot:
hot_cool = 'cool'
if hot_cool.upper()=='HOT':
    train_wys = sorted_wys[10:]
    test_wys = sorted_wys[:10]
elif hot_cool.upper()=='COOL':
    train_wys = sorted_wys[:-10]
    test_wys = sorted_wys[-10:]
else:
    print('NOT Valid')

helper = {}


topo = 'SNOTEL/raw_snotel_topo_30m.nc'
# topo = 'mountains/raw_wus_snotel_topo_clean_mountains.nc'
tmp_data = xa.open_dataset(topo)
target_mean = tmp_data.SWE.mean().data
target_std = tmp_data.SWE.std().data

helper['target_mean'] = target_mean
helper['target_std'] = target_std

save_path = 'cross-wy/'+hot_cool.upper()+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
    print('save path created. ')

print(len(train_wys), ' wys for training')
print(len(test_wys), ' wys for testing')

all_train_time = []
for year in train_wys:
    time_slice = slice(str(year)+'-10-01', str(year+1)+'-09-30')
    timestamp = tmean.sel(time=time_slice).time.data
    for time in timestamp:
        all_train_time.append(time)
helper['all_train_time'] = all_train_time

forcings = {'pr': 'gridMET/pr_wus_clean.nc', 'rmax': 'gridMET/rmax_wus_clean.nc', 'rmin': 'gridMET/rmin_wus_clean.nc',
            'sph': 'gridMET/sph_wus_clean.nc', 'srad': 'gridMET/srad_wus_clean.nc', 'tmmn': 'gridMET/tmmn_wus_clean.nc',
            'tmmx': 'gridMET/tmmx_wus_clean.nc', 'vpd': 'gridMET/vpd_wus_clean.nc', 'vs': 'gridMET/vs_wus_clean.nc'}
forcings_data = []
forcing_list = list(forcings.keys())
for forcing in forcing_list:
    forcing_data = xa.open_dataarray(forcings[forcing])
    forcings_data.append(forcing_data)

forcings_mean = []
forcings_std = []
for forcing_data in forcings_data:
    forcings_mean.append(forcing_data.sel(
        time=all_train_time).mean().data)
    forcings_std.append(forcing_data.sel(
        time=all_train_time).std().data)
    
helper['forcings_mean'] = forcings_mean
helper['forcings_std'] = forcings_std

with open(save_path+'helper.p', 'wb') as pfile:
    pickle.dump(helper, pfile)
print('Done')
print(helper)
