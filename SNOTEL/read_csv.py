import glob

import pandas as pd
import xarray as xa
from tqdm import tqdm

path = 'raw_data/'
x_data = []
for file in tqdm(glob.glob(path + '*')[:]):
    lat = float(file[13:21])
    lon = float(file[22:32])
    # print(file, ' ', lat, ' ', lon)
    data = pd.read_csv(file, names=['year', 'month', 'day', 'accu_precip', 'precip', 'max_temperature',
                                    'min_temperature', 'mean_temperature', 'SWE'], delim_whitespace=True)
    data['time'] = pd.to_datetime(dict(year=data.year, month=data.month, day=data.day))
    data = data.set_index('time')

    # print(data)
    dataset = xa.Dataset().from_dataframe(data)
    dataset = dataset.assign({'longitude': lon, 'latitude': lat})
    x_data.append(dataset)
x_data = xa.concat(x_data, join='outer', dim='n_stations')
print(x_data)
print(x_data.n_stations)
x_data.to_netcdf('raw_WUS_snotel.nc')
