from typing import Dict

import numpy as np
import torch
import xarray as xa
from torch.utils.data import Dataset
from tqdm import tqdm

def find_snow5km(lon, lat, snowcover: xa.DataArray):
    lat5km = snowcover.lat.data
    lon5km = snowcover.lon.data
    difflat = np.abs(lat5km - lat)
    difflon = np.abs(lon5km - lon)
    lon_ind = np.argmin(difflon)
    lat_ind = np.argmin(difflat)
    snow_series = snowcover.isel(lat=lat_ind, lon=lon_ind).data
    return snow_series


snow_cover = '/tempest/duan0000/SWE/NSIDC/MODIS/Aqua2008-2018_5km_snow_cover.nc'


class SWEDataset(Dataset):
    def __init__(self, nc_file, dyn_inputs, attributions, target, window_size,
                 station, snowcover=snow_cover, mode='Train', relu_flag=False):  # single station dataset.
        super(SWEDataset, self).__init__()
        self.relu_flag = relu_flag
        self.station = station
        self.window_size = window_size
        self.dyn_inputs = dyn_inputs
        self.attributions = attributions
        if snowcover == None:
            self.snow_flag = False
        else:
            self.snow_flag = True
            self.snow_cover = xa.open_dataarray(snowcover)
        data = xa.open_dataset(nc_file)
        self.lon = data.longitude.sel(n_stations=self.station).data
        # for snow extraction.
        self.lat = data.latitude.sel(n_stations=self.station).data
        # attribution preprocess
        attr_data = data[attributions]  # all attributions for normalization.
        attr_mean = attr_data.mean(skipna=True)
        attr_std = attr_data.std(skipna=True)
        attr_norm = (attr_data - attr_mean) / attr_std
        attr_norm = attr_norm.sel(n_stations=station)  # select this station
        x_attr = np.array([attr_norm[feature]
                          for feature in attributions]).astype('float32')
        x_attr = torch.from_numpy(x_attr)  # (6, )
        self.x_attr = x_attr.view(x_attr.nelement())  # (6)
        # dynamic data preprocess. Only for this station.
        self.dyn_data = data[dyn_inputs]
        self.target_data = data[target]
        self._split_normalize()  # normalization for all stations.

        self.mode = mode
        if self.snow_flag:
            # get the specific station.
            self.dyn_norm, self.tar_norm, snow_series = self._preprocess()
        else:
            self.dyn_norm, self.tar_norm = self._preprocess()

        self.x_d = self._get_feature_array(
            self.dyn_norm, dyn_inputs)  # xarray2numpy
        if self.snow_flag:
            # add snow_cover series.
            self.x_d = np.concatenate((self.x_d, snow_series), axis=1)
        self.y = self._get_feature_array(self.tar_norm, target)
        # iterate through features. fill in NAN with minimum
        for c in range(self.x_d.shape[1]):
            x_min = np.nanmin(self.x_d[:, c])
            # replace NAN.
            self.x_d[:, c] = np.nan_to_num(self.x_d[:, c], nan=0)
        # standard deviation for training loss function.
        qstd = np.array(np.nanstd(self.y))
        self.qstd = torch.from_numpy(qstd.astype('float32'))
        self.x_d_new, self.y_new = self._reshape()
        # delete y nan samples:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_new[:, -1, 0]))[0]
        self.y_new = np.delete(self.y_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d_new, self.y_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_new.astype('float32')))

        self.idx = len(idx)

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[feature].values for feature in features]).T
        return x

    def _reshape(self):
        self.n_samples = self.x_d.shape[0] - self.window_size + 1
        n_features = len(self.dyn_inputs)
        if self.snow_flag:
            n_features = n_features + 1
        x_d_new = np.zeros((self.n_samples, self.window_size, n_features))
        y_new = np.zeros((self.n_samples, self.window_size, 1))
        for i in range(self.n_samples):
            x_d_new[i, :, :] = self.x_d[i:i + self.window_size, :]
            y_new[i, :, :] = self.y[i:i + self.window_size, :]
        return x_d_new, y_new

    def __getitem__(self, item):
        return self.x_d_new[item], self.x_attr, self.y_new[item], self.qstd

    def __len__(self):
        return self.y_new.shape[0]

    # 11 years in total, 4 train, 3 val, 4 test.
    def _split_normalize(self):  # dyn and target for all stations.
        self.dyn_train = self.dyn_data.sel(day=np.arange(1, 1462))  # 1~1461
        self.dyn_val = self.dyn_data.sel(
            day=np.arange(1462, 2558))  # 1462~2557
        self.dyn_test = self.dyn_data.sel(
            day=np.arange(2558, 4019))  # 2558~4018
        self.dyn_mean = self.dyn_train.mean(skipna=True)
        self.dyn_std = self.dyn_train.std(skipna=True)
        self.target_train = self.target_data.sel(day=np.arange(1, 1462))
        self.target_val = self.target_data.sel(day=np.arange(1462, 2558))
        self.target_test = self.target_data.sel(day=np.arange(2558, 4019))
        self.target_mean = self.target_train.mean(skipna=True)
        self.target_std = self.target_train.std(skipna=True)

    def _preprocess(self):
        if self.snow_flag:
            snow_series = find_snow5km(
                lon=self.lon, lat=self.lat, snowcover=self.snow_cover)
            snow_series = snow_series / 100  # percentage
            snow_series = snow_series.reshape(-1, 1)

        if self.mode.upper() == 'TRAIN':
            dyn_norm = (self.dyn_train - self.dyn_mean) / self.dyn_std
            tar_norm = (self.target_train - self.target_mean) / self.target_std
            if self.snow_flag:
                snow_series = snow_series[0:1461]
            if self.relu_flag:
                tar_norm = self.target_train / self.target_std
        elif self.mode.upper() == 'VAL':
            dyn_norm = (self.dyn_val - self.dyn_mean) / self.dyn_std
            tar_norm = (self.target_val - self.target_mean) / self.target_std
            if self.snow_flag:
                snow_series = snow_series[1461:2557]
            if self.relu_flag:
                tar_norm = self.target_val / self.target_std
        elif self.mode.upper() == 'TEST':
            dyn_norm = (self.dyn_test - self.dyn_mean) / self.dyn_std
            tar_norm = (self.target_test - self.target_mean) / self.target_std
            if self.snow_flag:
                snow_series = snow_series[2557:4018]
            if self.relu_flag:
                tar_norm = self.target_test / self.target_std
        else:
            print(self.mode.upper())
        dyn_norm = dyn_norm.sel(n_stations=self.station)
        tar_norm = tar_norm.sel(n_stations=self.station)
        if self.snow_flag:
            return dyn_norm, tar_norm, snow_series
        else:
            return dyn_norm, tar_norm


class gridMETRelativeStation(Dataset):  # predict percentage of MAX SWE.
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train'):
        super(gridMETRelativeStation, self).__init__()
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        idx = np.where(np.isinf(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1]))
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_max = self.y_log.sel(time=slice(
            '1980-10-01', '1999-09-30')).max(dim='time', skipna=True)  # xarray
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_max = self.target_max.isel(n_stations=self.station)
            # relative to the maximum SWE.
            self.y_log = self.y_log / self.y_max
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_max = self.target_max.isel(n_stations=self.station)
            # relative to the maximum SWE.
            self.y_log = self.y_log / self.y_max
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_max = self.target_max.isel(n_stations=self.station)
            # relative to the maximum SWE.
            self.y_log = self.y_log / self.y_max

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd


class gridMETDatasetStation(Dataset):
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train',
                 permute=False, permute_id=0):
        super(gridMETDatasetStation, self).__init__()
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        if permute != False:
            rows = self.x_d.shape[0]  # total samples
            perm = np.random.permutation(rows)  # random index
            temp = np.empty_like(self.x_d)
            temp[:] = self.x_d
            permute_id_array = np.array(permute_id)
            permute_id_array = permute_id_array.reshape(1, -1)
            # permute indicates the feature
            temp[:, permute_id] = self.x_d[perm.reshape(-1, 1), permute_id_array]
            self.x_d = np.empty_like(temp)
            self.x_d[:] = temp
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1]))
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_mean = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).mean().data
        self.target_std = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).std().data
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'ALL': # used for spatial cross-validation. 
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd

class gridMETDatasetStationAveTemp(Dataset):
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train',
                 permute=False, permute_id=0):
        super(gridMETDatasetStationAveTemp, self).__init__()
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        # Average temperature:
        temp_min = xa.open_dataarray(forcings['tmmn'])
        temp_max = xa.open_dataarray(forcings['tmmx'])
        temp_ave = (temp_min+temp_max)/2
        self.forcings[5] = temp_ave # replave tmmn with temp_ave
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        if permute != False:
            rows = self.x_d.shape[0]  # total samples
            perm = np.random.permutation(rows)  # random index
            temp = np.empty_like(self.x_d)
            temp[:] = self.x_d
            permute_id_array = np.array(permute_id)
            permute_id_array = permute_id_array.reshape(1, -1)
            # permute indicates the feature
            temp[:, permute_id] = self.x_d[perm.reshape(-1, 1), permute_id_array]
            self.x_d = np.empty_like(temp)
            self.x_d[:] = temp
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1]))
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_mean = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).mean().data
        self.target_std = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).std().data
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd

class gridMETDatasetStationStaticP(Dataset):
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train',
                 permute=False, permute_id=0):
        super(gridMETDatasetStationStaticP, self).__init__()
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        if permute!=False:
            if type(permute_id)==list:
                for id in permute_id:
                    permute_attr = self.attr_norm[attributions[id]]
                    rows = permute_attr.shape[0] # n_stations
                    perm = np.random.permutation(rows)  # random index
                    temp = np.empty_like(permute_attr.data)
                    temp[:] = permute_attr.data
                    # permute indicates the feature
                    temp[:] = permute_attr[perm]
                    temp = xa.DataArray(temp, dims='n_stations', coords={'n_stations': self.attr[attributions[id]].n_stations})
                    self.attr_norm[attributions[id]] = temp
            else:
                permute_attr = self.attr_norm[attributions[permute_id]]
                rows = permute_attr.shape[0] # n_stations
                perm = np.random.permutation(rows)  # random index
                temp = np.empty_like(permute_attr.data)
                temp[:] = permute_attr.data
                # permute indicates the feature
                temp[:] = permute_attr[perm]
                temp = xa.DataArray(temp, dims='n_stations', coords={'n_stations': self.attr[attributions[permute_id]].n_stations})
                self.attr_norm[attributions[permute_id]] = temp
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1]))
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_mean = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).mean().data
        self.target_std = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).std().data
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd

class gridMETDatasetStationAveRH(Dataset):
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train',
                 permute=False, permute_id=0):
        super(gridMETDatasetStationAveRH, self).__init__()
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        # Average temperature:
        rh_min = xa.open_dataarray(forcings['rmin'])
        rh_max = xa.open_dataarray(forcings['rmax'])
        rh_ave = (rh_min+rh_max)/2
        self.forcings[1] = rh_ave # replave rmax with rh_ave
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        if permute != False:
            rows = self.x_d.shape[0]  # total samples
            perm = np.random.permutation(rows)  # random index
            temp = np.empty_like(self.x_d)
            temp[:] = self.x_d
            permute_id_array = np.array(permute_id)
            permute_id_array = permute_id_array.reshape(1, -1)
            # permute indicates the feature
            temp[:, permute_id] = self.x_d[perm.reshape(-1, 1), permute_id_array]
            self.x_d = np.empty_like(temp)
            self.x_d[:] = temp
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1]))
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_mean = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).mean().data
        self.target_std = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).std().data
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd


# Data assimilation (default SNOW17 run as input)
class gridMETDatasetStationSNOW17(Dataset):
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, window_size=180, mode='Train'):
        super(gridMETDatasetStationSNOW17, self).__init__()
        default = xa.open_dataarray('/tempest/duan0000/snow17/default17/default.nc')
        self.default = default.isel(n_stations=station_id) / 25.4 # mm to inch
        self.station = station_id
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        self._split_and_norm()  # x, y_log
        # time, features.
        self.x_d = self._get_feature_array(self.forcings_norm, forcing_list)
        self.x_d = np.concatenate((self.x_d, (self.default.data).reshape(-1, 1)), axis=-1) # concatenate snow17 into features. 
        self.y_log_d = np.array(self.y_log)  # time.
        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)
        # NANs:
        # last prediction, 0--> the SWE feature
        idx = np.where(np.isnan(self.y_d_new[:, -1, 0]))[0]
        self.y_d_new = np.delete(self.y_d_new, idx, axis=0)
        self.x_d_new = np.delete(self.x_d_new, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new, self.y_d_new = torch.from_numpy(self.x_d_new.astype('float32')), \
            torch.from_numpy((self.y_d_new.astype('float32')))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        # N, WS, features.
        self.x_d_new = np.zeros(
            (n_samples, self.window_size, self.x_d.shape[1])) 
        self.y_d_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]
            self.y_d_new[i, :, 0] = self.y_log_d[i:i + self.window_size]

    def _split_and_norm(self):  # target: log transform
        # self.y_log = np.log(self.y + 0.2)
        self.y_log = self.y  # linear Z-transform.
        self.forcings_mean = []
        self.forcings_std = []
        self.forcings_norm = []
        # Calculate mean and std with all stations.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=slice('1980-10-01', '1999-09-30')).std().data)
        self.target_mean = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).mean().data
        self.target_std = self.y_log.sel(
            time=slice('1980-10-01', '1999-09-30')).std().data
        # select stations in train-test-split.
        if self.mode.upper() == 'TRAIN':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1980-10-01', '1999-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1980-10-01', '1999-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
            self.default = self.default.sel(time=slice('1980-10-01', '1999-09-30'))
            self.default = (self.default - self.target_mean) / self.target_std
        elif self.mode.upper() == 'TEST':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('2008-10-01', '2018-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2008-10-01', '2018-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
            self.default = self.default.sel(time=slice('2008-10-01', '2018-09-30'))
            self.default = (self.default - self.target_mean) / self.target_std
        elif self.mode.upper() == 'VAL':
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice('1999-10-01', '2008-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('1999-10-01', '2008-09-30'))
            self.y_log = self.y_log.isel(n_stations=self.station)
            self.y_log = (self.y_log - self.target_mean) / self.target_std
            self.default = self.default.sel(time=slice('1999-10-01', '2008-09-30'))
            self.default = (self.default - self.target_mean) / self.target_std

    def _get_feature_array(self, data_in, features):
        x = np.array([data_in[i].values for i,
                     feature in enumerate(features)]).T
        return x

    def __len__(self):
        return self.x_d_new.shape[0]

    def __getitem__(self, item):  # x_d_new: N, WS, features.
        return self.x_d_new[item], self.attr_norm, self.y_d_new[item], self.qstd

import time
import gc
# Construct Dataset by Water Years. 
class gridMETDatasetStationWY(Dataset): 
    def __init__(self, forcings: Dict, topo_file, attributions, target, station_id, 
                train_wy, test_wy, target_mean, target_std,
                all_train_time, window_size=180, mode='Train') -> None:
        super(gridMETDatasetStationWY, self).__init__()
        self.station = station_id
        self.train_wy = train_wy
        self.test_wy = test_wy
        self.all_train_time = all_train_time
        self.target_mean = target_mean
        self.target_std = target_std
        # attrs and SWE target:
        self.attr = xa.open_dataset(topo_file)
        self.y = self.attr[target[0]]
        self.forcings = []
        self.mode = mode
        self.window_size = window_size
        forcing_list = list(forcings.keys())
        for forcing in forcing_list:
            forcing_data = xa.open_dataarray(forcings[forcing])
            self.forcings.append(forcing_data)
        self._split_and_norm()  # x, y_log
        idx = np.where(np.isnan(self.y_d[:, -1, 0]))[0]
        self.y_d = np.delete(self.y_d, idx, axis=0)
        self.x_d = np.delete(self.x_d, idx, axis=0)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.n_samples = self.x_d.shape[0]
        self.x_d, self.y_d = torch.from_numpy(self.x_d.astype('float32')), \
            torch.from_numpy((self.y_d.astype('float32')))
        # Static variables
        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
        self.qstd = torch.empty(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x_d[item], self.attr_norm, self.y_d[item], self.qstd

    def _split_and_norm(self):
        self.y_log = self.y  # linear Z-transform.
        # Transpose SWE to make time the first dimension for faster slice. 
        self.y_log = self.y_log.transpose('time','n_stations')
        self.forcings_mean = []
        self.forcings_std = []
        # Get training time period from train_wy
        '''all_train_time = []
        for year in self.train_wy:
            time_slice = slice(str(year)+'-10-01', str(year+1)+'-09-30')
            timestamp = self.forcings[0].sel(time=time_slice).time.data
            for time in timestamp:
                all_train_time.append(time)'''
        # Calculate mean and std with all stations over train_wy.
        for forcing_data in self.forcings:
            self.forcings_mean.append(forcing_data.sel(
                time=self.all_train_time).mean().data)
            self.forcings_std.append(forcing_data.sel(
                time=self.all_train_time).std().data)        
        # Get Data and Construct Array. 
        if self.mode.upper()=='TRAIN': # training with train_wy
            wy_list = self.train_wy
        elif self.mode.upper()=='TEST':
            wy_list = self.test_wy
        for ind, year in enumerate(wy_list): # iterate through water years
            self.forcings_norm_wy = []
            start_date = np.datetime64(str(year)+'-10-01')-np.timedelta64(180, 'D')
            end_date = np.datetime64(str(year+1)+'-09-30')
            for i, forcing_data in enumerate(self.forcings):
                forcing_norm = forcing_data.sel(
                    time=slice(start_date, end_date))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm_wy.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            # Concatenate features
            x = np.array([self.forcings_norm_wy[i].values for i,
                    feature in enumerate(self.forcings_norm_wy)]).T
            # print(x.shape, ' x shape') # time, features
            y_log_wy = self.y_log.sel(time=slice(start_date, end_date))
            y_log_wy = y_log_wy.isel(n_stations=self.station)
            y_log_wy = (y_log_wy - self.target_mean) / self.target_std
            x_wy, y_wy = self._reshape(x, y_log_wy) # N, WS, Features; N, WS, 1
            if ind ==0: # first water year:
                self.x_d = x_wy
                self.y_d = y_wy
                # print(x_wy.shape, y_wy.shape, ' Water Year')
            else:
                self.x_d = np.concatenate((self.x_d, x_wy), axis=0) # concatenate through samples
                self.y_d = np.concatenate((self.y_d, y_wy), axis=0)
        
    def _reshape(self, x, y):
        n_samples = x.shape[0] - self.window_size + 1
        # N, WS, features.
        x_new = np.zeros(
            (n_samples, self.window_size, x.shape[1])) 
        y_new = np.zeros((n_samples, self.window_size, 1))  # N, WS, 1
        for i in range(n_samples):
            x_new[i, :] = x[i:i + self.window_size, :]
            y_new[i, :, 0] = y[i:i + self.window_size]
        return x_new, y_new
