from typing import Dict

import numpy as np
import torch
import xarray as xa
from torch.utils.data import Dataset
from tqdm import tqdm

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
                    time=slice('2000-10-01', '2013-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2000-10-01', '2013-09-30'))
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
        self.qstd = self.y.isel(n_stations=self.station).std().data/7
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
        self.qstd = torch.empty(0)

        self._reshape()

        self.attr = self.attr[attributions]
        attr_mean = self.attr.mean()
        attr_std = self.attr.std()
        self.attr_norm = (self.attr - attr_mean) / attr_std  # stations.
        self.attr_norm = self.attr_norm.isel(n_stations=station_id)  # select
        self.attr_norm = np.array([self.attr_norm[feature]
                                  for feature in attributions])  # 5,
        self.attr_norm = torch.from_numpy(self.attr_norm.astype('float32'))
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
                    time=slice('2000-10-01', '2013-09-30'))
                forcing_norm = forcing_norm.isel(n_stations=self.station)
                self.forcings_norm.append(
                    (forcing_norm - self.forcings_mean[i]) / self.forcings_std[i])
            self.y_log = self.y_log.sel(time=slice('2000-10-01', '2013-09-30'))
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


# extrapolation dataset
class COgridMETDataset(Dataset):  # lat 168, lon 108
    def __init__(self, forcings: Dict, topo_file, attributions, lat, window_size=180,
                 scaler_mean='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_mean.nc',
                 scaler_std='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_std.nc', hist=False, proj=False):
        super(COgridMETDataset, self).__init__()
        self.window_size = window_size
        self.forcings = []
        self.scaler_mean = xa.open_dataset(scaler_mean)
        self.scaler_std = xa.open_dataset(scaler_std)
        for forcing in forcings:
            forcing_data = xa.open_dataarray(forcings[forcing])
            if hist:
                forcing_data = forcing_data.sel(time=slice('1982-10-01', '1999-09-30'))
            elif proj:
                forcing_data = forcing_data.sel(time=slice('1982-10-01', '2005-09-30'))
                # print('PROJ')
            else:
                forcing_data = forcing_data.sel(time=slice('1980-10-01', '2013-09-30'))
            forcing_data = forcing_data.isel(lat=lat)  # select the point.
            forcing_data = (forcing_data - self.scaler_mean[forcing]) / self.scaler_std[forcing]
            self.forcings.append(forcing_data)
            # print(forcing_data.shape)
        attribution = xa.open_dataset(topo_file)
        self.n_lat = attribution.latitude.shape[0]
        self.n_lon = attribution.lon.shape[0]
        attribution = attribution.isel(lat=lat)
        attr_data = (attribution[attributions] - self.scaler_mean[attributions]) / self.scaler_std[attributions]
        self.x_d = [var.values.reshape(var.shape[0], var.shape[1], 1) for var in
                    self.forcings]  # time, lons, features, 1
        self.x_attr = np.array([attr_data[attr].values.reshape(-1) for attr in attributions])  # 5, 108.
        self.x_d = np.concatenate(self.x_d, axis=-1)  # time, lons, features.
        print(self.x_d.shape)
        self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))
        print(self.x_d_new.shape)
        self.x_attr = torch.from_numpy(self.x_attr.astype('float32'))
        self.n_samples = n_samples = self.x_d.shape[0] - self.window_size + 1
    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        n_features = self.x_d.shape[-1]
        self.x_d_new = np.empty((n_samples, self.window_size, self.x_d.shape[1], n_features))
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]

    def __len__(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        lons = self.x_d.shape[1]
        return n_samples * lons

    def __getitem__(self, item):
        n = item % self.x_d_new.shape[0]
        k = item // self.x_d_new.shape[0]
        a = k % self.x_d_new.shape[2]
        return self.x_d_new[n, :, a], self.x_attr[:, a]


class LOCAgridMETDataset(Dataset):  # lat 168, lon 108
    def __init__(self, forcings: Dict, topo_file, attributions, lat, window_size=180,
                 scaler_mean='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_mean.nc',
                 scaler_std='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_std.nc', scenario='historical'):
        super(LOCAgridMETDataset, self).__init__()
        self.window_size = window_size
        self.forcings = []
        self.scaler_mean = xa.open_dataset(scaler_mean)
        self.scaler_std = xa.open_dataset(scaler_std)
        for forcing in forcings:
            forcing_data = xa.open_dataarray(forcings[forcing])
            if scenario=='historical' or scenario=='hist':
                forcing_data = forcing_data.sel(time=slice('1950-01-01', '2005-12-31'))
            elif scenario=='rcp85':
                forcing_data = forcing_data.sel(time=slice('2006-01-01', '2099-12-31'))
                # print('PROJ')
            forcing_data = forcing_data.isel(lat=lat)  # select the point.
            forcing_data = (forcing_data - self.scaler_mean[forcing]) / self.scaler_std[forcing]
            self.forcings.append(forcing_data)
            # print(forcing_data.shape)
        attribution = xa.open_dataset(topo_file)
        self.n_lat = attribution.latitude.shape[0]
        self.n_lon = attribution.lon.shape[0]
        attribution = attribution.isel(lat=lat)
        attr_data = (attribution[attributions] - self.scaler_mean[attributions]) / self.scaler_std[attributions]
        self.x_d = [var.values.reshape(var.shape[0], var.shape[1], 1) for var in
                    self.forcings]  # time, lons, features, 1
        self.x_attr = np.array([attr_data[attr].values.reshape(-1) for attr in attributions])  # 5, 108.
        self.x_d = np.concatenate(self.x_d, axis=-1)  # time, lons, features.
        print(self.x_d.shape)
        self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))
        print(self.x_d_new.shape)
        self.x_attr = torch.from_numpy(self.x_attr.astype('float32'))
        self.n_samples = n_samples = self.x_d.shape[0] - self.window_size + 1
    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        n_features = self.x_d.shape[-1]
        self.x_d_new = np.empty((n_samples, self.window_size, self.x_d.shape[1], n_features))
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]

    def __len__(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        lons = self.x_d.shape[1]
        return n_samples * lons

    def __getitem__(self, item):
        n = item % self.x_d_new.shape[0]
        k = item // self.x_d_new.shape[0]
        a = k % self.x_d_new.shape[2]
        return self.x_d_new[n, :, a], self.x_attr[:, a]
