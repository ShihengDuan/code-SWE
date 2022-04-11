from typing import Dict

import numpy as np
import torch
import xarray as xa
from torch.utils.data import Dataset

# snow_cover = '/tempest/duan0000/SWE/NSIDC/MODIS/Aqua2008-2018_5km_snow_cover.nc'
snow_cover_percent = '/tempest/duan0000/SWE/data/snow_series_percent'  # located snow file.


class CODataset(Dataset):  # dataset for CO PRISM. # Use normalized files.
    def __init__(self, ppt, tmax, tmin, tmean, attributions, topo_file, window_size,
                 lat_id, lon_id, snow_cover=snow_cover_percent):
        super(CODataset, self).__init__()
        if snow_cover != None:
            self.snow_flag = True
            self.snow_cover = xa.open_dataarray(snow_cover)
        else:
            self.snow_flag = False
        self.window_size = window_size
        ppt = xa.open_dataarray(ppt)
        self.ppt = ppt.isel(lat=lat_id, lon=lon_id)
        tmin = xa.open_dataarray(tmin)
        self.tmin = tmin.isel(lat=lat_id, lon=lon_id)
        tmax = xa.open_dataarray(tmax)
        self.tmax = tmax.isel(lat=lat_id, lon=lon_id)
        tmean = xa.open_dataarray(tmean)
        self.tmean = tmean.isel(lat=lat_id, lon=lon_id)
        # self.lon = self.ppt.lon.data
        # self.lat = self.ppt.lat.data
        if self.snow_flag:
            # snow_series = find_snow5km(self.lon, self.lat, self.snow_cover)
            # snow_series = snow_series / 100
            # snow_series = snow_series.reshape(-1, 1)
            snow_series = self.snow_cover.isel(lat=lat_id, lon=lon_id)
            snow_series_data = snow_series.data.reshape(-1, 1)
        attribution = xa.open_dataset(topo_file)
        attr_data = attribution[attributions]
        self.attr_data = attr_data.isel(lat=lat_id, lon=lon_id)
        # self._normalize()

        x_attr = np.array([self.attr_data[feature] for feature in attributions]).astype('float32')
        x_attr = torch.from_numpy(x_attr)  # (6, )
        self.x_attr = x_attr.view(x_attr.nelement())  # (6)

        self.x_d = np.array([var.values for var in [self.ppt, self.tmean, self.tmax, self.tmin]]).T
        length = self.x_d.shape[0]
        if self.snow_flag:
            self.x_d = np.concatenate((self.x_d, snow_series_data[:length]), axis=1)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self.x_d_new = self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))

    def _reshape(self):
        self.n_samples = self.x_d.shape[0] - self.window_size + 1
        n_features = 4
        if self.snow_flag:
            n_features += 1
        x_d_new = np.zeros((self.n_samples, self.window_size, n_features))
        for i in range(self.n_samples):
            x_d_new[i, :, :] = self.x_d[i:i + self.window_size, :]
        return x_d_new

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x_d_new[item], self.x_attr


class COWholeDataset(Dataset):
    def __init__(self, ppt, tmax, tmin, tmean, attributions, topo_file, lat_id, window_size=180,
                 snow_cover=snow_cover_percent):
        super(COWholeDataset, self).__init__()
        self.window_size = window_size
        self.ppt = xa.open_dataarray(ppt)  # <xarray.DataArray (time: 3653, lat: 839, lon: 540)>
        self.tmin = xa.open_dataarray(tmin)
        self.tmax = xa.open_dataarray(tmax)
        self.tmean = xa.open_dataarray(tmean)
        self.ppt = self.ppt.isel(lat=lat_id)  # time, lon.
        self.tmin = self.tmin.isel(lat=lat_id)
        self.tmax = self.tmax.isel(lat=lat_id)
        self.tmean = self.tmean.isel(lat=lat_id)

        attribution = xa.open_dataset(topo_file)
        attr_data = attribution[attributions]
        self.attr_data = attr_data.isel(lat=lat_id)
        self.lons = attr_data.lon.data
        self.lats = attr_data.lat.data
        if snow_cover != None:
            self.snow_flag = True
            self.snow_cover = xa.open_dataarray(snow_cover)
            self.snow_cover = self.snow_cover.isel(lat=lat_id)  # (time, lat)
            self.length = self.tmean.shape[0]

        else:
            self.snow_flag = False

        self.x_d = np.array([var.values for var in [self.ppt, self.tmean, self.tmax, self.tmin]])  # (4, 3653, 540)
        self.x_d = np.moveaxis(self.x_d, 0, -1)  # (3653, 540, 4)

        if self.snow_flag:
            self.snow_cover_data = self.snow_cover.data[:self.length]
            shape = self.snow_cover_data.shape
            self.snow_cover_data = self.snow_cover_data.reshape(shape[0], shape[1], 1)  # add feature channel.
            self.x_d = np.concatenate((self.x_d, self.snow_cover_data), axis=-1)
        self.x_d = np.nan_to_num(self.x_d, nan=0)  # replace NAN.
        self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))
        x_attr = []
        for feature in attributions:
            value = self.attr_data[feature].values.reshape(-1)
            if value.shape[0] < 540:  # Latitude
                value = np.repeat(value, 540)
            x_attr.append(value)
        x_attr = np.array(x_attr)  # (5, 540)
        self.x_attr = torch.from_numpy(x_attr.astype('float32'))

    def _reshape(self):
        n_samples = self.x_d.shape[0] - self.window_size + 1
        n_features = 4
        if self.snow_flag:
            n_features += 1
        self.x_d_new = np.empty((n_samples, self.window_size, self.lons.shape[0], n_features))
        for i in range(n_samples):
            self.x_d_new[i, :] = self.x_d[i:i + self.window_size, :]

    def __len__(self):
        return self.x_d_new.shape[0] * self.x_d_new.shape[2]  # N, lons

    def __getitem__(self, item):
        n = item % self.x_d_new.shape[0]
        k = item // self.x_d_new.shape[0]
        a = k % self.x_d_new.shape[2]

        return self.x_d_new[n, :, a], self.x_attr[:, a]


class COgridMETDataset(Dataset):  # lat 168, lon 108
    def __init__(self, forcings: Dict, topo_file, attributions, lat, window_size=180,
                 scaler_mean='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_mean.nc',
                 scaler_std='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_std.nc', hist=False):
        super(COgridMETDataset, self).__init__()
        self.window_size = window_size
        self.forcings = []
        self.scaler_mean = xa.open_dataset(scaler_mean)
        self.scaler_std = xa.open_dataset(scaler_std)
        for forcing in forcings:
            forcing_data = xa.open_dataarray(forcings[forcing])
            if hist:
                forcing_data = forcing_data.sel(time=slice('1982-10-01', '1999-09-30'))
            else:
                forcing_data = forcing_data.sel(time=slice('2008-10-01', '2018-09-30'))
            forcing_data = forcing_data.isel(lat=lat)  # select the point.
            forcing_data = (forcing_data - self.scaler_mean[forcing]) / self.scaler_std[forcing]
            self.forcings.append(forcing_data)
        attribution = xa.open_dataset(topo_file)
        attribution = attribution.isel(lat=lat)
        attr_data = (attribution[attributions] - self.scaler_mean[attributions]) / self.scaler_std[attributions]
        self.x_d = [var.values.reshape(var.shape[0], var.shape[1], var.shape[2]) for var in
                    self.forcings]  # time, lons, features, 1
        self.x_attr = np.array([attr_data[attr].values.reshape(-1) for attr in attributions])  # 5, 108.
        self.x_d = np.concatenate(self.x_d, axis=-1)  # time, lons, features.

        self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))
        self.x_attr = torch.from_numpy(self.x_attr.astype('float32'))

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


class CO_LOCADataset(Dataset):  # lat 112, lon 72
    def __init__(self, forcings: Dict, topo_file, attributions, lat, window_size=180, scenario='hist',
                 scaler_mean='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_mean.nc',
                 scaler_std='/tempest/duan0000/SWE/gridMET/Rocky/gridmet_std.nc'):
        super(CO_LOCADataset, self).__init__()
        self.window_size = window_size
        self.forcings = []
        self.scaler_mean = xa.open_dataset(scaler_mean)
        self.scaler_std = xa.open_dataset(scaler_std)
        for forcing in forcings:
            forcing_data = xa.open_dataarray(forcings[forcing])
            if scenario == 'hist':
                forcing_data = forcing_data.sel(time=slice('1981-10-01', '2001-09-30'))
            else:
                forcing_data = forcing_data.sel(time=slice('2071-10-01', '2091-09-30'))
            forcing_data = forcing_data.isel(lat=lat)  # select the point.
            forcing_data = (forcing_data - self.scaler_mean[forcing]) / self.scaler_std[forcing]
            self.forcings.append(forcing_data)
        attribution = xa.open_dataset(topo_file)
        attribution = attribution.isel(lat=lat)
        attr_data = (attribution[attributions] - self.scaler_mean[attributions]) / self.scaler_std[attributions]
        self.x_d = [var.values.reshape(var.shape[0], var.shape[1], var.shape[2]) for var in
                    self.forcings]  # time, lons, features, 1
        self.x_attr = np.array([attr_data[attr].values.reshape(-1) for attr in attributions])  # 5, 108.
        self.x_d = np.concatenate(self.x_d, axis=-1)  # time, lons, features.

        self._reshape()
        self.x_d_new = torch.from_numpy(self.x_d_new.astype('float32'))
        self.x_attr = torch.from_numpy(self.x_attr.astype('float32'))

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
