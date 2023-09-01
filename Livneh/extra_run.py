import inspect
import os
import sys
import warnings
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import COgridMETDataset
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
warnings.filterwarnings("ignore", category=FutureWarning)

from models.lstm import LSTM, LSTM_multi

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--model', type=str, default='lstm')
    args = vars(parser.parse_args())
    return args

args = get_args()
ens = args['ens']
device_id = args['device']
devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')
if device_id !=-1:
    device = devices[device_id] # device to use from the 4 GPUs.
else:
    device = devices[ens%4]

model_type = args['model'].upper()
pool_size = 4

path = '/p/gpfs1/shiduan/SWE/Livneh/LSTM/ori/CO/'
model_path = 'runs_PRISM/LSTM/model_ens_'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

HID = 128
RELU_FLAG = False

attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']

forcings = {'pr': 'spatialForcings/CO_prec.nc', 
            'tmmn': 'spatialForcings/CO_tmin.nc',
            'tmmx': 'spatialForcings/CO_tmax.nc',
            }
n_inputs = len(attributions) + len(forcings)
topo_file = 'spatialForcings/CO_topo_file.nc'
# topo_file = '../gridMET/Rocky/topo_file_30m.nc'
# topo_file = '../MACA/'+ 'rocky' + '_topo_file_prism_nn.nc' # PRISM topo file. 
def inference(lat_id, device=device):
    ds = COgridMETDataset(forcings=forcings, lat=lat_id, attributions=attributions, topo_file=topo_file,
                          window_size=180, hist=False, proj=True, 
                          scaler_mean='livneh_mean.nc',
                          scaler_std='livneh_std.nc')
    if model_type == 'LSTM':
        model = LSTM(hidden_units=HID, input_size=n_inputs,
                     relu_flag=RELU_FLAG)
        model_path_ens = model_path + str(ens)
    model.load_state_dict(torch.load(model_path_ens))
    model = model.to(device)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    pred = []
    prob = []
    for data in loader:
        x_d_new, x_attr = data
        x_d_new, x_attr = x_d_new.to(device), x_attr.to(device)
        output = model(x_d_new, x_attr)
        if model_type.upper() == 'LSTM-MULTI':
            y_hat_sub = output[0][:, -1:, 0:1]
            y_prob = output[-1][:, -1:, 1:2]
            prob.append(y_prob.to('cpu').data.numpy())
        else:
            y_hat_sub = output[0][:, -1:, :]
        pred.append(y_hat_sub.to('cpu').data.numpy())
        
    pred = np.concatenate(pred).flatten()
    if len(prob)>0:
        prob = np.concatenate(prob).flatten()
        return lat_id, pred, prob
    # np.save(path + str(lat_id) + '_' + str(ens), pred)
    return lat_id, pred


if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    ds = COgridMETDataset(forcings=forcings, lat=0, attributions=attributions, topo_file=topo_file,
                          window_size=180, hist=False, proj=True, 
                          scaler_mean='livneh_mean.nc',
                          scaler_std='livneh_std.nc')
    n_samples = ds.n_samples
    n_lat, n_lon = ds.n_lat, ds.n_lon
    prediction = np.empty((n_lat, n_lon, n_samples))
    print(n_samples, n_lat, n_lon)
    with Pool(pool_size) as pool:
        lat_ids = np.arange(n_lat)
        for results in tqdm(pool.imap(inference, lat_ids), total=len(lat_ids), position=0):
            # lat, pred = results
            lat = results[0]
            print(results[1].shape)
            split_data = np.split(results[1], n_lon)
            # print(lat, ' ', len(pred), ' ', len(pred) / 108, ' ', split_data[0].shape)
            for i in range(n_lon):
                prediction[lat, i] = split_data[i]
            if model_type.upper()=='LSTM-MULTI':
                split_data = np.split(results[2], n_lon)
                for i in range(n_lon):
                    probability[lat, i] = split_data[i]
    np.save(path + 'prediction_' + str(ens), prediction)
    if model_type.upper()=='LSTM-MULTI':
        np.save(path + 'probability_' + str(ens), probability)
