# dev test
from genericpath import exists
import os.path
import pickle
import xarray as xa
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from dataset import gridMETDatasetStation, gridMETRelativeStation
# NOT from the data.SWE_dataset. This file has different train-test time split. 
import sys
sys.path.append('../models')
from lstm import LSTM

import argparse
torch.backends.cudnn.benchmark=True # set the optimal algorithm for tasks. 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--rel', type=int, default=0)
    args = vars(parser.parse_args())
    return args

def train(model, ds, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=128, shuffle=True, 
                        num_workers=4, pin_memory=True)
    print('TRAINING')
    for epoch in tqdm(range(50), desc='train-'+str(ens)):
        loss_val = []
        for data in loader:
            x_d_new, x_attr, y_new, qstd = data
            x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                           y_new.to(device), qstd.to(device)

            y_sub = y_new[:, -1:]
            y_hat = model(x_d_new, x_attr)[0]
            y_hat_sub = y_hat[:, -1:, :]
            # loss = loss_fn(y_hat_sub, y_sub, qstd)
            loss = loss_fn(y_hat_sub, y_sub)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_val.append(loss.item())
        loss_val = np.mean(loss_val)
        if epoch == 47:
            model2 = model
        if writer is not None:
            writer.add_scalar('training mse', scalar_value=loss_val, global_step=epoch)
    return model, model2


def evaluate(model, ds, device=torch.device('cuda:0')):
    test_dl = DataLoader(ds, batch_size=128, shuffle=False)
    y_true = []
    y_pred = []
    model = model.eval()
    for data in test_dl:
        x_d_new, x_attr, y_new, qts = data
        x_d_new, x_attr, y_new = x_d_new.to(device), x_attr.to(device), y_new.to(device)
        y_sub = y_new[:, -1:]
        y_hat = model(x_d_new, x_attr)[0]
        y_hat_sub = y_hat[:, -1:, :]
        y_pred.append(y_hat_sub.cpu().data.numpy())
        y_true.append(y_sub.cpu().data.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return y_true, y_pred

args = get_args()
ens = args['ens']
device_id = args['device']
rel = args['rel']

devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')

# topo_file = 'SNOTEL/raw_snotel_topo_30m.nc'
# topo_file = 'SNOTEL/raw_snotel_topo_600m.nc'
topo_file = '../SNOTEL/raw_wus_snotel_topo_clean.nc' # PRISM DEM

WINDOW_SIZE = 180
RELU_FLAG = False
LR = 1e-4
HID = 128
ENS = 10
if device_id !=-1:
    device = devices[device_id] # device to use from the 4 GPUs.
else:
    device = devices[ens%4] 
model_type = 'LSTM'
if rel==0:
    path = 'runs_PRISM/' + model_type.upper() + '/'
else:
    path = 'runs_PRISM/' + model_type.upper() + '_rel/'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
    print('Make output directory')
else:
    print('Folder is here. ')
loss_fn = nn.MSELoss()

attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
# attributions = ['longitude', 'latitude', 'elevation_30m', 'dah_30m', 'trasp_30m']
# attributions = ['longitude', 'latitude', 'elevation_600m', 'dah_600m', 'trasp_600m']
# attributions = ['longitude', 'latitude', 'elevation_30m']

forcings = {'pr': 'Prec_wus_clean.nc', 
            'tmmn': 'Tmin_wus_clean.nc',
            'tmmx': 'Tmax_wus_clean.nc',
            }

n_inputs = len(attributions) + len(forcings)
target = ['SWE']
train_ds = []

# for station_id in mountain_ids:  # 765
if rel==0: # use SWE
    for station_id in range(581):
        ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                mode='TRAIN', topo_file=topo_file, station_id=station_id)
        train_ds.append(ds)
else: # use relative SWE
    for station_id in range(581):
        ds = gridMETRelativeStation(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                mode='TRAIN', topo_file=topo_file, station_id=station_id)
        train_ds.append(ds)
train_ds = ConcatDataset(train_ds)
print(train_ds.__len__())


print(ens, ' Start')
if model_type.lower() == 'lstm':
    model = LSTM(hidden_units=HID, input_size=n_inputs, relu_flag=RELU_FLAG)
elif model_type.lower() == 'tcnn':
    model = TCNN(kernal_size=KER, num_levels=LEVELS, num_channels=CHA,
                    input_size=n_inputs)
elif model_type.lower() == 'attention':
    model = Attention(num_att_layers=num, dim_feedforward=forward, embedding_size=embedding, n_head=head,
                        input_size=n_inputs)
model = model.to(device)
model1, model2 = train(model, train_ds, LR, device=device)
torch.save(model1.state_dict(), path + 'model_ens_' + str(ens))
torch.save(model2.state_dict(), path + 'model_ens_' + str(9-ens))
result_true = {}
result_pred = {}
result_pred2 = {}

for station_id in tqdm(range(581), desc='test_ds'):
    if rel==0:
        ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target,
                                    window_size=WINDOW_SIZE,
                                    mode='TEST', topo_file=topo_file,
                                    station_id=station_id)
    else:
        ds = gridMETRelativeStation(forcings=forcings, attributions=attributions, target=target,
                                    window_size=WINDOW_SIZE,
                                    mode='TEST', topo_file=topo_file,
                                    station_id=station_id)
    if ds.__len__() > 0:
        y_true, y_pred = evaluate(model1, ds, device=device)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        result_true[station_id] = y_true
        result_pred[station_id] = y_pred
        y_true, y_pred = evaluate(model2, ds, device=device)
        y_pred = y_pred.reshape(-1, 1)
        result_pred2[station_id] = y_pred
with open(path + 'result_true_' + str(ens), 'wb') as f:
    pickle.dump(result_true, f)
with open(path + 'result_pred_' + str(ens), 'wb') as f:
    pickle.dump(result_pred, f)
with open(path + 'result_pred_' + str(9-ens), 'wb') as f:
    pickle.dump(result_pred2, f)
