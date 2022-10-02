import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from models.lstm import LSTM
import xarray as xa
from data.SWE_Dataset import gridMETDatasetStation
import os
from torch import nn

import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    parser.add_argument('--mountain', type=int)
    args = vars(parser.parse_args())
    return args

def train(model, ds, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=128, shuffle=True, 
                        num_workers=2, pin_memory=True)
    print('TRAINING')
    for epoch in tqdm(range(50)):
        loss_val = []
        for data in loader:
            x_d_new, x_attr, y_new, qstd = data
            x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                           y_new.to(device), qstd.to(device)

            y_sub = y_new[:, -1:]
            y_hat = model(x_d_new, x_attr)[0]
            y_hat_sub = y_hat[:, -1:, :]
            loss = loss_fn(y_hat_sub, y_sub)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_val.append(loss.item())
        loss_val = np.mean(loss_val)
        if writer is not None:
            writer.add_scalar('training mse', scalar_value=loss_val, global_step=epoch)
        if epoch == 47:
            model2 = model
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
mountain_id = args['mountain']
devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')
print('Mountain: ', mountain_id)

WINDOW_SIZE = 180
model_type = 'lstm'
LR = 1e-4
HID = 128
ENS = 10
device = devices[(ens+mountain_id)%4] # device to use from the 4 GPUs. 
target = ['SWE']
swe_ds = xa.open_dataset(
    'mountains/raw_wus_snotel_topo_clean_mountains.nc')
mountains = swe_ds.mountain
del swe_ds
loss_fn = nn.MSELoss()

attributions = ['longitude', 'latitude', 'elevation_30m', 'dah_30m', 'trasp_30m']
attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
forcings = {'pr': 'gridMET/pr_wus_clean.nc', 'rmax': 'gridMET/rmax_wus_clean.nc', 'rmin': 'gridMET/rmin_wus_clean.nc',
            'sph': 'gridMET/sph_wus_clean.nc', 'srad': 'gridMET/srad_wus_clean.nc', 'tmmn': 'gridMET/tmmn_wus_clean.nc',
            'tmmx': 'gridMET/tmmx_wus_clean.nc', 'vpd': 'gridMET/vpd_wus_clean.nc', 'vs': 'gridMET/vs_wus_clean.nc'}
n_inputs = len(attributions) + len(forcings)

topo = 'SNOTEL/raw_snotel_topo_30m.nc'
topo = 'mountains/raw_wus_snotel_topo_clean_mountains.nc'
save_path = 'mountains_prism/output-all/'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
    print('save path created. ')
model_path = 'mountains_prism/mountain_'+str(mountain_id)+'/'
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    print('model path created. ')
# leave-one-mountain-out cross validation
# for mountain_id in range(1, 9):
train_id = []
test_id = []
for station in range(581):
    label = mountains.isel(n_stations=station)
    if label != -1 and label != mountain_id:  # train sample
        train_id.append(station)
    elif label != -1 and label == mountain_id:  # test sample
        test_id.append(station)
print(len(train_id), ' stations for training')
print(len(test_id), ' stations for testing')

train_ds = []
for station in tqdm(train_id, desc='train ds'):
    ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                mode='ALL', topo_file=topo, station_id=station) # mode set to ALL for cross
    train_ds.append(ds)

train_ds = ConcatDataset(train_ds)
print(train_ds.__len__(), ' samples for training')
print(ens, ' Start')
if model_type.lower() == 'lstm':
    model = LSTM(hidden_units=HID, input_size=n_inputs, relu_flag=False)
model = model.to(device)
model1, model2 = train(model, train_ds, LR, device=device)
torch.save(model1.state_dict(), model_path  + 'model_ens_' + str(ens))
torch.save(model2.state_dict(), model_path  + 'model_ens_' + str(9-ens))

# test:
for station in tqdm(test_id, desc='testing'):
    ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                mode='ALL', topo_file=topo, station_id=station) # mode set to ALL for cross
    if ds.__len__()>0:
        y_true, y_pred = evaluate(model1, ds, device=device)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        np.save(save_path+'pred_'+str(station)+'_ens_'+str(ens), y_pred)
        np.save(save_path+'obs_'+str(station), y_true)
        y_true, y_pred = evaluate(model2, ds, device=device)
        y_pred = y_pred.reshape(-1, 1)
        np.save(save_path+'pred_'+str(station)+'_ens_'+str(9-ens), y_pred)
        
        

