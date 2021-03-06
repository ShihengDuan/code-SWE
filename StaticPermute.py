import os.path
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.SWE_Dataset import gridMETDatasetStationStaticP
from models.lstm import LSTM


def train(model, ds, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
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
            writer.add_scalar(
                'training mse', scalar_value=loss_val, global_step=epoch)
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
        x_d_new, x_attr, y_new = x_d_new.to(
            device), x_attr.to(device), y_new.to(device)
        y_sub = y_new[:, -1:]
        y_hat = model(x_d_new, x_attr)[0]
        y_hat_sub = y_hat[:, -1:, :]
        y_pred.append(y_hat_sub.cpu().data.numpy())
        y_true.append(y_sub.cpu().data.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    return y_true, y_pred


WINDOW_SIZE = 180
RELU_FLAG = False
LR = 1e-4
HID = 128
ENS = 10
device = torch.device('cuda:0')
model_type = 'LSTM'
loss_fn = nn.MSELoss()

attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
forcings = {'pr': 'gridMET/pr_wus_clean.nc', 'rmax': 'gridMET/rmax_wus_clean.nc', 'rmin': 'gridMET/rmin_wus_clean.nc',
            'sph': 'gridMET/sph_wus_clean.nc', 'srad': 'gridMET/srad_wus_clean.nc', 'tmmn': 'gridMET/tmmn_wus_clean.nc',
            'tmmx': 'gridMET/tmmx_wus_clean.nc', 'vpd': 'gridMET/vpd_wus_clean.nc', 'vs': 'gridMET/vs_wus_clean.nc'}
n_inputs = len(attributions) + len(forcings)
target = ['SWE']

permute_id = [3, 4]
if type(permute_id)==list:
    print('Permute feature: ', permute_id)
else:
    print('Permute feature: ', attributions[permute_id])
path = '/tempest/duan0000/SWE/gridMET/runs_staticGroup/' + \
    model_type.upper() + '_1e-4_H128/Aspect/'
if not os.path.isdir(path):
    os.makedirs(path)

train_ds = []
for station_id in range(581):  # 765
    ds = gridMETDatasetStationStaticP(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                      mode='TRAIN', topo_file='SNOTEL/raw_wus_snotel_topo_clean.nc', station_id=station_id, permute=True,
                                      permute_id=permute_id)
    train_ds.append(ds)
train_ds = ConcatDataset(train_ds)
print(train_ds.__len__())
for e in [1, 2, 4]:  # ens number
    print(e, ' Start')
    if model_type.lower() == 'lstm':
        model = LSTM(hidden_units=HID, input_size=n_inputs,
                     relu_flag=RELU_FLAG)
    model = model.to(device)
    model1, model2 = train(model, train_ds, LR, device=device)
    torch.save(model1.state_dict(), path + 'model_ens_' + str(e))
    torch.save(model2.state_dict(), path + 'model_ens_' + str(9-e))
    result_true = {}
    result_pred = {}

    result_true = {}
    result_pred = {}
    result_pred2 = {}
    for station_id in tqdm(range(581), desc='test_ds'):
        ds = gridMETDatasetStationStaticP(forcings=forcings, attributions=attributions, target=target,
                                          window_size=WINDOW_SIZE,
                                          mode='TEST', topo_file='SNOTEL/raw_wus_snotel_topo_clean.nc',
                                          station_id=station_id, permute=True, permute_id=permute_id)
        if ds.__len__() > 0:
            y_true, y_pred = evaluate(model1, ds, device=device)
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            result_true[station_id] = y_true
            result_pred[station_id] = y_pred
            y_true, y_pred = evaluate(model2, ds, device=device)
            y_pred = y_pred.reshape(-1, 1)
            result_pred2[station_id] = y_pred
    with open(path + 'result_true_' + str(e), 'wb') as f:
        pickle.dump(result_true, f)
    with open(path + 'result_pred_' + str(e), 'wb') as f:
        pickle.dump(result_pred, f)
    with open(path + 'result_pred_' + str(9-e), 'wb') as f:
        pickle.dump(result_pred2, f)
