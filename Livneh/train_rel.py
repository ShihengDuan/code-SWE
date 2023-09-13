import os
import pickle
import warnings
import sys
import inspect
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
import json
from dataset import gridMETDatasetStation, gridMETRelativeStation
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from data.snowcover import SnowcoverDS
from models.lstm import LSTM_multi, LSTM_classifier
# from models.loss import L12loss, L12loss_native
warnings.filterwarnings("ignore", category=RuntimeWarning)
import argparse

from datetime import datetime

now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%m-%d-%H-%M-%S")
print("date and time =", dt_string)	

torch.backends.cudnn.benchmark=True # set the optimal algorithm for tasks. 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--style', type=str)
    args = vars(parser.parse_args())
    return args

def train_multi(model, ds1, ds2, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader1 = DataLoader(ds1, batch_size=128, shuffle=True)
    loader2 = DataLoader(ds2, batch_size=128, shuffle=True)
    print('TRAINING')
    # train on regression problem
    for epoch in tqdm(range(50)):
        if (epoch%2==0):
            loss_val = []
            for data in loader1:
                x_d_new, x_attr, y_new, qstd = data
                x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                            y_new.to(device), qstd.to(device)
                y_sub = y_new[:, -1:] # zero: -6.726943
                output = model(x_d_new, x_attr)
                y_hat_sub = output[0][:, -1:, 0:1] # first dimension for regression
                y_prob = output[-1][:, -1:, 1:]
                y_hat_sub_w_prob = y_prob * y_hat_sub
                loss = loss_fn1(y_hat_sub, y_sub)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                loss_val.append(loss.item())
            loss_val = np.mean(loss_val)
            print('Loss: ', loss_val)
            if writer is not None:
                writer.add_scalar('training mse', scalar_value=loss_val, global_step=epoch)
        else:
            # train on classification problem
            for data in loader2:
                x_d_new, x_attr, y_new, qstd = data
                # print(x_d_new.shape, x_attr.shape, y_new.shape)
                x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                            y_new.to(device), qstd.to(device)
                y_sub = y_new[:, -1:]
                y_hat = model(x_d_new, x_attr)[-1] # y_hat, h_n, c_n, y_hat_prob
                y_hat_sub = y_hat[:, -1:, 1:] # last dimension for regression
                loss = loss_fn2(y_hat_sub, y_sub)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
        if epoch==47:
            model2 = model
        
    return model, model2

def train_mix(model, ds1, ds2, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader1 = DataLoader(ds1, batch_size=128, shuffle=True)
    loader2 = DataLoader(ds2, batch_size=128, shuffle=True)
    print('TRAINING')
    # train on regression problem
    for epoch in tqdm(range(50)):
        if (epoch%2==0):
            loss_val = []
            for data in loader1:
                x_d_new, x_attr, y_new, qstd = data
                x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                            y_new.to(device), qstd.to(device)
                y_sub = y_new[:, -1:] # zero: -6.726943
                output = model(x_d_new, x_attr)
                y_hat_sub = output[0][:, -1:, 0:1] # first dimension for regression
                y_prob = output[-1][:, -1:, 1:]
                y_hat_sub_w_prob = y_prob * y_hat_sub
                # loss = loss_fn1(y_hat_sub, y_sub) + loss_fn2(y_prob, (y_sub>-6.726943).float())
                loss = loss_fn1(y_hat_sub, y_sub) + loss_fn2(y_prob, (y_sub>0).float())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                loss_val.append(loss.item())
            loss_val = np.mean(loss_val)
            print('Loss: ', loss_val)
            if writer is not None:
                writer.add_scalar('training mse', scalar_value=loss_val, global_step=epoch)
        else:
            # train on classification problem
            for data in loader2:
                x_d_new, x_attr, y_new, qstd = data
                # print(x_d_new.shape, x_attr.shape, y_new.shape)
                x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                            y_new.to(device), qstd.to(device)
                y_sub = y_new[:, -1:]
                y_hat = model(x_d_new, x_attr)[-1] # y_hat, h_n, c_n, y_hat_prob
                y_hat_sub = y_hat[:, -1:, 1:] # last dimension for regression
                loss = loss_fn2(y_hat_sub, y_sub)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
        if epoch==47:
            model2 = model
        
    return model, model2

def train_regress(model, ds1, ds2, lr, device=torch.device('cuda:0'), writer=None):
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader1 = DataLoader(ds1, batch_size=128, shuffle=True)
    # loader2 = DataLoader(ds2, batch_size=128, shuffle=True)
    print('TRAINING')
    # train on regression problem
    for epoch in tqdm(range(50)):
        loss_val = []
        for data in loader1:
            x_d_new, x_attr, y_new, qstd = data
            x_d_new, x_attr, y_new, qstd = x_d_new.to(device), x_attr.to(device), \
                                        y_new.to(device), qstd.to(device)
            y_sub = y_new[:, -1:] # zero: -6.726943
            output = model(x_d_new, x_attr)
            y_hat_sub = output[0][:, -1:, 0:1] # first dimension for regression
            y_prob = output[-1][:, -1:, 1:]
            # y_hat_sub_w_prob = y_prob * y_hat_sub
            loss = loss_fn1(y_hat_sub, y_sub)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_val.append(loss.item())
        loss_val = np.mean(loss_val)
        print('Loss: ', loss_val)
        if writer is not None:
            writer.add_scalar('training mse', scalar_value=loss_val, global_step=epoch)
        if epoch==47:
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
style = args['style']
loss_fn1 = nn.MSELoss()
# loss_fn1 = nn.L1Loss()
loss_fn2 = nn.BCELoss()
device_id = args['device']
devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')

topo_file = '../snowcover/snowcover_topo.nc' # PRISM DEM

WINDOW_SIZE = 180
LR = 1e-4
HID = 128
ENS = 10
device = devices[ens%4]
model_type = 'LSTM'


attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp'] # PRISM DEM

forcings = {'pr': 'pr_snowcover.nc', 
            'tmmn': 'tmmn_snowcover.nc',
            'tmmx': 'tmmx_snowcover.nc', 
            'srad': 'srad_snowcover.nc', 
            }


path = 'runs_PRISM/+'dt_string+'-'+style+'-relative/'
# path = dt_string+'-regress/'
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

n_inputs = len(attributions) + len(forcings)
# Save input information into JSON file. 
with open(path+'forcings.json', 'w') as fp:
    json.dump(forcings, fp)
att_dic = {'attribution': attributions}
with open(path+'attributions.json', 'w') as fp:
    json.dump(att_dic, fp)

target = ['SWE']
sc_ds = []

for point_id in tqdm(range(613), desc='building dataset'):# 613
    ds = SnowcoverDS(forcings, topo_file, attributions, point_id, target='snowcover', 
                 window_size=180, mode='Train')
    if ds.__len__() > 0:
        sc_ds.append(ds)
print('TOTAL Points: ', len(sc_ds))
sc_ds = ConcatDataset(sc_ds)
print(sc_ds.__len__())

swe_ds = []
forcings = {'pr': 'Prec_wus_clean.nc', 
            'tmmn': 'Tmin_wus_clean.nc',
            'tmmx': 'Tmax_wus_clean.nc',
            'netrad': 'NetRad_wus_clean.nc'
            }
topo_file = '../SNOTEL/raw_wus_snotel_topo_clean.nc' # PRISM DEM
for station_id in tqdm(range(581), desc='Load Data'):  # 581
    ds = gridMETRelativeStation(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                               mode='TRAIN', topo_file=topo_file, station_id=station_id)
    swe_ds.append(ds)
swe_ds = ConcatDataset(swe_ds)
print(swe_ds.__len__())

model = LSTM_multi(hidden_units=HID, input_size=n_inputs)
model = model.to(device)
if style=='mix':
    model1, model2 = train_mix(model, swe_ds, sc_ds, LR, device=device)
elif style=='regress':
    model1, model2 = train_regress(model, swe_ds, sc_ds, LR, device=device)
elif style=='multi':
    model1, model2 = train_multi(model, swe_ds, sc_ds, LR, device=device)
else:
    print('no such training style')

torch.save(model1.state_dict(), path + 'model_ens_' + str(ens))
torch.save(model2.state_dict(), path + 'model_ens_' + str(9-ens))
result_true = {}
result_pred1 = {}
result_pred2 = {}
topo_file = '../snowcover/snowcover_topo.nc' # PRISM DEM
for point_id in tqdm(range(613), desc='test_ds'):
    ds = SnowcoverDS(forcings, topo_file, attributions, point_id, target='snowcover', 
                 window_size=180, mode='Test')
    if ds.__len__() > 0:
        y_true, y_pred = evaluate(model1, ds, device=device)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        result_true[point_id] = y_true
        result_pred1[point_id] = y_pred
        y_true, y_pred = evaluate(model2, ds, device=device)
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        result_true[point_id] = y_true
        result_pred2[point_id] = y_pred

with open(path + 'result_true_' + str(9-ens), 'wb') as f:
    pickle.dump(result_true, f)
with open(path + 'result_pred_' + str(9-ens), 'wb') as f:
    pickle.dump(result_pred2, f)
with open(path + 'result_pred_' + str(ens), 'wb') as f:
    pickle.dump(result_pred1, f)
