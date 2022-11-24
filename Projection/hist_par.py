import argparse
import inspect
import os
import sys
import warnings
import json
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import DataLoader
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
warnings.filterwarnings("ignore", category=FutureWarning)
from data.CO_dataset import CO_LOCADataset
from models.lstm import LSTM
from models.tcnn import TCNN
from models.attention import Attention


def get_args(): # determine the ensemble nnumber, CMIP model, and mountain area. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--model', type=str, default='CanESM2')
    parser.add_argument('--mountain', type=str, default='rocky')
    ## Mountain: Rocky, Sierra, Northern, Western, Cascade
    args = vars(parser.parse_args())
    return args

args = get_args()
ens = args['ens']
# model_type = 'TCNN'
model_type = 'LSTM'
# model_type = 'Attention'
# path = 'LSTM_proj_max_L1_simple/'
# path = 'LSTM_proj_max_L2/'
# path = 'LSTM_proj_median_L1/'
# path = 'LSTM_proj_median_L2/'
# path = 'LSTM_proj_median_L3/'
# path = 'LSTM_proj_max_L3_simple/'
path = 'LSTM_proj_median_L3_simple/'
path = 'LSTM_proj_max_L1_simSta/'

print('PATH: ', path)

device_id = args['device']
devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')
if device_id !=-1:
    device = devices[device_id] # device to use from the 4 GPUs.
else:
    device = devices[ens%4]
mountain = args['mountain']
if mountain.upper()=='ROCKY':
    num_lat = 128
    num_lon = 85
elif mountain.upper()=='SIERRA':
    num_lat = 72
    num_lon = 64
elif mountain.upper()=='CASCADE':
    num_lat = 133
    num_lon = 52
elif mountain.upper()=='NORTH':
    num_lat = 90
    num_lon = 141
elif mountain.upper()=='UTAH':
    num_lat = 78
    num_lon = 86
else:
    print('Mountain Not Supported! Go back and write your code. ')
HID = 128
KER = 7
LEVELS = 5
CHA = 64
RELU_FLAG = False

head = 16
num = 3
forward = 32
embedding = 32
model = args['model']
# attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
attributions = ['longitude', 'latitude', 'elevation_30m', 'dah_30m', 'trasp_30m']
attributions = ['longitude', 'latitude', 'elevation_30m']

'''forcings = {'pr': model + '-hist-pr.nc', 'rave': model + '-hist-rave.nc',
            'sph': model + '-hist-sph.nc', 'srad': model + '-hist-srad.nc', 'tmmn': model + '-hist-tmmn.nc',
            'tmmx': model + '-hist-tmmx.nc', 'vs': model + '-hist-vs.nc'}'''
forcings = {'pr': '../LOCA/'+model+'/pr_'+mountain+'_1981-2100.nc', 
            'rave': '../LOCA/'+model+'/relHumid_'+mountain+'_1981-2100.nc',
            'sph': '../LOCA/'+model+'/Qair_'+mountain+'_1981-2100.nc',
            'srad': '../LOCA/'+model+'/shortwave_in_'+mountain+'_1981-2100.nc',
            'tmmn': '../LOCA/'+model+'/tasmax_'+mountain+'_1981-2100.nc',
            'tmmx': '../LOCA/'+model+'/tasmin_'+mountain+'_1981-2100.nc',
            'vs': '../LOCA/'+model+'/windspeed_'+mountain+'_1981-2100.nc',
            }
'''forcings = {'pr': '../LOCA/'+model+'/pr_'+mountain+'_1981-2100.nc', 
            'srad': '../LOCA/'+model+'/shortwave_in_'+mountain+'_1981-2100.nc',
            'tmmn': '../LOCA/'+model+'/tasmax_'+mountain+'_1981-2100.nc',
            'tmmx': '../LOCA/'+model+'/tasmin_'+mountain+'_1981-2100.nc',
            }'''
n_inputs = len(attributions) + len(forcings)
# topo_file = 'topo_file_LOCA.nc'

# topo_file = 'topo_file_LOCA_regrid.nc'
topo_file = '../LOCA/'+ mountain + '_topo_file_30m.nc'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L2/LSTM_median_1/model_ens_'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L2/LSTM/model_ens_'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L3/LSTM_median_1/11-13-15-15-32/model_ens_'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L3/LSTM_median_0/11-13-15-15-32/model_ens_'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L3/LSTM_median_0/11-13-16-07-39/model_ens_'
# mod_prefix = '../gridMET/runs_relative_proj_30m_L3/LSTM_median_1/11-13-16-07-35/model_ens_'
mod_prefix = '../gridMET/runs_relative_proj_30m_L1/LSTM_median_0/11-19-16-11-01/model_ens_'

config = {'model_prefix': mod_prefix, 'forcings': forcings, 
          'path':path, 'attribution': attributions,
         }
with open(path+'config.json', 'w') as fp:
    json.dump(config, fp)

def inference(lat_id, device=device):
    ds = CO_LOCADataset(forcings=forcings, lat=lat_id, attributions=attributions, topo_file=topo_file,
                        window_size=180, scenario='hist', adjust_unit=True, 
                        scaler_mean='../gridMET/Rocky/gridmet_mean_norm.nc', # this is mean from SNOTEL stations. should be static. 
                        scaler_std='../gridMET/Rocky/gridmet_std_norm.nc')
    if model_type == 'LSTM':
        model = LSTM(hidden_units=HID, input_size=n_inputs, relu_flag=RELU_FLAG)
        # model_path = '/tempest/duan0000/SWE/gridMET/runs_relative_proj/LSTM_1e-4/model_ens_' + str(ens)
        # model_path = '../gridMET/runs_relative_proj_30m_L2/LSTM/model_ens_' + str(ens)
        model_path = mod_prefix + str(ens)
        # model_path = '../gridMET/runs_relative_proj_30m_L1/LSTM_median_1/model_ens_' + str(ens)
        # model_path = '../gridMET/runs_relative_proj_30m_L1/LSTM_median_0/model_ens_' + str(ens)
        # model_path = '../gridMET/runs_relative_proj_30m_L1/LSTM_median_0/11-07-20-07-40/model_ens_' + str(ens)
        print('model_path: ', model_path)
    elif model_type == 'TCNN':
        model = TCNN(kernal_size=KER, num_levels=LEVELS, num_channels=CHA, input_size=n_inputs)
        model_path = '/tempest/duan0000/SWE/gridMET/runs_relative_clean/TCNN_1e-4/model_ens_' + str(ens)
    elif model_type.lower() == 'attention':
        model = Attention(num_att_layers=num, dim_feedforward=forward, embedding_size=embedding, n_head=head,
                          input_size=n_inputs)
        model_path = '/tempest/duan0000/SWE/gridMET/runs_relative_clean/Attention/model_ens_' \
                     + str(ens)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    pred = []
    for data in loader:
        x_d_new, x_attr = data
        x_d_new, x_attr = x_d_new.to(device), x_attr.to(device)
        y_hat = model(x_d_new, x_attr)[0]
        y_hat_sub = y_hat[:, -1:, :]
        pred.append(y_hat_sub.to('cpu').data.numpy())
    pred = np.concatenate(pred).flatten()
    # np.save(path + str(lat_id) + '_' + str(ens), pred)
    return lat_id, pred


if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    prediction = np.empty((num_lat, num_lon, 8860))

    with Pool(4) as pool:
        lat_ids = np.arange(num_lat)
        for results in tqdm(pool.imap(inference, lat_ids), total=len(lat_ids), position=0):
            lat, pred = results
            split_data = np.split(pred, num_lon)
            # print(lat, ' ', len(pred), ' ', len(pred) / 108, ' ', split_data[0].shape)
            for i in range(num_lon):
                prediction[lat, i] = split_data[i]
    np.save(path + model + '-' + mountain + '-hist_' + str(ens), prediction)
