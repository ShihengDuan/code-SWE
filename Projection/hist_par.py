import argparse
import inspect
import os
import sys
import warnings

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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--ens', type=int, help="model ensemble member")
args = vars(parser.parse_args())

ens = args['ens']
# model_type = 'TCNN'
model_type = 'LSTM'
# model_type = 'Attention'
path = 'LSTM_proj/'

device = torch.device('cuda:1')
HID = 128
KER = 7
LEVELS = 5
CHA = 64
RELU_FLAG = False

head = 16
num = 3
forward = 32
embedding = 32
# model = 'CanESM2'
model = args['model']
attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
forcings = {'pr': model + '-hist-pr.nc', 'rave': model + '-hist-rave.nc',
            'sph': model + '-hist-sph.nc', 'srad': model + '-hist-srad.nc', 'tmmn': model + '-hist-tmmn.nc',
            'tmmx': model + '-hist-tmmx.nc', 'vs': model + '-hist-vs.nc'}

n_inputs = len(attributions) + len(forcings)
topo_file = 'topo_file_LOCA.nc'


# topo_file = 'topo_file_LOCA_regrid.nc'


def inference(lat_id, device=device):
    ds = CO_LOCADataset(forcings=forcings, lat=lat_id, attributions=attributions, topo_file=topo_file,
                        window_size=180, scenario='hist')
    if model_type == 'LSTM':
        model = LSTM(hidden_units=HID, input_size=n_inputs, relu_flag=RELU_FLAG)
        model_path = '/tempest/duan0000/SWE/gridMET/runs_relative_proj/LSTM_1e-4/model_ens_' + str(ens)
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
    np.save(path + str(lat_id) + '_' + str(ens), pred)
    return lat_id, pred


if __name__ == '__main__':

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    prediction = np.empty((112, 72, 7126))

    with Pool(4) as pool:
        lat_ids = np.arange(112)
        for results in tqdm(pool.imap(inference, lat_ids), total=len(lat_ids), position=0):
            lat, pred = results
            split_data = np.split(pred, 72)
            # print(lat, ' ', len(pred), ' ', len(pred) / 108, ' ', split_data[0].shape)
            for i in range(72):
                prediction[lat, i] = split_data[i]
    np.save(path + model + '-hist_' + str(ens), prediction)
