# dev test
import os.path
import pickle

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import LSFEnvironment
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data.SWE_Dataset import gridMETDatasetStation
from models.PL_light import Attention_PL
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ens', type=int)
    args = vars(parser.parse_args())
    return args

args = get_args()
ens = args['ens']
devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
print(ens, ' STARTED')

topo_file = 'SNOTEL/raw_snotel_topo_30m.nc'

WINDOW_SIZE = 180
RELU_FLAG = False
LR = 1e-4
HID = 128
ENS = 10
device = devices[ens%4] # device to use from the 4 GPUs. 
model_type = 'Attention'
KER = 7
LEVELS = 5
CHA = 64

head = 16
num = 3
forward = 32
embedding = 32

path = 'gridMET/runs_30m/' + model_type.upper() + '/'
if not os.path.exists(path):
    os.makedirs(path)
    print('Make output directory')
else:
    print('Folder is here. ')
loss_fn = nn.MSELoss()
attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
attributions = ['longitude', 'latitude', 'elevation_30m', 'dah_30m', 'trasp_30m']
forcings = {'pr': 'gridMET/pr_wus_clean.nc', 'rmax': 'gridMET/rmax_wus_clean.nc', 'rmin': 'gridMET/rmin_wus_clean.nc',
            'sph': 'gridMET/sph_wus_clean.nc', 'srad': 'gridMET/srad_wus_clean.nc', 'tmmn': 'gridMET/tmmn_wus_clean.nc',
            'tmmx': 'gridMET/tmmx_wus_clean.nc', 'vpd': 'gridMET/vpd_wus_clean.nc', 'vs': 'gridMET/vs_wus_clean.nc'}
forcings = {'pr': 'gridMET/pr_wus_clean.nc', 'rmax': 'gridMET/rmax_wus_clean.nc', 'rmin': 'gridMET/rmin_wus_clean.nc'}
n_inputs = len(attributions) + len(forcings)
target = ['SWE']
train_ds = []

for station_id in range(581):  # 765
    ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target, 
                               window_size=WINDOW_SIZE,
                               mode='TRAIN', topo_file=topo_file, station_id=station_id)
    train_ds.append(ds)
train_ds = ConcatDataset(train_ds)
print(train_ds.__len__())

tb_logger = pl_loggers.TensorBoardLogger(path)
checkpoint_callback = ModelCheckpoint(
    dirpath=path, every_n_epochs=5, save_on_train_epoch_end=True, filename=None, save_top_k=-1)
print(ens, ' Start')
model = Attention_PL(num_att_layers=num, dim_feedforward=forward, embedding_size=embedding, n_head=head,
                        input_size=n_inputs)
trainer = Trainer(max_epochs=50, gpus=[ens%4], logger=tb_logger, 
                default_root=path, gradient_clip_val=1., 
                callbacks=[checkpoint_callback],
                plugins=[LSFEnvironment()],
                )
loader = DataLoader(ds, batch_size=128, shuffle=True)
trainer.fit(model, loader)
trainer.save_checkpoint(path + 'att.ckpt')

result_true = {}
result_pred = {}
trainer = Trainer(gpus=[ens%4], enable_progress_bar=False)
# for station in range(1, 285):  # or 814
for station_id in tqdm(range(581), desc='test_ds'):
    ds = gridMETDatasetStation(forcings=forcings, attributions=attributions, target=target,
                                window_size=WINDOW_SIZE,
                                mode='TEST', topo_file=topo_file,
                                station_id=station_id)
    if ds.__len__() > 0:
        test_dl = DataLoader(ds, batch_size=128, shuffle=False)
        output = trainer.predict(model, dataloaders=test_dl)
        real = [x[0].data.numpy() for x in output]
        pred = [x[1].data.numpy() for x in output]
        real = np.concatenate(real, axis=0)
        pred = np.concatenate(pred, axis=0)
        y_true = real.reshape(-1, 1)
        y_pred = pred.reshape(-1, 1)
        result_true[station_id] = y_true
        result_pred[station_id] = y_pred

with open(path + 'result_true_' + str(ens), 'wb') as f:
    pickle.dump(result_true, f)
with open(path + 'result_pred_' + str(ens), 'wb') as f:
    pickle.dump(result_pred, f)
    
