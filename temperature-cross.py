import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from models.lstm_light import LSTM_PL
import xarray as xa
from data.SWE_Dataset import gridMETDatasetStationWY
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

WINDOW_SIZE = 180
target = ['SWE']

attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']
forcings = {'pr': '/tempest/duan0000/SWE/gridMET/pr_wus_clean.nc', 'rmax': '/tempest/duan0000/SWE/gridMET/rmax_wus_clean.nc',
            'rmin': '/tempest/duan0000/SWE/gridMET/rmin_wus_clean.nc',
            'sph': '/tempest/duan0000/SWE/gridMET/sph_wus_clean.nc', 'srad': '/tempest/duan0000/SWE/gridMET/srad_wus_clean.nc',
            'tmmn': '/tempest/duan0000/SWE/gridMET/tmmn_wus_clean.nc',
            'tmmx': '/tempest/duan0000/SWE/gridMET/tmmx_wus_clean.nc', 'vpd': '/tempest/duan0000/SWE/gridMET/vpd_wus_clean.nc',
            'vs': '/tempest/duan0000/SWE/gridMET/vs_wus_clean.nc'}

topo = '/tempest/duan0000/SWE/SNOTEL/raw_wus_snotel_topo_clean.nc'
swe = xa.open_dataset(topo).SWE
# temperature split train
tave = xa.open_dataarray('/tempest/duan0000/SWE/gridMET/tave_wus_clean.nc')
wy_tave = []
for i in range(1980, 2018):
    t_year = tave.sel(time=slice(np.datetime64(str(i)+'-10-01'), np.datetime64(str(i+1)+'-09-30'))).mean()
    wy_tave.append(t_year.data)
hot_test = []
cold_train = []
for year in range(1980, 2018):
    if wy_tave[year-1980]>np.quantile(wy_tave, .7):
        hot_test.append(year)
    else:
        cold_train.append(year)
print('COLD temperature train ds: ', len(cold_train))
print('HOT temperature test ds: ', len(hot_test))

hot_train = []
cold_test = []
for year in range(1980, 2018):
    if wy_tave[year-1980]<np.quantile(wy_tave, .3):
        cold_test.append(year)
    else:
        hot_train.append(year)
print('HOT temperature train ds: ', len(hot_train))
print('COLD temperature test ds: ', len(cold_test))


names = ['hot-train-cold-test', 'cold-train-hot-test']
train_ids = [hot_train, cold_train]
test_ids = [cold_test, hot_test]

for scheme in range(2):
    train_wy = train_ids[scheme]
    test_wy = test_ids[scheme]
    all_train_time = []
    tave = xa.open_dataarray('/tempest/duan0000/SWE/gridMET/tave_wus_clean.nc')
    for year in train_wy:
        time_slice = slice(str(year)+'-10-01', str(year+1)+'-09-30')
        timestamp = tave.sel(time=time_slice).time.data
        for time in timestamp:
            all_train_time.append(time)
    target_mean = swe.sel(time=all_train_time).mean().data
    target_std = swe.sel(time=all_train_time).std().data
    print(names[scheme])
    save_path = '/tempest/duan0000/SWE/mountains/output-temperature/'+names[scheme]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('save path created.')
    train_ds = []
    for station in tqdm(range(581), desc='train ds'):
        ds = gridMETDatasetStationWY(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                   mode='TRAIN', topo_file=topo, station_id=station, 
                                   train_wy=train_wy, test_wy=test_wy, 
                                   target_mean=target_mean, target_std=target_std,
                                   all_train_time=all_train_time) 
        train_ds.append(ds)
    print(train_ds.__len__(), ' samples for training')
    
    train_ds = ConcatDataset(train_ds)
    loader = DataLoader(train_ds, shuffle=True, batch_size=128, num_workers=4)
    model = LSTM_PL(hidden=256, in_channels=len(
        forcings)+len(attributions), zero=None) # +1 if plus snow17
    log_path = '/tempest/duan0000/SWE/mountains/PL_log_temperature/' + \
        str(names[scheme])+'/'
    tb_logger = pl_loggers.TensorBoardLogger(log_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path, every_n_epochs=5, save_on_train_epoch_end=True, filename=None, save_top_k=-1)
    trainer = Trainer(max_epochs=50, gpus=[1], logger=tb_logger, default_root_dir=log_path,
                  gradient_clip_val=1., callbacks=[checkpoint_callback])
    trainer.fit(model, loader)
    trainer.save_checkpoint(log_path + '/LSTM.ckpt')
    # test:
    for station in tqdm(range(581), desc='testing'):
        ds = gridMETDatasetStationWY(forcings=forcings, attributions=attributions, target=target, window_size=WINDOW_SIZE,
                                   mode='TEST', topo_file=topo, station_id=station, 
                                   train_wy=train_wy, test_wy=test_wy, 
                                   target_mean=target_mean, target_std=target_std,
                                   all_train_time=all_train_time)
        if ds.__len__()>0:
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            trainer = Trainer(gpus=[1], enable_progress_bar=False)
            output = trainer.predict(model, dataloaders=loader)
            real = [x[0].data.numpy() for x in output]
            pred = [x[1].data.numpy() for x in output]
            real = np.concatenate(real, axis=0)
            pred = np.concatenate(pred, axis=0)

            np.save(save_path+'/pred_'+str(station), pred)
            np.save(save_path+'/obs_'+str(station), real)
        

