This is using Livneh dataset to train the LSTM model, since it is used as the reference data for LOCA. We want to keep the climatology the same.   
Livneh 2015 provides daily precipitaiton, min and max temperature. Wind speed is also included, but LOCA uses VIC to simulate wind speed (it is not directly downscaled).   
  
```Prec_wus_clean.nc```: precipitation. ```Tmax_wus_clean.nc```: maximum temperature. ```Tmin_wus_clean.nc```: minimum temperature. All the three files are for the forcing data (shape: time, 581 stations).  
```livneh_mean.nc``` and ```livneh_std.nc``` are mean and std for the training Livneh forcings.  
