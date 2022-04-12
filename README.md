# Introduction


This repo contains the necessary code for snow water equivalent (SWE) prediction and projection in the Western CONUS. A corresponding presentation can be found at [here](https://www.essoar.org/doi/abs/10.1002/essoar.10509011.1). We will update our manuscript when it is ready.  

The 'ensemble' files performes initial-weight ensemble run for the deep learning models. 

The 'Projection' is for projection analysis with RCP8.5 LOCA datasets.  

The 'models' folder contains necessary code for model definition in PyTorch.  

The 'data' folder contains files for dataset definition, including SNOTEL and Rocky Mountains.

'NSIDC' is used to process NSIDC-UA dataset and ExtraGirdMet is extrapolation over the Rocky Mountains. 

'Projection' stands for historical and future projections forced by LOCA downscaled CMIP5 data. 

Associated model weights and outputs: 10.5281/zenodo.6419931 and 10.5281/zenodo.6430612 


