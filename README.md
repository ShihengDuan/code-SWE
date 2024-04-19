# Introduction


This repo contains the necessary code for snow water equivalent (SWE) prediction and projection in the Western CONUS. Corresponding [presentation](https://www.essoar.org/doi/abs/10.1002/essoar.10509011.1) and [manuscript](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023WR035009) are available.   

Files begin with ```ensemble``` performe initial-weight ensemble run for the deep learning models. 

```Projection``` is the folder for projection analysis with historical and RCP8.5 LOCA datasets.  

```models``` contains necessary code for model definition in PyTorch.  

```data``` contains files for dataset definition, including SNOTEL and Rocky Mountains.

```NSIDC``` is used to process NSIDC-UA dataset and ```ExtraGirdMet``` is extrapolation over the Rocky Mountains. 


Associated model weights and outputs: 10.5281/zenodo.6419931 and 10.5281/zenodo.6430612 


