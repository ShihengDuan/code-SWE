# Introduction


This repo contains the necessary code for snow water equivalent (SWE) prediction and projection in the Western CONUS. A corresponding presentation can be found at [here](https://www.essoar.org/doi/abs/10.1002/essoar.10509011.1). We will update our manuscript when it is ready.  

Files begin with ```ensemble``` performe initial-weight ensemble run for the deep learning models. 

```Projection``` is the folder for projection analysis with historical and RCP8.5 LOCA datasets.  

```models``` contains necessary code for model definition in PyTorch.  

```data``` contains files for dataset definition, including SNOTEL and Rocky Mountains.

```NSIDC``` is used to process NSIDC-UA dataset and ```ExtraGirdMet``` is extrapolation over the Rocky Mountains. 


Associated model weights and outputs: 10.5281/zenodo.6419931 and 10.5281/zenodo.6430612 


# Major Mountain Ranges

Southern Rocky (Colorado): 35N-42N, 109W-104.5W    
Sierra Nevada: 35.5N-39.5N, 120.5W-118W    
Northern Rocky (Wyoming): 42N-47N, 116.5W-108.5W    
Cascade: 41N-49N, 123W-120W    
Western Rocky (Utah): 37N-41.5N, 114W-109W    
![Mountains](https://github.com/ShihengDuan/code-SWE/blob/dev/Mountains.png?raw=true, "MountainRanges")    


