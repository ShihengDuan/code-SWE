# Python files:
hyper.py: hyper parameter searching for LSTM.  
hyper_attention: attention model hyper parameter searching.  
hyper_tcnn: tcnn model hyper parameter searching.  
data/SWE_Dataset: Dataset definition.  
data/preprocess: get mean and std from different training datasets (Rocky/WUS).
The corresponding preprocess python files use different mean and std.  
data/CO_dataset: Dataset for extrapolation.

# The order matters:
Should always keep:  
dyn_inputs = ['precip_prism', 'Tmean_prism', 'Tmax_prism', 'Tmin_prism']  
attributions = ['longitude', 'latitude', 'elevation_prism', 'dah', 'trasp']

# Files with extend:
Use gridMET forcing.  

123
