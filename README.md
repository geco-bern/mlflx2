# An effective machine learning approach for predicting ecosystem CO2 assimilation across space and time

Piersilvio De Bartolomeis, Alexandru Meterez, Zixin Shu, Fanny Yang and Benjamin D. Stocker

## Abstract

Accurate predictions of environmental controls on ecosystem photosyn-
thesis are essential for understanding impacts of climate change and extreme events
on the carbon cycle and the provisioning of ecosystem services. Using time-series mea-
surements of ecosystem fluxes paired with measurements of meteorological variables
from a network of globally distributed sites and remotely sensed vegetation indices, we
train a recurrent deep neural network (Long-Short-Term Memory, LSTM), a simple
deep neural network (DNN), and a mechanistic, theory-based photosynthesis model
with the aim to predict ecosystem gross primary production (GPP). We test the mod-
els’ ability to spatially and temporally generalise across a wide range of environmental
conditions. Both neural network models outperform the theory-based model consider-
ing a leave-site-out cross-validation (LSOCV). The LSTM performs best and achieved
a mean R2of 0.78 across sites in the LSOCV and an average R2of 0.82 across rel-
atively moist temperate and boreal sites. This suggests that recurrent deep neural
networks provide a basis for a robust data-driven ecosystem photosynthesis modelling
in respective biomes. However, limits to global model upscaling are identified using
a cross-validation by vegetation types and by continents and we identified weakest
model performances at relatively arid sites where unknown vegetation exposure to
water limitation limits model reliability.

## Use

The repo contains code to run experiments on 2 models:

- FCN: A fully connected model
- LSTM: A recurrent model using and LSTM neural network

### Input files
We have 2 input files:

- `data/df_20210510.csv`: contains the raw data, without any imputation, and is used to compute the results excluding the imputed values
- `data/df_imputed.csv`: contains the imputed data, used to train the model


### Commands

These are the commands needed to train each of the models.

#### Leave-site-out
We save one CSV file, where each row contains a site and the model prediction when that site is left out of training.

###### LSTM Model
This model takes as input a sequence (the input data) and predicts another sequence, representing the GPP across the length of time.

```
python src/evaluation/leave_site_out_rnn.py
```

###### FCN Model
This model is an MLP neural network trained on the data. It was used as a baseline to compare to the rest of the methods.

```
python src/evaluation/leave_site_out_fcn.py
```

###### Random Forest 
In the `src/evaluation/random_forest` directory, there is a notebook that trains a Random Forest model, also used as a baseline.

###### Leave-vegetation-out
Run `python src/evaluation/leave_vegetation_out_rnn.py`, which will output a directory named `leave_{group_name}_out` (where `group_name` corresponds to the vegetation group that the model will be trained on. In this directory, we save separate CSV files for each in-distribution site (which is left out of training) and for each out-of-distribution vegetation group. 

###### Leave-continent-out
Run `python src/evaluation/leave_continent_out_rnn.py`, which will output a directory named `leave_{group_name}_out` (where `group_name` corresponds to the continent name that the model will be tested on). In this directory, we save for both the in-distribution continent and out-of-distribution continent, one CSV file per each in-distribution site (which is left out of training).

###### Run on cluster
To run on the cluster, execute:
```
bsub -R rusage[mem=10000,ngpus_excl_p=1] python SCRIPT_NAME_HERE.py
```

### Disclaimer
For any bug reports, send an [email to one of the authors](mailto:ameterez@student.ethz.ch).
