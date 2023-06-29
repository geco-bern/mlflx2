# An effective machine learning approach for predicting ecosystem CO2 assimilation across space and time

This repository contains the code for the experiments presented in the paper authored by Piersilvio De Bartolomeis, Alexandru Meterez, Zixin Shu, Fanny Yang and Benjamin D. Stocker, affiliated with ETH Zurich. 

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
