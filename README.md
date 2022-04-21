# An effective machine learning approach for predicting ecosystem CO2 assimilation across space and time

This repository contains the code for the experiments presented in the paper authored by Piersilvio De Bartolomeis, Alexandru Meterez, Zixin Shu, Fanny Yang and Benjamin D. Stocker, affiliated with ETH Zurich. 

The repo contains code to run experiments on 2 models:

- FCN: A fully connected model
- LSTM: A recurrent model using and LSTM neural network

### Commands

These are the commands needed to train each of the models.

###### LSTM Model
This model takes as input a sequence (the input data) and predicts another sequence, representing the GPP across the length of time.

```
python evaluation/leave_site_out_rnn.py
```

###### FCN Model
This model is an MLP neural network trained on the data. It was used as a baseline to compare to the rest of the methods.

```
python evaluation/leave_site_out_fcn.py
```

###### Random Forest 
In the `evaluation/random_forest` directory, there is a notebook that trains a Random Forest model, also used as a baseline.

###### Other experiments
The experiments that test leave-vegetation-out and leave-continent-out can be run in a similar fashion. To see the list of arguments (names of vegetations/continejts) look inside the files.

### Disclaimer
While the models have been extensively tested, we are merely humans. For any bug reports, send an [email to one of the authors](mailto:ameterez@student.ethz.ch).