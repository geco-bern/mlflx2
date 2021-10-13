from model.model import Model
from preprocess import prepare_df_lgocv
from sklearn.metrics import r2_score
import torch
import pandas as pd
import argparse
import torch.nn.functional as F
import numpy as np
import operator
import pdb
import pickle
from tqdm import tqdm
from copy import deepcopy

EUROPE = [8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
35,36,37,39,40]
US = [42,43,44,45,46,47,48,49,50,51,52]

def compute_bias(x_test,y_test,conditional_test):
    i = 0
    bias = []
    for (x, y,conditional) in zip(x_test, y_test, conditional_test):
        x = torch.FloatTensor(x).to(DEVICE)
        y = torch.FloatTensor(y).to(DEVICE)
        c = torch.FloatTensor(conditional).to(DEVICE)
        y_pred = model(x, c)
        bias.append((y_pred - y).detach().cpu().numpy())
        i += 1
    
    return np.concatenate(bias)

# Parse arguments 
parser = argparse.ArgumentParser(description='CV LSTM')

parser.add_argument('-gpu', '--gpu', default=None, type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=None, type=int,
                      help='number of cv epochs ()')

parser.add_argument('-c', '--conditional',  type=int,
                      help='enable conditioning')

args = parser.parse_args()
DEVICE = torch.device("cuda:" + args.gpu)
torch.manual_seed(40)

#Importing data
data = pd.read_csv('utils/df_imputed.csv', index_col=0)
data = data.drop(columns='date')
raw = pd.read_csv('../data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
raw = raw[raw.index != 'CN-Cng']

df_sensor_europe,df_sensor_us, df_meta_europe, df_meta_us, df_gpp_europe,df_gpp_us = prepare_df_lgocv(data)
sites = raw.index.unique()

#define model parameter
INPUT_FEATURES = len(df_sensor_europe[0].columns) 
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = len(df_meta_europe[0].columns)

masks = []
for s in sites:
    mask = raw[raw.index == s].isna().values
    masks.append(list(map(operator.not_, mask)))

cv_r2 = []
sites = []
bias_europe = []
bias_us  = []

for k in tqdm(range(len(df_sensor_europe))):

    train_sites = deepcopy(df_sensor_europe)
    conditional_sites  = deepcopy(df_meta_europe)
    gpp_sites  =  deepcopy(df_gpp_europe)
    
    train_sites = np.delete(train_sites, k)
    test_sites_in = [df_sensor_europe[k]]
    conditional_sites =  np.delete(conditional_sites, k)
    conditional_sites_in = [df_meta_europe[k]]
    gpp_sites = np.delete(gpp_sites, k)
    gpp_sites_in = [df_gpp_europe[k]]


    x_train = [x.values for x in train_sites]
    conditional_train = [x.values for x in conditional_sites]
    y_train = [x.values.reshape(-1,1) for x in gpp_sites]

    x_test_eu = [x.values for x in test_sites_in]
    conditional_test_eu =  [x.values for x in conditional_sites_in]
    y_test_eu = [x.values.reshape(-1, 1) for x in gpp_sites_in]

    x_test_us = [x.values for x in df_sensor_us]
    conditional_test_us =  [x.values for x in df_meta_us]
    y_test_us = [x.values.reshape(-1, 1) for x in df_gpp_us]

    model = Model(INPUT_FEATURES, CONDITIONAL_FEATURES, HIDDEN_DIM, args.conditional, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    best_r2 = 0
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        train_r2 = 0.0
        model.train()
        for (x, y, conditional) in zip(x_train, y_train, conditional_train):
            x = torch.FloatTensor(x).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            c = torch.FloatTensor(conditional).to(DEVICE)
            y_pred = model(x, c)
            optimizer.zero_grad()
            loss = F.mse_loss( y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
        
        model.eval()
        with torch.no_grad():
                i = 0
                r2 = 0
                for (x, y, conditional) in zip(x_test_eu, y_test_eu, conditional_test_eu):
                    x = torch.FloatTensor(x).to(DEVICE)
                    y = torch.FloatTensor(y).to(DEVICE)
                    c = torch.FloatTensor(conditional).to(DEVICE)
                    y_pred = model(x, c)
                    test_loss = F.mse_loss( y_pred, y)
                    r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                    #r2[str(df_sensor_europe[i].index[0])] = test_r2
                    #print(f"Test Site: {df_sensor_us[i].index[0]} R2: {r2[str(df_sensor_us[i].index[0])]}")
                    #print("-------------------------------------------------------------------")
                    i += 1
                r2 /= i

                if r2 >= best_r2:
                    best_r2 = r2
                    temp_bias_us = compute_bias(x_test_us,y_test_us,conditional_test_us)
                    temp_bias_europe = compute_bias(x_test_eu,y_test_eu,conditional_test_eu)

    bias_europe.append(temp_bias_europe)
    bias_us.append(temp_bias_us)  


    
bias_europe = np.concatenate(bias_europe).reshape(-1)

bias_us = np.concatenate(bias_us).reshape(-1)

np.save("train_eu_bias_europe.npy",bias_europe)
np.save("train_eu_bias_us.npy",bias_us)




    
