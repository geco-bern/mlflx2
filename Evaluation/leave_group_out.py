import sys
sys.path.append("..")
from model.model import Model
from preprocess_new import prepare_df
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
from random import shuffle

torch.manual_seed(40)

EUROPE = [8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
35,36,37,39,40]
US = [42,43,44,45,46,47,48,49,50,51,52]
ALL_IDX = list(range(0, 53))
INPUT_FEATURES = 11
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = 21

def compute_bias(model, x_test, y_test, device):
    bias = []
    for (x, y) in zip(x_test, y_test):
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        y_pred = model(x, None)
        bias.append((y_pred - y).detach().cpu().numpy())
    
    return np.concatenate(bias)

if __name__ == '__main__':
    # Parse arguments 
    parser = argparse.ArgumentParser(description='CV LSTM')

    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--n_epochs', default=5, type=int, help='number of cv epochs', required=True)
    parser.add_argument('--conditional',  type=int, default=0, help='enable conditioning')
    parser.add_argument('--group_name', type=str)

    args = parser.parse_args()

    TEST_IDX = None
    if args.group_name == "Europe":
        TEST_IDX = EUROPE
    elif args.group_name == "US":
        TEST_IDX = US
    assert TEST_IDX is not None, "Please select a group."
    TRAIN_IDX = list(set(ALL_IDX) - set(TEST_IDX))
    DEVICE = args.device
    
    #Importing data
    data = pd.read_csv('../utils/df_imputed.csv', index_col=0)
    data = data.drop(columns='date')
    sites = data.index.unique().values
    test_out_sites = sites[TEST_IDX]

    raw = pd.read_csv('../data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
    raw = raw[raw.index != 'CN-Cng']

    #Prepare the metadata
    meta_data = pd.get_dummies(data[['classid','igbp_land_use']])
    masks = []
    for s in sites:
        mask = raw[raw.index == s].isna().values
        masks.append(list(map(operator.not_, mask)))

    cv_r2 = []
    bias_in_all = []
    bias_out_all = []

    for s in TRAIN_IDX:
        PROC_TRAIN_IDX = np.asarray(list(set(TRAIN_IDX) - set([s]))) # remove one site from the train set
        # PROC_TEST_IDX = np.asarray(TEST_IDX + [s]) # and add it to the test set
        PROC_TEST_IDX = np.asarray([s]) # and add it to the test set
        proc_train_sites = sites[PROC_TRAIN_IDX]
        proc_test_sites = sites[PROC_TEST_IDX]

        # From the test set, split in in_data and out_data
        test_in_sites = [sites[s]]
        # test_out_sites = sites[TEST_IDX]
        test_in_data = pd.concat([data[data.index == site] for site in test_in_sites if data[data.index == site].size != 0])
        test_out_data = pd.concat([data[data.index == site] for site in test_out_sites if data[data.index == site].size != 0])
        
        # Get the train/test data based on the processed sites
        train_data = pd.concat([data[data.index == site] for site in proc_train_sites if data[data.index == site].size != 0])
        test_data = pd.concat([data[data.index == site] for site in proc_test_sites if data[data.index == site].size != 0])

        train_metadata = pd.concat([meta_data[meta_data.index == site] for site in proc_train_sites if meta_data[meta_data.index == site].size != 0])
        test_metadata = pd.concat([meta_data[meta_data.index == site] for site in proc_test_sites if meta_data[meta_data.index == site].size != 0])

        train_sensor, test_sensor, train_gpp, test_gpp = prepare_df(train_data, test_data)
        x_train = [x.values for x in train_sensor]
        y_train = [x.values.reshape(-1,1) for x in train_gpp]

        x_test = [x.values for x in test_sensor]
        y_test = [x.values.reshape(-1,1) for x in test_gpp]


        # Preprocess the in/out test sets
        _, test_in_sensor, _, test_in_gpp = prepare_df(train_data, test_in_data)
        _, test_out_sensor, _, test_out_gpp = prepare_df(train_data, test_out_data)
        x_test_in = [x.values for x in test_in_sensor]
        y_test_in = [x.values.reshape(-1,1) for x in test_in_gpp]
        x_test_out = [x.values for x in test_out_sensor]
        y_test_out = [x.values.reshape(-1,1) for x in test_out_gpp]

        # Init model
        model = Model(INPUT_FEATURES, CONDITIONAL_FEATURES, HIDDEN_DIM, False, 1).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters())

        best_r2 = 0
        bias_in = 0
        bias_out = 0
        for epoch in tqdm(range(args.n_epochs)):
            
            train_loss = 0.0
            train_r2 = 0.0

            model.train()
            train_dataset = list(zip(x_train, y_train))
            shuffle(train_dataset)
            print("> Training")
            for (x, y) in train_dataset:
                x = torch.FloatTensor(x).to(DEVICE)
                y = torch.FloatTensor(y).to(DEVICE)
                c = None
                y_pred = model(x, None)
                optimizer.zero_grad()
                loss = F.mse_loss(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
            
            model.eval()
            with torch.no_grad():
                r2 = 0
                test_dataset = list(zip(x_test, y_test))
                print("> Test")
                for (x, y) in test_dataset:
                    x = torch.FloatTensor(x).to(DEVICE)
                    y = torch.FloatTensor(y).to(DEVICE)
                    y_pred = model(x, None)
                    test_loss = F.mse_loss( y_pred, y)
                    r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                r2 /= len(test_dataset)

                if r2 >= best_r2:
                    best_r2 = r2
                    bias_in = compute_bias(model, x_test_in, y_test_in, DEVICE)
                    bias_out = compute_bias(model, x_test_out, y_test_out, DEVICE)
        
        bias_in_all.append(bias_in)
        bias_out_all.append(bias_out)
    
    np.save(f"in_bias_leave_{args.group_name}.npy", np.concatenate(bias_in_all).reshape(-1))
    np.save(f"out_bias_leave_{args.group_name}.npy", np.concatenate(bias_out_all).reshape(-1))
