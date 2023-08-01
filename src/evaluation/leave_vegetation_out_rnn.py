import sys
sys.path.append("..")
from model.rnn_model import Model
from preprocess import prepare_df
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
import os
import csv

torch.manual_seed(40)

ALL_IDX = list(range(0, 53))
INPUT_FEATURES = 11
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = 21
DBF = [17, 23, 26, 30, 34, 43, 44, 50, 51]
ENF = [0, 13, 19, 22, 24, 25, 27, 31, 35, 36, 40, 41, 42, 45]
GRA = [1, 5, 10, 12, 16, 20, 32, 37, 39, 47, 52]
MF = [8, 9, 11, 38, 46]
def compute_bias(model, x_test, y_test, device):
    bias = []
    for (x, y) in zip(x_test, y_test):
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        y_pred = model(x, None)
        bias.append((y_pred - y).detach().cpu().numpy().squeeze().tolist())
    
    return bias

if __name__ == '__main__':
    # Parse arguments 
    parser = argparse.ArgumentParser(description='CV LSTM')

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--n_epochs', default=1, type=int, help='number of cv epochs')
    parser.add_argument('--conditional',  type=int, default=0, help='enable conditioning')
    parser.add_argument('--group_name', type=str)

    args = parser.parse_args()

    TRAIN_IDX = None
    if args.group_name == "DBF":
        TRAIN_IDX = DBF
    elif args.group_name == "ENF":
        TRAIN_IDX = ENF
    elif args.group_name == "GRA":
        TRAIN_IDX = GRA
    elif args.group_name == "MF":
        TRAIN_IDX = MF
    assert TRAIN_IDX is not None, "Please select a group."

    # Debug printing
    print("Starting leave-vegetation-out on RNN model:")
    print(f"> Device: {args.device}")
    print(f"> Epochs: {args.n_epochs}")
    print(f"> Condition on metadata: {args.conditional}")
    print(f"> Vegetation to train on (tested on the rest): {args.group_name}")
    
    TEST_IDX = list(set(ALL_IDX) - set(TRAIN_IDX))
    DEVICE = args.device
     
    #Importing data
    data = pd.read_csv('../data/df_imputed.csv', index_col=0)
    dates = data.groupby('sitename')['date'].apply(list)
    data = data.drop(columns='date')
    sites = data.index.unique().values
    DBF = sites[DBF]
    ENF = sites[ENF]
    GRA = sites[GRA]
    MF = sites[MF]
    raw = pd.read_csv('../data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
    raw = raw[raw.index != 'CN-Cng']

    #Prepare the metadata
    meta_data = pd.get_dummies(data[['classid','igbp_land_use']])
    masks = []
    for s in sites:
        mask = raw[raw.index == s].isna().values
        masks.append(list(map(operator.not_, mask)))

    cv_r2 = []
    bias_test_all = []
    bias_DBF_all = []
    bias_ENF_all = []
    bias_GRA_all = []
    bias_MF_all = []

    # Vegetation type data
    DBF_data = pd.concat([data[data.index == site] for site in DBF if data[data.index == site].size != 0])
    ENF_data = pd.concat([data[data.index == site] for site in ENF if data[data.index == site].size != 0])
    GRA_data = pd.concat([data[data.index == site] for site in GRA if data[data.index == site].size != 0])
    MF_data = pd.concat([data[data.index == site] for site in MF if data[data.index == site].size != 0])
    
    output_dir = f"leave_{args.group_name}_out"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for s in TRAIN_IDX:
        PROC_TRAIN_IDX = np.asarray(list(set(TRAIN_IDX) - set([s]))) # remove one site from the train set
        PROC_TEST_IDX = np.asarray([s]) # and add it to the test set
        proc_train_sites = sites[PROC_TRAIN_IDX]
        proc_test_sites = sites[PROC_TEST_IDX]

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
        _, DBF_sensor, _, DBF_gpp = prepare_df(train_data, DBF_data)
        _, ENF_sensor, _, ENF_gpp = prepare_df(train_data, ENF_data)
        _, GRA_sensor, _, GRA_gpp = prepare_df(train_data, GRA_data)
        _, MF_sensor, _, MF_gpp = prepare_df(train_data, MF_data)

        x_DBF = [x.values for x in DBF_sensor]
        y_DBF = [x.values.reshape(-1,1) for x in DBF_gpp]

        x_ENF = [x.values for x in ENF_sensor]
        y_ENF = [x.values.reshape(-1,1) for x in ENF_gpp]

        x_GRA = [x.values for x in GRA_sensor]
        y_GRA = [x.values.reshape(-1,1) for x in GRA_gpp]

        x_MF = [x.values for x in MF_sensor]
        y_MF = [x.values.reshape(-1,1) for x in MF_gpp]

        # Init model
        model = Model(INPUT_FEATURES, CONDITIONAL_FEATURES, HIDDEN_DIM, False, 1).to(DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

        best_r2 = 0
        bias_test = 0
        bias_DBF = 0
        bias_ENF = 0
        bias_GRA = 0
        bias_MF = 0

        for epoch in tqdm(range(args.n_epochs)):
            train_loss = 0.0
            train_r2 = 0.0
            model.train()
            train_dataset = list(zip(x_train, y_train))
            shuffle(train_dataset)
            preds = []
            gts = []
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
            print(f"Train loss: {train_loss}")
            model.eval()
            with torch.no_grad():
                r2 = 0
                test_dataset = list(zip(x_test, y_test))
                for (x, y) in test_dataset:
                    x = torch.FloatTensor(x).to(DEVICE)
                    y = torch.FloatTensor(y).to(DEVICE)
                    y_pred = model(x, None)
                    test_loss = F.mse_loss(y_pred, y)
                    score = r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
                    print(f"Loss: {test_loss}")
                    r2 += score
                r2 /= len(test_dataset)
                if r2 >= best_r2:
                    print(f'Found better at epoch {epoch}: {r2}')
                    best_r2 = r2
                    bias_test = compute_bias(model, x_test, y_test, DEVICE)
                    bias_DBF = compute_bias(model, x_DBF, y_DBF, DEVICE)
                    bias_ENF = compute_bias(model, x_ENF, y_ENF, DEVICE)
                    bias_GRA = compute_bias(model, x_GRA, y_GRA, DEVICE)
                    bias_MF = compute_bias(model, x_MF, y_MF, DEVICE)

        bias_DBF_dict = []
        for i, site in enumerate(DBF):
            assert len(dates[site]) == len(bias_DBF[i]), print(len(dates[site]), len(bias_DBF[i]))
            sitename_col = [site for _ in range(len(dates[site]))]
            bias_DBF_dict += list(zip(sitename_col, dates[site], bias_DBF[i]))

        bias_ENF_dict = []
        for i, site in enumerate(ENF):
            assert len(dates[site]) == len(bias_ENF[i]), print(len(dates[site]), len(bias_ENF[i]))
            sitename_col = [site for _ in range(len(dates[site]))]
            bias_ENF_dict += list(zip(sitename_col, dates[site], bias_ENF[i]))

        bias_GRA_dict = []
        for i, site in enumerate(GRA):
            assert len(dates[site]) == len(bias_GRA[i]), print(len(dates[site]), len(bias_GRA[i]))
            sitename_col = [site for _ in range(len(dates[site]))]
            bias_GRA_dict += list(zip(sitename_col, dates[site], bias_GRA[i]))

        bias_MF_dict = []
        for i, site in enumerate(MF):
            assert len(dates[site]) == len(bias_MF[i]), print(len(dates[site]), len(bias_MF[i]))
            sitename_col = [site for _ in range(len(dates[site]))]
            bias_MF_dict += list(zip(sitename_col, dates[site], bias_MF[i]))

        bias_test_dict = []
        for i, site in enumerate(proc_test_sites):
            assert len(dates[site]) == len(bias_test[i]), print(len(dates[site]), len(bias_test[i]))
            sitename_col = [site for _ in range(len(dates[site]))]
            bias_test_dict += list(zip(sitename_col, dates[site], bias_test[i]))

        with open(f"{output_dir}/{sites[s]}_DBF_bias.csv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(bias_DBF_dict)

        with open(f"{output_dir}/{sites[s]}_ENF_bias.csv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(bias_ENF_dict)

        with open(f"{output_dir}/{sites[s]}_GRA_bias.csv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(bias_GRA_dict)

        with open(f"{output_dir}/{sites[s]}_MF_bias.csv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(bias_MF_dict)

        with open(f"{output_dir}/{sites[s]}_test_bias.csv", "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(bias_test_dict)