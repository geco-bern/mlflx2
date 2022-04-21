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

torch.manual_seed(40)

ALL_IDX = list(range(0, 53))
INPUT_FEATURES = 11
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = 21
DBF = [17, 23, 26, 30, 34, 43, 44, 50, 51]
ENF = [ 0, 13, 19, 22, 24, 25, 27, 31, 35, 36, 40, 41, 42, 45]
GRA = [ 1,  5, 10, 12, 16, 20, 32, 37, 39, 47, 52]
MF = [ 8,  9, 11, 38, 46]
LAMBDA = 10
def compute_bias(model, x_test, y_test, device):
    bias = []
    for (x, y) in zip(x_test, y_test):
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        y_pred = model(x, None) * LAMBDA
        bias.append((y_pred - y).detach().cpu().numpy())
    
    return np.concatenate(bias)

if __name__ == '__main__':
    # Parse arguments 
    parser = argparse.ArgumentParser(description='CV LSTM')

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--n_epochs', default=50, type=int, help='number of cv epochs')
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
    print(f"Training on: {args.group_name}")
    TEST_IDX = list(set(ALL_IDX) - set(TRAIN_IDX))
    DEVICE = args.device
     
    #Importing data
    data = pd.read_csv('../data/df_imputed.csv', index_col=0)
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

    for s in TRAIN_IDX:
        #PROC_TRAIN_IDX = np.asarray([s]) # remove one site from the train set
        PROC_TRAIN_IDX = np.asarray(list(set(TRAIN_IDX) - set([s]))) # remove one site from the train set
        PROC_TEST_IDX = np.asarray([s]) # and add it to the test set
        proc_train_sites = sites[PROC_TRAIN_IDX]
        proc_test_sites = sites[PROC_TEST_IDX]

        # Get the train/test data based on the processed sites
        train_data = pd.concat([data[data.index == site] for site in proc_train_sites if data[data.index == site].size != 0])
        test_data = pd.concat([data[data.index == site] for site in proc_test_sites if data[data.index == site].size != 0])

        train_metadata = pd.concat([meta_data[meta_data.index == site] for site in proc_train_sites if meta_data[meta_data.index == site].size != 0])
        test_metadata = pd.concat([meta_data[meta_data.index == site] for site in proc_test_sites if meta_data[meta_data.index == site].size != 0])

        train_sensor, test_sensor, train_gpp, test_gpp, normalized_train_gpp, means, stds = prepare_df(train_data, test_data)
        x_train = [x.values for x in train_sensor]
        y_train = [x.values.reshape(-1,1) for x in train_gpp]
        normalized_y_train = [x.values.reshape(-1,1) for x in normalized_train_gpp]

        x_test = [x.values for x in test_sensor]
        y_test = [x.values.reshape(-1,1) for x in test_gpp]

        # Preprocess the in/out test sets
        _, DBF_sensor, _, DBF_gpp, _, _, _ = prepare_df(train_data, DBF_data)
        _, ENF_sensor, _, ENF_gpp, _, _, _ = prepare_df(train_data, ENF_data)
        _, GRA_sensor, _, GRA_gpp, _, _, _ = prepare_df(train_data, GRA_data)
        _, MF_sensor, _, MF_gpp, _, _, _ = prepare_df(train_data, MF_data)

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
                y = torch.FloatTensor(y).to(DEVICE) / LAMBDA
                c = None
                y_pred = model(x, None)
                optimizer.zero_grad()
                loss = F.mse_loss(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy() * LAMBDA)
            print(f"Train loss: {train_loss}")
            model.eval()
            with torch.no_grad():
                r2 = 0
                test_dataset = list(zip(x_test, y_test))
                for (x, y) in test_dataset:
                    x = torch.FloatTensor(x).to(DEVICE)
                    y = torch.FloatTensor(y).to(DEVICE)
                    y_pred = model(x, None) * LAMBDA
                    test_loss = F.mse_loss(y_pred, y)
                    score = r2_score(y_true=y.detach().cpu().numpy()[masks[s]], y_pred=y_pred.detach().cpu().numpy()[masks[s]])
                    print(f"Loss: {test_loss}")
                    r2 += score
                r2 /= len(test_dataset)
                print(f'R2 at epoch {epoch}: {r2}')
                if r2 >= best_r2:
                    print(f'Found better at epoch {epoch}: {r2}')
                    best_r2 = r2
                    bias_test = compute_bias(model, x_test, y_test, DEVICE)
                    bias_DBF = compute_bias(model, x_DBF, y_DBF, DEVICE)
                    bias_ENF = compute_bias(model, x_ENF, y_ENF, DEVICE)
                    bias_GRA = compute_bias(model, x_GRA, y_GRA, DEVICE)
                    bias_MF = compute_bias(model, x_MF, y_MF, DEVICE)
        bias_test_all.append(bias_test)
        bias_DBF_all.append(bias_DBF)
        bias_ENF_all.append(bias_ENF)
        bias_GRA_all.append(bias_GRA)
        bias_MF_all.append(bias_MF)
    
    np.save(f"train_{args.group_name}_bias_DBF.npy", np.concatenate(bias_DBF_all).reshape(-1))
    np.save(f"train_{args.group_name}_bias_ENF.npy", np.concatenate(bias_ENF_all).reshape(-1))
    np.save(f"train_{args.group_name}_bias_GRA.npy", np.concatenate(bias_GRA_all).reshape(-1))
    np.save(f"train_{args.group_name}_bias_MF.npy", np.concatenate(bias_MF_all).reshape(-1))
    np.save(f"train_{args.group_name}_bias_test.npy", np.concatenate(bias_test_all).reshape(-1))
