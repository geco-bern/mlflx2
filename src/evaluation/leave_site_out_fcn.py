#This is the Fully Connected Model with leave-one-site-out cross-validation

from model.fcn_model import Model
from preprocess import prepare_df
from sklearn.metrics import r2_score
import torch
import pandas as pd
import argparse
import torch.nn.functional as F
import numpy as np
import operator
import pickle 

torch.manual_seed(40)
INPUT_FEATURES = 11

if __name__ == '__main__':
    # Parse arguments 
    parser = argparse.ArgumentParser(description='CV DNN') 
    parser.add_argument('--device', default='cuda:0', type=str,
                          help='indices of GPU to enable ')
    parser.add_argument('-e', '--n_epochs', default=150, type=int,
                          help='number of cv epochs ()')


    args = parser.parse_args()
    DEVICE = args.device

    #importing data
    data = pd.read_csv('../data/df_imputed.csv', index_col=0)
    data = data.drop(columns='date')
    raw = pd.read_csv('../data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
    raw = raw[raw.index != 'CN-Cng']
    sites = raw.index.unique()

    masks = []
    for s in sites:
        mask = raw[raw.index == s].isna().values
        masks.append(list(map(operator.not_, mask)))

    cv_r2 = []
    cv_pred = []
    for s in range(len(sites)):
        sites_to_train_list = list(range(len(sites)))
        sites_to_train_list.remove(s)
        sites_to_train=sites[sites_to_train_list]
        site_to_test=sites[s]
        
        #Prepare and standardise the sensor data
        df_train=[data[data.index == site] for site in sites_to_train]
        df_train=pd.concat(df_train)
        df_test=data[data.index ==site_to_test]

        df_sensor, df_sensor_test, df_gpp, df_gpp_test = prepare_df(df_train,df_test)
        
        #Prepare dataframe for training
        x_train = [df_sensor[i].values for i in range(len(sites)-1)]
        y_train = [df_gpp[i].values.reshape(-1,1) for i in range(len(sites)-1)]

        x_test = df_sensor_test[0].values 
        y_test = df_gpp_test[0].values.reshape(-1,1) 

        #import the model
        model = Model(INPUT_FEATURES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters())

        r2 = []
        pred = []
        
        for epoch in range(args.n_epochs):
            train_loss = 0.0
            train_r2 = 0.0
            model.train()
            for (x, y) in zip(x_train, y_train):
                x = torch.FloatTensor(x).to(DEVICE)
                y = torch.FloatTensor(y).to(DEVICE)
                y_pred = model(x)
                optimizer.zero_grad()
                loss = F.mse_loss( y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
            
            model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(x_test).to(DEVICE)
                y = torch.FloatTensor(y_test).to(DEVICE)
                y_pred = model(x)
                test_loss = F.mse_loss( y_pred, y)
                test_r2 = r2_score(y_true=y.detach().cpu().numpy()[masks[s]], y_pred=y_pred.detach().cpu().numpy()[masks[s]])
                r2.append(test_r2)
                pred.append(y_pred)
        
        cv_r2.append(max(r2))
        cv_pred.append(pred[np.argmax(r2)].detach().cpu().numpy())
        print(f"Test Site: {s} R2: {cv_r2[s]}")
        print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
        print("-------------------------------------------------------------------")
        
    #save the prediction dataframe
    d = {"Site": sites, "Predictions": cv_pred}
    df = pd.DataFrame(d)
    df.to_csv(f"notime_epochs_{args.n_epochs}.csv")


