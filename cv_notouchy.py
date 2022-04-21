#This is the final LSTM model with leave-one-site-out cross-validation
#standardising new

from model.model import Model
from preprocess_new import prepare_df
from sklearn.metrics import r2_score
import torch
import pandas as pd
import argparse
import torch.nn.functional as F
import numpy as np
from plotly import graph_objects as go
import operator



# Parse arguments 
parser = argparse.ArgumentParser(description='CV LSTM')

parser.add_argument('-gpu', '--gpu', default='cuda:0' ,type=str,
                      help='indices of GPU to enable ')

parser.add_argument('-e', '--n_epochs', default=150, type=int,
                      help='number of cv epochs ()')

parser.add_argument('-c', '--conditional',  type=int,
                      help='enable conditioning')

args = parser.parse_args()
DEVICE = args.gpu
torch.manual_seed(40)

#importing data
data = pd.read_csv('./utils/df_imputed.csv', index_col=0)
data = data.drop(columns='date')
raw = pd.read_csv('./data/df_20210510.csv', index_col=0)['GPP_NT_VUT_REF']
raw = raw[raw.index != 'CN-Cng']

sites = raw.index.unique()

INPUT_FEATURES = 11
HIDDEN_DIM = 256
CONDITIONAL_FEATURES = 21
masks = []
for s in sites:
    mask = raw[raw.index == s].isna().values
    masks.append(list(map(operator.not_, mask)))

cv_r2 = []
sites_out = []
cv_pred = [[] for s in range(len(sites))]

for s in range(len(sites)):
    #remove the site for testing
    sites_to_train_list = list(range(len(sites)))
    sites_to_train_list.remove(s)
    sites_to_train=sites[sites_to_train_list]
    site_to_test=sites[s]
    
    #Prepare the metadata
    meta_data = pd.get_dummies(data[['classid','igbp_land_use']])
    df_meta_all = [meta_data[meta_data.index==site] for site in sites if meta_data[meta_data.index == site].size != 0]
    df_meta_test=df_meta_all[s]
    df_meta_all.pop(s)
    df_meta=df_meta_all
    
    #Prepare and standardise the sensor data
    df_train=[data[data.index ==site] for site in sites_to_train]
    df_train=pd.concat(df_train)
    df_test=data[data.index ==site_to_test]
    
    df_sensor, df_sensor_test, df_gpp, df_gpp_test=prepare_df(df_train,df_test)

    #Prepare dataframe for training
    x_train = [df_sensor[i].values for i in range(len(sites)-1)]
    conditional_train = [df_meta[i].values for i in range(len(sites)-1)]
    y_train = [df_gpp[i].values.reshape(-1,1) for i in range(len(sites)-1)]

    x_test = df_sensor_test[0].values 
    conditional_test = df_meta_test.values
    y_test = df_gpp_test[0].values.reshape(-1,1)  
    
    #import the model
    model = Model(INPUT_FEATURES, CONDITIONAL_FEATURES, HIDDEN_DIM, args.conditional, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    r2 = []
    for epoch in range(args.n_epochs):
        print(f"Epoch: {epoch+1}/{args.n_epochs}")
        train_loss = 0.0
        train_r2 = 0.0
        model.train()
        for (x, y, conditional) in zip(x_train, y_train, conditional_train):
            x = torch.FloatTensor(x).to(DEVICE)
            y = torch.FloatTensor(y).to(DEVICE)
            c = torch.FloatTensor(conditional).to(DEVICE)
            y_pred = model(x, c)
            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_r2 += r2_score(y_true=y.detach().cpu().numpy(), y_pred=y_pred.detach().cpu().numpy())
        
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x_test).to(DEVICE)
            y = torch.FloatTensor(y_test).to(DEVICE)
            c = torch.FloatTensor(conditional_test).to(DEVICE)
            y_pred = model(x, c)
            test_loss = F.mse_loss(y_pred, y)
            test_r2 = r2_score(y_true=y.detach().cpu().numpy()[masks[s]], y_pred=y_pred.detach().cpu().numpy()[masks[s]])
            r2.append(test_r2)
            if test_r2 >= max(r2):
                cv_pred[s] = y_pred.detach().cpu().numpy().flatten().tolist()
    
    cv_r2.append(max(r2))
    sites_out.append(sites[s])
    print(f"Test Site: {sites[s]} R2: {cv_r2[s]}")
    print("CV R2 cumulative mean: ", np.mean(cv_r2), " +- ", np.std(cv_r2))
    print("-------------------------------------------------------------------")
    
    
#save the dataframe of the prediction   
d = {"Site": sites_out, "Predictions": cv_pred}
df = pd.DataFrame(d)
df.to_csv(f"notouchy_epochs_{args.n_epochs}_conditional_{args.conditional}.csv")

# normal={"mean":train_mean, "std":train_std}
# out= pd.DataFrame(normal)
# out.to_csv("denormalising tool_notouchy.csv")    
