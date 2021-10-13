import pandas as pd
import numpy as np


def normalize(df):
    #Normalising the dataframe by features
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
    return result


def prepare_df(data, meta_columns=['classid','igbp_land_use']):
    #Group the data by sites and seprerate time dependented non-time dependented and the target variable
    # Site Data
    sites = data.index.unique()
    meta_data = pd.get_dummies(data[meta_columns])
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    
    # Batch by site
    df_sensor = [normalize(sensor_data[sensor_data.index == site]) for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
    df_meta = [meta_data[meta_data.index == site] for site in sites if meta_data[meta_data.index == site].size != 0]
    df_gpp = [data[data.index == site]['GPP_NT_VUT_REF'] for site in sites if data[data.index == site].size != 0]   
    df_gpp = [(df_gpp[i]-df_gpp[i].mean())/df_gpp[i].std() for i in range(len(df_gpp))]
        

    return df_sensor, df_meta, df_gpp


EUROPE = [8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
35,36,37,39,40]
US = [42,43,44,45,46,47,48,49,50,51,52]

def prepare_df_lgocv(data, meta_columns=['classid','igbp_land_use']):
    #prepare the leave-group-out-cross-validation dataframe
    
    # Site Data
    sites = data.index.unique()
    meta_data = pd.get_dummies(data[meta_columns])
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])

    # Batch by site
    df_sensor = [normalize(sensor_data[sensor_data.index == site]) for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
    df_meta = [meta_data[meta_data.index == site] for site in sites if meta_data[meta_data.index == site].size != 0]
    df_gpp = [data[data.index == site]['GPP_NT_VUT_REF'] for site in sites if data[data.index == site].size != 0]   
    df_gpp = [(df_gpp[i]-df_gpp[i].mean())/df_gpp[i].std() for i in range(len(df_gpp))]

    df_sensor_europe = np.asarray(df_sensor, dtype=object)[EUROPE]
    df_sensor_us = np.asarray(df_sensor, dtype=object)[US]                    

    df_meta_europe = np.asarray(df_meta, dtype=object)[EUROPE]
    df_meta_us = np.asarray(df_meta, dtype=object)[US]    
    
    df_gpp_europe = np.asarray(df_gpp,  dtype=object)[EUROPE]
    df_gpp_us = np.asarray(df_gpp,  dtype=object)[US]    

    return df_sensor_europe,df_sensor_us, df_meta_europe, df_meta_us, df_gpp_europe,df_gpp_us


