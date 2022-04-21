import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize(df,df_test):
    #Normalising the dataframe by features
    result = df.copy()
    result_test=df_test.copy()
    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - df[feature_name].mean()) / df[feature_name].std()
        result_test[feature_name]=(df_test[feature_name]- df[feature_name].mean())/ df[feature_name].std()
    return result,result_test

def prepare_df(data, data_test, meta_columns=['classid','igbp_land_use']):
    #Group the data by sites and seprerate time dependented non-time dependented and the target variable
    # Site Data
    sites = data.index.unique()
    sensor_data = data.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    df_gpp = data['GPP_NT_VUT_REF']
    
    sites_test = data_test.index.unique()
    sensor_data_test = data_test.drop(columns=['plant_functional_type', 'classid', 'koeppen_code','igbp_land_use', 'GPP_NT_VUT_REF'])
    df_gpp_test = data_test['GPP_NT_VUT_REF']
    
    #Standardising
    df_sensor, df_sensor_test = normalize(sensor_data, sensor_data_test)
    
    # Batch by site
    df_sensor = [df_sensor[df_sensor.index==site] for site in sites if sensor_data[sensor_data.index == site].size != 0 ]
    df_gpp = [df_gpp[df_gpp.index==site] for site in sites]
    
    df_sensor_test = [df_sensor_test[df_sensor_test.index==site] for site in sites_test if sensor_data_test[sensor_data_test.index == site].size != 0 ]
    df_gpp_test = [df_gpp_test[df_gpp_test.index==site] for site in sites_test]

    return df_sensor, df_sensor_test, df_gpp, df_gpp_test