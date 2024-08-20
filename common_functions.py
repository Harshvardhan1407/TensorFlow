import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def add_lags(dff):
    try:
        target_map = dff['consumed_unit'].to_dict()
        # 15 minutes, 30 minutes, 1 hour
        dff['lag1'] = (dff.index - pd.Timedelta('15 minutes')).map(target_map)
        dff['lag2'] = (dff.index - pd.Timedelta('30 minutes')).map(target_map)
        dff['lag3'] = (dff.index - pd.Timedelta('1 day')).map(target_map)
        dff['lag4'] = (dff.index - pd.Timedelta('7 days')).map(target_map)
        # df['lag5'] = (df.index - pd.Timedelta('15 days')).map(target_map)
        # df['lag6'] = (df.index - pd.Timedelta('30 days')).map(target_map)
        # df['lag7'] = (df.index - pd.Timedelta('45 days')).map(target_map)
        return dff
    
    except KeyError as e:
        print(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def create_features(hourly_data):
    hourly_data = hourly_data.copy()

    # Check if the index is in datetime format
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        hourly_data.index = pd.to_datetime(hourly_data.index)

    hourly_data['day'] = hourly_data.index.day
    hourly_data['hour'] = hourly_data.index.hour
    hourly_data['month'] = hourly_data.index.month
    hourly_data['dayofweek'] = hourly_data.index.dayofweek
    # hourly_data['quarter'] = hourly_data.index.quarter
    hourly_data['dayofyear'] = hourly_data.index.dayofyear
    # hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
    # hourly_data['year'] = hourly_data.index.year
    return hourly_data

def data_validation_filtering(dfs):
    try:
        """ validation and filtering """
        # frequency and pf
        condition = (
            (dfs['Frequency'] > 51) | (dfs['Frequency'] < 49)  |
            (dfs['R_PF'] > 1) | (dfs['Y_PF'] > 1) | (dfs['B_PF'] > 1) |  # Check if any PF is greater than 1
            (dfs['R_PF'] < 0) | (dfs['Y_PF'] < 0) | (dfs['B_PF'] < 0)    # Check if any PF is less than 0
        )   
        # Apply the condition to set 'KWh' to NaN
        dfs.loc[condition, 'kWh'] = np.nan
        dfs.drop(['R_PF','Y_PF','B_PF','Frequency'],axis= 1 ,inplace= True)
        # voltage
        no_voltage_df = dfs[(dfs['R_Voltage'] == 0) & (dfs['Y_Voltage'] == 0) & (dfs['B_Voltage'] == 0)]
        if not no_voltage_df.empty:
            no_voltage_but_current = no_voltage_df[(no_voltage_df['R_Current'] != 0) & (no_voltage_df['B_Current'] != 0) & (no_voltage_df['Y_Current'] != 0)]

            if not no_voltage_but_current.empty:
                dfs.loc[no_voltage_but_current.index, 'kWh'] = np.nan
        # current
        no_current_df = dfs[(dfs['R_Current'] == 0) & (dfs['Y_Current'] == 0) & (dfs['B_Current'] == 0)]
        if not no_current_df.empty:
            load_with_no_current_df = no_current_df[(no_current_df['Load_kW']>0.03) & (no_current_df['Load_kVA']>0.03)]
            
            if not load_with_no_current_df.empty:
                    dfs.loc[load_with_no_current_df.index, 'kWh'] = np.nan
        dfs.drop(['R_Voltage','Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current','B_Current','Load_kW','Load_kVA'],axis= 1 ,inplace= True)
        return dfs
    except Exception as e:
        print("error in validation :",e,e.args())

def kWh_validation(clean_df):
    try:
        clean_df.loc[clean_df['kWh'] == 0, "kWh"] = np.nan
        clean_df.loc[clean_df['kWh'].first_valid_index():]
        clean_df.bfill(inplace=True)

        # missing packet
        sensor_df = clean_df.resample(rule="1H").bfill()
        if sensor_df.isna().sum().sum() !=0:
            sensor_df.interpolate(method="linear", inplace=True)

        # previous value of opening_KWh
        sensor_df['prev_KWh'] = sensor_df['kWh'].shift(1)
        sensor_df.dropna(inplace=True)

        if not sensor_df[sensor_df['prev_KWh'] > sensor_df['kWh']].empty:
            print("prev kwh > kwh")

        # consumed unit
        sensor_df['consumed_unit'] = sensor_df['kWh'] - sensor_df['prev_KWh']
        epsilon = 11
        min_samples = 3
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        sensor_df['db_outlier'] = dbscan.fit_predict(sensor_df[['consumed_unit']])
        sensor_df.loc[sensor_df['db_outlier'] == -1, 'consumed_unit'] = np.nan
        sensor_df.bfill(inplace=True)
        sensor_df.drop(['kWh','prev_KWh','db_outlier'],axis=1, inplace= True)
        # sensor_df.reset_index(inplace=True)

        sensor_df_with_lags = add_lags(sensor_df)
        final_df = create_features(sensor_df_with_lags)
        final_df.fillna(0,inplace= True)
        final_df.reset_index(inplace=True,drop=True)
        return final_df
    except Exception as e:
         print("error in kWh validation:",e,e.args)