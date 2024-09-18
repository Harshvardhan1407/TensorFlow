import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import logger

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

def data_ingstion():
    try:
        dfs = pd.read_parquet(r"D:\RND\Machine_Learning\Datasets/clean_dataset_ml_50_id.parquet")
        dfs.fillna(0,inplace=True)
        df = dfs.copy()
        df['creation_time'] = pd.to_datetime(df['creation_time'])
        df.set_index(['creation_time'],drop= True, inplace= True)
        # df["min"] = df.index.minute
        df.sort_index(inplace=True)
        df.reset_index(drop=True,inplace=True)
        logger.info(f"columns in df : {tuple(df.columns)}")
        logger.info(f"shape of dataframe : {tuple(df.shape)}")
        return df
    except Exception as e:
        logger.error(f"error in data ingestion:{e}",exc_info=True)

def correlation_matrix(df):
    try:
        correlation_matrix = df.corr()
        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))
        # Draw the heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
        # Show the plot
        plt.show()
    
    except Exception as e:
        logger.error(f"error in correlation_plot: {e}",exc_info=True)

def tensor_conversion(dataframe):
    try:
        dataframe_tensor = tf.convert_to_tensor(dataframe, dtype=tf.float32)
        return dataframe_tensor
    except Exception as e:
        logger.error(f"error in tensor_conversion: {e}",exc_info=True)
# train test split
def train_test_split(df):
    try:
        # column_indices = {name: i for i, name in enumerate(df.columns)}
        n = len(df)
        train_df = df[0:int(n*0.9)]
        # val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]
        train_features, test_features = train_df.copy(),test_df.copy() #,val_df.copy()
        train_labels = train_features.pop('consumed_unit')
        # val_labels = val_features.pop('consumed_unit')
        test_labels = test_features.pop('consumed_unit')
        train_features = tensor_conversion(train_features)
        test_features = tensor_conversion(test_features)
        train_labels = tensor_conversion(train_labels)
        test_labels = tensor_conversion(test_labels)
        print(f"train_features shape:{train_features.shape},train_label shape: {train_labels.shape}")
        # print(f"val_features shape:{val_features.shape},val_label shape: {val_labels.shape},val_labels type: {type(val_labels)}")
        print(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
        logger.info(f"data split done")
        logger.info(f"train_features shape:{train_features.shape},train_label shape: {train_labels.shape}")
        # logger.info(f"val_features shape:{val_features.shape},val_label shape: {val_labels.shape},val_labels type: {type(val_labels)}")
        logger.info(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
        return train_features,test_features,train_labels, test_labels
    except Exception as e:
        logger.error(f"error in split: {e}",exc_info=True)

def normalizer_function(df):
    try:
        # df_mean, df_std = df.mean(), df.std()
        # df = (df - df_mean) / df_std
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(df))
        return normalizer
    
    except Exception as e:
        logger.error(f"error in normalizer: {e}",exc_info=True)

def compile_and_fit(model,
                    train_features = None,
                    train_labels = None, 
                    patience=5,
                    batch_size = None,
                    MAX_EPOCHS = 20,
                    learning_rate = 0.0001,                    
                    ):
    try:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(learning_rate),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

        history = model.fit(train_features,train_labels, epochs=MAX_EPOCHS,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=1,
                            )
        return model, history
    except Exception as e:
        logger.info(f"error in compiling and fitting: {e}",exc_info=True)

def training_score(history):
    print(f"training_score:loss:{history.history['loss'][-1]:.3f},mae:{history.history['mean_absolute_error'][-1]:.3f}")

def evaluate(model,features,labels,verbose=0):
    score = model.evaluate(features,labels,verbose=verbose)
    print("test_score:",score)
    return score

def prediction(model, input_data,input_label=None,verbose=1):
    y_pred = model.predict(input_data,verbose)
    # Calculate MAE, MSE, and R²
    mae = mean_absolute_error(input_label, y_pred)
    mse = mean_squared_error(input_label, y_pred)
    r2 = r2_score(input_label, y_pred)
    print(f"mae: {mae:.3f}")
    print(f"mse: {mse:.3f}")
    print(f"R2: {r2:.3f}")
    return y_pred

def predict_func(model, input_data, input_label=None, verbose=1):
    # Predict using the model
    y_pred = model.predict(input_data, verbose=verbose)
    
    # Convert the predictions and input labels to TensorFlow tensors
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
    
    # Calculate MAE using TensorFlow
    mae_fn = tf.keras.losses.MeanAbsoluteError()
    mae = mae_fn(input_label, y_pred).numpy()
    
    # Calculate MSE using TensorFlow
    mse_fn = tf.keras.losses.MeanSquaredError()
    mse = mse_fn(input_label, y_pred).numpy()
    
    # Calculate R² score manually
    total_error = tf.reduce_sum(tf.square(input_label - tf.reduce_mean(input_label)))
    unexplained_error = tf.reduce_sum(tf.square(input_label - y_pred))
    r2 = 1 - tf.divide(unexplained_error, total_error).numpy()
    return mae, mse, r2

# Example usage
# mae, mse, r2 = prediction(model, input_data, input_label)
# print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
