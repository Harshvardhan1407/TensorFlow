import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logger import logger
from tabulate import tabulate
import requests
from sklearn.cluster import DBSCAN
import holidays
from datetime import timedelta, datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
logger.info(f"tf version:{tf.__version__}")
from tensorflow import keras
from tensorflow.keras import layers                         # type: ignore
from tensorflow.keras.models import Sequential              # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout    # type: ignore
from tensorflow.keras import regularizers                   # type: ignore
from sklearn.inspection import permutation_importance
np.set_printoptions(precision=3, suppress=True)


class common():
    def __init__(self) -> None:
        pass
    
    # def holidays_list(self,start_date_str, end_date_str):
    #     logger.info("Generating holidays list")
    #     try:
    #         start_date = start_date_str.date()
    #         end_date = end_date_str.date()
    #         holiday_list = []
    #         # Get the holiday dates in India for the specified year
    #         india_holidays = holidays.CountryHoliday('India', years=start_date.year)
    #         # Iterate through each date from start_date to end_date
    #         current_date = start_date
    #         while current_date <= end_date:
    #             # Check if the current date is a holiday in India or a Sunday
    #             if current_date in india_holidays or current_date.weekday() == 6:
    #                 holiday_list.append(current_date)
    #             current_date += timedelta(days=1)
    #         return holiday_list

    #     except Exception as e:
    #         logger.error(f"Error in holidays_list: {e}")
    #         return None

    def add_lags(self, dff, target_col, large_data= False):
        try:
            # target_map = dff['consumed_unit'].to_dict()
            target_map = dff[target_col].to_dict()

            # 1 Hour, 2 Hours, 6 Hours
            dff['lag1_hour']   =  (dff.index - pd.Timedelta('1 hour')).map(target_map)
            dff['lag2_hours']  =  (dff.index - pd.Timedelta('2 hours')).map(target_map)
            dff['lag3_hours']  =  (dff.index - pd.Timedelta('3 hours')).map(target_map)
            dff['lag6_hours']  =  (dff.index - pd.Timedelta('6 hours')).map(target_map)
            dff['lag12_hours'] =  (dff.index - pd.Timedelta('12 hours')).map(target_map)
            dff['lag1_day']    =  (dff.index - pd.Timedelta('1 day')).map(target_map)
            dff['lag1_week']   =  (dff.index - pd.Timedelta('7 days')).map(target_map)

            if large_data:
                dff['lag_15_day'] = (dff.index - pd.Timedelta('15 days')).map(target_map)
                dff['lag_30_day'] = (dff.index - pd.Timedelta('30 days')).map(target_map)
                dff['lag_45_day'] = (dff.index - pd.Timedelta('45 days')).map(target_map)
                logger.info(f" lags added for large data")
                return dff
            
            dff['lag2_days'] = (dff.index - pd.Timedelta('2 days')).map(target_map)
            dff['lag3_days'] = (dff.index - pd.Timedelta('3 days')).map(target_map)

            logger.info(f"lags added")
            return dff
        
        except KeyError as e:
            logger.error(f"Error: {e}. 'consumed_unit' column not found in the DataFrame.",exc_info=True)
        except Exception as ex:
            logger.error(f"An unexpected error occurred: {ex}",exc_info= True)


    def create_features(self,hourly_data):
        try:
            hourly_data = hourly_data.copy()
            # Check if the index is in datetime format
            if not isinstance(hourly_data.index, pd.DatetimeIndex):
                hourly_data.index = pd.to_datetime(hourly_data.index)
            
            hourly_data['hour'] = hourly_data.index.hour
            hourly_data['day'] = hourly_data.index.day
            hourly_data['dayofweek'] = hourly_data.index.dayofweek
            hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
            hourly_data['hour'] = hourly_data.index.hour
            # hourly_data['month'] = hourly_data.index.month
            hourly_data['dayofweek'] = hourly_data.index.dayofweek
            # hourly_data['quarter'] = hourly_data.index.quarter
            hourly_data['dayofyear'] = hourly_data.index.dayofyear
            hourly_data['is_weekend'] = hourly_data['dayofweek'].isin([5, 6]).astype(int)
            hourly_data['holiday'] = 0
            # hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
            # hourly_data['year'] = hourly_data.index.year
            return hourly_data

        except Exception as e:
            logger.info(f"error in feature creation: {e}",exc_info= True)

    def pca_function(self,data,n_components=0.95,col_to_drop= None):
        try:
            if col_to_drop is not None:
                data.drop(col_to_drop, axis=1, inplace= True)
            data_for_pca = data.copy()
            # Step 1: Standardize the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_for_pca)
            # Step 2: Apply PCA
            pca = PCA(n_components)  # Retain 95% of the variance
            pca_result = pca.fit_transform(data_scaled)
            # Step 3: Explained variance ratio (how much variance is captured by each component)
            explained_variance = pca.explained_variance_ratio_
            # Print the results
            logger.info(f"Explained variance by each principal component: {explained_variance}")
            logger.info(f"Transformed data shape (after PCA): {pca_result.shape}")
            # You can also convert the PCA result back into a DataFrame
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
            # To see how much variance each component explains
            logger.info(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")
            logger.info(f"pca done")
            return pca_df ,pca_result
        except Exception as e:
            logger.error(f"error in pca:{e}",exc_info=True)

    def pca_plot(self,pca_result):
        try:
            # Visualize the first two principal components
            plt.figure(figsize=(8,6))
            plt.scatter(pca_result[:,0], pca_result[:,1], c='blue', edgecolor='k', s=50)
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('PCA on Meter Data')
            plt.show()
        except Exception as e:
            logger.error(f"error in pca plotting: {e}")
    
    def holidays_list(self,start_date, end_date):
        logger.info("Generating holidays list")
        try:
            holiday_list = []
            india_holidays = holidays.CountryHoliday('India', years=start_date.year)
            # Iterate through each date from start_date to end_date
            current_date = start_date
            while current_date <= end_date:
                # Check if the current date is a holiday in India or a Sunday
                if current_date in india_holidays or current_date.weekday() == 6:
                    holiday_list.append(current_date)
                current_date += timedelta(days=1)

            return holiday_list
        except Exception as e:
            logger.error(f"Error in holidays_list: {e}",exc_info=True)
            return None
        
    def data_split(self, dataset, target_variable= "Load_kW"): 
        """
            Splits the given dataset into training and testing sets, separating features and the target variable.

            Args:
                dataset (pandas.DataFrame): The dataset containing both features and the target variable.
                target_variable (str): The name of the column representing the target variable to be predicted. Default is 'Load_kW'.

            Returns:
                X_train (pandas.DataFrame): Training set features.
                X_test (pandas.DataFrame): Testing set features.
                y_train (pandas.Series): Training set target variable.
                y_test (pandas.Series): Testing set target variable.
            
            Raises:
                Exception: If there's an issue during the data split, an error message is logged.

            Example:
                >>> df = pd.DataFrame({...})
                >>> X_train, X_test, y_train, y_test = obj.data_split(df, target_variable="Load_kW")
        """
        try:
            dataset_features = dataset.copy()
            dataset_label = dataset_features.pop(target_variable)
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(dataset_features, dataset_label, test_size=0.2, 
                                                                # random_state=42
                                                                )
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.info(f"error in data split : {e}",exc_info=True)

    def data_split_function(self,dataset, target_variable= "Load_kW"):
        """
            Splits the given dataset into training and testing sets, separating features and the target variable.

            Args:
                dataset (pandas.DataFrame): The dataset containing both features and the target variable.
                target_variable (str): The name of the column representing the target variable to be predicted. Default is 'Load_kW'.

            Returns:
                X_train (pandas.DataFrame): Training set features.
                X_test (pandas.DataFrame): Testing set features.
                y_train (pandas.Series): Training set target variable.
                y_test (pandas.Series): Testing set target variable.
            
            Raises:
                Exception: If there's an issue during the data split, an error message is logged.

            Example:
                >>> df = pd.DataFrame({...})
                >>> X_train, X_test, y_train, y_test = obj.data_split(df, target_variable="Load_kW")
        """
        try:
            n = len(dataset)
            train_df = dataset[0:int(n*0.9)]
            test_df = dataset[int(n*0.9):]
            train_dataset, test_dataset = train_df.copy(),test_df.copy() #,val_df.copy()
            # train_labels = train_features.pop(target_variable)
            # test_labels = test_features.pop(target_variable)
            
            # print(f"train_features shape:{train_features.shape},train_label shape: {train_labels.shape}")
            # print(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
            logger.info(f"data split done")
            # logger.info(f"train_features shape:{train_features.shape} ,train_label shape: {train_labels.shape}")
            # logger.info(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
            logger.info(f"train_features shape:{train_dataset.shape}")
            logger.info(f"test_features shape:{test_dataset.shape}")

            return train_dataset,test_dataset
            # return train_features,test_features,train_labels, test_labels
        
        except Exception as e:
            logger.error(f"error in split: {e}",exc_info=True)


    def model_trainer(self, train_dataset, target_variable, model_name=None):
        try:
            # dataset = train_fetures.merge(train_label, on="creation_time").copy()
            # print(dataset.columns)
            # Split the dataset into training and test sets
            train_features, test_features = self.data_split_function(dataset=train_dataset)
            train_labels = train_features.pop(target_variable)
            test_labels = test_features.pop(target_variable)
            
            if model_name =="RFR":
                # Step 4: Initialize the RandomForestRegressor model
                model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can tweak hyperparameters

            # Step 5: Train the model
            model.fit(train_features, train_labels)
            logger.info(f"model trained")
            
            self.prediction(model= model,
                            input_data= test_features,
                            scoring= True,
                            test_data= test_labels)
            # Save the model to a file
            joblib.dump(model, 'forecasting_model.pkl')

            return model
        except Exception as e:
            logger.error(f"error in model trainer: {e}",exc_info=True)

    def prediction(self,model, input_data, scoring=False, test_data= None,prediction= False):
        try:
            if prediction:
                input_data = self.scaling_layer(input_data)

            # Step 6: Make predictions
            y_pred = model.predict(input_data)
            if y_pred is not None:
                logger.info("predition_succesful")

            if scoring :
                # Step 7: Evaluate the model
                print("\nModel Power Evaluation")
                print("Mean Squared Error (MSE):", mean_squared_error(test_data, y_pred))
                print("Mean Absolute Error (MAE):", mean_absolute_error(test_data, y_pred))
                print("R-squared:", r2_score(test_data, y_pred))
            
                # Scatter plot
                plt.figure(figsize=(10, 6))
                # Plot actual values in red
                plt.plot(range(len(test_data)), test_data, color='red', label='Actual Power')
                # Plot predicted values in blue
                plt.plot(range(len(y_pred)), y_pred, color='blue', label='Predicted Power')
                # Adding labels and title
                plt.xlabel('Time')
                plt.ylabel('load')
                plt.title('Actual vs Predicted load')
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            return y_pred
        except Exception as e:
            logger.error(f"error in prediction: {e}",exc_info=True)

    def scaling_layer(self,datset):
        try:
            scaler = MinMaxScaler()
            features_to_normalize = datset.columns
            datset[features_to_normalize] = scaler.fit_transform(datset[features_to_normalize])
            return datset
        
        except Exception as e:
            logger.error(f"error in scaling: {e}",exc_info= True)    


    def correlation_matrix(self,df):
        try:
            correlation_matrix_data = df.corr()
            # Set up the matplotlib figure
            plt.figure(figsize=(18, 10))
            # Draw the heatmap
            sns.heatmap(correlation_matrix_data, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
            # Show the plot
            plt.show()
        
        except Exception as e:
            logger.error(f"error in correlation_plot: {e}",exc_info=True)

class NPCL():
    def __init__(self) -> None:
        pass

    def data_ingestion_npcl(self,data_path=None):
        df = pd.read_csv(data_path, parse_dates=['creation_time'])
        data = df.copy()
        logger.info(f"stats at starting: \n total data points: {len(data)}, \n no of location_id: {data['location_id'].nunique()}.")
        # data.drop(["status","id","error_reason"],axis=1,inplace= True)
        data.drop(['id'],axis= 1,inplace= True)    
        # data = data.loc[data['location_id']!= 100000011999]
        # data = data.loc[~data['location_id'].isin([100000011999, 100000190001, 100000030201, 100000011203])]
        null_values = data.isna().sum().sum()
        logger.info(f"null values: {null_values}")
        if null_values != 0:
            data.dropna(inplace= True)
            logger.info(f"dropped null values")
            logger.info(f"null values: {data.isna().sum().sum()}") 
        duplicate_values = data.duplicated().sum()
        logger.info(f"duplicate values: {duplicate_values}")
        if duplicate_values != 0:
            data.drop_duplicates(inplace = True)
            logger.info(f"duplicate values dropped")
            logger.info(f"duplicate values: {data.duplicated().sum()}")
        
        # Dictionary to rename columns
        rename_dict = {
            'R_voltage': 'R_Voltage',
            'Y_volatge': 'Y_Voltage',
            'B_voltage': 'B_Voltage',
            'r_current': 'R_Current',
            'y_current': 'Y_Current',
            'b_current': 'B_Current',
            'r_pf': 'R_PF',
            'y_pf': 'Y_PF',
            'b_pf': 'B_PF',
            'instant_cum_Kw': 'Load_kW',
            'instant_cum_Kva': 'Load_KVA',
            'grid_reading_kwh': 'grid_reading_kwh',  
            'creation_time': 'creation_time'
        }
        # Rename columns
        data.rename(columns=rename_dict, inplace=True)
        data = data[['location_id', 'creation_time', 'frequency', 'R_Voltage', 'Y_Voltage', 'B_Voltage',
                'R_Current', 'Y_Current', 'B_Current', 'R_PF', 'Y_PF', 'B_PF', 'Load_kW', 'Load_KVA',
                'grid_reading_kwh','grid_reading_kvah'
                ,"error_reason"
                ]]
        data.sort_values("creation_time", inplace= True)
        data.reset_index(drop=True,inplace=True)
        logger.info(f"stats after: \n total data points: {len(data)}, \n no of location_id: {data['location_id'].nunique()}.")
        # data.sort_index(inplace=True)
        logger.info("#############--data ingestion done--".ljust(180,"#"))
        return data

    def data_filter_condition(self,data= None):
        try:
            # Combining all conditions into one
            logger.info(f"length in starting: {len(data)}")
            combined_condition = (
                # Frequency should be between 49 and 51
                (data['frequency'] !=0) & ((data['frequency'] > 51) | (data['frequency'] < 49))  |

                # Any PF outside the range [-1, 1]
                (data[['R_PF', 'Y_PF', 'B_PF']].abs() > 1).any(axis=1) |

                # All voltages are zero and either Load kW is non-zero or all currents are non-zero
                ((data[['R_Voltage', 'Y_Voltage', 'B_Voltage']].eq(0).all(axis=1)) & ((data['Load_kW'] != 0) | 
                (data[['R_Current', 'Y_Current', 'B_Current']].ne(0).all(axis=1)))) |

                # All currents are zero and either Load kW or Load KVA is greater than 0.03
                ((data[['R_Current', 'Y_Current', 'B_Current']].eq(0).all(axis=1)) & ((data['Load_kW'] > 0.03) | (data['Load_KVA'] > 0.03))) |

                # Load kW cannot be greater than Load KVA
                (data['Load_kW'] > data['Load_KVA']) |
                
                # load zero where current consumption
                (data['Load_kW'] ==0)&
                ((data['R_Current']!=0) |  (data['Y_Current']!=0) | (data['B_Current']!=0))
            )
            # Applying the combined condition
            data = data.loc[~combined_condition]
            logger.info(f"length in ending: {len(data)}")
            data.reset_index(drop=True,inplace=True)
            summary = data.iloc[:, 2:14].describe()
            # summary = data.iloc[:,:].describe()
            logger.info(f"after cleaning dataframe stats")
            logger.info(f"\n {tabulate(summary.round(2), headers='keys', tablefmt='pretty')}")
            logger.info("#############--data filtering done--".ljust(180,"#"))
            return data
        except Exception as e:
            logger.info(f"error in filter condition: {e}",exc_info=True)

    def data_cleaning_and_validation(self,df=None):
        try:
            logger.info(f"date_time data type: {df['creation_time'].dtype}")
            if df['creation_time'].dtype not in ['<M8[ns]',"datetime64[ns]"]:
                df['creation_time'] = pd.to_datetime(df['creation_time'])
            df[['R_PF', 'Y_PF', 'B_PF']] = df[['R_PF', 'Y_PF', 'B_PF']].abs()
            for col in df.columns:
                if df[col].dtype == object:
                    logger.info(f"columns with categorical values: {col}")
            summary = df.iloc[:, 2:14].describe()
            # summary = df.iloc[:,:].describe()
            logger.info(f"before cleaning dataframe stats :")
            logger.info(f" \n {tabulate(summary.round(2), headers='keys', tablefmt='pretty')}")
            df = self.data_filter_condition(df)
            columns_to_drop = ['frequency',"Load_KVA","grid_reading_kvah","grid_reading_kwh","error_reason"]
            df.drop(columns_to_drop,axis=1, inplace= True)
            logger.info(f"dropped columns: {columns_to_drop}")
            logger.info(f"columns after cleaning: {np.array(df.columns)}")
            logger.info("#############--data cleaning and validation done--".ljust(180,"#"))
            return df
        except Exception as e:
            logger.info(f"error in data cleaing and filtering: {e}",exc_info= True)

    def weather_data_api(self, latitude, longitude, from_date, to_date, duration="hour"):
        try:
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={from_date}&end_date={to_date}&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,wind_speed_10m,wind_speed_100m"
            logger.info(f"weather api url : {url}")
            response = requests.get(url)
            response.raise_for_status()
            weather_data = response.json()
        
            for i in range(len(weather_data['hourly']['time'])):
                            hour_data = {
                # "_id": f"{site_data['_id']}_{weather_data['hourly']['time'][i]}",  # MongoDB's unique identifier
                # "site_id": site_data["_id"],
                "time": weather_data['hourly']['time'][i],
                "temperature_2m": weather_data['hourly'].get('temperature_2m', [])[i],
                "relative_humidity_2m": weather_data['hourly'].get('relative_humidity_2m', [])[i],
                "apparent_temperature": weather_data['hourly'].get('apparent_temperature', [])[i],
                "precipitation": weather_data['hourly'].get('precipitation', [])[i],
                "wind_speed_10m": weather_data['hourly'].get('wind_speed_10m', [])[i],
                "wind_speed_100m": weather_data['hourly'].get('wind_speed_100m', [])[i],
                "creation_time_iso": datetime.utcfromtimestamp(
                    datetime.strptime(weather_data['hourly']['time'][i],
                                        '%Y-%m-%dT%H:%M').timestamp()).isoformat()
            }
            weather_df = pd.DataFrame(weather_data['hourly'])
            weather_df['time'] = pd.to_datetime(weather_df['time'])
            weather_df.rename(columns={"time":"creation_time"}, inplace=True)
            weather_df.set_index("creation_time",inplace=True,drop=True)
            logger.info(f"weather data:{len(weather_df)} ")

            if duration != "hour":
                resampled_df = weather_df.resample(rule=duration).ffill()
                resampled_df.reset_index(inplace=True)
                return resampled_df
            logger.info(f"weather data done")
            return weather_df
                
        except Exception as e:
                logger.info(f"error in weather data: {e}",exc_info=True)

######################################################################################################################################


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

# def kWh_validation(clean_df):
#     try:
#         clean_df.loc[clean_df['kWh'] == 0, "kWh"] = np.nan
#         clean_df.loc[clean_df['kWh'].first_valid_index():]
#         clean_df.bfill(inplace=True)

#         # missing packet
#         sensor_df = clean_df.resample(rule="1H").bfill()
#         if sensor_df.isna().sum().sum() !=0:
#             sensor_df.interpolate(method="linear", inplace=True)

#         # previous value of opening_KWh
#         sensor_df['prev_KWh'] = sensor_df['kWh'].shift(1)
#         sensor_df.dropna(inplace=True)

#         if not sensor_df[sensor_df['prev_KWh'] > sensor_df['kWh']].empty:
#             print("prev kwh > kwh")

#         # consumed unit
#         sensor_df['consumed_unit'] = sensor_df['kWh'] - sensor_df['prev_KWh']
#         epsilon = 11
#         min_samples = 3
#         dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#         sensor_df['db_outlier'] = dbscan.fit_predict(sensor_df[['consumed_unit']])
#         sensor_df.loc[sensor_df['db_outlier'] == -1, 'consumed_unit'] = np.nan
#         sensor_df.bfill(inplace=True)
#         sensor_df.drop(['kWh','prev_KWh','db_outlier'],axis=1, inplace= True)
#         # sensor_df.reset_index(inplace=True)

#         sensor_df_with_lags = add_lags(sensor_df)
#         final_df = create_features(sensor_df_with_lags)
#         final_df.fillna(0,inplace= True)
#         final_df.reset_index(inplace=True,drop=True)
#         return final_df
#     except Exception as e:
#          print("error in kWh validation:",e,e.args)

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


def tensor_conversion(dataframe):
    try:
        dataframe_tensor = tf.convert_to_tensor(dataframe, dtype=tf.float32)
        return dataframe_tensor
    except Exception as e:
        logger.error(f"error in tensor_conversion: {e}",exc_info=True)

# train test split
# def train_test_split(df):
#     try:
#         # column_indices = {name: i for i, name in enumerate(df.columns)}
#         n = len(df)
#         train_df = df[0:int(n*0.9)]
#         # val_df = df[int(n*0.7):int(n*0.9)]
#         test_df = df[int(n*0.9):]
#         train_features, test_features = train_df.copy(),test_df.copy() #,val_df.copy()
#         train_labels = train_features.pop('consumed_unit')
#         # val_labels = val_features.pop('consumed_unit')
#         test_labels = test_features.pop('consumed_unit')
#         train_features = tensor_conversion(train_features)
#         test_features = tensor_conversion(test_features)
#         train_labels = tensor_conversion(train_labels)
#         test_labels = tensor_conversion(test_labels)
#         print(f"train_features shape:{train_features.shape},train_label shape: {train_labels.shape}")
#         # print(f"val_features shape:{val_features.shape},val_label shape: {val_labels.shape},val_labels type: {type(val_labels)}")
#         print(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
#         logger.info(f"data split done")
#         logger.info(f"train_features shape:{train_features.shape},train_label shape: {train_labels.shape}")
#         # logger.info(f"val_features shape:{val_features.shape},val_label shape: {val_labels.shape},val_labels type: {type(val_labels)}")
#         logger.info(f"test_features shape:{test_features.shape} ,test_label shape: {test_labels.shape}")
#         return train_features,test_features,train_labels, test_labels
#     except Exception as e:
#         logger.error(f"error in split: {e}",exc_info=True)

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
