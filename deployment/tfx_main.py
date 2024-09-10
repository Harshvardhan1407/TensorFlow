import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
print("tf version:",tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
np.set_printoptions(precision=3, suppress=True)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import logger
logger.info(f"tf version:{tf.__version__}")

# data ingestion 
def data_ingstion():
    try:
        dfs = pd.read_parquet(r"D:\RND\Machine_Learning\Datasets/clean_dataset_ml_100_id.parquet")
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
        logger.error(f"error in correlation_plot: {e}")

# train test split
def train_test_split(df):
    try:
        # column_indices = {name: i for i, name in enumerate(df.columns)}
        n = len(df)
        train_df = df[0:int(n*0.9)]
        # val_df = df[int(n*0.7):int(n*0.9)]
        test_df = df[int(n*0.9):]
        # print(f"train_df:,{len(train_df)},val_df:,{len(val_df)},test_df:{len(test_df)}")
        print(f"train_df:,{len(train_df)},test_df:{len(test_df)}")
        train_features, test_features = train_df.copy(),test_df.copy() #,val_df.copy()
        train_labels = train_features.pop('consumed_unit')
        # val_labels = val_features.pop('consumed_unit')
        test_labels = test_features.pop('consumed_unit')
        logger.info(f"data split done")
        logger.info(f"traning data shape:{train_features.shape},test data shape: {test_features.shape}")
        return train_features,test_features,train_labels, test_labels
    except Exception as e:
        logger.error(f"error in split: {e}")

# tensor conversion
def tensor_conversion(df):
    try:
        tf_dataset = tf.data.Dataset.from_tensor_slices(df)
        return tf_dataset
    except Exception as e:
        logger.error(f"error in tensor_conversion :{e}")

# feature engineering
# normalization
def normalizer_function(df):
    try:
        df_mean, df_std = df.mean(), df.std()
        # df = (df - df_mean) / df_std
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(df))
        return normalizer
    
    except Exception as e:
        logger.error(f"error in normalizer: {e}")

def compile_and_fit(model,
                    train_features = train_features,
                    train_labels = train_labels, 
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
        logger.info(f"error in compiling and fitting: {e}")
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


df = data_ingstion()
train_features,val_features,test_features,train_labels,val_labels,test_labels = train_test_split(df)
print(train_features.shape,val_features.shape,test_features.shape)
normalizer = normalizer_function(train_features)
model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu',input_shape=(17,)),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1)])

model, history= compile_and_fit(model,batch_size=16)


training_score(history)
score = evaluate(model, val_features,val_labels)
predicted_value = prediction(model,val_features,input_label=val_labels)