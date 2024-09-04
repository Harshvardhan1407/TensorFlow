import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
np.set_printoptions(precision=3, suppress=True)

# data ingestion 
dfs = pd.read_parquet(r"D:\RND\Machine_Learning\Datasets/clean_dataset_ml_50_id.parquet")
dfs.fillna(0,inplace=True)
df = dfs.copy()
df['creation_time'] = pd.to_datetime(df['creation_time'])
df.set_index(['creation_time'],drop= True, inplace= True)
df["min"] = df.index.minute
df.sort_index(inplace=True)
df.reset_index(drop=True,inplace=True)

# feature engineering
# normalization
df_mean = df.mean()
df_std = df.std()
df = (df - df_mean) / df_std

# train test split
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.8)]
# val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.8):]
print("train_df:",len(train_df),"test_df:",len(test_df))
num_features = df.shape[1]
train_features = train_df.copy()
test_features = test_df.copy()
train_labels = train_features.pop('consumed_unit')
test_labels = test_features.pop('consumed_unit')

linear_model = tf.keras.Sequential([
    layers.Dense(units=1)
])
test_results = {}