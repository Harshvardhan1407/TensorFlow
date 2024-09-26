import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

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
from common_functions import data_ingstion, train_test_split, normalizer_function, compile_and_fit, training_score, evaluate, predict_func, tensor_convertor

BATCH_SIZE = 32
EPOCHS=50

df = data_ingstion()
train_features,test_features,train_labels,test_labels = train_test_split(df)
normalizer = normalizer_function(train_features)


# train_features_array = train_features.to_numpy()
# test_features_array = test_features.to_numpy()
n_timesteps = 1  # Each sample has 1 timestep (since features are already lagged)
n_features = (16,)  # Number of features

# X_train = train_features_array.reshape((train_features.shape[0], n_timesteps, n_features))
# X_test = test_features_array.reshape((test_features.shape[0], n_timesteps, n_features))

# Define LSTM model
model = Sequential([
    layers.LSTM(64, activation='relu', input_shape=(n_timesteps,n_features), return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),  # Dropout to reduce overfitting
    layers.LSTM(32, activation='relu', return_sequences=False, kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# model, history= compile_and_fit(model,batch_size=16)



# # model = keras.Sequential([
# #       normalizer,
# #       layers.Dense(64, activation='relu',input_shape=(17,)),
# #       layers.Dense(64, activation='relu'),
# #       layers.Dense(32, activation='relu'),
# #       layers.Dense(16, activation='relu'),
# #       layers.Dense(1)])



# # training_score(history)
# # score = evaluate(model, val_features,val_labels)
# # predicted_value = prediction(model,val_features,input_label=val_labels)