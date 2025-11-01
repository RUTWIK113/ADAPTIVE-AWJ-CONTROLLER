# In control/ann_model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import os

# Use the modern .keras format
MODEL_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'awj_model.keras')


def build_ann_model():
    """
    Builds the 4-5-1 ANN architecture.
    """
    model = Sequential(name="AWJ_Depth_Predictor")
    model.add(InputLayer(shape=(4,), name="Inputs"))
    model.add(Dense(5, activation='relu', name="Hidden_Layer"))
    model.add(Dense(1, activation='linear', name="Output_Depth"))
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("ANN Model Summary:")
    model.summary()
    return model